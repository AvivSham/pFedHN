from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn


def quantize_tensor(
    tensor: torch.Tensor,
    nbits: int = 4,
    block_size: int = 64,
    residual: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    tuple[int, ...],
    torch.Tensor | None,
]:
    """Quantize a float tensor using block-wise symmetric or affine uniform quantization with sub-byte bit packing.

    Parameters:
        tensor: Floating-point input tensor.
        nbits: Quantization bit width (e.g. 4 for INT4 or 8 for INT8).
        block_size: Number of elements per independent scaling block.
        residual: Optional residual error accumulation tensor from previous round (error-feedback).

    Returns:
        packed: Packed quantized tensor (torch.uint8).
        scales: Per-block float32 scaling factors.
        zeros: Per-block float32 zero points (None for symmetric 4-bit).
        orig_shape: Original shape tuple of the tensor.
        new_residual: Updated error-feedback residual tensor.
    """
    orig_shape = tensor.shape
    orig_device = tensor.device
    orig_dtype = tensor.dtype

    work_tensor = tensor.detach().clone().to(torch.float32)
    if residual is not None and residual.shape == orig_shape:
        work_tensor = work_tensor + residual.to(device=orig_device, dtype=torch.float32)

    flat = work_tensor.reshape(-1)
    numel = flat.numel()

    if numel == 0:
        empty_packed = torch.empty((0,), dtype=torch.uint8, device=orig_device)
        empty_scales = torch.empty((0,), dtype=torch.float32, device=orig_device)
        empty_res = torch.zeros(orig_shape, dtype=orig_dtype, device=orig_device)
        return empty_packed, empty_scales, None, orig_shape, empty_res

    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        padded = torch.nn.functional.pad(flat, (0, pad_len), mode="constant", value=0.0)
    else:
        padded = flat

    blocks = padded.view(-1, block_size)
    qmax = (1 << (nbits - 1)) - 1  # 7 for 4-bit signed symmetric
    qmin = -(1 << (nbits - 1))  # -8 for 4-bit signed symmetric

    max_val = torch.max(torch.abs(blocks), dim=1, keepdim=True)[0]
    scales = torch.clamp(max_val / float(qmax), min=1e-8)

    scaled_blocks = blocks / scales
    quantized_blocks = torch.clamp(torch.round(scaled_blocks), qmin, qmax)

    # Compute reconstructed values for error-feedback residual
    dequantized_padded = (quantized_blocks * scales).view(-1)
    dequantized_flat = dequantized_padded[:numel]
    new_residual = (flat - dequantized_flat).view(orig_shape).to(orig_dtype)

    # Offset to unsigned representation [0, 2^nbits - 1]
    unsigned_blocks = (quantized_blocks - qmin).to(torch.uint8)
    unsigned_flat = unsigned_blocks.view(-1)

    if nbits == 4:
        # Pack two 4-bit nibbles into one uint8: lower nibble in bits [3:0], higher nibble in bits [7:4]
        even_nibbles = unsigned_flat[0::2]
        odd_nibbles = unsigned_flat[1::2]
        packed = (even_nibbles & 0x0F) | ((odd_nibbles & 0x0F) << 4)
    elif nbits == 8:
        packed = unsigned_flat
    else:
        raise ValueError(f"Unsupported nbits: {nbits}. Supported: 4, 8.")

    return packed, scales.squeeze(-1), None, orig_shape, new_residual


def dequantize_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    orig_shape: tuple[int, ...] | None = None,
    nbits: int = 4,
    block_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize packed integers back into floating-point representation.

    Parameters:
        packed: Packed quantized tensor (torch.uint8).
        scales: Per-block scaling factors.
        zeros: Optional per-block zero points.
        orig_shape: Original shape tuple.
        nbits: Quantization bit width.
        block_size: Independent block size.
        dtype: Output float dtype.

    Returns:
        Dequantized tensor reshaped to orig_shape.
    """
    if packed.numel() == 0:
        if orig_shape is not None:
            return torch.zeros(orig_shape, dtype=dtype, device=packed.device)
        return torch.empty((0,), dtype=dtype, device=packed.device)

    qmin = -(1 << (nbits - 1))

    if nbits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        unsigned = torch.stack([low, high], dim=-1).view(-1)
    elif nbits == 8:
        unsigned = packed
    else:
        raise ValueError(f"Unsupported nbits: {nbits}. Supported: 4, 8.")

    signed = unsigned.to(torch.float32) + float(qmin)
    blocks = signed.view(-1, block_size)

    scale_view = scales.view(-1, 1)
    recon_blocks = blocks * scale_view

    recon_flat = recon_blocks.view(-1)

    if orig_shape is not None:
        numel = 1
        for dim in orig_shape:
            numel *= dim
        recon_flat = recon_flat[:numel]
        return recon_flat.view(orig_shape).to(dtype)

    return recon_flat.to(dtype)


def quantize_hypernetwork_weights(
    weights: dict[str, torch.Tensor],
    nbits: int = 4,
    block_size: int = 64,
    residuals: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, float]]:
    """Quantize hypernetwork-generated client model weights before downlink transmission.

    Parameters:
        weights: Dictionary of generated weights (from hnet(node_id)).
        nbits: Bit width (default 4).
        block_size: Block size (default 64).
        residuals: Optional residuals dict for error feedback.

    Returns:
        quantized_dict: Serialized dictionary with packed tensors and scales.
        new_residuals: Updated residuals dictionary.
        stats: Compression statistics dictionary.
    """
    quantized_dict: dict[str, Any] = {
        "nbits": nbits,
        "block_size": block_size,
        "layers": {},
    }
    new_residuals: dict[str, torch.Tensor] = {}

    total_orig_bytes = 0
    total_quant_bytes = 0

    if residuals is None:
        residuals = {}

    for name, param in weights.items():
        res = residuals.get(name, None)
        orig_bytes = param.numel() * param.element_size()
        total_orig_bytes += orig_bytes

        if not param.is_floating_point():
            quantized_dict["layers"][name] = {
                "type": "passthrough",
                "tensor": param.detach().clone(),
            }
            total_quant_bytes += orig_bytes
            continue

        packed, scales, zeros, orig_shape, new_res = quantize_tensor(
            param, nbits=nbits, block_size=block_size, residual=res
        )

        quant_bytes = (
            packed.numel() * packed.element_size()
            + scales.numel() * scales.element_size()
        )
        total_quant_bytes += quant_bytes

        quantized_dict["layers"][name] = {
            "type": "quantized",
            "packed": packed,
            "scales": scales,
            "zeros": zeros,
            "orig_shape": orig_shape,
            "dtype": str(param.dtype),
        }
        new_residuals[name] = new_res

    compression_ratio = float(total_orig_bytes) / max(total_quant_bytes, 1)
    stats = {
        "orig_bytes": total_orig_bytes,
        "quant_bytes": total_quant_bytes,
        "compression_ratio": compression_ratio,
        "bandwidth_savings_pct": (1.0 - (total_quant_bytes / max(total_orig_bytes, 1)))
        * 100.0,
    }

    return quantized_dict, new_residuals, stats


def dequantize_hypernetwork_weights(
    quantized_dict: dict[str, Any],
    target_net: nn.Module | None = None,
    device: torch.device | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Dequantize compressed hypernetwork payload into model weight tensors.

    Parameters:
        quantized_dict: Quantized payload dictionary.
        target_net: Optional target module to directly load weights into.
        device: Optional target device for tensors.

    Returns:
        OrderedDict of reconstructed weight tensors suitable for net.load_state_dict.
    """
    nbits = quantized_dict.get("nbits", 4)
    block_size = quantized_dict.get("block_size", 64)
    layers = quantized_dict.get("layers", {})

    dequantized_state: OrderedDict[str, torch.Tensor] = OrderedDict()

    dtype_map = {
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.float16": torch.float16,
    }

    for name, data in layers.items():
        if data.get("type") == "passthrough":
            tensor = data["tensor"]
            if device is not None:
                tensor = tensor.to(device)
            dequantized_state[name] = tensor
        else:
            packed = data["packed"]
            scales = data["scales"]
            zeros = data.get("zeros")
            orig_shape = data["orig_shape"]
            dtype = dtype_map.get(data.get("dtype", "torch.float32"), torch.float32)

            recon = dequantize_tensor(
                packed=packed,
                scales=scales,
                zeros=zeros,
                orig_shape=orig_shape,
                nbits=nbits,
                block_size=block_size,
                dtype=dtype,
            )
            if device is not None:
                recon = recon.to(device)
            dequantized_state[name] = recon

    if target_net is not None:
        target_net.load_state_dict(dequantized_state)

    return dequantized_state


def quantize_delta_theta(
    delta_theta: dict[str, torch.Tensor],
    nbits: int = 4,
    block_size: int = 64,
    residuals: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, float]]:
    """Quantize client parameter differences (delta theta) before uplink transmission to hypernetwork.

    Parameters:
        delta_theta: Parameter difference dictionary: theta_initial - theta_final.
        nbits: Quantization bit width (default 4).
        block_size: Block size (default 64).
        residuals: Residual accumulation dictionary.

    Returns:
        quantized_dict: Quantized delta payload.
        new_residuals: Updated residuals.
        stats: Compression statistics.
    """
    return quantize_hypernetwork_weights(
        weights=delta_theta,
        nbits=nbits,
        block_size=block_size,
        residuals=residuals,
    )


def dequantize_delta_theta(
    quantized_dict: dict[str, Any],
    device: torch.device | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Dequantize uplink delta theta for hypernetwork backpropagation via autograd.grad."""
    return dequantize_hypernetwork_weights(
        quantized_dict=quantized_dict,
        target_net=None,
        device=device,
    )


def calc_compression_stats(
    original_dict: dict[str, torch.Tensor],
    quantized_dict: dict[str, Any],
) -> dict[str, float]:
    """Calculate exact wire size savings between uncompressed tensors and quantized payload."""
    orig_bytes = sum(t.numel() * t.element_size() for t in original_dict.values())
    quant_bytes = 0

    layers = quantized_dict.get("layers", {})
    for data in layers.values():
        if data.get("type") == "passthrough":
            t = data["tensor"]
            quant_bytes += t.numel() * t.element_size()
        else:
            packed = data["packed"]
            scales = data["scales"]
            quant_bytes += packed.numel() * packed.element_size()
            quant_bytes += scales.numel() * scales.element_size()
            zeros = data.get("zeros")
            if zeros is not None:
                quant_bytes += zeros.numel() * zeros.element_size()

    ratio = float(orig_bytes) / max(quant_bytes, 1)
    savings = (1.0 - (quant_bytes / max(orig_bytes, 1))) * 100.0
    return {
        "orig_bytes": orig_bytes,
        "quant_bytes": quant_bytes,
        "compression_ratio": ratio,
        "bandwidth_savings_pct": savings,
    }
