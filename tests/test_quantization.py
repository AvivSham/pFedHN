import unittest
from collections import OrderedDict

import torch

from experiments.pfedhn.models import CNNHyper, CNNTarget
from experiments.quantization import (
    calc_compression_stats,
    dequantize_delta_theta,
    dequantize_hypernetwork_weights,
    dequantize_tensor,
    quantize_delta_theta,
    quantize_hypernetwork_weights,
    quantize_tensor,
)


class TestQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_tensor_quantize_dequantize_roundtrip(self):
        tensor = torch.randn(16, 3, 5, 5, dtype=torch.float32)
        packed, scales, zeros, orig_shape, _residual = quantize_tensor(
            tensor, nbits=4, block_size=64
        )

        self.assertEqual(orig_shape, (16, 3, 5, 5))
        self.assertEqual(packed.dtype, torch.uint8)

        dequant = dequantize_tensor(
            packed, scales, zeros, orig_shape, nbits=4, block_size=64
        )
        self.assertEqual(dequant.shape, tensor.shape)
        self.assertEqual(dequant.dtype, torch.float32)

        # Cosine similarity between reconstructed and original should be high (> 0.95)
        cos_sim = torch.cosine_similarity(
            tensor.flatten(), dequant.flatten(), dim=0
        ).item()
        self.assertGreater(cos_sim, 0.95)

    def test_int4_subbyte_packing_exact_size(self):
        # 128 elements in float32 = 512 bytes
        tensor = torch.randn(128, dtype=torch.float32)
        packed, scales, _, _, _ = quantize_tensor(tensor, nbits=4, block_size=64)

        # Packed uint8 should contain 64 bytes (2 elements per byte)
        self.assertEqual(packed.numel(), 64)
        self.assertEqual(packed.element_size(), 1)
        self.assertEqual(scales.numel(), 2)  # 128 / 64 = 2 blocks

    def test_int8_quantization(self):
        tensor = torch.randn(100, dtype=torch.float32)
        packed, scales, zeros, orig_shape, _ = quantize_tensor(
            tensor, nbits=8, block_size=32
        )

        self.assertEqual(packed.dtype, torch.uint8)
        dequant = dequantize_tensor(
            packed, scales, zeros, orig_shape, nbits=8, block_size=32
        )
        cos_sim = torch.cosine_similarity(
            tensor.flatten(), dequant.flatten(), dim=0
        ).item()
        self.assertGreater(cos_sim, 0.99)

    def test_non_multiple_block_sizes(self):
        # Tensor size 77 (not multiple of block_size 32)
        tensor = torch.randn(77, dtype=torch.float32)
        packed, scales, zeros, orig_shape, _ = quantize_tensor(
            tensor, nbits=4, block_size=32
        )

        dequant = dequantize_tensor(
            packed, scales, zeros, orig_shape, nbits=4, block_size=32
        )
        self.assertEqual(dequant.shape, (77,))
        self.assertFalse(torch.isnan(dequant).any())

    def test_hypernetwork_weights_quantization(self):
        num_nodes = 5
        embed_dim = 16
        hnet = CNNHyper(n_nodes=num_nodes, embedding_dim=embed_dim, n_kernels=16)
        target_net = CNNTarget(n_kernels=16)

        node_id = torch.tensor([2], dtype=torch.long)
        weights = hnet(node_id)

        quant_dict, _residuals, stats = quantize_hypernetwork_weights(
            weights, nbits=4, block_size=64
        )

        self.assertIn("orig_bytes", stats)
        self.assertIn("quant_bytes", stats)
        self.assertGreater(stats["compression_ratio"], 5.0)
        self.assertGreater(stats["bandwidth_savings_pct"], 80.0)

        # Dequantize and load directly into target net
        dequant_weights = dequantize_hypernetwork_weights(
            quant_dict, target_net=target_net
        )
        self.assertEqual(set(dequant_weights.keys()), set(weights.keys()))

        # Run forward pass on target net with dequantized weights
        dummy_input = torch.randn(4, 3, 32, 32)
        output = target_net(dummy_input)
        self.assertEqual(output.shape, (4, 10))
        self.assertFalse(torch.isnan(output).any())

    def test_delta_theta_quantization_and_autograd(self):
        num_nodes = 5
        embed_dim = 16
        hnet = CNNHyper(n_nodes=num_nodes, embedding_dim=embed_dim, n_kernels=16)

        node_id = torch.tensor([1], dtype=torch.long)
        weights = hnet(node_id)

        # Simulated client delta theta
        delta_theta = OrderedDict(
            {k: torch.randn_like(v) * 0.01 for k, v in weights.items()}
        )

        quant_delta, _, _ = quantize_delta_theta(delta_theta, nbits=4, block_size=64)
        dequant_delta = dequantize_delta_theta(quant_delta)

        # Hypernetwork backward gradient calculation via autograd.grad
        hnet_grads = torch.autograd.grad(
            list(weights.values()),
            hnet.parameters(),
            grad_outputs=list(dequant_delta.values()),
        )

        self.assertEqual(len(hnet_grads), len(list(hnet.parameters())))
        for g in hnet_grads:
            self.assertIsNotNone(g)
            self.assertFalse(torch.isnan(g).any())

    def test_error_feedback_residuals(self):
        tensor = torch.randn(64, dtype=torch.float32)
        residuals = None

        # Simulate 3 rounds of residual tracking
        for _ in range(3):
            _packed, _scales, _zeros, _orig_shape, residuals = quantize_tensor(
                tensor, nbits=4, block_size=32, residual=residuals
            )
            self.assertIsNotNone(residuals)
            self.assertEqual(residuals.shape, tensor.shape)
            # Residual norm should remain bounded
            self.assertLess(torch.norm(residuals).item(), torch.norm(tensor).item())

    def test_empty_tensor(self):
        tensor = torch.empty((0,), dtype=torch.float32)
        packed, scales, zeros, orig_shape, _res = quantize_tensor(
            tensor, nbits=4, block_size=64
        )
        dequant = dequantize_tensor(
            packed, scales, zeros, orig_shape, nbits=4, block_size=64
        )
        self.assertEqual(dequant.numel(), 0)

    def test_calc_compression_stats(self):
        orig_dict = {"layer1": torch.randn(128, dtype=torch.float32)}
        quant_dict, _, _ = quantize_hypernetwork_weights(
            orig_dict, nbits=4, block_size=64
        )
        stats = calc_compression_stats(orig_dict, quant_dict)
        self.assertGreater(stats["compression_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
