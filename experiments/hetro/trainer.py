import argparse
import json
import logging
import os
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import wandb
from tqdm import trange

from experiments.hetro.models import CNNHyper, CNNTargetLook
from experiments.lookahead import Lookahead
from experiments.node import BaseNodes
from experiments.utils import (common_parser, get_device, set_logger, set_seed,
                               str2bool)


def eval_model(nodes, num_nodes, hnet, net, criteria, device, split):
    curr_results = evaluate(nodes, num_nodes, hnet, net, criteria, device, split=split)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_loss = np.mean([val['loss'] for val in curr_results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]

    return curr_results, avg_loss, avg_acc, all_acc


@torch.no_grad()
def evaluate(nodes: BaseNodes, num_nodes, hnet, net, criteria, device, split='test'):
    hnet.eval()
    results = defaultdict(lambda: defaultdict(list))

    for node_id in range(num_nodes):  # iterating over nodes

        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = nodes.test_loaders[node_id]
        elif split == 'val':
            curr_data = nodes.val_loaders[node_id]
        else:
            curr_data = nodes.train_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            img, label = tuple(t.to(device) for t in batch)

            weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
            net.load_state_dict(weights)
            pred = net(img)
            running_loss += criteria(pred, label).item()
            running_correct += pred.argmax(1).eq(label).sum().item()
            running_samples += len(label)

        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples

    return results


def train(data_name: str, data_path: str, num_nodes: int, steps: int, optim: str,
          lr: float, n_kernels: int, bs: int, device, eval_every: int, save_path: Path) -> None:

    # --------------------------
    # Datasets + Subsets Indexes
    # --------------------------
    nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=args.classes_per_user,
                      batch_size=bs)

    embed_dim = args.embed_dim
    if embed_dim == -1:
        logging.info("auto embedding size")
        embed_dim = int(1 + num_nodes / 4)

    if data_name == "cifar10":
        hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=args.hyper_hid, n_hidden=args.n_hidden, n_kernels=n_kernels)
        net = CNNTargetLook(n_kernels=n_kernels)
    elif data_name == "cifar100":
        hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=args.hyper_hid,
                        n_hidden=args.n_hidden, n_kernels=n_kernels, out_dim=100)
        net = CNNTargetLook(n_kernels=n_kernels, out_dim=100)
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100']")

    hnet = hnet.to(device)
    net = net.to(device)

    embed_lr = args.embed_lr if args.embed_lr is not None else lr
    optimizers = {
        'sgd': torch.optim.SGD(
            [
                {'params': [p for n, p in hnet.named_parameters() if 'embed' not in n]},
                {'params': [p for n, p in hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
            ], lr=lr, momentum=0.9, weight_decay=args.wd
        ),
        'adam': torch.optim.Adam(params=hnet.parameters(), lr=lr)
    }
    optimizer = optimizers[optim]
    criteria = torch.nn.CrossEntropyLoss()
    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(steps)

    results = defaultdict()
    for step in step_iter:
        hnet.train()
        node_id = random.choice(range(num_nodes))

        weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
        net.load_state_dict(weights)
        # look_optim = Lookahead(
        #     torch.optim.SGD(
        #         net.parameters(), lr=args.inner_lr, momentum=.9, weight_decay=args.la_wd
        #     ),
        #     la_steps=args.la_steps, la_alpha=args.la_lr
        # )

        inner_optim = torch.optim.SGD(
            net.parameters(), lr=args.inner_lr, momentum=.9, weight_decay=args.la_wd
        )

        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

        # state = defaultdict(dict)
        # # Cache the current optimizer parameters
        # for group in inner_optim.param_groups:
        #     for p in group['params']:
        #         param_state = state[p]
        #         param_state['cached_params'] = torch.zeros_like(p.data)
        #         param_state['cached_params'].copy_(p.data)

        # theta = [p.detach().clone() for p in net.parameters()]

        # NOTE: evaluation on sent model
        with torch.no_grad():
            net.eval()
            batch = next(iter(nodes.test_loaders[node_id]))
            img, label = tuple(t.to(device) for t in batch)
            pred = net(img)
            prvs_loss = criteria(pred, label)
            prvs_acc = pred.argmax(1).eq(label).sum().item() / len(label)
            net.train()

        for i in range(args.la_steps):
            net.train()
            inner_optim.zero_grad()
            optimizer.zero_grad()

            batch = next(iter(nodes.train_loaders[node_id]))
            img, label = tuple(t.to(device) for t in batch)

            pred = net(img)

            loss = criteria(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 50)

            inner_optim.step()

        optimizer.zero_grad()

        final_state = net.state_dict()

        delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

        # delta_theta = OrderedDict()
        # Lookahead and cache the current optimizer parameters
        # for group in inner_optim.param_groups:
        #     for p in group['params']:
        #         param_state = state[p]
        #
        #         # NOTE: update "grads"
        #         delta_theta[p] = param_state['cached_params'] - p.data

        hnet_grads = torch.autograd.grad(
            list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
        )
        # delta_theta = [p_i - p for p, p_i in zip(net.parameters(), theta) if p.requires_grad]
        # hnet_grads = torch.autograd.grad(
        #     list(weights.values()), hnet.parameters(), grad_outputs=delta_theta

        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g

        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer.step()

        step_iter.set_description(
            f"Step: {step+1}, Node ID: {node_id}, Loss: {prvs_loss:.4f},  Acc: {prvs_acc:.4f}"
        )

        if step % eval_every == 0:
            last_eval = step
            step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="test")
            logging.info(f"\nStep: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")
            results[step+1] = step_results

            if args.wandb:
                wandb.log(
                    {'test_avg_loss': avg_loss, 'test_avg_acc': avg_acc},
                    step=step
                )

            _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="val")
            if best_acc < val_avg_acc:
                best_acc = val_avg_acc
                best_step = step
                test_best_based_on_step = avg_acc
                test_best_min_based_on_step = np.min(all_acc)
                test_best_max_based_on_step = np.max(all_acc)
                test_best_std_based_on_step = np.std(all_acc)

            if args.wandb:
                wandb.log(
                    {
                        'val_avg_loss': val_avg_loss, 'val_avg_acc': val_avg_acc,
                        "best_step": best_step, "best_val_acc": best_acc,
                        'best_test_acc_based_on_val_beststep': test_best_based_on_step,
                        'test_best_min_based_on_step': test_best_min_based_on_step,
                        'test_best_max_based_on_step': test_best_max_based_on_step,
                        'test_best_std_based_on_step': test_best_std_based_on_step
                    },
                    step=step
                )

    if step != last_eval:
        _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="val")
        step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="test")
        logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")
        results[step + 1] = step_results
        if best_acc < val_avg_acc:
            best_acc = val_avg_acc
            best_step = step
            test_best_based_on_step = avg_acc
            test_best_min_based_on_step = np.min(all_acc)
            test_best_max_based_on_step = np.max(all_acc)
            test_best_std_based_on_step = np.std(all_acc)

        if args.wandb:
            wandb.log(
                {
                    'test_avg_loss': avg_loss, 'test_avg_acc': avg_acc,
                    "val_avg_loss": val_avg_loss, "val_avg_acc": val_avg_acc,
                    "best_step": best_step, "best_val_acc": best_acc,
                    'best_test_acc_based_on_val_beststep': test_best_based_on_step,
                    'test_best_min_based_on_step': test_best_min_based_on_step,
                    'test_best_max_based_on_step': test_best_max_based_on_step,
                    'test_best_std_based_on_step': test_best_std_based_on_step
                },
                step=step
            )

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(str(save_path / f"results_{args.la_steps}_la_steps_seed_{args.seed}.json"), "w") as file:
        json.dump(results, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Hypernetwork with Lookahead experiment", parents=[common_parser]
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="cifar10", choices=['cifar10', 'cifar100'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
    parser.add_argument("--classes_per_user", type=int, default=2, help="N classes assigned to each user")
    parser.add_argument("--num-nodes", type=int, default=20, help="number of simulated nodes")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--la-steps", type=int, default=50, help="lookahead steps")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    # parser.add_argument("--la-lr", type=float, default=1e-2, help="lookahead learning rate")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--la-wd", type=float, default=5e-5, help="lookahead weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=10, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="fhn_hetro", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    if args.wandb:
        name = f"lookahead_{args.data_name}_lr_{args.lr}_inlr_{args.inner_lr}_embedlr_{args.embed_lr}" \
               f"_la_steps_{args.la_steps}_seed_{args.seed}" \
               f"_num_nodes_{args.num_nodes}_kernels_{args.nkernels}"
        wandb.init(project="fhn", entity='ax2', name=name)
        wandb.config.update(args)

    train(
        data_name=args.data_name,
        data_path=args.data_path,
        num_nodes=args.num_nodes,
        steps=args.num_steps,
        optim=args.optim,
        lr=args.lr,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path
    )
