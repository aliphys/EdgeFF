#!/usr/bin/env python3
"""Training entrypoint wrapper for basicRun."""

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader, random_split

ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from basicRun import (
    load_config,
    load_env,
    Net,
    overlay_y_on_x,
    overlay_on_x_neutral,
    eval_train_set,
    eval_test_set,
    eval_val_set,
    eval_with_inference_measurement,
    INA3221PowerMonitor,
    TegratsMonitor,
)
from basicRun.data import get_dataset


def resolve_config(config_path):
    candidate = Path(config_path)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    candidate = PACKAGE_ROOT / config_path
    if candidate.exists():
        return str(candidate)

    candidate = Path.cwd() / config_path
    if candidate.exists():
        return str(candidate)

    return str(config_path)


def dataset_loaders(dataset_name, train_batch_size, test_batch_size, val_size, seed):
    full_train, is_color = get_dataset(dataset_name, train=True)
    test_dataset, _ = get_dataset(dataset_name, train=False)

    if val_size <= 0 or val_size >= len(full_train):
        raise ValueError('val_size must be positive and smaller than the training dataset size')

    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, is_color


def main():
    parser = argparse.ArgumentParser(description='Train or evaluate a Forward-Forward model.')
    parser.add_argument('--config', type=str, default='configs/run.yaml', help='Path to run config YAML file')
    parser.add_argument('--project', type=str, default='edgeff-refactor', help='W&B project name')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity (username or team name)')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA even if available.')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--layers', type=str, default=None, help='Comma-separated layer dimensions')
    parser.add_argument('--rep-epochs', type=int, default=None, help='Number of representation training epochs')
    parser.add_argument('--softmax-epochs', type=int, default=None, help='Number of softmax training epochs')
    parser.add_argument('--train-batch-size', type=int, default=None, help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=None, help='Test/validation batch size')
    parser.add_argument('--val-size', type=int, default=None, help='Validation dataset size')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval during training')
    parser.add_argument('--final-train-sample', type=int, default=None, help='Number of final train samples to evaluate')
    parser.add_argument('--final-val-sample', type=int, default=None, help='Number of final validation samples to evaluate')
    parser.add_argument('--enable-hw-monitor', action='store_true', default=False, help='Enable hardware monitoring during evaluation')
    parser.add_argument('--hw-interval-ms', type=int, default=None, help='Hardware monitoring polling interval in milliseconds')
    args = parser.parse_args()

    load_env()
    config = load_config(args.config)
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        current_value = getattr(args, key)
        default_value = parser.get_default(key)
        if current_value == default_value:
            setattr(args, key, value)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using device: {device}')
    if use_cuda:
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
        print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

    layers = [int(x.strip()) for x in args.layers.split(',') if x.strip()]
    if not layers:
        raise ValueError('layers configuration must contain at least one dimension')

    expected_input_dim = 784 if args.dataset in ['MNIST', 'FMNIST'] else 3072
    if layers[0] != expected_input_dim:
        print(f'WARNING: First layer dimension ({layers[0]}) does not match dataset {args.dataset} (expected {expected_input_dim}).')
        layers[0] = expected_input_dim

    train_loader, val_loader, test_loader, is_color = dataset_loaders(
        dataset_name=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    model = Net(layers, device=device, onehot_max_value=10.0, is_color=is_color)

    wandb_run = wandb.init(
        project=args.project,
        settings=wandb.Settings(init_timeout=1800),
        job_type='train',
        config={
            'layers': layers,
            'representation_epochs': args.rep_epochs,
            'softmax_epochs': args.softmax_epochs,
            'train_batch_size': args.train_batch_size,
            'test_batch_size': args.test_batch_size,
            'val_size': args.val_size,
            'dataset': args.dataset,
            'seed': args.seed,
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': torch.cuda.get_device_name(0) if use_cuda else None,
        },
        tags=['forward-forward', args.dataset.lower()],
    )

    hw_monitor = None
    if args.enable_hw_monitor:
        try:
            hw_monitor = TegratsMonitor(interval_ms=args.hw_interval_ms)
            hw_monitor.start()
            print('Hardware monitoring enabled.')
        except Exception as exc:
            print(f'Hardware monitoring unavailable: {exc}')
            hw_monitor = None

    # Training loop
    for epoch in range(args.rep_epochs):
        total_train = len(train_loader.dataset)
        for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            x_pos_batch = overlay_y_on_x(batch_inputs, batch_targets, is_color=is_color)
            y_neg_batch = batch_targets.clone()
            for idx, y_samp in enumerate(batch_targets):
                allowed = list(range(10))
                allowed.remove(y_samp.item())
                y_neg_batch[idx] = torch.tensor(random.choice(allowed), device=device)
            x_neg_batch = overlay_y_on_x(batch_inputs, y_neg_batch, is_color=is_color)

            model.train(x_pos_batch, x_neg_batch)

            if batch_idx % args.log_interval == 0:
                processed = (batch_idx + 1) * batch_inputs.shape[0]
                pct = processed / total_train
                print(f'Rep Epoch: {epoch+1}/{args.rep_epochs} [{processed}/{total_train} ({pct:.0%})]')

        with torch.no_grad():
            val_subset_inputs, val_subset_targets = next(iter(val_loader))
            val_subset_inputs = val_subset_inputs.to(device)
            val_subset_targets = val_subset_targets.to(device)
            preds = model.predict_one_pass(val_subset_inputs, batch_size=val_subset_inputs.shape[0])
            sample_val_acc = (preds.cpu() == val_subset_targets.cpu()).float().mean().item()
        wandb_run.log({'epoch': epoch, 'phase': 'representation', 'sample_val_accuracy': sample_val_acc})

    for epoch in range(args.softmax_epochs):
        total_train = len(train_loader.dataset)
        for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            x_neutral_batch = overlay_on_x_neutral(batch_inputs, is_color=is_color)
            model.train_softmax_layer(x_neutral_batch, batch_targets, batch_inputs.shape[0], layers)

            if batch_idx % args.log_interval == 0:
                processed = (batch_idx + 1) * batch_inputs.shape[0]
                pct = processed / total_train
                print(f'Softmax Epoch: {epoch+1}/{args.softmax_epochs} [{processed}/{total_train} ({pct:.0%})]')

        with torch.no_grad():
            val_subset_inputs, val_subset_targets = next(iter(val_loader))
            val_subset_inputs = val_subset_inputs.to(device)
            val_subset_targets = val_subset_targets.to(device)
            preds = model.predict_one_pass(val_subset_inputs, batch_size=val_subset_inputs.shape[0])
            sample_val_acc = (preds.cpu() == val_subset_targets.cpu()).float().mean().item()
        wandb_run.log({'epoch': epoch, 'phase': 'softmax', 'sample_val_accuracy': sample_val_acc})

    # Save model
    model_dir = Path(__file__).resolve().parent / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'temp_'
    torch.save(model, model_path)

    # Evaluation
    train_inputs = []
    train_targets = []
    for batch_inputs, batch_targets in train_loader:
        train_inputs.append(batch_inputs)
        train_targets.append(batch_targets)
    train_inputs_full = torch.cat(train_inputs, dim=0).to(device)
    train_targets_full = torch.cat(train_targets, dim=0).to(device)

    test_inputs = []
    test_targets = []
    for batch_inputs, batch_targets in test_loader:
        test_inputs.append(batch_inputs)
        test_targets.append(batch_targets)
    test_inputs_full = torch.cat(test_inputs, dim=0).to(device)
    test_targets_full = torch.cat(test_targets, dim=0).to(device)

    val_inputs = []
    val_targets = []
    for batch_inputs, batch_targets in val_loader:
        val_inputs.append(batch_inputs)
        val_targets.append(batch_targets)
    val_inputs_full = torch.cat(val_inputs, dim=0).to(device)
    val_targets_full = torch.cat(val_targets, dim=0).to(device)

    if hw_monitor:
        train_metrics = eval_with_inference_measurement(
            model,
            train_inputs_full,
            train_targets_full,
            hw_monitor=hw_monitor,
            set_name='train',
        )
        wandb_run.log(train_metrics)
        test_metrics = eval_with_inference_measurement(
            model,
            test_inputs_full,
            test_targets_full,
            hw_monitor=hw_monitor,
            set_name='test',
        )
        wandb_run.log(test_metrics)
        val_metrics = eval_with_inference_measurement(
            model,
            val_inputs_full,
            val_targets_full,
            hw_monitor=hw_monitor,
            set_name='validation',
        )
        wandb_run.log(val_metrics)
    else:
        eval_train_set(model, inputs=train_inputs_full, targets=train_targets_full)
        eval_test_set(model, inputs=test_inputs_full, targets=test_targets_full)
        eval_val_set(model, inputs=val_inputs_full, targets=val_targets_full)

    final_metrics = {
        'final/train_accuracy_sample': (model.predict_one_pass(train_inputs_full[:args.final_train_sample], batch_size=min(5000, args.final_train_sample)).cpu() == train_targets_full[:args.final_train_sample].cpu()).float().mean().item(),
        'final/val_accuracy_sample': (model.predict_one_pass(val_inputs_full[:args.final_val_sample], batch_size=min(5000, args.final_val_sample)).cpu() == val_targets_full[:args.final_val_sample].cpu()).float().mean().item(),
        'final/test_accuracy': (model.predict_one_pass(test_inputs_full, batch_size=min(5000, test_inputs_full.shape[0])).cpu() == test_targets_full.cpu()).float().mean().item(),
    }
    wandb_run.log(final_metrics)

    if hw_monitor:
        try:
            hw_monitor.stop()
        except Exception as exc:
            print(f'Error stopping hardware monitor: {exc}')

    wandb_run.finish()


if __name__ == '__main__':
    main()
