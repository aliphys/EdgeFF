import torch
from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import wandb
import argparse
import yaml
import os
from pathlib import Path
import subprocess
from dotenv import load_dotenv

import Evaluation

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def dataset_loaders(dataset_name='MNIST', test_batch_size=512):
    if dataset_name == 'MNIST':
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x))])
        test_ds = MNIST('./data/', train=False, download=True, transform=transform)
        onehot_max_value = 10.0
        is_color = False
    elif dataset_name == 'FMNIST':
        transform = Compose([
            ToTensor(),
            Normalize((0.2860,), (0.3530,)),
            Lambda(lambda x: torch.flatten(x))])
        test_ds = FashionMNIST('./data/', train=False, download=True, transform=transform)
        onehot_max_value = 10.0
        is_color = False
    elif dataset_name == 'SVHN':
        transform = Compose([
            ToTensor(),
            Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            Lambda(lambda x: torch.flatten(x))])
        test_ds = SVHN('./data/', split='test', download=True, transform=transform)
        onehot_max_value = 10.0
        is_color = True
    elif dataset_name == 'CIFAR10':
        transform = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            Lambda(lambda x: torch.flatten(x))])
        test_ds = CIFAR10('./data/', train=False, download=True, transform=transform)
        onehot_max_value = 10.0
        is_color = True
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
    return test_loader, onehot_max_value, is_color

def main():
    parser = argparse.ArgumentParser(description='Evaluate models from wandb for inference energy/latency')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Load .env
    root_dir = Path(__file__).resolve().parent.parent.parent
    dotenv_path = root_dir / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)

    # Init wandb
    wandb.init(project=config.get('project', 'edgeff-network-width'), job_type='eval')

    # Hardware monitor
    try:
        from tegrats_monitor import INA3221PowerMonitor, TegratsMonitor
        hw_monitor = TegratsMonitor(interval_ms=config.get('hw_interval_ms', 500))
        hw_monitor.start()
    except ImportError:
        print("Hardware monitoring not available")
        hw_monitor = None

    dataset_name = config.get('dataset', 'MNIST')
    test_loader, onehot_max_value, is_color = dataset_loaders(dataset_name, test_batch_size=512)

    # Load test data
    test_inputs = torch.cat([d for d, _ in test_loader], dim=0)
    test_targets = torch.cat([t for _, t in test_loader], dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    batch_sizes = config.get('inference_batch_sizes', [1, 2, 8, 16, 32, 64, 128, 256, 512])

    for run_id in config.get('run_ids', []):
        # Download model
        artifact_name = f"run-{run_id}-trained-model:latest"
        artifact = wandb.use_artifact(artifact_name)
        artifact_dir = artifact.download()
        model_path = os.path.join(artifact_dir, 'temp_')
        model = torch.load(model_path, map_location=device)

        # Set model attributes
        model.device = device
        for layer in model.layers:
            layer.to(device)
        for softmax_layer in model.softmax_layers:
            softmax_layer.to(device)
        model.onehot_max_value = onehot_max_value
        model.is_color = is_color

        width = artifact.metadata.get('width', 0)

        for batch_size in batch_sizes:
            print(f"Evaluating run {run_id}, width {width}, batch_size {batch_size}")
            test_loader_bs = DataLoader(torch.utils.data.TensorDataset(test_inputs, test_targets), batch_size=batch_size, shuffle=False)

            if hw_monitor:
                metrics = Evaluation.eval_with_inference_measurement(
                    model, test_inputs, test_targets,
                    hw_monitor=hw_monitor, set_name='test', batch_size=batch_size
                )
                # Log metrics
                wandb.log({
                    'run_id': run_id,
                    'width': width,
                    'batch_size': batch_size,
                    'energy_per_sample_mj': metrics.get('test/inference_energy_per_sample_mj', 0),
                    'latency_per_sample_ms': metrics.get('test/inference_latency_per_sample_ms', 0),
                    'avg_power_mw': metrics.get('test/inference_avg_power_mw', 0),
                    'memory_mb': metrics.get('test/inference_memory_mb', 0),
                    'accuracy': metrics.get('test/accuracy', 0)
                })
            else:
                # Without hardware, just accuracy
                with torch.no_grad():
                    preds = model.predict_one_pass(test_inputs, batch_size=batch_size)
                    acc = (preds.cpu() == test_targets.cpu()).float().mean().item()
                wandb.log({
                    'run_id': run_id,
                    'width': width,
                    'batch_size': batch_size,
                    'accuracy': acc
                })

    if hw_monitor:
        hw_monitor.stop()
    wandb.finish()

if __name__ == '__main__':
    main()