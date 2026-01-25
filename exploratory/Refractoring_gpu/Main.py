import torch
from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader, random_split
import wandb  # experiment tracking
import argparse  # CLI argument parsing
import subprocess  # for checking tegrastats availability
from dotenv import load_dotenv  # load API key from .env
from pathlib import Path

import Train
import Evaluation

from sklearn.model_selection import train_test_split

import tools
import os

import numpy as np


def main():
    """Main training/evaluation function compatible with W&B sweeps"""
    print('Forward-Forward Training')
    # Load .env file from project root before initializing wandb
    root_dir = Path(__file__).resolve().parent.parent.parent  # EdgeFF project root
    dotenv_path = root_dir / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}; proceeding without it.")

    import os  # after potential dotenv load
    if 'WANDB_API_KEY' not in os.environ:
        print('Warning: WANDB_API_KEY not set; wandb may prompt for manual authentication.')
    ############################
    # Argument Parsing
    ############################
    parser = argparse.ArgumentParser(description='Forward-Forward MNIST one-pass experiment')
    parser.add_argument('--layers', type=str, default='784,100,100,100', help='Comma separated layer sizes including input dimension. Use 784 for MNIST/FMNIST, 3072 for SVHN/CIFAR10.')
    parser.add_argument('--rep-epochs', type=int, default=10, help='Number of representation training epochs.')
    parser.add_argument('--softmax-epochs', type=int, default=10, help='Number of softmax training epochs.')
    parser.add_argument('--train-batch-size', type=int, default=256, help='Mini-batch size for training.')
    parser.add_argument('--test-batch-size', type=int, default=512, help='Batch size for validation/test loaders.')
    parser.add_argument('--val-size', type=int, default=10000, help='Validation set size split from original training set.')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'FMNIST', 'SVHN', 'CIFAR10'], help='Dataset to use for training and evaluation.')
    parser.add_argument('--no-train', action='store_true', help='Skip training; only load existing model and evaluate.')
    parser.add_argument('--project', type=str, default='edgeff', help='wandb project name.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for splitting and numpy.')
    parser.add_argument('--log-val-samples', type=int, default=1000, help='Number of validation samples for epoch logging.')
    parser.add_argument('--final-train-sample', type=int, default=5000, help='Number of train samples for final sampled accuracy metric.')
    parser.add_argument('--final-val-sample', type=int, default=2000, help='Number of validation samples for final sampled accuracy metric.')
    parser.add_argument('--log-interval', type=int, default=10, help='How many batches between terminal progress prints.')
    parser.add_argument('--enable-hw-monitor', action='store_true', default=True, help='Enable tegrastats + INA3221 hardware monitoring if available.')
    parser.add_argument('--hw-interval-ms', type=int, default=500, help='Sampling interval for tegrastats in milliseconds (default 1000).')
    parser.add_argument('--no-cuda', type=lambda x: str(x).lower() == 'true', default=False, help='Disable CUDA even if available.')
    args = parser.parse_args()

    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    if use_cuda:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    layers = [int(x.strip()) for x in args.layers.split(',') if x.strip()]
    length_network = len(layers) - 1
    print('layers: ' + str(length_network))
    print(layers)
    
    # Validate input dimension matches dataset
    expected_input_dim = 784 if args.dataset in ['MNIST', 'FMNIST'] else 3072
    if layers[0] != expected_input_dim:
        print(f"WARNING: First layer dimension ({layers[0]}) doesn't match dataset {args.dataset} (expected {expected_input_dim})")
        print(f"Automatically adjusting layers to match dataset...")
        layers[0] = expected_input_dim
        print(f"Updated layers: {layers}")
    
    Train_flag = not args.no_train

    ############################
    # Set seeds
    ############################
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Attempt hardware monitor import
    try:
        from tegrats_monitor import INA3221PowerMonitor, TegratsMonitor
        print("Successfully imported TegratsMonitor")
    except ImportError as e:
        print(f"Failed to import tegrats_monitor: {e}")
        INA3221PowerMonitor = None
        TegratsMonitor = None

    # Initialize Weights & Biases run
    wandb_run = wandb.init(
        project=args.project,
        settings=wandb.Settings(init_timeout=600),
        job_type='train' if Train_flag else 'evaluate',
        config={
            'layers': layers,
            'representation_epochs': args.rep_epochs,
            'softmax_epochs': args.softmax_epochs,
            'train_batch_size': args.train_batch_size,
            'test_batch_size': args.test_batch_size,
            'val_size': args.val_size,
            'log_val_samples': args.log_val_samples,
            'final_train_sample': args.final_train_sample,
            'final_val_sample': args.final_val_sample,
            'seed': args.seed,
            'dataset': args.dataset,
            'train_flag': Train_flag,
            'enable_hw_monitor': args.enable_hw_monitor,
            'hw_interval_ms': args.hw_interval_ms,
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': torch.cuda.get_device_name(0) if use_cuda else None,
        },
        tags=['forward-forward', args.dataset.lower()]
    )

    # If running in a sweep, override args with sweep config
    if wandb_run.sweep_id:
        print(f"Running as part of sweep: {wandb_run.sweep_id}")
        # Override parameters from sweep
        sweep_config = wandb.config
        if hasattr(sweep_config, 'layers'):
            args.layers = sweep_config.layers
            layers = [int(x.strip()) for x in args.layers.split(',') if x.strip()]
            length_network = len(layers) - 1
            print(f"Sweep override - layers: {layers}")
        if hasattr(sweep_config, 'no_cuda'):
            args.no_cuda = sweep_config.no_cuda
            use_cuda = not args.no_cuda and torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            print(f"Sweep override - device: {device}")
        if hasattr(sweep_config, 'rep_epochs'):
            args.rep_epochs = sweep_config.rep_epochs
            print(f"Sweep override - rep_epochs: {args.rep_epochs}")
        if hasattr(sweep_config, 'softmax_epochs'):
            args.softmax_epochs = sweep_config.softmax_epochs
            print(f"Sweep override - softmax_epochs: {args.softmax_epochs}")
        if hasattr(sweep_config, 'train_batch_size'):
            args.train_batch_size = sweep_config.train_batch_size
            print(f"Sweep override - train_batch_size: {args.train_batch_size}")
        if hasattr(sweep_config, 'seed'):
            args.seed = sweep_config.seed
            print(f"Sweep override - seed: {args.seed}")
            # Re-set seeds with new value
            import random
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if use_cuda:
                torch.cuda.manual_seed(args.seed)
        
        # Update wandb config with overridden values
        wandb_run.config.update({
            'layers': layers,
            'representation_epochs': args.rep_epochs,
            'softmax_epochs': args.softmax_epochs,
            'train_batch_size': args.train_batch_size,
            'seed': args.seed,
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': torch.cuda.get_device_name(0) if use_cuda else None,
        }, allow_val_change=True)

        # Set run name based on width
        width = layers[1] if len(layers) > 1 else 0
        wandb_run.name = f"FF-Width-{width}-Seed-{args.seed}"

    # Log source code artifact (Main, Train, Evaluation, tools, tegrats_monitor if exists)
    code_artifact = wandb.Artifact('ff-source', type='code')
    current_dir = Path(__file__).resolve().parent  # Refractoring directory
    files_to_add = [
        current_dir / 'Main.py',
        current_dir / 'Train.py',
        current_dir / 'Evaluation.py',
        current_dir / 'tools.py',
        current_dir / 'tegrats_monitor.py',
    ]
    for f in files_to_add:
        if f.exists():
            code_artifact.add_file(str(f))
        else:
            print(f"Info: source file not found for artifact: {f}")
    wandb_run.log_artifact(code_artifact)

    # Initialize hardware monitoring if requested (MUST be after wandb.init())
    hw_monitor = None
    power_monitor = None
    if args.enable_hw_monitor and TegratsMonitor:
        # Check tegrastats presence
        try:
            result = subprocess.run(['which', 'tegrastats'], capture_output=True, timeout=2)
            if result.returncode == 0:
                # Power monitor (optional)
                try:
                    power_monitor = INA3221PowerMonitor() if INA3221PowerMonitor else None
                except Exception as e:
                    print(f"INA3221PowerMonitor init failed: {e}")
                    power_monitor = None

                # Start tegrastats monitor
                hw_monitor = TegratsMonitor(
                    power_monitor=power_monitor,
                    interval_ms=args.hw_interval_ms
                )
                hw_monitor.start()
                print("Hardware monitoring enabled.")
            else:
                print("Hardware monitoring requested but tegrastats not found.")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Hardware monitoring initialization error: {e}")
    elif args.enable_hw_monitor:
        print("tegrats_monitor module not available; skipping hardware monitoring.")

    """Data loading now uses mini-batches instead of loading entire dataset at once."""
    def dataset_loaders(dataset_name='MNIST', train_batch_size=64, test_batch_size=256, val_size=10000, seed=0):
        """
        Load dataset with support for MNIST, Fashion-MNIST, SVHN, and CIFAR10.
        
        Args:
            dataset_name: One of 'MNIST', 'FMNIST', 'SVHN', 'CIFAR10'
            train_batch_size: Batch size for training
            test_batch_size: Batch size for validation and test
            val_size: Size of validation set to split from training data
            seed: Random seed for splitting
            
        Returns:
            train_loader, val_loader, test_loader, onehot_max_value, is_color
        """
        if dataset_name == 'MNIST':
            transform = Compose([
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
                Lambda(lambda x: torch.flatten(x))])
            full_train = MNIST('./data/', train=True, download=True, transform=transform)
            test_ds = MNIST('./data/', train=False, download=True, transform=transform)
            # Max normalized value: (1.0 - 0.1307) / 0.3081 ≈ 2.82, use 10.0 for strong signal
            onehot_max_value = 10.0
            is_color = False
            
        elif dataset_name == 'FMNIST':
            transform = Compose([
                ToTensor(),
                Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST stats
                Lambda(lambda x: torch.flatten(x))])
            full_train = FashionMNIST('./data/', train=True, download=True, transform=transform)
            test_ds = FashionMNIST('./data/', train=False, download=True, transform=transform)
            # Max normalized value: (1.0 - 0.2860) / 0.3530 ≈ 2.02, use 10.0 for strong signal
            onehot_max_value = 10.0
            is_color = False
            
        elif dataset_name == 'SVHN':
            transform = Compose([
                ToTensor(),
                Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),  # SVHN stats
                Lambda(lambda x: torch.flatten(x))])
            full_train = SVHN('./data/', split='train', download=True, transform=transform)
            test_ds = SVHN('./data/', split='test', download=True, transform=transform)
            # Max normalized value: ~2.84 (red channel), use 10.0 for strong signal
            onehot_max_value = 10.0
            is_color = True
            
        elif dataset_name == 'CIFAR10':
            transform = Compose([
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # CIFAR10 stats
                Lambda(lambda x: torch.flatten(x))])
            full_train = CIFAR10('./data/', train=True, download=True, transform=transform)
            test_ds = CIFAR10('./data/', train=False, download=True, transform=transform)
            # Max normalized value: ~2.13 (green channel), use 10.0 for strong signal
            onehot_max_value = 10.0
            is_color = True
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Split off validation set
        train_size = len(full_train) - val_size
        train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

        train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=test_batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
        return train_loader, val_loader, test_loader, onehot_max_value, is_color


    def overlay_y_on_x(x, y, max_value=10.0, is_color=False):
        """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
        
        For grayscale images (MNIST, Fashion-MNIST), modifies the first 10 pixels.
        For color images (SVHN, CIFAR10), modifies the first 10 pixels of each color channel.
        
        Args:
            x: Input tensor (flattened)
            y: Label tensor
            max_value: Maximum value to use for one-hot encoding (default: 10.0)
            is_color: True for RGB images (SVHN, CIFAR10), False for grayscale (MNIST, FMNIST)
        """
        x_ = x.clone()
        
        if is_color:
            # For RGB images: 3 channels × 32 × 32 = 3072 pixels
            # Apply one-hot encoding to first 10 pixels of each channel
            pixels_per_channel = x_.shape[1] // 3  # 1024 for 32×32 images
            
            # Red channel (pixels 0-9)
            x_[:, :10] *= 0.0
            x_[range(x.shape[0]), y] = max_value
            
            # Green channel (pixels 1024-1033)
            x_[:, pixels_per_channel:pixels_per_channel+10] *= 0.0
            x_[range(x.shape[0]), pixels_per_channel + y] = max_value
            
            # Blue channel (pixels 2048-2057)
            x_[:, 2*pixels_per_channel:2*pixels_per_channel+10] *= 0.0
            x_[range(x.shape[0]), 2*pixels_per_channel + y] = max_value
        else:
            # For grayscale images: apply to first 10 pixels only
            x_[:, :10] *= 0.0
            x_[range(x.shape[0]), y] = max_value
            
        return x_


    def overlay_on_x_neutral(x, is_color=False):
        """Replace the first 10 pixels of data [x] with 0.1s
        
        For grayscale images (MNIST, Fashion-MNIST), modifies the first 10 pixels.
        For color images (SVHN, CIFAR10), modifies the first 10 pixels of each color channel.
        
        Args:
            x: Input tensor (flattened)
            is_color: True for RGB images (SVHN, CIFAR10), False for grayscale (MNIST, FMNIST)
        """
        x_ = x.clone()
        
        if is_color:
            # For RGB images: apply to first 10 pixels of each channel
            pixels_per_channel = x_.shape[1] // 3  # 1024 for 32×32 images
            
            # Red channel
            x_[:, :10] *= 0.0
            x_[:, :10] = 0.1
            
            # Green channel
            x_[:, pixels_per_channel:pixels_per_channel+10] *= 0.0
            x_[:, pixels_per_channel:pixels_per_channel+10] = 0.1
            
            # Blue channel
            x_[:, 2*pixels_per_channel:2*pixels_per_channel+10] *= 0.0
            x_[:, 2*pixels_per_channel:2*pixels_per_channel+10] = 0.1
        else:
            # For grayscale images
            x_[:, :10] *= 0.0
            x_[:, :10] = 0.1
            
        return x_


    train_loader, val_loader, test_loader, onehot_max_value, is_color = dataset_loaders(dataset_name=args.dataset,
                                                                                        train_batch_size=args.train_batch_size,
                                                                                        test_batch_size=args.test_batch_size,
                                                                                        val_size=args.val_size,
                                                                                        seed=args.seed)


    # train
    if Train_flag:
        # Streaming training (mini-batch) using Train.Net directly
        model = Train.Net(layers, device=device, onehot_max_value=onehot_max_value, is_color=is_color)
        representation_epochs = args.rep_epochs
        softmax_epochs = args.softmax_epochs

        # Prepare a small validation subset (first batch) for quick logging
        val_subset_inputs, val_subset_targets = next(iter(val_loader))
        # Move to device
        val_subset_inputs = val_subset_inputs.to(device)
        val_subset_targets = val_subset_targets.to(device)
        # Optionally truncate for logging speed
        if val_subset_inputs.shape[0] > args.log_val_samples:
            val_subset_inputs = val_subset_inputs[:args.log_val_samples]
            val_subset_targets = val_subset_targets[:args.log_val_samples]

        for epoch in range(representation_epochs):
            total_train = len(train_loader.dataset)
            for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
                # Move batch to device
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                # Create positive and negative samples for FF representation training
                x_pos_batch = overlay_y_on_x(batch_inputs, batch_targets, onehot_max_value, is_color)
                # Negative labels: choose random incorrect label per sample
                y_neg_batch = batch_targets.clone()
                for idx, y_samp in enumerate(batch_targets):
                    allowed = list(range(10))
                    allowed.remove(y_samp.item())
                    y_neg_batch[idx] = torch.tensor(np.random.choice(allowed))
                x_neg_batch = overlay_y_on_x(batch_inputs, y_neg_batch, onehot_max_value, is_color)
                model.train(x_pos_batch, x_neg_batch)

                if batch_idx % args.log_interval == 0:
                    processed = (batch_idx + 1) * batch_inputs.shape[0]
                    pct = processed / total_train
                    print(f'Rep Epoch: {epoch+1}/{representation_epochs} '\
                          f'[{processed}/{total_train} ({pct:.0%})]')

            # Log sample validation accuracy after each epoch
            with torch.no_grad():
                preds = model.predict_one_pass(val_subset_inputs, batch_size=val_subset_inputs.shape[0])
                sample_val_acc = (preds.cpu() == val_subset_targets.cpu()).float().mean().item()
            wandb_run.log({'epoch': epoch, 'phase': 'representation', 'sample_val_accuracy': sample_val_acc})

        for epoch in range(softmax_epochs):
            total_train = len(train_loader.dataset)
            for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
                # Move batch to device
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                x_neutral_batch = overlay_on_x_neutral(batch_inputs, is_color)
                model.train_softmax_layer(x_neutral_batch, batch_targets, batch_inputs.shape[0], layers)
                if batch_idx % args.log_interval == 0:
                    processed = (batch_idx + 1) * batch_inputs.shape[0]
                    pct = processed / total_train
                    print(f'Softmax Epoch: {epoch+1}/{softmax_epochs} '\
                          f'[{processed}/{total_train} ({pct:.0%})]')
            with torch.no_grad():
                preds = model.predict_one_pass(val_subset_inputs, batch_size=val_subset_inputs.shape[0])
                sample_val_acc = (preds.cpu() == val_subset_targets.cpu()).float().mean().item()
            wandb_run.log({'epoch': epoch, 'phase': 'softmax', 'sample_val_accuracy': sample_val_acc})

        # Save model
        name = 'temp_'
        model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, name)
        torch.save(model, model_path)
        
        # Log model as wandb artifact
        model_artifact = wandb.Artifact('trained-model', type='model')
        model_artifact.add_file(model_path)
        # Add metadata
        width = layers[1] if len(layers) > 1 else 0  # Hidden layer width
        model_artifact.metadata = {
            'layers': layers,
            'width': width,
            'dataset': args.dataset,
            'seed': args.seed
        }
        wandb_run.log_artifact(model_artifact)
    else:
        name = 'temp_'
        model = torch.load(os.path.split(os.path.realpath(__file__))[0] + '/model/' + name)
        # Ensure loaded model is on the correct device
        if hasattr(model, 'device'):
            model.device = device
            # Move all layers to device
            for layer in model.layers:
                layer.to(device)
            for softmax_layer in model.softmax_layers:
                softmax_layer.to(device)
        # Set encoding parameters if not present (for backward compatibility)
        if not hasattr(model, 'onehot_max_value'):
            model.onehot_max_value = onehot_max_value
        if not hasattr(model, 'is_color'):
            model.is_color = is_color

    # evaluation
    # Build full tensors for train set evaluation (optional; could stream instead)
    train_inputs_full = torch.cat([d for d, _ in train_loader], dim=0).to(device)
    train_targets_full = torch.cat([t for _, t in train_loader], dim=0).to(device)

    # Use new measurement-aware evaluation if hw_monitor is available
    if hw_monitor:
        train_metrics = Evaluation.eval_with_inference_measurement(
            model, train_inputs_full, train_targets_full, 
            hw_monitor=hw_monitor, set_name='train'
        )
        wandb_run.log(train_metrics)
    else:
        Evaluation.eval_train_set(model, inputs=train_inputs_full, targets=train_targets_full)

    # test data
    test_inputs_full = torch.cat([d for d, _ in test_loader], dim=0).to(device)
    test_targets_full = torch.cat([t for _, t in test_loader], dim=0).to(device)

    if hw_monitor:
        test_metrics = Evaluation.eval_with_inference_measurement(
            model, test_inputs_full, test_targets_full,
            hw_monitor=hw_monitor, set_name='test'
        )
        wandb_run.log(test_metrics)
    else:
        Evaluation.eval_test_set(model, inputs=test_inputs_full, targets=test_targets_full)

    # validation data
    val_inputs_full = torch.cat([d for d, _ in val_loader], dim=0).to(device)
    val_targets_full = torch.cat([t for _, t in val_loader], dim=0).to(device)

    if hw_monitor:
        val_metrics = Evaluation.eval_with_inference_measurement(
            model, val_inputs_full, val_targets_full,
            hw_monitor=hw_monitor, set_name='validation'
        )
        wandb_run.log(val_metrics)
    else:
        Evaluation.eval_val_set(model, inputs=val_inputs_full, targets=val_targets_full)

    # analysis of validation data
    mean, std = tools.analysis_val_set(model, inputs=val_inputs_full, targets=val_targets_full)

    confidence_mean_vec = mean
    confidence_std_vec = std
    # Evaluation.eval_val_set_light(model, inputs=X_val, targets=y_val,
    #                               confidence_mean_vec=confidence_mean_vec, confidence_std_vec=confidence_std_vec)
    Evaluation.eval_val_set_light(model, inputs=test_inputs_full, targets=test_targets_full,
                                  confidence_mean_vec=confidence_mean_vec,
                                  confidence_std_vec=confidence_std_vec)  ## temporary use

    # Log final summary metrics to wandb
    def quick_accuracy(model, x, y):
        with torch.no_grad():
            preds = model.predict_one_pass(x, batch_size=min(5000, x.shape[0]))
        return (preds.cpu() == y.cpu()).float().mean().item()

    final_metrics = {
        'final/train_accuracy_sample': quick_accuracy(model, train_inputs_full[:args.final_train_sample], train_targets_full[:args.final_train_sample]),
        'final/val_accuracy_sample': quick_accuracy(model, val_inputs_full[:args.final_val_sample], val_targets_full[:args.final_val_sample]),
        'final/test_accuracy': quick_accuracy(model, test_inputs_full, test_targets_full),
    }
    wandb_run.log(final_metrics)
    # Stop hardware monitor before finishing wandb
    if hw_monitor:
        try:
            hw_monitor.stop()
        except Exception as e:
            print(f"Error stopping hardware monitor: {e}")
    wandb_run.finish()


if __name__ == '__main__':
    main()