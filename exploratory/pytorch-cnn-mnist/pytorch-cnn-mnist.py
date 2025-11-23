from __future__ import print_function

import argparse
import glob
import os
import re
import subprocess
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from torchvision import datasets, transforms

# Load .env file from the root directory
root_dir = Path(__file__).resolve().parent.parent.parent
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)


class INA3221PowerMonitor:
    """Monitor power consumption using INA3221 sysfs nodes on Jetson Orin devices."""
    
    def __init__(self):
        self.hwmon_path = None
        self.channels = {
            1: "VDD_IN",              # Total Module Power
            2: "VDD_CPU_GPU_CV",      # CPU/GPU/CV cores
            3: "VDD_SOC"              # Memory & Engines
        }
        
        # Try to find hwmon path
        pattern = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*"
        matches = glob.glob(pattern)
        
        if matches:
            self.hwmon_path = matches[0]
            print(f"Found INA3221 power monitor at: {self.hwmon_path}")
        else:
            print("INA3221 power monitor not found (not a Jetson Orin device)")
    
    def read_channel(self, channel_num):
        """Read voltage and current for a specific channel."""
        if not self.hwmon_path:
            return None
        
        try:
            # Read voltage in millivolts
            voltage_path = f"{self.hwmon_path}/in{channel_num}_input"
            with open(voltage_path, 'r') as f:
                voltage_mv = float(f.read().strip())
            
            # Read current in milliamperes
            current_path = f"{self.hwmon_path}/curr{channel_num}_input"
            with open(current_path, 'r') as f:
                current_ma = float(f.read().strip())
            
            # Calculate power in milliwatts
            power_mw = (voltage_mv * current_ma) / 1000.0
            
            return {
                'voltage_mv': voltage_mv,
                'current_ma': current_ma,
                'power_mw': power_mw
            }
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"Error reading channel {channel_num}: {e}")
            return None
    
    def get_power_metrics(self):
        """Get power metrics for all channels."""
        if not self.hwmon_path:
            return {}
        
        metrics = {}
        for channel_num, channel_name in self.channels.items():
            data = self.read_channel(channel_num)
            if data:
                metrics[f"{channel_name}_power_mw"] = data['power_mw']
                metrics[f"{channel_name}_voltage_mv"] = data['voltage_mv']
                metrics[f"{channel_name}_current_ma"] = data['current_ma']
        
        return metrics


class TegratsMonitor:
    """Monitor system metrics using tegrastats on Jetson devices."""
    
    def __init__(self, power_monitor=None):
        self.running = False
        self.thread = None
        self.process = None
        self.power_monitor = power_monitor
        
    def parse_tegrastats(self, line):
        """Parse tegrastats output and extract metrics."""
        metrics = {}
        
        # RAM usage: RAM 1234/4096MB
        ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
        if ram_match:
            metrics['ram_used_mb'] = int(ram_match.group(1))
            metrics['ram_total_mb'] = int(ram_match.group(2))
            metrics['ram_usage_percent'] = (int(ram_match.group(1)) / int(ram_match.group(2))) * 100
        
        # GPU usage: GR3D_FREQ 99%
        gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)
        if gpu_match:
            metrics['gpu_usage_percent'] = int(gpu_match.group(1))
        
        # CPU usage: CPU [25%@102,26%@102,24%@102,23%@102]
        cpu_match = re.search(r'CPU \[([\d%@,]+)\]', line)
        if cpu_match:
            cpu_values = re.findall(r'(\d+)%', cpu_match.group(1))
            if cpu_values:
                metrics['cpu_avg_percent'] = sum(int(v) for v in cpu_values) / len(cpu_values)
        
        # Temperature: temp@30.5C or Tdiode@30.5C
        temp_match = re.search(r'(?:temp|Tdiode)@([\d.]+)C', line)
        if temp_match:
            metrics['temperature_celsius'] = float(temp_match.group(1))
        
        # Power from tegrastats (fallback if INA3221 not available): VDD_IN 2345mW
        if not self.power_monitor:
            power_match = re.search(r'VDD_IN (\d+)mW', line)
            if power_match:
                metrics['VDD_IN_power_mw'] = int(power_match.group(1))
        
        return metrics
    
    def monitor_loop(self):
        """Background thread to monitor tegrastats."""
        try:
            self.process = subprocess.Popen(
                ['tegrastats', '--interval', '1000'],  # 1 second interval
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in iter(self.process.stdout.readline, ''):
                if not self.running:
                    break
                    
                # Parse tegrastats output
                metrics = self.parse_tegrastats(line.strip())
                
                # Add INA3221 power metrics if available
                if self.power_monitor:
                    power_metrics = self.power_monitor.get_power_metrics()
                    metrics.update(power_metrics)
                
                # Log to wandb if we have metrics
                if metrics:
                    wandb.log(metrics)
                    
        except FileNotFoundError:
            print("Warning: tegrastats not found. Running on non-Jetson device.")
        except Exception as e:
            print(f"Error monitoring tegrastats: {e}")
    
    def start(self):
        """Start monitoring tegrastats."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
        print("Started tegrastats monitoring")
    
    def stop(self):
        """Stop monitoring tegrastats."""
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()
        if self.thread:
            self.thread.join(timeout=2)
        print("Stopped tegrastats monitoring")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                batch_idx / len(train_loader), loss.item()))
            wandb.log({"batch_loss": loss.item(), "epoch": epoch})


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            example_images.append(wandb.Image(
                data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0%})\n'.format(
        test_loss, correct, len(test_loader.dataset),
        correct / len(test_loader.dataset)))
    wandb.log({
        "Examples": example_images,
        "Test Accuracy": 100. * correct / len(test_loader.dataset),
        "Test Loss": test_loss})


def main():
    wandb.init(project="pytorch-cnn-mnist.py")
    
    # Log code as artifact
    code_artifact = wandb.Artifact('training-code', type='code')
    code_artifact.add_file(__file__)
    wandb.log_artifact(code_artifact)
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    wandb.config.update(args)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize system monitoring for Jetson devices
    tegra_monitor = None
    power_monitor = None
    
    # Try to initialize INA3221 power monitor (Jetson Orin)
    try:
        power_monitor = INA3221PowerMonitor()
    except Exception as e:
        print(f"Could not initialize INA3221 power monitor: {e}")
    
    # Try to start tegrastats monitoring
    try:
        result = subprocess.run(['which', 'tegrastats'], capture_output=True, timeout=2)
        if result.returncode == 0:
            tegra_monitor = TegratsMonitor(power_monitor=power_monitor)
            tegra_monitor.start()
            if power_monitor and power_monitor.hwmon_path:
                print("Hardware monitoring: tegrastats + INA3221 power monitor active")
            else:
                print("Hardware monitoring: tegrastats active (INA3221 not available)")
        else:
            print("Hardware monitoring: Not available (not running on Jetson device)")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Hardware monitoring: Not available ({e})")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    wandb.watch(model)

    try:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)
    finally:
        # Stop monitoring when training completes or is interrupted
        if tegra_monitor:
            tegra_monitor.stop()


if __name__ == '__main__':
    main()