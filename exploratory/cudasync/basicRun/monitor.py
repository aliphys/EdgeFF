"""Hardware monitoring helpers for Jetson devices."""

import glob
import re
import subprocess
import threading
import time

import torch
import wandb


class INA3221PowerMonitor:
    def __init__(self):
        self.hwmon_path = None
        self.channels = {
            1: "VDD_IN",
            2: "VDD_CPU_GPU_CV",
            3: "VDD_SOC",
        }
        pattern = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*"
        matches = glob.glob(pattern)
        if matches:
            self.hwmon_path = matches[0]
            print(f"Found INA3221 power monitor at: {self.hwmon_path}")
        else:
            print("INA3221 power monitor not found (not a Jetson Orin device)")

    def read_channel(self, channel_num):
        if not self.hwmon_path:
            return None
        try:
            voltage_path = f"{self.hwmon_path}/in{channel_num}_input"
            with open(voltage_path, 'r') as f:
                voltage_mv = float(f.read().strip())
            current_path = f"{self.hwmon_path}/curr{channel_num}_input"
            with open(current_path, 'r') as f:
                current_ma = float(f.read().strip())
            power_mw = (voltage_mv * current_ma) / 1000.0
            return {
                'voltage_mv': voltage_mv,
                'current_ma': current_ma,
                'power_mw': power_mw,
            }
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"Error reading channel {channel_num}: {e}")
            return None

    def get_power_metrics(self):
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


class InferenceMetrics:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.power_samples = []

    def add_power_sample(self, timestamp, power_mw):
        self.power_samples.append((timestamp, power_mw))

    def calculate_metrics(self, batch_size):
        if not self.start_time or not self.end_time or not self.power_samples:
            return None
        relevant_samples = [
            (ts, power) for ts, power in self.power_samples
            if self.start_time <= ts <= self.end_time
        ]
        if not relevant_samples:
            before = [s for s in self.power_samples if s[0] <= self.start_time]
            after = [s for s in self.power_samples if s[0] >= self.end_time]
            if before and after:
                relevant_samples = [before[-1], after[0]]
            elif before:
                relevant_samples = [before[-1]]
            elif after:
                relevant_samples = [after[0]]
        if not relevant_samples:
            return None
        avg_power_mw = sum(power for _, power in relevant_samples) / len(relevant_samples)
        latency_s = self.end_time - self.start_time
        latency_ms = latency_s * 1000.0
        total_energy_mj = avg_power_mw * latency_s
        energy_per_sample_mj = total_energy_mj / batch_size if batch_size > 0 else 0
        latency_per_sample_ms = latency_ms / batch_size if batch_size > 0 else 0
        return {
            'inference/total_batch_latency_ms': latency_ms,
            'inference/latency_per_sample_ms': latency_per_sample_ms,
            'inference/total_batch_energy_mj': total_energy_mj,
            'inference/energy_per_sample_mj': energy_per_sample_mj,
            'inference/avg_power_during_inference_mw': avg_power_mw,
            'inference/batch_size': batch_size,
            'inference/num_power_samples': len(relevant_samples),
        }


class TegratsMonitor:
    def __init__(self, power_monitor=None, interval_ms=500, wandb_run=None):
        self.running = False
        self.thread = None
        self.process = None
        self.power_monitor = power_monitor
        self.interval_ms = interval_ms
        self.wandb_run = wandb_run
        self.power_mode = None
        self.inference_metrics = None
        self.power_history = []
        self.max_history_size = 1000

    def parse_tegrastats(self, line):
        metrics = {}
        ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
        if ram_match:
            metrics['ram_used_mb'] = int(ram_match.group(1))
            metrics['ram_total_mb'] = int(ram_match.group(2))
            metrics['ram_usage_percent'] = (int(ram_match.group(1)) / int(ram_match.group(2))) * 100
        gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)
        if gpu_match:
            metrics['gpu_usage_percent'] = int(gpu_match.group(1))
        cpu_match = re.search(r'CPU \[([\d%@,]+)\]', line)
        if cpu_match:
            cpu_values = re.findall(r'(\d+)%', cpu_match.group(1))
            if cpu_values:
                metrics['cpu_avg_percent'] = sum(int(v) for v in cpu_values) / len(cpu_values)
        temp_match = re.search(r'(?:temp|Tdiode)@([\d.]+)C', line)
        if temp_match:
            metrics['temperature_celsius'] = float(temp_match.group(1))
        if not self.power_monitor:
            power_match = re.search(r'VDD_IN (\d+)mW', line)
            if power_match:
                metrics['VDD_IN_power_mw'] = int(power_match.group(1))
        return metrics

    def get_power_mode(self):
        try:
            output = subprocess.check_output(['nvpmodel', '-q'], text=True, stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None
        mode_match = re.search(r'current mode\s*:\s*(.+)', output, re.IGNORECASE)
        if mode_match:
            mode_value = mode_match.group(1).strip()
            label_match = re.search(r'\b(7W|7W-AI|7W-CPU|10W|15W|20W|25W|30W|40W|50W|MAXN|MAXN_SUPER)\b', output, re.IGNORECASE)
            if label_match:
                return label_match.group(1).upper()
            return mode_value
        label_match = re.search(r'\b(7W|7W-AI|7W-CPU|10W|15W|20W|25W|30W|40W|50W|MAXN|MAXN_SUPER)\b', output, re.IGNORECASE)
        return label_match.group(1).upper() if label_match else None

    def monitor_loop(self):
        try:
            self.process = subprocess.Popen(
                ['tegrastats', '--interval', str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )
            for line in iter(self.process.stdout.readline, ''):
                if not self.running:
                    break
                metrics = self.parse_tegrastats(line.strip())
                if self.power_monitor:
                    power_metrics = self.power_monitor.get_power_metrics()
                    metrics.update(power_metrics)
                try:
                    if torch.cuda.is_available():
                        metrics['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
                        metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
                        metrics['gpu_memory_peak_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
                except Exception as e:
                    print(f"Error getting PyTorch GPU memory: {e}")
                try:
                    import psutil
                    memory = psutil.virtual_memory()
                    metrics['cpu_memory_used_mb'] = memory.used / (1024 ** 2)
                    metrics['cpu_memory_percent'] = memory.percent
                except ImportError:
                    pass
                except Exception as e:
                    print(f"Error getting CPU memory: {e}")
                current_time = time.time()
                total_power_mw = metrics.get('VDD_IN_power_mw', 0)
                if total_power_mw > 0:
                    self.power_history.append((current_time, total_power_mw))
                    if len(self.power_history) > self.max_history_size:
                        self.power_history.pop(0)
                    if self.inference_metrics:
                        self.inference_metrics.add_power_sample(current_time, total_power_mw)
                if metrics:
                    wandb.log(metrics)
        except FileNotFoundError:
            print("Warning: tegrastats not found. Running on non-Jetson device.")
        except Exception as e:
            print(f"Error monitoring tegrastats: {e}")

    def start(self):
        if self.running:
            return
        self.power_mode = self.get_power_mode()
        self.running = True
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
        print("Started tegrastats monitoring")

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()
        if self.thread:
            self.thread.join(timeout=2)
        print("Stopped tegrastats monitoring")

    def start_inference_measurement(self):
        self.inference_metrics = InferenceMetrics()
        self.inference_metrics.start_time = time.time()
        for timestamp, power_mw in self.power_history:
            self.inference_metrics.add_power_sample(timestamp, power_mw)
        return self.inference_metrics

    def stop_inference_measurement(self, batch_size):
        if not self.inference_metrics:
            return None
        self.inference_metrics.end_time = time.time()
        metrics = self.inference_metrics.calculate_metrics(batch_size)
        self.inference_metrics = None
        return metrics
