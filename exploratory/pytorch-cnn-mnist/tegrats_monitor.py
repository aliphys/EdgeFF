import glob
import re
import subprocess
import threading

import wandb


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
