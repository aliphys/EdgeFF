"""
Power monitoring module for federated learning using jtop (Jetson devices).
Provides real-time power consumption tracking during federated training.
"""

import json
import time
import threading
import subprocess
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from jtop import jtop

# Monkey-patch jtop to always use interval=0.1s (100ms) on creation
_original_jtop_init = jtop.__init__

def _patched_jtop_init(self, *args, **kwargs):
    kwargs['interval'] = 0.2  # Set to 200ms for a balance between responsiveness and overhead
    _original_jtop_init(self, *args, **kwargs)

jtop.__init__ = _patched_jtop_init

# In order to reduce the sampling time to a lower interval than 250ms, we modify the /etc/systemd/jtop.service file
# Change the line: ExecStart=/usr/local/bin/jtop -i 100

def check_jtop_service_status():
    """Check if jtop service is active and running"""
    try:
        result = subprocess.run(['systemctl', 'is-active', 'jtop.service'], 
                              capture_output=True, text=True, timeout=5)
        return result.stdout.strip() == 'active'
    except:
        return False

def wait_for_jtop_service(max_wait_time=30):
    """Wait for jtop service to become active, with timeout"""
    print("Waiting for jtop service to become active...")
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        if check_jtop_service_status():
            print("jtop service is now active")
            return True
        time.sleep(1.0)
    print(f"jtop service did not become active within {max_wait_time} seconds")
    return False

@dataclass
class PowerMeasurement:
    """Data class for storing power measurements from Jetson devices"""
    timestamp: float
    client_id: int
    round_num: int
    epoch: int
    power_cpu_gpu_cv: Optional[float] = None  # VDD_CPU_GPU_CV: CPU, GPU and CV cores (DLA, PVA)
    power_soc: Optional[float] = None         # VDD_SOC: Memory subsystem and engines (nvdec, nvenc, vi, vic, isp)
    total_power: Optional[float] = None       # VDD_IN: Total Module Power
    gpu_utilization: Optional[float] = None
    cpu_utilization: Optional[float] = None
    memory_usage: Optional[float] = None
    temperature: Optional[float] = None


class PowerMonitor:
    """Monitor power consumption using jtop for Jetson devices"""
    
    def __init__(self, client_id: int):
        """
        Initialize power monitor for a specific federated client.
        Args:
            client_id (int): Unique identifier for the federated client
        """
        self.client_id = client_id
        self.measurements: List[PowerMeasurement] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.current_round = 0
        self.current_epoch = 0
        self._lock = threading.Lock()
        self._thread_ready = threading.Event()
        
    def start_monitoring(self, round_num: int, epoch: int = 0):
        """
        Start power monitoring for a specific round and epoch.
        Args:
            round_num (int): Current federated learning round
            epoch (int): Current local training epoch
        """
        with self._lock:
            self.current_round = round_num
            self.current_epoch = epoch
            self.monitoring = True
            
        self._thread_ready.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_power)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Wait for the thread to be ready (with timeout)
        self._thread_ready.wait(timeout=2.0)
        
    def stop_monitoring(self):
        """Stop power monitoring and wait for thread to complete"""
        with self._lock:
            self.monitoring = False
            
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)  # 5 second timeout
            
        # Reset thread reference to None after stopping
        self.monitor_thread = None
            
    def update_epoch(self, epoch: int):
        """
        Update current epoch for measurements.
        Args:
            epoch (int): New epoch number
        """
        with self._lock:
            self.current_epoch = epoch
            
    def _monitor_power(self):
        """Internal method to collect power measurements from jtop"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with jtop() as jetson:
                    # Give jtop a moment to initialize
                    time.sleep(0.1)
                    
                    # Signal that the thread is ready (only on first successful connection)
                    if retry_count == 0:
                        self._thread_ready.set()
                    
                    while True:
                        with self._lock:
                            if not self.monitoring:
                                return  # Exit cleanly
                            current_round = self.current_round
                            current_epoch = self.current_epoch
                        
                        # Continuously collect data when jetson.ok() is true for highest resolution
                        if jetson.ok():
                            # Extract power measurements from jtop using correct API
                            power_data = jetson.power
                            
                            # Get total power (in mW, convert to W)
                            total_power = None
                            total_power_info = power_data.get('tot', {})
                            if total_power_info and 'power' in total_power_info:
                                total_power = total_power_info['power'] / 1000.0  # Convert mW to W
                            
                            # Get individual power rail measurements (convert mW to W)
                            power_cpu_gpu_cv = None  # VDD_CPU_GPU_CV: CPU, GPU and CV cores
                            power_soc = None         # VDD_SOC: SOC core (memory subsystem and engines)
                            
                            rails = power_data.get('rail', {})
                            for rail_name, rail_data in rails.items():
                                if isinstance(rail_data, dict) and rail_data.get('online', False):
                                    power_val = rail_data.get('power', 0)
                                    if power_val is not None:
                                        power_w = power_val / 1000.0  # Convert mW to W
                                        
                                        # Store specific power rail measurements based on Jetson Orin Nano channels
                                        # Channel 1: VDD_IN - Total Module Power
                                        # Channel 2: VDD_CPU_GPU_CV - CPU, GPU and CV cores (DLA, PVA) 
                                        # Channel 3: VDD_SOC - SOC core (memory, nvdec, nvenc, vi, vic, isp, etc.)
                                        
                                        if 'VDD_IN' in rail_name:
                                            # This is total module power - we'll use it for total_power if available
                                            pass  # Handle separately below
                                        elif 'VDD_CPU_GPU_CV' in rail_name:
                                            power_cpu_gpu_cv = power_w    # CPU, GPU and CV cores
                                        elif 'VDD_SOC' in rail_name:
                                            power_soc = power_w           # SOC core (memory subsystem and engines)
                            
                            # Extract GPU utilization
                            gpu_util = None
                            gpu_info = jetson.gpu.get('gpu', {})
                            status = gpu_info.get('status', {})
                            if 'load' in status:
                                gpu_util = status['load']
                            
                            # Extract CPU utilization (overall usage)
                            cpu_util = None
                            cpu_info = jetson.cpu
                            total_cpu = cpu_info.get('total', {})
                            if 'idle' in total_cpu:
                                cpu_util = 100.0 - total_cpu['idle']  # Convert idle to usage
                            
                            # Extract memory usage
                            memory_usage = None
                            memory_info = jetson.memory
                            ram = memory_info.get('RAM', {})
                            if 'used' in ram and 'tot' in ram and ram['tot'] > 0:
                                memory_usage = (ram['used'] / ram['tot']) * 100.0
                            
                            # Extract temperature (use CPU temperature as primary)
                            temperature = None
                            temp_info = jetson.temperature
                            # Try different temperature sensor names
                            for sensor_name, sensor_data in temp_info.items():
                                if sensor_data.get('online', False) and sensor_data.get('temp', -256) > -200:
                                    temperature = sensor_data['temp']
                                    break  # Use the first available temperature
                            
                            measurement = PowerMeasurement(
                                timestamp=time.time(),
                                client_id=self.client_id,
                                round_num=current_round,
                                epoch=current_epoch,
                                power_cpu_gpu_cv=power_cpu_gpu_cv,
                                power_soc=power_soc,
                                total_power=total_power,
                                gpu_utilization=gpu_util,
                                cpu_utilization=cpu_util,
                                memory_usage=memory_usage,
                                temperature=temperature
                            )
                            
                            with self._lock:
                                self.measurements.append(measurement)
                        
                        # High-resolution sampling: 100ms sleep for 10Hz data collection
                        # This maximizes time resolution while maintaining jtop service stability
                        time.sleep(0.1)
                        
                # If we get here, jtop connection closed normally
                return
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                print(f"Error in power monitoring for client {self.client_id} (attempt {retry_count}/{max_retries}): {error_msg}")
                
                # Check if it's a jtop service issue
                if "jtop.service is not active" in error_msg or "service" in error_msg.lower():
                    print(f"jtop service issue detected. Waiting for service recovery...")
                    # Wait for jtop service to recover
                    if not wait_for_jtop_service(max_wait_time=15):
                        print("jtop service recovery timeout - continuing with next retry")
                elif not check_jtop_service_status():
                    print("jtop service is not active - waiting for recovery...")
                    wait_for_jtop_service(max_wait_time=10)
                        
                if retry_count >= max_retries:
                    print(f"Power monitoring failed after {max_retries} attempts for client {self.client_id}")
                    # Signal thread ready even on failure to prevent deadlock
                    self._thread_ready.set()
                    return
                else:
                    time.sleep(1.0)  # Brief wait before retry
            
    def get_measurements(self) -> List[Dict]:
        """
        Get all measurements as dictionaries.
        Returns:
            List[Dict]: List of measurement dictionaries
        """
        with self._lock:
            return [asdict(m) for m in self.measurements]
        
    def save_measurements(self, filepath: str):
        """
        Save measurements to JSON file.
        Args:
            filepath (str): Path to save the measurements
        """
        measurements = self.get_measurements()
        with open(filepath, 'w') as f:
            json.dump(measurements, f, indent=2)
            
    def get_round_summary(self, round_num: int) -> Dict:
        """
        Get power consumption summary for a specific round.
        Args:
            round_num (int): Round number to summarize
        Returns:
            Dict: Power consumption summary statistics
        """
        with self._lock:
            round_measurements = [m for m in self.measurements if m.round_num == round_num]
        
        if not round_measurements:
            return {}
            
        # Calculate statistics for total power
        total_powers = [m.total_power for m in round_measurements if m.total_power is not None]
        timestamps = [m.timestamp for m in round_measurements]
        
        if not total_powers or not timestamps:
            return {
                'client_id': self.client_id,
                'round_num': round_num,
                'sample_count': len(round_measurements)
            }
        
        duration = max(timestamps) - min(timestamps)
        avg_power = sum(total_powers) / len(total_powers)
        
        return {
            'client_id': self.client_id,
            'round_num': round_num,
            'duration_seconds': duration,
            'avg_power_watts': avg_power,
            'max_power_watts': max(total_powers),
            'min_power_watts': min(total_powers),
            'energy_joules': avg_power * duration,  # Energy = Power * Time
            'sample_count': len(round_measurements),
            'avg_gpu_utilization': sum([m.gpu_utilization for m in round_measurements if m.gpu_utilization is not None]) / len([m for m in round_measurements if m.gpu_utilization is not None]) if any(m.gpu_utilization is not None for m in round_measurements) else None,
            'avg_cpu_utilization': sum([m.cpu_utilization for m in round_measurements if m.cpu_utilization is not None]) / len([m for m in round_measurements if m.cpu_utilization is not None]) if any(m.cpu_utilization is not None for m in round_measurements) else None,
            'avg_temperature': sum([m.temperature for m in round_measurements if m.temperature is not None]) / len([m for m in round_measurements if m.temperature is not None]) if any(m.temperature is not None for m in round_measurements) else None
        }
        
    def clear_measurements(self):
        """Clear all stored measurements"""
        with self._lock:
            self.measurements.clear()