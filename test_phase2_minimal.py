#!/usr/bin/env python3
"""
Minimal test to isolate the Phase 2 hanging issue
"""

import torch
import sys
import os
import time
from collections import defaultdict

# Add the current directory to Python path
sys.path.insert(0, '/home/jetson/Documents/github/EdgeFF')

from power_monitor import PowerMonitor, PowerMeasurement

class MinimalEnergyMonitor:
    """Minimal energy monitor for testing"""
    
    def __init__(self, client_id: int = 0):
        self.power_monitor = PowerMonitor(client_id)
        self.layer_measurements = defaultdict(list)
        
    def start_layer_monitoring(self, layer_idx: int, sample_idx: int = 0):
        print(f"      DEBUG: Starting power monitor for layer {layer_idx}")
        self.power_monitor.start_monitoring(round_num=0, epoch=layer_idx)
        print(f"      DEBUG: Power monitor started successfully")
        
    def stop_layer_monitoring(self, layer_idx: int):
        print(f"      DEBUG: Stopping power monitor for layer {layer_idx}")
        self.power_monitor.stop_monitoring()
        print(f"      DEBUG: Power monitor stopped successfully")
        
    def get_layer_energy_summary(self, layer_idx: int) -> dict:
        return {
            'total_energy_joules': 0.001,
            'avg_total_power_watts': 5.0,
            'avg_cpu_gpu_power_watts': 2.0,
            'avg_soc_power_watts': 1.5
        }
        
    def clear_measurements(self):
        self.power_monitor.clear_measurements()

def test_power_monitoring_cycles():
    """Test multiple start/stop cycles"""
    print("=== Testing Power Monitoring Start/Stop Cycles ===")
    
    monitor = MinimalEnergyMonitor()
    
    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1} ---")
        
        # Start monitoring
        print("Starting monitoring...")
        monitor.start_layer_monitoring(0, cycle)
        
        # Simulate some work
        print("Simulating work for 2 seconds...")
        time.sleep(2)
        
        # Stop monitoring
        print("Stopping monitoring...")
        monitor.stop_layer_monitoring(0)
        
        # Get results
        print("Getting energy summary...")
        summary = monitor.get_layer_energy_summary(0)
        print(f"Energy summary: {summary}")
        
        # Clear measurements
        print("Clearing measurements...")
        monitor.clear_measurements()
        
        print(f"Cycle {cycle + 1} completed successfully")
        
        # Small delay between cycles
        time.sleep(0.1)
    
    print("\n=== All cycles completed successfully! ===")

def test_sample_processing():
    """Test processing a small batch of dummy samples"""
    print("\n=== Testing Sample Processing Loop ===")
    
    # Create dummy data
    batch_size = 10
    dummy_samples = torch.randn(batch_size, 784)  # MNIST-like data
    
    monitor = MinimalEnergyMonitor()
    
    print(f"Processing batch of {batch_size} samples...")
    
    # Start monitoring
    monitor.start_layer_monitoring(0, 0)
    
    batch_start_time = time.perf_counter()
    
    # Process samples (this is where it might hang)
    print("Starting sample processing loop...")
    predictions = []
    for i, sample in enumerate(dummy_samples):
        print(f"  Processing sample {i+1}/{batch_size}")
        # Simulate some processing
        prediction = torch.tensor([i % 10])  # Dummy prediction
        predictions.append(prediction)
        
        # Add a small delay to simulate real processing
        time.sleep(0.01)
    
    batch_duration = time.perf_counter() - batch_start_time
    print(f"Sample processing completed in {batch_duration:.3f} seconds")
    
    # Stop monitoring
    monitor.stop_layer_monitoring(0)
    
    print("Sample processing test completed successfully!")

if __name__ == "__main__":
    print("Starting minimal Phase 2 test...")
    
    try:
        # Test 1: Power monitoring cycles
        test_power_monitoring_cycles()
        
        # Test 2: Sample processing
        test_sample_processing()
        
        print("\n✅ All tests passed! The issue is not in basic power monitoring.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()