#!/usr/bin/env python3
"""
Simple test script to demonstrate inference energy and latency measurement.
This script shows how the measurement works without running full training.
"""

import torch
import time
import numpy as np

# Import the monitoring classes
try:
    from exploratory.Refractoring.tegrats_monitor import INA3221PowerMonitor, TegratsMonitor
    print("✓ Successfully imported TegratsMonitor")
except ImportError as e:
    print(f"✗ Failed to import tegrats_monitor: {e}")
    print("  This test requires a Jetson device with tegrastats")
    exit(1)


def simulate_inference(batch_size, duration_ms=100):
    """Simulate inference by doing some computation."""
    # Create random tensors and do some operations
    x = torch.randn(batch_size, 784)
    w = torch.randn(784, 100)
    
    # Simulate inference computation
    start = time.time()
    while (time.time() - start) * 1000 < duration_ms:
        _ = torch.matmul(x, w)
        _ = torch.relu(_)
    
    return torch.argmax(torch.randn(batch_size, 10), dim=1)


def main():
    print("\n" + "="*60)
    print("Inference Energy & Latency Measurement Test")
    print("="*60 + "\n")
    
    # Initialize power monitor
    print("1. Initializing power monitor...")
    power_monitor = INA3221PowerMonitor()
    
    if not power_monitor.hwmon_path:
        print("   ⚠ INA3221 not available, using tegrastats power readings")
    else:
        print("   ✓ INA3221 power monitor initialized")
    
    # Initialize tegrastats monitor
    print("\n2. Starting tegrastats monitor...")
    hw_monitor = TegratsMonitor(
        power_monitor=power_monitor,
        interval_ms=500,  # 500ms sampling for this test
        wandb_run=None  # No wandb for this test
    )
    hw_monitor.start()
    
    # Wait for some power samples to accumulate
    print("   Waiting 2 seconds for power samples to accumulate...")
    time.sleep(2)
    print(f"   ✓ Power history size: {len(hw_monitor.power_history)} samples")
    
    # Test inference measurement with different batch sizes
    test_cases = [
        (1, 50),      # 1 sample, ~50ms
        (10, 100),    # 10 samples, ~100ms
        (100, 200),   # 100 samples, ~200ms
    ]
    
    print("\n3. Testing inference measurements:")
    print("-" * 60)
    
    for batch_size, duration_ms in test_cases:
        print(f"\n   Batch size: {batch_size}, Target duration: {duration_ms}ms")
        
        # Start measurement
        hw_monitor.start_inference_measurement()
        
        # Simulate inference
        predictions = simulate_inference(batch_size, duration_ms)
        
        # Stop measurement and get metrics
        metrics = hw_monitor.stop_inference_measurement(batch_size)
        
        if metrics:
            print(f"   Results:")
            print(f"      Total latency: {metrics['inference/total_batch_latency_ms']:.2f} ms")
            print(f"      Latency per sample: {metrics['inference/latency_per_sample_ms']:.4f} ms")
            print(f"      Total energy: {metrics['inference/total_batch_energy_mj']:.4f} mJ")
            print(f"      Energy per sample: {metrics['inference/energy_per_sample_mj']:.4f} mJ")
            print(f"      Avg power: {metrics['inference/avg_power_during_inference_mw']:.2f} mW")
            print(f"      Power samples used: {metrics['inference/num_power_samples']}")
        else:
            print("   ✗ Failed to get metrics (no power samples available)")
        
        # Wait between tests
        time.sleep(1)
    
    # Stop monitor
    print("\n4. Stopping hardware monitor...")
    hw_monitor.stop()
    print("   ✓ Monitor stopped")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
