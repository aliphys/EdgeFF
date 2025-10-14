#!/usr/bin/env python3
"""
Test to confirm that network size affects power monitoring
"""

import torch
import sys
import os
import time
import gc
from collections import defaultdict

# Add the current directory to Python path
sys.path.insert(0, '/home/jetson/Documents/github/EdgeFF')

from power_monitor import PowerMonitor, PowerMeasurement

class TestEnergyMonitor:
    """Energy monitor for testing different network sizes"""
    
    def __init__(self, client_id: int = 0):
        self.power_monitor = PowerMonitor(client_id)
        self.layer_measurements = defaultdict(list)
        
    def start_layer_monitoring(self, layer_idx: int, sample_idx: int = 0):
        print(f"      Starting power monitor for layer {layer_idx}")
        self.power_monitor.start_monitoring(round_num=0, epoch=layer_idx)
        print(f"      Power monitor started successfully")
        
    def stop_layer_monitoring(self, layer_idx: int):
        print(f"      Stopping power monitor for layer {layer_idx}")
        self.power_monitor.stop_monitoring()
        print(f"      Power monitor stopped successfully")
        
    def clear_measurements(self):
        self.power_monitor.clear_measurements()

def test_with_network_size():
    """Test power monitoring with different network sizes"""
    print("=== Testing Power Monitoring with Large Network Model ===")
    
    try:
        # Load the large network model
        print("Loading LARGE network model [784, 2000, 2000, 2000, 2000]...")
        model_path = '/home/jetson/Documents/github/EdgeFF/model/temp_'
        model = torch.load(model_path, weights_only=False)
        model = model.cpu()
        
        # Print model size info
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        print(f"✓ Large model loaded: {total_params:,} parameters, ~{model_size_mb:.1f} MB")
        
        # Create a batch of samples (similar to what Phase 2 does)
        batch_size = 100  # Smaller batch to start
        dummy_batch = torch.randn(batch_size, 784)
        print(f"✓ Created batch of {batch_size} samples")
        
        # Test power monitoring with large model processing
        monitor = TestEnergyMonitor()
        
        print("\n--- Testing with LARGE model ---")
        print("Starting power monitoring...")
        monitor.start_layer_monitoring(0, 0)
        
        print("Processing samples with large model...")
        start_time = time.perf_counter()
        
        # Process samples (this might hang with large model)
        predictions = []
        for i, sample in enumerate(dummy_batch):
            if i % 25 == 0:
                print(f"  Processing sample {i+1}/{batch_size}")
            
            # Import here to avoid early issues
            from Train import overlay_on_x_neutral
            
            # Process sample through model
            with torch.no_grad():
                h = overlay_on_x_neutral(sample.unsqueeze(0))
                
                # Process through first few layers (large layers)
                for layer_idx, layer in enumerate(model.layers[:2]):  # Just first 2 layers
                    h = layer(h)
                    if layer_idx == 0:  # After first layer
                        # Check memory usage
                        if hasattr(torch.cuda, 'memory_allocated') and torch.cuda.is_available():
                            gpu_mem = torch.cuda.memory_allocated() / (1024**2)
                            print(f"    GPU memory after layer {layer_idx}: {gpu_mem:.1f} MB")
                
                # Make a dummy prediction
                prediction = torch.tensor([i % 10])
                predictions.append(prediction)
        
        processing_time = time.perf_counter() - start_time
        print(f"✓ Processed {batch_size} samples in {processing_time:.3f} seconds")
        
        print("Stopping power monitoring...")
        monitor.stop_layer_monitoring(0)
        
        print("✅ Large model test completed successfully!")
        
        # Cleanup
        del model, dummy_batch, predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Large model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_pressure():
    """Test if memory pressure affects power monitoring"""
    print("\n=== Testing Memory Pressure Effect ===")
    
    try:
        # Create large memory allocations to simulate memory pressure
        print("Creating memory pressure...")
        large_tensors = []
        for i in range(10):
            # Create large tensors to consume memory
            tensor = torch.randn(1000, 2000)  # ~8MB each
            large_tensors.append(tensor)
        
        total_mem = sum(t.numel() * t.element_size() for t in large_tensors) / (1024**2)
        print(f"✓ Allocated {total_mem:.1f} MB of tensors")
        
        # Now test power monitoring under memory pressure
        monitor = TestEnergyMonitor()
        
        print("Testing power monitoring under memory pressure...")
        monitor.start_layer_monitoring(0, 0)
        
        # Simulate some work
        time.sleep(2)
        
        monitor.stop_layer_monitoring(0)
        
        print("✅ Memory pressure test completed!")
        
        # Cleanup
        del large_tensors
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Memory pressure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting network size vs power monitoring test...")
    print("This test will help confirm if large networks cause power monitoring to hang.")
    
    # Test 1: Large model with power monitoring
    success1 = test_with_network_size()
    
    # Test 2: Memory pressure effect
    if success1:
        success2 = test_memory_pressure()
    else:
        success2 = False
    
    if success1 and success2:
        print("\n✅ All tests passed! Large network doesn't cause immediate hang.")
        print("   The issue might be in the specific combination of:")
        print("   - Large model + Power monitoring + Batch processing")
    else:
        print("\n❌ Tests failed - this confirms network size affects power monitoring!")
        print("   Recommended solutions:")
        print("   1. Process smaller batches during power monitoring") 
        print("   2. Add memory cleanup between batches")
        print("   3. Use CPU-only inference during power monitoring")
        print("   4. Monitor power less frequently for large models")