#!/usr/bin/env python3
"""
Test to replicate the exact Phase 2 conditions that cause hanging
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

class ExactPhase2Monitor:
    """Exact replica of RealPowerEnergyMonitor from Main.py"""
    
    def __init__(self, client_id: int = 0):
        self.power_monitor = PowerMonitor(client_id)
        self.layer_measurements = defaultdict(list)
        self.layer_times = defaultdict(list)
        self.confidence_check_times = []
        self.early_exit_stats = {'layer_1': 0, 'layer_2': 0, 'layer_3': 0}
        self.current_round = 0
        self.current_sample_idx = 0
        
    def start_layer_monitoring(self, layer_idx: int, sample_idx: int = 0):
        self.current_sample_idx = sample_idx
        self.power_monitor.start_monitoring(
            round_num=self.current_round, 
            epoch=layer_idx
        )
        
    def stop_layer_monitoring(self, layer_idx: int):
        self.power_monitor.stop_monitoring()
        measurements = self.power_monitor.get_measurements()
        if measurements:
            layer_measurements = [m for m in measurements if m['epoch'] == layer_idx]
            if layer_measurements:
                self.layer_measurements[layer_idx].extend(layer_measurements)
        self.power_monitor.clear_measurements()
        
    def get_layer_energy_summary(self, layer_idx: int) -> dict:
        measurements = self.layer_measurements[layer_idx]
        if not measurements:
            return {
                'layer_idx': layer_idx,
                'sample_count': 0,
                'avg_total_power_watts': 0.0,
                'avg_cpu_gpu_power_watts': 0.0,
                'avg_soc_power_watts': 0.0,
                'total_energy_joules': 0.0,
                'duration_seconds': 0.0
            }
        
        total_powers = [m['total_power'] for m in measurements if m['total_power'] is not None]
        cpu_gpu_powers = [m['power_cpu_gpu_cv'] for m in measurements if m['power_cpu_gpu_cv'] is not None]
        soc_powers = [m['power_soc'] for m in measurements if m['power_soc'] is not None]
        timestamps = [m['timestamp'] for m in measurements]
        
        if not total_powers or not timestamps:
            return {
                'layer_idx': layer_idx,
                'sample_count': len(measurements),
                'avg_total_power_watts': 0.0,
                'avg_cpu_gpu_power_watts': 0.0,
                'avg_soc_power_watts': 0.0,
                'total_energy_joules': 0.0,
                'duration_seconds': 0.0
            }
        
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.25
        avg_total_power = sum(total_powers) / len(total_powers)
        avg_cpu_gpu_power = sum(cpu_gpu_powers) / len(cpu_gpu_powers) if cpu_gpu_powers else 0.0
        avg_soc_power = sum(soc_powers) / len(soc_powers) if soc_powers else 0.0
        total_energy = avg_total_power * duration
        
        return {
            'layer_idx': layer_idx,
            'sample_count': len(measurements),
            'duration_seconds': duration,
            'avg_total_power_watts': avg_total_power,
            'avg_cpu_gpu_power_watts': avg_cpu_gpu_power,
            'avg_soc_power_watts': avg_soc_power,
            'total_energy_joules': total_energy,
        }
    
    def clear_measurements(self):
        self.layer_measurements.clear()
        self.power_monitor.clear_measurements()

def _process_sample_to_target_layer(model, sample, target_exit_layer, confidence_mean_vec, confidence_std_vec):
    """Exact copy of the function from Main.py"""
    from Train import overlay_on_x_neutral
    
    h = overlay_on_x_neutral(sample)
    softmax_layer_input = None
    
    target_layer_idx = target_exit_layer - 1
    
    for layer_idx, (layer, softmax_layer) in enumerate(zip(model.layers, model.softmax_layers)):
        h = layer(h)
        
        if softmax_layer_input is None:
            softmax_layer_input = h.cpu()
        else:
            softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
        
        if layer_idx == target_layer_idx:
            _, softmax_output = softmax_layer(softmax_layer_input)
            return softmax_output.argmax(1)
    
    _, softmax_output = model.softmax_layers[-1](softmax_layer_input)
    return softmax_output.argmax(1)

def test_exact_phase2_conditions():
    """Test with exact Phase 2 conditions that cause hanging"""
    print("=== Testing EXACT Phase 2 Conditions ===")
    print("This replicates the exact scenario where Phase 2 hangs...")
    
    try:
        # Load the large model
        print("Loading large model...")
        model_path = '/home/jetson/Documents/github/EdgeFF/model/temp_'
        model = torch.load(model_path, weights_only=False)
        model = model.cpu()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded: {total_params:,} parameters")
        
        # Create large batch - same size as Phase 2 uses
        batch_size = 1000  # This is the batch size used in Phase 2
        print(f"Creating batch of {batch_size} samples (same as Phase 2)...")
        
        dummy_batch = torch.randn(batch_size, 784)
        exit_layer = 3  # Layer 3 is where all samples exit in the real scenario
        
        print(f"✓ Created batch of {batch_size} samples for layer {exit_layer}")
        
        # Create exact monitor as used in Phase 2
        monitor = ExactPhase2Monitor()
        
        print(f"\n--- Testing Layer {exit_layer} with {batch_size} samples ---")
        print("This should replicate the exact hanging condition...")
        
        # Start power monitoring
        print("Starting power monitoring...")
        monitor.start_layer_monitoring(exit_layer - 1, 0)
        print("✓ Power monitoring started")
        
        # Process the batch exactly as Phase 2 does
        batch_start_time = time.perf_counter()
        
        print("Processing samples (this is where it likely hangs)...")
        predictions = []
        
        with torch.no_grad():
            for i, sample in enumerate(dummy_batch):
                if i % 100 == 0:
                    print(f"  Processing sample {i+1}/{batch_size}")
                    
                    # Check memory usage periodically
                    if torch.cuda.is_available():
                        gpu_mem = torch.cuda.memory_allocated() / (1024**2)
                        print(f"    GPU memory: {gpu_mem:.1f} MB")
                
                # This is the exact function call from Phase 2
                prediction = _process_sample_to_target_layer(
                    model, sample.unsqueeze(0), exit_layer, None, None
                )
                predictions.append(prediction)
        
        batch_duration = time.perf_counter() - batch_start_time
        print(f"✓ Processed {batch_size} samples in {batch_duration:.3f} seconds")
        
        # Stop power monitoring
        print("Stopping power monitoring...")
        monitor.stop_layer_monitoring(exit_layer - 1)
        print("✓ Power monitoring stopped")
        
        # Get energy summary
        print("Getting energy summary...")
        energy_summary = monitor.get_layer_energy_summary(exit_layer - 1)
        print(f"✓ Energy summary retrieved: {energy_summary.get('total_energy_joules', 0):.6f} J")
        
        print("✅ EXACT Phase 2 test completed successfully!")
        print("   This means the hang is likely caused by something else...")
        
        return True
        
    except Exception as e:
        print(f"❌ EXACT Phase 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_smaller_batches():
    """Test if smaller batch sizes work"""
    print("\n=== Testing with Smaller Batches ===")
    
    try:
        model_path = '/home/jetson/Documents/github/EdgeFF/model/temp_'
        model = torch.load(model_path, weights_only=False)
        model = model.cpu()
        
        # Test with progressively larger batch sizes
        batch_sizes = [50, 100, 250, 500]
        
        for batch_size in batch_sizes:
            print(f"\n--- Testing batch size: {batch_size} ---")
            
            dummy_batch = torch.randn(batch_size, 784)
            monitor = ExactPhase2Monitor()
            
            monitor.start_layer_monitoring(2, 0)  # Layer 3 (0-indexed = 2)
            
            start_time = time.perf_counter()
            predictions = []
            
            with torch.no_grad():
                for i, sample in enumerate(dummy_batch):
                    prediction = _process_sample_to_target_layer(
                        model, sample.unsqueeze(0), 3, None, None
                    )
                    predictions.append(prediction)
            
            duration = time.perf_counter() - start_time
            monitor.stop_layer_monitoring(2)
            
            print(f"  ✓ Batch size {batch_size}: {duration:.3f}s ({duration/batch_size*1000:.1f}ms/sample)")
            
            # Cleanup
            del dummy_batch, predictions
            gc.collect()
        
        print("✅ All batch sizes completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Batch size test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing EXACT Phase 2 hanging conditions...")
    print("This will help identify the exact cause of the hang.")
    
    # Test 1: Exact Phase 2 conditions (1000 samples)
    print("\n" + "="*60)
    success1 = test_exact_phase2_conditions()
    
    # Test 2: Different batch sizes
    if success1:
        print("\n" + "="*60)
        success2 = test_with_smaller_batches()
    else:
        success2 = False
    
    print("\n" + "="*60)
    if success1 and success2:
        print("✅ All tests passed!")
        print("   The hang is NOT caused by batch size or large model alone.")
        print("   It might be caused by:")
        print("   1. Specific timing issues in the power monitoring thread")
        print("   2. Resource contention between jtop and model processing")
        print("   3. Memory fragmentation over time")
        print("   4. Some other environmental factor")
    else:
        print("❌ Tests failed - this confirms the exact hang condition!")
        print("   The hang occurs with:")
        print(f"   - Large model: [784, 2000, 2000, 2000, 2000]")
        print(f"   - Large batch size: 1000 samples")
        print(f"   - Power monitoring during processing")