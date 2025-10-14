#!/usr/bin/env python3
"""
Test to isolate the _process_sample_to_target_layer function
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '/home/jetson/Documents/github/EdgeFF')

def test_model_processing():
    """Test the model processing function that might be hanging"""
    print("=== Testing Model Processing Function ===")
    
    try:
        # Load the model
        print("Loading model...")
        model_path = '/home/jetson/Documents/github/EdgeFF/model/temp_'
        model = torch.load(model_path, weights_only=False)
        model = model.cpu()
        print("✓ Model loaded successfully")
        
        # Create dummy sample
        print("Creating dummy sample...")
        dummy_sample = torch.randn(1, 784)  # MNIST-like sample
        print("✓ Dummy sample created")
        
        # Test the overlay_on_x_neutral function
        print("Testing overlay_on_x_neutral function...")
        from Train import overlay_on_x_neutral
        processed_sample = overlay_on_x_neutral(dummy_sample)
        print("✓ overlay_on_x_neutral function works")
        
        # Test processing through layers
        print("Testing layer processing...")
        h = processed_sample
        softmax_layer_input = None
        target_layer_idx = 2  # Target exit layer 3 (0-indexed = 2)
        
        print(f"Model has {len(model.layers)} layers and {len(model.softmax_layers)} softmax layers")
        
        for layer_idx, (layer, softmax_layer) in enumerate(zip(model.layers, model.softmax_layers)):
            print(f"  Processing layer {layer_idx}...")
            
            # This might be where it hangs
            h = layer(h)
            print(f"    Layer {layer_idx} forward pass completed, output shape: {h.shape}")
            
            # Accumulate features for softmax
            if softmax_layer_input is None:
                softmax_layer_input = h.cpu()
            else:
                softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
            print(f"    Accumulated features shape: {softmax_layer_input.shape}")
            
            # If we've reached the target exit layer, make prediction and exit
            if layer_idx == target_layer_idx:
                print(f"    Reached target layer {target_layer_idx}, making prediction...")
                _, softmax_output = softmax_layer(softmax_layer_input)
                prediction = softmax_output.argmax(1)
                print(f"    Prediction: {prediction}")
                break
        
        print("✅ Model processing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """Test processing a small batch"""
    print("\n=== Testing Batch Processing ===")
    
    try:
        # Load the model
        print("Loading model...")
        model_path = '/home/jetson/Documents/github/EdgeFF/model/temp_'
        model = torch.load(model_path, weights_only=False)
        model = model.cpu()
        print("✓ Model loaded successfully")
        
        # Create dummy batch
        batch_size = 5
        dummy_batch = torch.randn(batch_size, 784)
        print(f"✓ Created batch of {batch_size} samples")
        
        # Process each sample in the batch
        predictions = []
        for i, sample in enumerate(dummy_batch):
            print(f"  Processing sample {i+1}/{batch_size}...")
            
            # This is the exact function call from the main code
            prediction = _process_sample_to_target_layer(
                model, sample.unsqueeze(0), 3, None, None  # target_exit_layer=3
            )
            predictions.append(prediction)
            print(f"    Sample {i+1} prediction: {prediction}")
        
        print("✅ Batch processing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def _process_sample_to_target_layer(model, sample, target_exit_layer, confidence_mean_vec, confidence_std_vec):
    """Copy of the function from Main.py for testing"""
    from Train import overlay_on_x_neutral
    
    h = overlay_on_x_neutral(sample)
    softmax_layer_input = None
    
    # Process up to the target exit layer
    target_layer_idx = target_exit_layer - 1  # Convert to 0-indexed
    
    for layer_idx, (layer, softmax_layer) in enumerate(zip(model.layers, model.softmax_layers)):
        h = layer(h)
        
        # Accumulate features for softmax
        if softmax_layer_input is None:
            softmax_layer_input = h.cpu()
        else:
            softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
        
        # If we've reached the target exit layer, make prediction and exit
        if layer_idx == target_layer_idx:
            _, softmax_output = softmax_layer(softmax_layer_input)
            return softmax_output.argmax(1)
    
    # Fallback (shouldn't reach here)
    _, softmax_output = model.softmax_layers[-1](softmax_layer_input)
    return softmax_output.argmax(1)

if __name__ == "__main__":
    print("Starting model processing test...")
    
    # Test 1: Single sample processing
    success1 = test_model_processing()
    
    # Test 2: Batch processing (this might hang)
    if success1:
        success2 = test_batch_processing()
    
    if success1 and success2:
        print("\n✅ All model processing tests passed!")
    else:
        print("\n❌ Some tests failed - this is likely where the hang occurs.")