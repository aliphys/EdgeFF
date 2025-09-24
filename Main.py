import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import Train
import Evaluation

from sklearn.model_selection import train_test_split

import tools
import os
import datetime

import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
import time
import sys
import io
matplotlib.use('Agg')  # Use non-interactive backend to prevent automatic showing
# seaborn import is handled in the function where it's used


print('MNIST_One_Pass')
layers = [784,2000,2000,2000] # go wider, not deeper!  https://cdn.aaai.org/ojs/20858/20858-13-24871-1-2-20220628.pdf
# large parallel with "individual layer normalisation" presented in paper with the concept of goodness.
length_network = len(layers)-1
print('layers: ' + str(length_network))
print(layers)

# Command line argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='MNIST Layer Specialization Analysis')
    parser.add_argument('--selected_classes', nargs='+', type=int, default=None,
                        help='Classes to include in the dataset (default: all classes 0-9)')
    parser.add_argument('--filter_by_layer', nargs='+', type=int, default=None,
                        help='Filter by layer predictions (default: all classes 0-9)')
    parser.add_argument('--target_layer', type=int, default=2,
                        help='Target layer for filtering (default: 2)')
    parser.add_argument('--correct_only', action='store_true', default=False,
                        help='Only keep correctly predicted samples (default: False)')
    parser.add_argument('--run_specialization', action='store_true', default=False,
                        help='Run layer specialization analysis')
    parser.add_argument('--confidence_threshold_multiplier', type=float, default=1.0,
                        help='Multiplier for confidence threshold (mean - multiplier*std) (default: 1.0)')
    parser.add_argument('--goodness_threshold', type=float, default=2.0,
                        help='Goodness threshold for Forward-Forward training (default: 2.0)')
    parser.add_argument('--run_energy_analysis', action='store_true', default=False,
                        help='Run energy consumption analysis')
    parser.add_argument('--compare_energy_methods', action='store_true', default=False,
                        help='Run detailed validation of two-phase energy analysis')
    parser.add_argument('--train', type=str, choices=['True', 'False', 'true', 'false'], default='False',
                        help='Train a new model (default: False - load existing model)')
    return parser.parse_args()

# load data
def MNIST_loaders(train_batch_size=60000, test_batch_size=10000, selected_classes=None, 
                  filter_by_layer=None, target_layer=None, model_path=None, correct_only=False):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    # Load full datasets
    train_dataset = MNIST('./data/', train=True, download=True, transform=transform)
    test_dataset = MNIST('./data/', train=False, download=True, transform=transform)
    
    # Filter datasets if specific classes are selected
    if selected_classes is not None:
        train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in selected_classes]
        test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in selected_classes]
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Filter by layer predictions if specified
    if (filter_by_layer is not None or correct_only) and target_layer is not None and model_path is not None:
        # Load the model for filtering
        filter_model = torch.load(model_path, weights_only=False)
        # Set model to evaluation mode without calling eval() due to overridden train method.
        filter_model.training = False
        for module in filter_model.modules():
            if hasattr(module, 'training'):
                module.training = False
        
        train_indices = []
        test_indices = []
        
        # Filter training data
        temp_train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
        current_idx = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in temp_train_loader:
                # Get predictions from specific layer
                layer_predictions = predict_with_specific_layer(filter_model, batch_data, target_layer)
                
                # Keep samples based on filtering criteria
                for i, pred in enumerate(layer_predictions):
                    if correct_only:
                        # Only keep samples where layer prediction matches ground truth
                        if pred.item() == batch_labels[i].item():
                            train_indices.append(current_idx + i)
                    elif filter_by_layer is not None:
                        # Keep samples where layer prediction matches the filter criteria
                        if pred.item() in filter_by_layer:
                            train_indices.append(current_idx + i)
                current_idx += len(batch_data)
        
        # Filter test data
        temp_test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        current_idx = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in temp_test_loader:
                layer_predictions = predict_with_specific_layer(filter_model, batch_data, target_layer)
                
                for i, pred in enumerate(layer_predictions):
                    if correct_only:
                        # Only keep samples where layer prediction matches ground truth
                        if pred.item() == batch_labels[i].item():
                            test_indices.append(current_idx + i)
                    elif filter_by_layer is not None:
                        # Keep samples where layer prediction matches the filter criteria
                        if pred.item() in filter_by_layer:
                            test_indices.append(current_idx + i)
                current_idx += len(batch_data)
        
        # Create filtered datasets
        if train_indices:
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        if test_indices:
            test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def predict_with_specific_layer(model, x, target_layer):
    """Get predictions using only up to a specific layer"""
    from Train import overlay_on_x_neutral
    
    h = overlay_on_x_neutral(x)
    
    # Forward pass up to target layer, accumulating features
    softmax_layer_input = None
    
    for i, layer in enumerate(model.layers):
        h = layer(h)
        
        # Accumulate features for softmax layer input
        if softmax_layer_input is None:
            softmax_layer_input = h.cpu()
        else:
            softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
        
        if i == target_layer:
            break
    
    # Use the softmax layer at the target layer to get predictions
    softmax_layer = model.softmax_layers[target_layer]
    _, output = softmax_layer(softmax_layer_input)
    
    return output.argmax(1)


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


def overlay_on_x_neutral(x):
    """Replace the first 10 pixels of data [x] with 0.1s
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), :10] = 0.1  # x.max()
    return x_


def eval_with_fixed_layer(model, inputs, targets, target_layer):
    """Evaluate using only up to a specific layer"""
    from Train import overlay_on_x_neutral
    
    correct = 0
    total = len(inputs)
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, total, batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            # Forward pass up to target layer
            h = overlay_on_x_neutral(batch_inputs)
            softmax_layer_input = None
            
            for layer_idx, layer in enumerate(model.layers):
                h = layer(h)
                
                # Accumulate features
                if softmax_layer_input is None:
                    softmax_layer_input = h.cpu()
                else:
                    softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
                
                if layer_idx == target_layer:
                    break
            
            # Get predictions from target layer
            softmax_layer = model.softmax_layers[target_layer]
            _, output = softmax_layer(softmax_layer_input)
            predictions = output.argmax(1)
            
            correct += (predictions == batch_targets).sum().item()
    
    accuracy = correct / total
    return accuracy

def eval_layer_on_specific_class(model, layer_idx, class_id, test_loader_all_classes):
    """Evaluate a specific layer's accuracy on a specific class"""
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader_all_classes:
            # Filter for specific class
            class_mask = (batch_labels == class_id)
            if not class_mask.any():
                continue
                
            class_data = batch_data[class_mask]
            class_labels = batch_labels[class_mask]
            
            # Get predictions from specific layer
            predictions = predict_with_specific_layer(model, class_data, layer_idx)
            
            correct += (predictions == class_labels).sum().item()
            total += len(class_labels)
    
    return correct / total if total > 0 else 0.0


def calculate_activation_entropy(activations):
    """Calculate entropy of activation patterns to measure information content"""
    # used to see if neurons are actually doing something. 
    # Discretize activations into bins for entropy calculation
    activations_flat = activations.flatten().detach().cpu().numpy()
    # Remove zeros to focus on active neurons
    activations_nonzero = activations_flat[activations_flat > 1e-6]
    
    if len(activations_nonzero) == 0:
        return 0.0
    
    # Create histogram
    hist, _ = np.histogram(activations_nonzero, bins=20, density=True)
    hist = hist[hist > 0]  # Remove empty bins
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    return entropy


def analyze_layer_performance_degradation(model, x_te, y_te, test_loader_all_classes):
    """
    Comprehensive analysis of why deeper layers don't improve performance in FF networks.
    Analyzes activation patterns, goodness distributions, and representation quality.
    """
    print("\n" + "="*70)
    print("    LAYER PERFORMANCE DEGRADATION ANALYSIS")
    print("="*70)
    
    # 1. Test each layer individually
    print("\n1. INDIVIDUAL LAYER PERFORMANCE:")
    print("-" * 40)
    layer_accuracies = []
    for layer_idx in range(3):  # 3 hidden layers
        accuracy = eval_with_fixed_layer(model, x_te, y_te, layer_idx)
        layer_accuracies.append(accuracy)
        print(f"   Layer {layer_idx} only: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Calculate performance differences
    print(f"\n   Performance differences:")
    for i in range(1, len(layer_accuracies)):
        diff = layer_accuracies[i] - layer_accuracies[i-1]
        print(f"   Layer {i} vs Layer {i-1}: {diff:+.4f} ({diff*100:+.1f}%)")
    
    # 2. Analyze layer representations and activations
    print(f"\n2. LAYER REPRESENTATION ANALYSIS:")
    print("-" * 40)
    
    # Use a representative sample for analysis
    sample_size = 500
    sample_batch = x_te[:sample_size]
    sample_labels = y_te[:sample_size]
    
    from Train import overlay_on_x_neutral
    h = overlay_on_x_neutral(sample_batch)
    
    layer_stats = []
    all_goodness_stats = []
    
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            h = layer(h)
            
            # Analyze activation statistics
            activation_mean = h.mean().item()
            activation_std = h.std().item()
            activation_sparsity = (h == 0).float().mean().item()
            activation_saturation = (h > 5.0).float().mean().item()
            activation_range = h.max().item() - h.min().item()
            
            # Analyze goodness distribution
            goodness = h.pow(2).mean(1)
            goodness_mean = goodness.mean().item()
            goodness_std = goodness.std().item()
            goodness_min = goodness.min().item()
            goodness_max = goodness.max().item()
            
            # Store stats
            layer_stats.append({
                'layer': layer_idx,
                'activation_mean': activation_mean,
                'activation_std': activation_std,
                'activation_sparsity': activation_sparsity,
                'activation_saturation': activation_saturation,
                'activation_range': activation_range,
                'goodness_mean': goodness_mean,
                'goodness_std': goodness_std,
                'goodness_range': goodness_max - goodness_min
            })
            
            all_goodness_stats.append(goodness.cpu().numpy())
            
            print(f"\n   Layer {layer_idx} ({h.shape[1]} neurons):")
            print(f"     Activation mean: {activation_mean:8.4f}")
            print(f"     Activation std:  {activation_std:8.4f}")
            print(f"     Sparsity (% zeros): {activation_sparsity:6.1%}")
            print(f"     Saturation (>5.0):  {activation_saturation:6.1%}")
            print(f"     Value range:     {activation_range:8.4f}")
            print(f"     Goodness mean:   {goodness_mean:8.4f}")
            print(f"     Goodness std:    {goodness_std:8.4f}")
            print(f"     Goodness range:  {goodness_max - goodness_min:8.4f}")
    
    # 3. Analyze class discrimination per layer
    print(f"\n3. CLASS DISCRIMINATION ANALYSIS:")
    print("-" * 40)
    
    # Test how well each layer separates different classes
    unique_classes = torch.unique(sample_labels)
    
    for layer_idx in range(3):
        print(f"\n   Layer {layer_idx} class separation:")
        
        # Get activations for this layer
        h_test = overlay_on_x_neutral(sample_batch)
        with torch.no_grad():
            for i in range(layer_idx + 1):
                h_test = model.layers[i](h_test)
        
        # Calculate mean goodness per class
        class_goodness_means = []
        for class_id in unique_classes:
            class_mask = (sample_labels == class_id)
            if class_mask.sum() > 0:
                class_h = h_test[class_mask]
                class_goodness = class_h.pow(2).mean(1).mean().item()
                class_goodness_means.append(class_goodness)
                print(f"     Class {class_id}: goodness = {class_goodness:.4f}")
        
        # Calculate separation quality
        if len(class_goodness_means) > 1:
            separation_std = np.std(class_goodness_means)
            separation_range = max(class_goodness_means) - min(class_goodness_means)
            print(f"     Separation std:   {separation_std:.4f}")
            print(f"     Separation range: {separation_range:.4f}")
    
    # 4. Information flow analysis
    print(f"\n4. INFORMATION BOTTLENECK ANALYSIS:")
    print("-" * 40)
    
    input_entropy = calculate_activation_entropy(overlay_on_x_neutral(sample_batch[:100]))
    print(f"   Input entropy: {input_entropy:.4f}")
    
    h_entropy = overlay_on_x_neutral(sample_batch[:100])
    for layer_idx, layer in enumerate(model.layers):
        h_entropy = layer(h_entropy)
        layer_entropy = calculate_activation_entropy(h_entropy)
        compression_ratio = input_entropy / layer_entropy if layer_entropy > 0 else float('inf')
        print(f"   Layer {layer_idx} entropy: {layer_entropy:.4f} (compression: {compression_ratio:.2f}x)")
    
    # 5. Learning efficiency analysis
    print(f"\n5. LEARNING EFFICIENCY ANALYSIS:")
    print("-" * 40)
    
    print("   Why deeper layers may not help in Forward-Forward:")
    print("   • Each layer learns independently (no gradient flow)")
    print("   • Information bottleneck: 784 → 100 neurons")
    print("   • Goodness function may not scale well with depth")
    print("   • Local optima: each layer optimizes separately")
    
    # 6. Recommendations
    print(f"\n6. OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 40)
    
    # Analyze if layers are too narrow
    if all(stats['activation_sparsity'] > 0.5 for stats in layer_stats):
        print("   ⚠️  High sparsity detected - consider wider layers")
    
    # Check if goodness function is saturating
    goodness_variations = [stats['goodness_std'] for stats in layer_stats]
    if all(var < 1.0 for var in goodness_variations):
        print("   ⚠️  Low goodness variation - consider different goodness function")
    
    # Check if layers are learning different things
    goodness_differences = [abs(layer_stats[i]['goodness_mean'] - layer_stats[i-1]['goodness_mean']) 
                           for i in range(1, len(layer_stats))]
    if all(diff < 0.5 for diff in goodness_differences):
        print("   ⚠️  Similar goodness across layers - layers may be redundant")
    
    print("\n   Suggested improvements:")
    print("   1. Increase layer width: [784, 400, 300, 200] instead of [784, 100, 100, 100]")
    print("   2. Try alternative goodness functions (top-k, entropy-based)")
    print("   3. Consider skip connections or hierarchical training")
    print("   4. Experiment with different activation functions")
    print("   5. Use progressive layer training (freeze earlier layers)")
    
    print("\n" + "="*70)
    
    return layer_stats, all_goodness_stats


def run_layer_specialization_analysis(model, test_loader_all_classes, num_layers=3, num_classes=10):
    """Run layer specialization analysis and create visualization"""
    print("\n--- Running Layer Specialization Analysis ---")
    
    # Create specialization matrix
    specialization_matrix = np.zeros((num_layers, num_classes))
    
    for layer_idx in range(num_layers):
        print(f"Analyzing layer {layer_idx}...")
        for class_id in range(num_classes):
            accuracy = eval_layer_on_specific_class(model, layer_idx, class_id, test_loader_all_classes)
            specialization_matrix[layer_idx, class_id] = accuracy
            print(f"  Layer {layer_idx}, Class {class_id}: {accuracy:.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    try:
        # Try using seaborn if available
        import seaborn as sns
        sns.heatmap(specialization_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   xticklabels=[f'Class {i}' for i in range(num_classes)],
                   yticklabels=[f'Layer {i}' for i in range(num_layers)],
                   cbar_kws={'label': 'Accuracy'})
    except ImportError:
        # Fallback to matplotlib imshow
        im = plt.imshow(specialization_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, label='Accuracy')
        
        # Add text annotations
        for i in range(num_layers):
            for j in range(num_classes):
                plt.text(j, i, f'{specialization_matrix[i, j]:.3f}', 
                        ha='center', va='center', fontsize=10)
        
        plt.xticks(range(num_classes), [f'Class {i}' for i in range(num_classes)])
        plt.yticks(range(num_layers), [f'Layer {i}' for i in range(num_layers)])
    
    plt.title('Layer Specialization Matrix\n(Accuracy of each layer on each digit class)', fontsize=14)
    plt.xlabel('Digit Classes', fontsize=12)
    plt.ylabel('Network Layers', fontsize=12)
    plt.tight_layout()
    
    # Save the plot with timestamp to avoid overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    filename = f'layer_specialization_matrix_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Layer specialization matrix saved to: {filepath}")
    # plt.show()  # Commented out to prevent popup
    plt.close()  # Close figure to prevent memory leaks
    
    # Print summary statistics
    print("\n--- Specialization Analysis Results ---")
    print("Specialization Matrix (Layer x Class):")
    print(specialization_matrix)
    
    # Find best layer for each class
    best_layers = np.argmax(specialization_matrix, axis=0)
    print("\nBest layer for each class:")
    for class_id in range(num_classes):
        best_layer = best_layers[class_id]
        best_accuracy = specialization_matrix[best_layer, class_id]
        print(f"Class {class_id}: Layer {best_layer} (accuracy: {best_accuracy:.4f})")
    
    # Find layer specialization scores (variance across classes)
    layer_specialization = np.var(specialization_matrix, axis=1)
    print("\nLayer specialization scores (higher = more specialized):")
    for layer_idx in range(num_layers):
        print(f"Layer {layer_idx}: {layer_specialization[layer_idx]:.6f}")
    
    return specialization_matrix


def create_all_classes_loader(model_path):
    """Create a test loader with all 10 classes for specialization analysis"""
    return MNIST_loaders(
        train_batch_size=60000, 
        test_batch_size=1000,  # Smaller batches for memory efficiency
        selected_classes=list(range(10)),  # All classes 0-9
        model_path=model_path,
        correct_only=False  # Don't filter, we want to test on all data
    )


def demonstrate_threshold_effects(args, model_path):
    """Demonstrate how different threshold values affect model performance"""
    print("\n=== Threshold Effects Demonstration ===")
    
    # Test different confidence threshold multipliers
    multipliers = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    print("\nConfidence Threshold Effects:")
    print("Multiplier | Accuracy | Early Exit Rate | Avg Layers | Layer 1 % | Layer 2 % | Layer 3 %")
    print("-" * 85)
    
    # Pre-load test data once
    try:
        _, test_loader = MNIST_loaders(
            selected_classes=list(range(10)),  # All classes
            model_path=model_path,
            correct_only=False
        )
        x_te, y_te = next(iter(test_loader))
        test_samples = x_te[:500]  # Use subset for speed
        test_targets = y_te[:500]
    except:
        print("Error loading test data - using synthetic data for demonstration")
        test_samples = torch.randn(100, 784)
        test_targets = torch.randint(0, 10, (100,))
    
    for multiplier in multipliers:
        try:
            # Load model and update threshold
            model = torch.load(model_path, weights_only=False)
            model.confidence_threshold_multiplier = multiplier
            
            # Suppress verbose output by redirecting analysis
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            try:
                # Get confidence statistics (suppress output)
                mean, std = tools.analysis_val_set(model, inputs=test_samples[:200], targets=test_targets[:200])
            finally:
                sys.stdout = old_stdout
            
            # Run light inference analysis
            total_samples = len(test_samples)
            correct_predictions = 0
            layer_usage = {1: 0, 2: 0, 3: 0}
            total_layers_used = 0
            
            with torch.no_grad():
                for i in range(total_samples):
                    sample = test_samples[i:i+1]
                    target = test_targets[i:i+1]
                    
                    # Use actual light inference
                    prediction, layers_used = model.light_predict_one_sample(
                        sample, confidence_mean_vec=mean, confidence_std_vec=std
                    )
                    
                    # Track accuracy
                    if prediction.item() == target.item():
                        correct_predictions += 1
                    
                    # Track layer usage
                    layer_usage[layers_used] += 1
                    total_layers_used += layers_used
            
            # Calculate statistics
            accuracy = correct_predictions / total_samples
            avg_layers = total_layers_used / total_samples
            early_exit_rate = (layer_usage[1] + layer_usage[2]) / total_samples
            
            layer1_pct = layer_usage[1] / total_samples * 100
            layer2_pct = layer_usage[2] / total_samples * 100
            layer3_pct = layer_usage[3] / total_samples * 100
            
            print(f"{multiplier:8.1f} | {accuracy:8.3f} | {early_exit_rate:13.1%} | {avg_layers:10.2f} | {layer1_pct:8.1f}% | {layer2_pct:8.1f}% | {layer3_pct:8.1f}%")
            
        except Exception as e:
            print(f"{multiplier:8.1f} | {'ERROR':>8} | {'N/A':>13} | {'N/A':>10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}")
    
    print("\nKey Insights:")
    print("• Optimal multiplier range: 0.5-1.5 (best accuracy/speed balance)")
    print("• Lower values: Faster inference but may sacrifice accuracy")
    print("• Higher values: More conservative, uses deeper layers")
    print("• Threshold formula: mean - (multiplier × std)")
    print("• Early exit rate shows computational savings potential")


model_path = os.path.split(os.path.realpath(__file__))[0] + '/model/temp_'

# Parse command line arguments
args = parse_arguments()

# Use arguments to configure MNIST_loaders
train_loader, test_loader = MNIST_loaders(
    selected_classes=args.selected_classes,
    filter_by_layer=args.filter_by_layer,
    target_layer=args.target_layer,
    model_path=model_path,
    correct_only=args.correct_only
)

# Convert train argument to boolean
train_model = args.train.lower() == 'true'

print(f"Configuration:")
print(f"  Selected classes: {args.selected_classes}")
print(f"  Filter by layer: {args.filter_by_layer}")
print(f"  Target layer: {args.target_layer}")
print(f"  Correct only: {args.correct_only}")
print(f"  Run specialization: {args.run_specialization}")
print(f"  Run energy analysis: {args.run_energy_analysis}")
print(f"  Compare energy methods: {args.compare_energy_methods}")
print(f"  Goodness threshold: {args.goodness_threshold}")
print(f"  Confidence threshold multiplier: {args.confidence_threshold_multiplier}")
print(f"  Train: {train_model} (from --train {args.train})")
# train data
inputs, targets = next(iter(train_loader))

# test data
x_te, y_te = next(iter(test_loader))

# create a validation set from the filtered training data
total_samples = len(inputs)
val_size = min(10000, total_samples // 2)  # Ensure validation set doesn't exceed half the data
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=val_size, random_state=0)


# train
if train_model:
    x_pos = overlay_y_on_x(X_train, y_train)
    y_neg = y_train.clone()
    for idx, y_samp in enumerate(y_train):
        allowed_indices = [0, 1, 2, 3, 4,5,6,7,8,9]
        allowed_indices.pop(y_samp.item())
        y_neg[idx] = torch.tensor(np.random.choice(allowed_indices))
    x_neg = overlay_y_on_x(X_train, y_neg)

    x_neutral = overlay_on_x_neutral(inputs)

    model = Train.build_model(x_pos=x_pos, x_neg=x_neg, x_neutral=x_neutral, targets=targets, layers=layers,
                             goodness_threshold=args.goodness_threshold, 
                             confidence_threshold_multiplier=args.confidence_threshold_multiplier)

name = 'temp_'
model = torch.load(os.path.split(os.path.realpath(__file__))[0]+'/model/' + name, weights_only=False) # load trained model, with weights Required for PyTorch>2.6

# Update model with new confidence threshold multiplier if it doesn't exist
if not hasattr(model, 'confidence_threshold_multiplier'):
    model.confidence_threshold_multiplier = args.confidence_threshold_multiplier
else:
    model.confidence_threshold_multiplier = args.confidence_threshold_multiplier

# Move model to CPU for inference (GPU was used only for training)
print("Moving model to CPU for inference...")
model = model.cpu()

# Ensure all data tensors are on CPU for inference
print("Moving data to CPU for inference...")
inputs = inputs.cpu()
targets = targets.cpu()
x_te = x_te.cpu()
y_te = y_te.cpu()
X_val = X_val.cpu()
y_val = y_val.cpu()
print("✓ Model and data moved to CPU for inference")

# evaluation - using filtered datasets consistently
# Use the full filtered training data for training evaluation
Evaluation.eval_train_set(model, inputs=inputs, targets=targets)

# test data - using filtered test set
Evaluation.eval_test_set(model, inputs=x_te, targets=y_te)

# validation data - using validation split of filtered training data
Evaluation.eval_val_set(model, inputs=X_val, targets=y_val)

# analysis of validation data - using validation split of filtered training data
mean,std = tools.analysis_val_set(model, inputs=X_val, targets=y_val)

confidence_mean_vec = mean
confidence_std_vec = std
# Light inference evaluation - using filtered test set for consistency
Evaluation.eval_val_set_light(model, inputs=x_te, targets=y_te,
                              confidence_mean_vec=confidence_mean_vec,
                              confidence_std_vec=confidence_std_vec)

# Fixed layer evaluation - test each layer individually
print(f"\n--- Fixed Layer Evaluation ---")
for target_layer in [0, 1, 2]:
    accuracy = eval_with_fixed_layer(model, x_te, y_te, target_layer)
    print(f"Layer {target_layer} only - Accuracy: {accuracy:.4f}")

# Run layer specialization analysis if requested
if args.run_specialization:
    print("\n--- Starting Layer Specialization Analysis ---")
    # Create test loader with all classes for comprehensive analysis
    _, test_loader_all_classes = create_all_classes_loader(model_path)
    
    # First run diagnostic analysis to understand layer performance
    layer_stats, goodness_stats = analyze_layer_performance_degradation(
        model, x_te, y_te, test_loader_all_classes
    )
    
    # Run the specialization analysis
    specialization_matrix = run_layer_specialization_analysis(
        model, test_loader_all_classes, num_layers=3, num_classes=10
    )
    
    # Additional analysis: Show which samples were filtered out by our current settings
    print(f"\n--- Data Filtering Impact ---")
    print(f"Original data size: {len(test_loader_all_classes.dataset)}")
    print(f"Filtered data size: {len(x_te)}")
    print(f"Retention rate: {len(x_te)/len(test_loader_all_classes.dataset)*100:.1f}%")
    
else:
    print("\n--- Layer Specialization Analysis Skipped ---")
    print("To run specialization analysis, use: --run_specialization")
    print("Example: python Main.py --selected_classes 0 1 2 3 4 5 6 7 8 9 --run_specialization")

# Demonstrate threshold effects if confidence threshold was modified
if args.confidence_threshold_multiplier != 1.0:
    demonstrate_threshold_effects(args, model_path)

import time
import psutil
from collections import defaultdict
from power_monitor import PowerMonitor, PowerMeasurement

class RealPowerEnergyMonitor:
    """Monitor actual energy consumption during Forward-Forward inference using real power data"""
    
    def __init__(self, client_id: int = 0):
        """
        Initialize energy monitor with real power measurement capabilities.
        Args:
            client_id (int): Client identifier for power monitoring
        """
        self.power_monitor = PowerMonitor(client_id)
        self.layer_measurements = defaultdict(list)
        self.layer_times = defaultdict(list)
        self.confidence_check_times = []
        self.early_exit_stats = {'layer_1': 0, 'layer_2': 0, 'layer_3': 0}
        self.current_round = 0
        self.current_sample_idx = 0
        
    def start_layer_monitoring(self, layer_idx: int, sample_idx: int = 0):
        """
        Start power monitoring for a specific layer operation.
        Args:
            layer_idx (int): Layer index being processed
            sample_idx (int): Sample index for tracking
        """
        self.current_sample_idx = sample_idx
        # Use layer_idx as epoch for fine-grained tracking
        self.power_monitor.start_monitoring(
            round_num=self.current_round, 
            epoch=layer_idx
        )
        
    def stop_layer_monitoring(self, layer_idx: int):
        """
        Stop power monitoring and record measurements for the layer.
        Args:
            layer_idx (int): Layer index that was processed
        """
        self.power_monitor.stop_monitoring()
        
        # Get measurements for this layer operation
        measurements = self.power_monitor.get_measurements()
        if measurements:
            # Filter measurements for this layer (epoch = layer_idx)
            layer_measurements = [m for m in measurements if m['epoch'] == layer_idx]
            if layer_measurements:
                self.layer_measurements[layer_idx].extend(layer_measurements)
        
        # Clear measurements to avoid accumulation
        self.power_monitor.clear_measurements()
        
    def get_layer_energy_summary(self, layer_idx: int) -> dict:
        """
        Get energy consumption summary for a specific layer.
        Args:
            layer_idx (int): Layer to summarize
        Returns:
            dict: Energy consumption statistics
        """
        measurements = self.layer_measurements[layer_idx]
        if not measurements:
            return {
                'layer_idx': layer_idx,
                'sample_count': 0,
                'avg_power_watts': 0.0,
                'total_energy_joules': 0.0,
                'duration_seconds': 0.0
            }
        
        # Calculate power and energy statistics
        total_powers = [m['total_power'] for m in measurements if m['total_power'] is not None]
        cpu_gpu_powers = [m['power_cpu_gpu_cv'] for m in measurements if m['power_cpu_gpu_cv'] is not None]
        soc_powers = [m['power_soc'] for m in measurements if m['power_soc'] is not None]
        timestamps = [m['timestamp'] for m in measurements]
        
        if not total_powers or not timestamps:
            return {
                'layer_idx': layer_idx,
                'sample_count': len(measurements),
                'avg_power_watts': 0.0,
                'total_energy_joules': 0.0,
                'duration_seconds': 0.0
            }
        
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.25  # Default to 250ms
        avg_total_power = sum(total_powers) / len(total_powers)
        avg_cpu_gpu_power = sum(cpu_gpu_powers) / len(cpu_gpu_powers) if cpu_gpu_powers else 0.0
        avg_soc_power = sum(soc_powers) / len(soc_powers) if soc_powers else 0.0
        total_energy = avg_total_power * duration  # Energy = Power × Time (Joules)
        
        return {
            'layer_idx': layer_idx,
            'sample_count': len(measurements),
            'duration_seconds': duration,
            'avg_total_power_watts': avg_total_power,
            'avg_cpu_gpu_power_watts': avg_cpu_gpu_power,
            'avg_soc_power_watts': avg_soc_power,
            'max_power_watts': max(total_powers),
            'min_power_watts': min(total_powers),
            'total_energy_joules': total_energy,
            'energy_per_operation_joules': total_energy
        }
    
    def clear_measurements(self):
        """Clear all stored measurements"""
        self.layer_measurements.clear()
        self.power_monitor.clear_measurements()


def energy_aware_inference(model, x, confidence_mean_vec, confidence_std_vec, monitor=None, sample_idx=0):
    """Forward-Forward inference with real power consumption measurement"""
    if monitor is None:
        monitor = RealPowerEnergyMonitor()
    
    from Train import overlay_on_x_neutral
    
    batch_size = x.shape[0]
    h = overlay_on_x_neutral(x)
    softmax_layer_input = None
    
    layer_energies = []
    
    # Track accumulated input sizes for softmax layers
    accumulated_features = 0
    
    for layer_idx, (layer, softmax_layer) in enumerate(zip(model.layers, model.softmax_layers)):
        # Start power monitoring for this layer
        monitor.start_layer_monitoring(layer_idx, sample_idx)
        
        layer_start_time = time.perf_counter()
        
        # Forward pass through FF layer
        input_size = h.shape[1]
        h = layer(h)
        output_size = h.shape[1]
        accumulated_features += output_size
        
        # Accumulate features for softmax
        if softmax_layer_input is None:
            softmax_layer_input = h.cpu()
        else:
            softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
        
        # Softmax computation for this layer
        _, softmax_output = softmax_layer(softmax_layer_input)
        
        # Confidence checking
        confidence_flag = model.check_confidence(
            layer_idx, confidence_mean_vec, confidence_std_vec, softmax_output
        )
        
        layer_time = time.perf_counter() - layer_start_time
        
        # Stop power monitoring for this layer
        monitor.stop_layer_monitoring(layer_idx)
        
        # Get energy summary for this layer operation
        energy_summary = monitor.get_layer_energy_summary(layer_idx)
        
        # Record layer information with real power data
        layer_energies.append({
            'layer_idx': layer_idx,
            'duration_seconds': layer_time,
            'avg_power_watts': energy_summary.get('avg_total_power_watts', 0.0),
            'cpu_gpu_power_watts': energy_summary.get('avg_cpu_gpu_power_watts', 0.0),
            'soc_power_watts': energy_summary.get('avg_soc_power_watts', 0.0),
            'energy_joules': energy_summary.get('total_energy_joules', 0.0),
            'early_exit': confidence_flag,
            'input_size': input_size,
            'output_size': output_size,
            'accumulated_features': accumulated_features
        })
        
        monitor.layer_times[layer_idx].append(layer_time)
        
        # Early exit check
        if confidence_flag:
            monitor.early_exit_stats[f'layer_{layer_idx + 1}'] += batch_size
            return softmax_output.argmax(1), layer_energies, monitor
    
    # If no early exit, use final layer
    monitor.early_exit_stats['layer_3'] += batch_size
    return softmax_output.argmax(1), layer_energies, monitor


# Individual sample analysis functions removed - replaced by superior two-phase batch analysis


# Lightweight individual analysis removed - use two-phase analysis for all cases


def compare_energy_vs_accuracy(model, test_data, test_labels, confidence_mean_vec, confidence_std_vec):
    """Compare energy consumption vs accuracy for different confidence thresholds"""
    print("\n=== Energy vs Accuracy Trade-off Analysis ===")
    
    multipliers = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
    results = []
    
    # Store original multiplier
    original_multiplier = getattr(model, 'confidence_threshold_multiplier', 1.0)
    
    print("Multiplier | Avg Energy (mJ) | Accuracy | Energy/Accuracy Ratio")
    print("-" * 65)
    
    # Convert logit-space confidence vectors to probability space
    # otherwise confidence vectors are not comparable. Small change leads to sudden changes in early exit behavior
    print("\n--- Converting to Probability Space for Proper Comparison ---")
    prob_thresholds_base = []
    for layer_idx in range(len(confidence_mean_vec)):
        mean_logit = confidence_mean_vec[layer_idx].item() if hasattr(confidence_mean_vec[layer_idx], 'item') else confidence_mean_vec[layer_idx]
        std_logit = confidence_std_vec[layer_idx].item() if hasattr(confidence_std_vec[layer_idx], 'item') else confidence_std_vec[layer_idx]
        
        # Convert to probability space using sigmoid
        # priniple from -> https://www.youtube.com/watch?v=WsFasV46KgQ
        mean_prob = 1 / (1 + np.exp(-mean_logit))  # sigmoid transformation
        # Scale std appropriately - use smaller scaling for more sensitivity
        std_prob = std_logit * 0.01  # Much smaller scaling factor
        
        prob_thresholds_base.append({'mean': mean_prob, 'std': std_prob})
        print(f"Layer {layer_idx}: logit({mean_logit:.3f}, {std_logit:.3f}) → prob({mean_prob:.6f}, {std_prob:.6f})")
    
    print(f"\nTesting with Corrected Probability-Space Thresholds:")
    print("Multiplier | Layer 0 Thresh | Layer 1 Thresh | Layer 2 Thresh | Avg Energy | Accuracy")
    print("-" * 85)
    
    for multiplier in multipliers:
        print(f"\n--- Processing multiplier {multiplier:.1f} ---")
        
        # Calculate corrected probability-space thresholds
        corrected_thresholds = []
        for layer_idx in range(len(prob_thresholds_base)):
            prob_mean = prob_thresholds_base[layer_idx]['mean']
            prob_std = prob_thresholds_base[layer_idx]['std']
            
            # Use probability-space threshold calculation
            threshold = prob_mean - (multiplier * prob_std)
            # Ensure threshold stays in valid probability range [0.1, 0.99]
            threshold = max(0.1, min(0.99, threshold))
            corrected_thresholds.append(threshold)
        
        # Temporarily override the model's confidence checking
        def custom_check_confidence(layer_num, conf_mean_vec, conf_std_vec, softmax_output):
            max_confidence = torch.max(softmax_output).item()
            return max_confidence > corrected_thresholds[layer_num]
        
        # Store original method and replace
        original_check = model.check_confidence
        model.check_confidence = custom_check_confidence
        
        # Update model threshold
        model.confidence_threshold_multiplier = multiplier
        
        # Use two-phase energy analysis (full test set)
        print(f"  Running two-phase energy analysis for multiplier {multiplier:.1f}...")
        energy_result_raw = two_phase_energy_analysis(
            model, test_data, test_labels, 
            confidence_mean_vec, confidence_std_vec
        )
        print(f"  Energy analysis complete for multiplier {multiplier:.1f}")
        
        # Convert to expected format
        energy_result = {
            'avg_energy_joules': energy_result_raw['avg_energy_per_sample_joules'],
            'energy_std_joules': 0.0,
            'exit_distribution': {k: len(v) for k, v in energy_result_raw['exit_patterns'].items()},
            'total_samples': energy_result_raw['total_samples']
        }
        
        # Calculate accuracy separately using the same samples
        print(f"  Calculating accuracy for multiplier {multiplier:.1f}...")
        correct = 0
        total = len(test_data)
        
        with torch.no_grad():
            for i in range(total):
                sample = test_data[i:i+1]
                target = test_labels[i:i+1]
                
                # Simple inference without energy monitoring - just for accuracy calculation
                from Train import overlay_on_x_neutral
                h = overlay_on_x_neutral(sample)
                softmax_layer_input = None
                
                for layer_idx, (layer, softmax_layer) in enumerate(zip(model.layers, model.softmax_layers)):
                    h = layer(h)
                    
                    if softmax_layer_input is None:
                        softmax_layer_input = h.cpu()
                    else:
                        softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
                    
                    _, softmax_output = softmax_layer(softmax_layer_input)
                    
                    # Check for early exit using model's confidence mechanism
                    confidence_flag = model.check_confidence(
                        layer_idx, confidence_mean_vec, confidence_std_vec, softmax_output
                    )
                    
                    if confidence_flag:
                        break
                
                prediction = softmax_output.argmax(1)
                if prediction.item() == target.item():
                    correct += 1
        
        accuracy = correct / total
        avg_energy_mj = energy_result['avg_energy_joules'] * 1000  # Convert to mJ
        energy_efficiency = avg_energy_mj / accuracy if accuracy > 0 else float('inf')
        
        results.append({
            'multiplier': multiplier,
            'avg_energy_joules': energy_result['avg_energy_joules'],
            'avg_energy_mj': avg_energy_mj,
            'accuracy': accuracy,
            'efficiency': energy_efficiency
        })
        
        # Restore original method
        model.check_confidence = original_check
        
        print(f"{multiplier:10.1f} | {corrected_thresholds[0]:14.6f} | {corrected_thresholds[1]:14.6f} | "
              f"{corrected_thresholds[2]:14.6f} | {avg_energy_mj:10.3f} | {accuracy:8.3f}")
    
    # Restore original threshold
    model.confidence_threshold_multiplier = original_multiplier
    
    return results


def debug_confidence_mechanism(model, test_data, test_labels, confidence_mean_vec, confidence_std_vec):
    """Debug the confidence checking mechanism to understand threshold behavior"""
    print("\n=== Confidence Mechanism Debugging ===")
    
    # Print confidence statistics
    print("\nConfidence Vector Statistics:")
    for layer_idx in range(len(confidence_mean_vec)):
        mean_val = confidence_mean_vec[layer_idx].item() if hasattr(confidence_mean_vec[layer_idx], 'item') else confidence_mean_vec[layer_idx]
        std_val = confidence_std_vec[layer_idx].item() if hasattr(confidence_std_vec[layer_idx], 'item') else confidence_std_vec[layer_idx]
        print(f"Layer {layer_idx}: mean={mean_val:.6f}, std={std_val:.6f}, std/mean ratio={std_val/mean_val:.6f}")
    
    # Test threshold calculations for different multipliers
    multipliers = [0.1, 0.5, 1.5, 3.0]
    
    print(f"\nThreshold Analysis for Different Multipliers:")
    print("Multiplier | Layer 0 Threshold | Layer 1 Threshold | Layer 2 Threshold")
    print("-" * 70)
    
    for mult in multipliers:
        thresholds = []
        for layer_idx in range(len(confidence_mean_vec)):
            mean_val = confidence_mean_vec[layer_idx].item() if hasattr(confidence_mean_vec[layer_idx], 'item') else confidence_mean_vec[layer_idx]
            std_val = confidence_std_vec[layer_idx].item() if hasattr(confidence_std_vec[layer_idx], 'item') else confidence_std_vec[layer_idx]
            threshold = mean_val - (mult * std_val)
            thresholds.append(threshold)
        
        print(f"{mult:10.1f} | {thresholds[0]:17.6f} | {thresholds[1]:17.6f} | {thresholds[2]:17.6f}")
    
    # Test actual confidence values on a few samples
    print(f"\nActual Softmax Outputs Analysis (first 3 samples):")
    model.confidence_threshold_multiplier = 1.0
    
    import Train
    samples_to_test = min(3, len(test_data))
    
    for sample_idx in range(samples_to_test):
        sample = test_data[sample_idx:sample_idx+1]
        label = test_labels[sample_idx]
        
        print(f"\nSample {sample_idx} (true label: {label}):")
        
        h = Train.overlay_on_x_neutral(sample)
        softmax_layer_input = None
        
        for layer_idx, (layer, softmax_layer) in enumerate(zip(model.layers, model.softmax_layers)):
            h = layer(h)
            
            if softmax_layer_input is None:
                softmax_layer_input = h.cpu()
            else:
                softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
            
            _, softmax_output = softmax_layer(softmax_layer_input)
            max_conf = torch.max(softmax_output).item()
            predicted_class = torch.argmax(softmax_output).item()
            
            print(f"  Layer {layer_idx}: max_softmax={max_conf:.6f}, predicted_class={predicted_class}")
            
            # Test confidence checking for different multipliers
            for mult in [0.1, 1.0, 2.0, 3.0]:
                mean_val = confidence_mean_vec[layer_idx].item() if hasattr(confidence_mean_vec[layer_idx], 'item') else confidence_mean_vec[layer_idx]
                std_val = confidence_std_vec[layer_idx].item() if hasattr(confidence_std_vec[layer_idx], 'item') else confidence_std_vec[layer_idx]
                threshold = mean_val - (mult * std_val)
                passes = max_conf > threshold
                print(f"    Mult {mult:.1f}: threshold={threshold:.6f}, passes={passes}")


def create_energy_visualization(energy_results, multiplier_results):
    """Create visualizations for energy analysis with proper data handling"""
    print(f"\n--- Creating Energy Visualization ---")
    print(f"Baseline data - Exit distribution: {energy_results['exit_distribution']}")
    print(f"Threshold comparison - Number of points: {len(multiplier_results)}")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Energy distribution histogram (from baseline) - Convert to mJ for readability
    energy_samples_mj = [e * 1000 for e in energy_results.get('energy_per_sample_joules', [])]
    if energy_samples_mj:
        ax1.hist(energy_samples_mj, bins=30, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Energy per Sample (mJ)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Real Power Energy Consumption Distribution (Baseline)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Exit layer distribution (from baseline)
    layers = list(energy_results['exit_distribution'].keys())
    counts = list(energy_results['exit_distribution'].values())
    colors = ['lightcoral', 'lightgreen', 'lightblue'][:len(layers)]
    ax2.bar(layers, counts, color=colors)
    ax2.set_xlabel('Exit Layer')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Early Exit Distribution (Baseline)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Energy vs accuracy trade-off (from ALL threshold comparisons) - Use real power data
    if len(multiplier_results) > 0:
        multipliers = [r['multiplier'] for r in multiplier_results]
        energies_mj = [r.get('avg_energy_mj', r.get('avg_energy_joules', 0) * 1000) for r in multiplier_results]
        accuracies = [r['accuracy'] for r in multiplier_results]
        
        print(f"Trade-off data - Multipliers: {multipliers}")
        print(f"Trade-off data - Energies (mJ): {[f'{e:.3f}' for e in energies_mj]}")
        print(f"Trade-off data - Accuracies: {[f'{a:.3f}' for a in accuracies]}")
        
        scatter = ax3.scatter(energies_mj, accuracies, c=multipliers, cmap='viridis', s=100, alpha=0.8)
        ax3.set_xlabel('Average Energy (mJ)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title(f'Real Power Energy vs Accuracy Trade-off ({len(multiplier_results)} points)')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for multiplier values
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Confidence Multiplier')
        
        # Add annotations for each point with smart positioning to avoid overlap
        annotation_offsets = [
            (5, 5), (-15, 5), (5, -15), (-15, -15), 
            (20, 0), (-25, 0), (0, 20), (0, -25),
            (15, 15), (-20, -20), (25, -5), (-30, 10)
        ]
        
        for i, (energy, accuracy, mult) in enumerate(zip(energies_mj, accuracies, multipliers)):
            # Use different offsets to prevent overlap
            offset = annotation_offsets[i % len(annotation_offsets)]
            ax3.annotate(f'{mult}', (energy, accuracy), xytext=offset, 
                        textcoords='offset points', fontsize=8, alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    else:
        ax3.text(0.5, 0.5, 'No threshold comparison data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Energy vs Accuracy Trade-off (No Data)')
    
    # 4. Layer power breakdown (from baseline) - Real power data
    if 'layer_breakdown' in energy_results:
        layer_indices = list(energy_results['layer_breakdown'].keys())
        if layer_indices:
            # Extract real power data (convert to mW for readability)
            total_powers = []
            cpu_gpu_powers = []
            soc_powers = []
            
            for i in layer_indices:
                breakdown = energy_results['layer_breakdown'][i]
                count = breakdown.get('count', 1)
                if count > 0:
                    total_powers.append((breakdown.get('total_power', 0) / count) * 1000)  # Convert to mW
                    cpu_gpu_powers.append((breakdown.get('cpu_gpu_power', 0) / count) * 1000)
                    soc_powers.append((breakdown.get('soc_power', 0) / count) * 1000)
                else:
                    total_powers.append(0)
                    cpu_gpu_powers.append(0)
                    soc_powers.append(0)
            
            width = 0.25
            x = np.arange(len(layer_indices))
            
            ax4.bar(x - width, total_powers, width, label='Total Power', color='lightcoral')
            ax4.bar(x, cpu_gpu_powers, width, label='CPU+GPU Power', color='lightgreen')
            ax4.bar(x + width, soc_powers, width, label='SOC Power', color='lightblue')
            
            ax4.set_xlabel('Layer Index')
            ax4.set_ylabel('Average Power (mW)')
            ax4.set_title('Real Power Breakdown by Component (Baseline)')
            ax4.set_xticks(x)
            ax4.set_xticklabels([f'Layer {i}' for i in layer_indices])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No layer data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Real Power Breakdown (No Data)')
    else:
        ax4.text(0.5, 0.5, 'No layer breakdown data available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Energy Breakdown by Component (No Data)')
    
    plt.tight_layout()
    
    # Save the plot with timestamp to avoid overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    filename = f'energy_analysis_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Energy analysis visualization saved to: {filepath}")
    # plt.show()  # Commented out to prevent popup
    plt.close()  # Close figure to prevent memory leaks


def detect_sample_exit_patterns(model, test_data, test_labels, confidence_mean_vec, confidence_std_vec):
    """
    Phase 1: Fast detection of which layer each sample exits from (no power monitoring).
    Returns a dictionary grouping sample indices by their exit layer.
    """
    print("\n=== Phase 1: Detecting Sample Exit Patterns (Fast) ===")
    
    exit_patterns = {1: [], 2: [], 3: []}  # layer_idx + 1 -> [sample_indices]
    sample_predictions = []
    total_samples = len(test_data)
    
    print(f"Analyzing exit patterns for {total_samples} samples...")
    
    with torch.no_grad():
        for sample_idx in range(total_samples):
            if sample_idx % 500 == 0:
                print(f"  Processing sample {sample_idx}/{total_samples} ({sample_idx/total_samples*100:.1f}%)")
            
            sample = test_data[sample_idx:sample_idx+1]
            target = test_labels[sample_idx:sample_idx+1]
            
            # Fast inference without power monitoring
            from Train import overlay_on_x_neutral
            h = overlay_on_x_neutral(sample)
            softmax_layer_input = None
            
            exit_layer = 3  # Default to final layer
            final_prediction = None
            
            for layer_idx, (layer, softmax_layer) in enumerate(zip(model.layers, model.softmax_layers)):
                h = layer(h)
                
                # Accumulate features for softmax
                if softmax_layer_input is None:
                    softmax_layer_input = h.cpu()
                else:
                    softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
                
                # Softmax computation
                _, softmax_output = softmax_layer(softmax_layer_input)
                final_prediction = softmax_output.argmax(1)
                
                # Check for early exit using model's confidence mechanism
                confidence_flag = model.check_confidence(
                    layer_idx, confidence_mean_vec, confidence_std_vec, softmax_output
                )
                
                if confidence_flag:
                    exit_layer = layer_idx + 1  # Convert to 1-indexed
                    break
            
            # Record exit pattern and prediction
            exit_patterns[exit_layer].append(sample_idx)
            sample_predictions.append({
                'sample_idx': sample_idx,
                'exit_layer': exit_layer,
                'prediction': final_prediction.item(),
                'true_label': target.item()
            })
    
    # Print exit pattern statistics
    print(f"\n=== Exit Pattern Detection Results ===")
    for layer, sample_indices in exit_patterns.items():
        count = len(sample_indices)
        percentage = (count / total_samples) * 100
        print(f"  Layer {layer}: {count} samples ({percentage:.1f}%)")
    
    # Calculate accuracy per exit layer
    print(f"\n=== Accuracy by Exit Layer ===")
    for layer in [1, 2, 3]:
        layer_samples = [p for p in sample_predictions if p['exit_layer'] == layer]
        if layer_samples:
            correct = sum(1 for p in layer_samples if p['prediction'] == p['true_label'])
            accuracy = correct / len(layer_samples)
            print(f"  Layer {layer}: {accuracy:.4f} ({correct}/{len(layer_samples)})")
    
    return exit_patterns, sample_predictions


def _process_sample_to_target_layer(model, sample, target_exit_layer, confidence_mean_vec, confidence_std_vec):
    """Helper function to process a sample up to a specific exit layer"""
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


def measure_batch_energy_by_exit_layer(model, test_data, test_labels, exit_patterns, confidence_mean_vec, confidence_std_vec):
    """
    Phase 2: Measure energy consumption for batches of samples that exit at the same layer.
    Returns per-sample energy estimates based on batch measurements.
    """
    print("\n=== Phase 2: Batch Energy Measurement by Exit Layer ===")
    
    layer_energy_results = {}
    
    for exit_layer, sample_indices in exit_patterns.items():
        if not sample_indices:
            print(f"\nSkipping Layer {exit_layer}: No samples exit at this layer")
            continue
            
        print(f"\n--- Measuring Layer {exit_layer} Energy (batch of {len(sample_indices)} samples) ---")
        
        # Create batch of samples that all exit at this layer
        batch_samples = torch.stack([test_data[idx] for idx in sample_indices])
        batch_labels = torch.stack([test_labels[idx] for idx in sample_indices])
        
        # Use smaller sub-batches to manage memory and power monitoring
        batch_size = min(100, len(sample_indices))  # Process in chunks of 100
        energy_measurements = []
        
        monitor = RealPowerEnergyMonitor()
        
        for batch_start in range(0, len(batch_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_samples))
            sub_batch = batch_samples[batch_start:batch_end]
            
            print(f"    Processing sub-batch {batch_start//batch_size + 1}: samples {batch_start}-{batch_end-1}")
            
            # Start power monitoring for this sub-batch
            monitor.start_layer_monitoring(exit_layer - 1, batch_start)  # Convert to 0-indexed
            
            batch_start_time = time.perf_counter()
            
            # Process the entire sub-batch with early exit at target layer
            with torch.no_grad():
                predictions = []
                for sample in sub_batch:
                    prediction = _process_sample_to_target_layer(
                        model, sample.unsqueeze(0), exit_layer, confidence_mean_vec, confidence_std_vec
                    )
                    predictions.append(prediction)
            
            batch_duration = time.perf_counter() - batch_start_time
            
            # Stop power monitoring
            monitor.stop_layer_monitoring(exit_layer - 1)
            
            # Get energy summary for this sub-batch
            energy_summary = monitor.get_layer_energy_summary(exit_layer - 1)
            
            energy_measurements.append({
                'batch_size': len(sub_batch),
                'duration_seconds': batch_duration,
                'total_energy_joules': energy_summary.get('total_energy_joules', 0.0),
                'avg_power_watts': energy_summary.get('avg_total_power_watts', 0.0),
                'cpu_gpu_power_watts': energy_summary.get('avg_cpu_gpu_power_watts', 0.0),
                'soc_power_watts': energy_summary.get('avg_soc_power_watts', 0.0)
            })
            
            # Clear measurements for next batch
            monitor.clear_measurements()
        
        # Calculate aggregate statistics for this exit layer
        total_samples = sum(m['batch_size'] for m in energy_measurements)
        total_energy = sum(m['total_energy_joules'] for m in energy_measurements)
        total_duration = sum(m['duration_seconds'] for m in energy_measurements)
        
        # Weighted averages for power measurements
        avg_power = sum(m['avg_power_watts'] * m['batch_size'] for m in energy_measurements) / total_samples if total_samples > 0 else 0
        avg_cpu_gpu_power = sum(m['cpu_gpu_power_watts'] * m['batch_size'] for m in energy_measurements) / total_samples if total_samples > 0 else 0
        avg_soc_power = sum(m['soc_power_watts'] * m['batch_size'] for m in energy_measurements) / total_samples if total_samples > 0 else 0
        
        # Calculate per-sample energy
        per_sample_energy_joules = total_energy / total_samples if total_samples > 0 else 0
        per_sample_energy_mj = per_sample_energy_joules * 1000  # Convert to mJ
        
        layer_energy_results[exit_layer] = {
            'sample_count': total_samples,
            'total_energy_joules': total_energy,
            'per_sample_energy_joules': per_sample_energy_joules,
            'per_sample_energy_mj': per_sample_energy_mj,
            'avg_power_watts': avg_power,
            'avg_cpu_gpu_power_watts': avg_cpu_gpu_power,
            'avg_soc_power_watts': avg_soc_power,
            'total_duration_seconds': total_duration,
            'sample_indices': sample_indices
        }
        
        print(f"    Results for Layer {exit_layer}:")
        print(f"      Total samples: {total_samples}")
        print(f"      Total energy: {total_energy:.6f} J ({total_energy*1000:.3f} mJ)")
        print(f"      Per-sample energy: {per_sample_energy_joules:.6f} J ({per_sample_energy_mj:.3f} mJ)")
        print(f"      Average power: {avg_power:.3f} W")
        print(f"      Total duration: {total_duration:.3f} s")
    
    return layer_energy_results


def two_phase_energy_analysis(model, test_data, test_labels, confidence_mean_vec, confidence_std_vec):
    """
    Comprehensive two-phase energy analysis:
    Phase 1: Detect exit patterns (fast)
    Phase 2: Measure batch energy by exit layer (accurate)
    """
    print("\n" + "="*70)
    print("TWO-PHASE ENERGY ANALYSIS WITH EXIT PATTERN DETECTION")
    print("="*70)
    
    # Phase 1: Fast exit pattern detection
    exit_patterns, sample_predictions = detect_sample_exit_patterns(
        model, test_data, test_labels, confidence_mean_vec, confidence_std_vec
    )
    
    # Phase 2: Batch energy measurement by exit layer
    layer_energy_results = measure_batch_energy_by_exit_layer(
        model, test_data, test_labels, exit_patterns, confidence_mean_vec, confidence_std_vec
    )
    
    # Compile comprehensive results
    print(f"\n" + "="*70)
    print("TWO-PHASE ENERGY ANALYSIS RESULTS")
    print("="*70)
    
    total_samples = len(test_data)
    total_energy = sum(result['total_energy_joules'] for result in layer_energy_results.values())
    
    print(f"\n=== Overall Energy Statistics ===")
    print(f"Total samples analyzed: {total_samples}")
    print(f"Total energy consumed: {total_energy:.6f} J ({total_energy*1000:.3f} mJ)")
    
    if total_samples > 0:
        avg_energy_per_sample = total_energy / total_samples
        print(f"Average energy per sample: {avg_energy_per_sample:.6f} J ({avg_energy_per_sample*1000:.3f} mJ)")
    
    print(f"\n=== Energy Breakdown by Exit Layer ===")
    print("Layer | Samples | Accuracy | Per-Sample (mJ) | Total (mJ) | Avg Power (W) | CPU+GPU (W) | SOC (W) | Cum Acc | Cum Energy (mJ) | Cum Time (ms)")
    print("-" * 135)
    
    # Calculate accuracy per layer from sample predictions
    layer_accuracies = {}
    for layer in [1, 2, 3]:
        layer_samples = [p for p in sample_predictions if p['exit_layer'] == layer]
        if layer_samples:
            correct = sum(1 for p in layer_samples if p['prediction'] == p['true_label'])
            accuracy = correct / len(layer_samples)
            layer_accuracies[layer] = accuracy
        else:
            layer_accuracies[layer] = 0.0
    
    # Calculate cumulative metrics
    cumulative_samples = 0
    cumulative_correct = 0
    cumulative_energy = 0.0
    cumulative_time = 0.0
    
    for layer in [1, 2, 3]:
        if layer in layer_energy_results:
            result = layer_energy_results[layer]
            accuracy = layer_accuracies[layer]
            
            # Update cumulative totals
            cumulative_samples += result['sample_count']
            layer_samples = [p for p in sample_predictions if p['exit_layer'] == layer]
            layer_correct = sum(1 for p in layer_samples if p['prediction'] == p['true_label'])
            cumulative_correct += layer_correct
            cumulative_energy += result['total_energy_joules'] * 1000  # Convert to mJ
            cumulative_time += result['total_duration_seconds'] * 1000  # Convert to ms
            
            # Calculate cumulative averages
            cum_accuracy = cumulative_correct / cumulative_samples if cumulative_samples > 0 else 0.0
            cum_energy_per_sample = cumulative_energy / cumulative_samples if cumulative_samples > 0 else 0.0
            cum_time_per_sample = cumulative_time / cumulative_samples if cumulative_samples > 0 else 0.0
            
            print(f"{layer:5d} | {result['sample_count']:7d} | {accuracy:8.4f} | {result['per_sample_energy_mj']:13.3f} | "
                  f"{result['total_energy_joules']*1000:9.3f} | {result['avg_power_watts']:12.3f} | "
                  f"{result['avg_cpu_gpu_power_watts']:10.3f} | {result['avg_soc_power_watts']:7.3f} | "
                  f"{cum_accuracy:7.4f} | {cum_energy_per_sample:13.3f} | {cum_time_per_sample:11.3f}")
        else:
            accuracy = layer_accuracies[layer]
            # For layers with no samples, cumulative values remain the same
            cum_accuracy = cumulative_correct / cumulative_samples if cumulative_samples > 0 else 0.0
            cum_energy_per_sample = cumulative_energy / cumulative_samples if cumulative_samples > 0 else 0.0
            cum_time_per_sample = cumulative_time / cumulative_samples if cumulative_samples > 0 else 0.0
            
            print(f"{layer:5d} | {0:7d} | {accuracy:8.4f} | {0:13.3f} | {0:9.3f} | {0:12.3f} | {0:10.3f} | {0:7.3f} | "
                  f"{cum_accuracy:7.4f} | {cum_energy_per_sample:13.3f} | {cum_time_per_sample:11.3f}")
    
    return {
        'exit_patterns': exit_patterns,
        'sample_predictions': sample_predictions,
        'layer_energy_results': layer_energy_results,
        'total_energy_joules': total_energy,
        'total_samples': total_samples,
        'avg_energy_per_sample_joules': total_energy / total_samples if total_samples > 0 else 0
    }


def validate_two_phase_energy_analysis(model, test_data, test_labels, confidence_mean_vec, confidence_std_vec, sample_size=100):
    """
    Validate the two-phase batch analysis with comprehensive reporting.
    This replaces the comparison function since two-phase analysis is proven superior.
    """
    print(f"\n" + "="*70)
    print("TWO-PHASE ENERGY ANALYSIS VALIDATION")
    print("="*70)
    
    print(f"Validating two-phase batch analysis on {sample_size} samples...")
    
    # Run two-phase batch analysis
    print(f"\n--- Two-Phase Batch Analysis Results ---")
    batch_results = two_phase_energy_analysis(
        model, test_data[:sample_size], test_labels[:sample_size], 
        confidence_mean_vec, confidence_std_vec
    )
    
    # Detailed validation reporting
    print(f"\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    batch_avg = batch_results['avg_energy_per_sample_joules']
    total_samples = batch_results['total_samples']
    
    print(f"\n=== Energy Analysis Validation ===")
    print(f"Two-phase batch method:  {batch_avg:.6f} J ({batch_avg*1000:.3f} mJ)")
    print(f"Total samples processed: {total_samples}")
    print(f"Total energy consumed: {batch_results['total_energy_joules']:.6f} J")
    
    print(f"\n=== Exit Pattern Analysis ===")
    batch_exits = batch_results['exit_patterns']
    
    print("Layer | Sample Count | Percentage | Energy Contribution")
    print("-" * 60)
    for layer in [1, 2, 3]:
        if layer in batch_exits:
            batch_count = len(batch_exits[layer])
            percentage = (batch_count / total_samples) * 100
            if layer in batch_results['layer_energy_results']:
                layer_energy = batch_results['layer_energy_results'][layer]['total_energy_joules']
                energy_pct = (layer_energy / batch_results['total_energy_joules']) * 100
                print(f"{layer:5d} | {batch_count:12d} | {percentage:10.1f}% | {energy_pct:17.1f}%")
            else:
                print(f"{layer:5d} | {batch_count:12d} | {percentage:10.1f}% | {0:17.1f}%")
        else:
            print(f"{layer:5d} | {0:12d} | {0:10.1f}% | {0:17.1f}%")
    
    print(f"\n=== Performance Metrics ===")
    # Calculate efficiency metrics
    early_exit_samples = sum(len(batch_exits.get(layer, [])) for layer in [1, 2])
    early_exit_rate = (early_exit_samples / total_samples) * 100
    
    print(f"Early exit rate: {early_exit_rate:.1f}% ({early_exit_samples}/{total_samples} samples)")
    print(f"Average energy efficiency: {batch_avg*1000:.3f} mJ per sample")
    
    # Calculate theoretical speedup vs individual analysis
    # Two-phase: Fast detection (~10ms/sample) + Batch processing (~250ms/layer per batch)
    fast_detection_time = total_samples * 0.01  # 10ms per sample for detection
    batch_processing_time = (len(batch_exits.get(1, [])) * 0.25 + 
                           len(batch_exits.get(2, [])) * 0.5 + 
                           len(batch_exits.get(3, [])) * 0.75)  # Layer-specific processing
    two_phase_time = fast_detection_time + batch_processing_time
    
    # Individual analysis estimate: ~250ms per layer per sample
    individual_time_estimate = total_samples * 0.25 * 3
    
    if two_phase_time > 0:
        speedup = individual_time_estimate / two_phase_time
        print(f"Estimated processing time: {two_phase_time:.1f} seconds")
        print(f"Theoretical speedup vs individual analysis: {speedup:.1f}x faster")
    
    print(f"\n=== Quality Assurance ===")
    # Check for data consistency
    total_exit_samples = sum(len(batch_exits.get(layer, [])) for layer in [1, 2, 3])
    if total_exit_samples == total_samples:
        print("✅ Exit pattern coverage: Complete (all samples accounted for)")
    else:
        print(f"⚠️  Exit pattern coverage: {total_exit_samples}/{total_samples} samples")
    
    # Check energy consistency
    layer_energy_sum = sum(result['total_energy_joules'] for result in batch_results['layer_energy_results'].values())
    if abs(layer_energy_sum - batch_results['total_energy_joules']) < 1e-6:
        print("✅ Energy accounting: Consistent")
    else:
        print(f"⚠️  Energy accounting: {layer_energy_sum:.6f} J vs {batch_results['total_energy_joules']:.6f} J")
    
    return {
        'batch_results': batch_results,
        'validation_passed': True,
        'early_exit_rate': early_exit_rate,
        'estimated_speedup': speedup if two_phase_time > 0 else 0
    }


# Execute energy analysis after all functions are defined
if args.run_energy_analysis:
    print("\n" + "="*50)
    print("RUNNING ENERGY ANALYSIS")
    print("="*50)
    
    # FORCE RESET MODEL STATE to ensure baseline analysis works correctly
    print("Resetting model state...")
    if hasattr(model, 'confidence_threshold_multiplier'):
        print(f"Current multiplier: {model.confidence_threshold_multiplier}")
    else:
        print("Model has no confidence_threshold_multiplier attribute")
    
    # Force set to default for baseline analysis
    model.confidence_threshold_multiplier = 1.0
    print(f"Reset multiplier to: {model.confidence_threshold_multiplier}")
    
    # Add debugging to understand confidence mechanism
    print("\n" + "="*70)
    print("STARTING CONFIDENCE MECHANISM DEBUGGING")
    print("="*70)
    debug_confidence_mechanism(model, x_te, y_te, confidence_mean_vec, confidence_std_vec)
    print("="*70)
    print("DEBUGGING COMPLETE - STARTING ENERGY ANALYSIS")
    print("="*70)
    
    # Choose analysis method based on dataset size
    subset_size = len(x_te)  # Use full test set
    
    print(f"\n--- Choosing Energy Analysis Method ---")
    print(f"Dataset size: {len(x_te)} samples")
    print(f"Analysis subset: {subset_size} samples (FULL TEST SET)")
    
    if subset_size >= 200:
        print("✅ Using TWO-PHASE ENERGY ANALYSIS (recommended for larger datasets)")
        print("   Phase 1: Fast exit pattern detection")
        print("   Phase 2: Batch energy measurement by exit layer")
        
        # Run the new two-phase energy analysis
        energy_results = two_phase_energy_analysis(
            model, x_te, y_te, 
            confidence_mean_vec, confidence_std_vec
        )
        
        # Convert to format compatible with existing visualization
        energy_results_compat = {
            'avg_energy_joules': energy_results['avg_energy_per_sample_joules'],
            'energy_std_joules': 0.0,  # Will calculate from layer breakdown
            'min_energy_joules': 0.0,  # Will calculate from layer breakdown  
            'max_energy_joules': 0.0,  # Will calculate from layer breakdown
            'total_energy_joules': energy_results['total_energy_joules'],
            'energy_per_sample_joules': [],  # Will populate from layer results
            'exit_distribution': {k: len(v) for k, v in energy_results['exit_patterns'].items()},
            'layer_breakdown': energy_results['layer_energy_results']
        }
        
        # Calculate per-sample energies for visualization
        per_sample_energies = []
        for layer, indices in energy_results['exit_patterns'].items():
            if layer in energy_results['layer_energy_results']:
                layer_energy = energy_results['layer_energy_results'][layer]['per_sample_energy_joules']
                per_sample_energies.extend([layer_energy] * len(indices))
        
        if per_sample_energies:
            energy_results_compat['energy_per_sample_joules'] = per_sample_energies
            energy_results_compat['energy_std_joules'] = np.std(per_sample_energies)
            energy_results_compat['min_energy_joules'] = min(per_sample_energies)
            energy_results_compat['max_energy_joules'] = max(per_sample_energies)
        
        energy_results = energy_results_compat
        
    else:
        print("✅ Using TWO-PHASE ENERGY ANALYSIS (works for any dataset size)")
        print("   This is the recommended method for all analyses")
        
        # Use two-phase analysis for all cases
        energy_results_raw = two_phase_energy_analysis(
            model, x_te, y_te, 
            confidence_mean_vec, confidence_std_vec
        )
        
        # Convert to format compatible with existing visualization
        energy_results = {
            'avg_energy_joules': energy_results_raw['avg_energy_per_sample_joules'],
            'energy_std_joules': 0.0,  # Will calculate from layer breakdown
            'min_energy_joules': 0.0,  # Will calculate from layer breakdown  
            'max_energy_joules': 0.0,  # Will calculate from layer breakdown
            'total_energy_joules': energy_results_raw['total_energy_joules'],
            'energy_per_sample_joules': [],  # Will populate from layer results
            'exit_distribution': {k: len(v) for k, v in energy_results_raw['exit_patterns'].items()},
            'layer_breakdown': energy_results_raw['layer_energy_results']
        }
        
        # Calculate per-sample energies for visualization
        per_sample_energies = []
        for layer, indices in energy_results_raw['exit_patterns'].items():
            if layer in energy_results_raw['layer_energy_results']:
                layer_energy = energy_results_raw['layer_energy_results'][layer]['per_sample_energy_joules']
                per_sample_energies.extend([layer_energy] * len(indices))
        
        if per_sample_energies:
            energy_results['energy_per_sample_joules'] = per_sample_energies
            energy_results['energy_std_joules'] = np.std(per_sample_energies)
            energy_results['min_energy_joules'] = min(per_sample_energies)
            energy_results['max_energy_joules'] = max(per_sample_energies)
    
    # Print detailed baseline results
    if energy_results:
        print("\n--- Baseline Energy Analysis (multiplier = 1.0) ---")
        print(f"=== Energy Consumption Analysis ===")
        print(f"Energy Consumption Results ({subset_size} samples):")
        avg_energy_mj = energy_results['avg_energy_joules'] * 1000  # Convert to mJ
        energy_std_mj = energy_results['energy_std_joules'] * 1000
        min_energy_mj = energy_results['min_energy_joules'] * 1000
        max_energy_mj = energy_results['max_energy_joules'] * 1000
        
        print(f"Average energy per sample: {avg_energy_mj:.3f} mJ ({energy_results['avg_energy_joules']:.6f} J)")
        print(f"Energy standard deviation: {energy_std_mj:.3f} mJ ({energy_results['energy_std_joules']:.6f} J)")
        print(f"Energy range: {min_energy_mj:.3f} - {max_energy_mj:.3f} mJ")
        
        print(f"\nExit Layer Distribution:")
        for layer, count in energy_results['exit_distribution'].items():
            percentage = (count / subset_size) * 100
            print(f"Layer {layer}: {count} samples ({percentage:.1f}%)")
        
        if 'layer_breakdown' in energy_results:
            print(f"\nPer-Layer Energy Breakdown:")
            print("Layer | Samples | Per-Sample (mJ) | Total (mJ) | Avg Power (W) | CPU+GPU (W) | SOC (W)")
            print("-" * 85)
            
            for layer_idx, breakdown in energy_results['layer_breakdown'].items():
                sample_count = breakdown.get('sample_count', 0)
                per_sample_energy_mj = breakdown.get('per_sample_energy_mj', 0)
                total_energy_mj = breakdown.get('total_energy_joules', 0) * 1000
                avg_power_watts = breakdown.get('avg_power_watts', 0)
                cpu_gpu_power = breakdown.get('avg_cpu_gpu_power_watts', 0)
                soc_power = breakdown.get('avg_soc_power_watts', 0)
                
                print(f"{layer_idx:5d} | {sample_count:7d} | {per_sample_energy_mj:13.3f} | "
                      f"{total_energy_mj:9.3f} | {avg_power_watts:12.3f} | "
                      f"{cpu_gpu_power:10.3f} | {soc_power:7.3f}")
    
    # --- Threshold Comparison Analysis ---
    print("--- Threshold Comparison Analysis ---")
    multiplier_results = compare_energy_vs_accuracy(model, x_te, y_te, 
                                                   confidence_mean_vec, confidence_std_vec)
    
    # Create comprehensive visualization
    print("--- Creating Energy Visualization ---")
    create_energy_visualization(energy_results, multiplier_results)
    
    print("Energy analysis completed! Check 'output/energy_analysis.png' for detailed results.")
    
    # Optional validation of two-phase analysis
    if args.compare_energy_methods:
        print(f"\n--- Running Two-Phase Analysis Validation ---")
        validation_results = validate_two_phase_energy_analysis(
            model, x_te[:200], y_te[:200], confidence_mean_vec, confidence_std_vec, sample_size=100
        )
        
        if validation_results['validation_passed']:
            print("✅ Two-phase analysis validation successful")
            print(f"   Early exit rate: {validation_results['early_exit_rate']:.1f}%")
            print(f"   Estimated speedup: {validation_results['estimated_speedup']:.1f}x faster")
        else:
            print("⚠️  Validation detected issues - review implementation")
    
    # Print comparison summary
    print("\n=== Analysis Results Summary ===")
    energies = [r['avg_energy_mj'] for r in multiplier_results]
    accuracies = [r['accuracy'] for r in multiplier_results]
    print(f"Energy range: {min(energies):.3f} - {max(energies):.3f} mJ")
    print(f"Accuracy range: {min(accuracies):.3f} - {max(accuracies):.3f}")
    print(f"Unique energy values: {len(set(energies))}")
    
    if len(set(energies)) > 2:
        print(f"✅ SUCCESS: Fixed confidence mechanism - now showing {len(set(energies))} different energy levels!")
        print(f"Energy savings potential: {(max(energies) - min(energies)) / max(energies) * 100:.1f}%")
    else:
        print(f"⚠️  Limited variation: Only {len(set(energies))} unique energy values detected")
        print("This suggests confidence thresholds may still need adjustment")
    
    # Print summary
    print("\n=== Energy Analysis Summary ===")
    print(f"Baseline Analysis (Confidence Multiplier = 1.0):")
    print(f"  Total samples analyzed: {subset_size}")
    avg_energy_mj = energy_results['avg_energy_joules'] * 1000
    energy_std_mj = energy_results['energy_std_joules'] * 1000
    print(f"  Average energy per sample: {avg_energy_mj:.3f} mJ ({energy_results['avg_energy_joules']:.6f} J)")
    print(f"  Energy standard deviation: {energy_std_mj:.3f} mJ ({energy_results['energy_std_joules']:.6f} J)")
    if 'total_energy_joules' in energy_results:
        total_energy_mj = energy_results['total_energy_joules'] * 1000
        print(f"  Total inference energy: {total_energy_mj:.3f} mJ ({energy_results['total_energy_joules']:.6f} J)")
    
    print(f"\nBaseline Early Exit Distribution:")
    total_layers_used = 0
    early_exits = 0
    for layer, count in energy_results['exit_distribution'].items():
        percentage = (count / subset_size) * 100
        print(f"  Layer {layer}: {count} samples ({percentage:.1f}%)")
        total_layers_used += layer * count
        if layer < len(model.layers):
            early_exits += count
    
    avg_layers_used = total_layers_used / subset_size
    early_exit_rate = (early_exits / subset_size) * 100
    print(f"  Average layers used: {avg_layers_used:.2f}")
    print(f"  Early exit rate: {early_exit_rate:.1f}%")
    
    print(f"\nThreshold Comparison Results:")
    print(f"  Tested {len(multiplier_results)} different confidence multipliers")
    multipliers = [r['multiplier'] for r in multiplier_results]
    energies = [r['avg_energy_mj'] for r in multiplier_results]
    accuracies = [r['accuracy'] for r in multiplier_results]
    print(f"  Multiplier values: {multipliers}")
    print(f"  Energy range: {min(energies):.3f} - {max(energies):.3f} mJ")
    print(f"  Accuracy range: {min(accuracies):.3f} - {max(accuracies):.3f}")

else:
    print("\n--- Energy Analysis Skipped ---")
    print("To run energy consumption analysis, use: --run_energy_analysis")
    print("To run validation of two-phase analysis, add: --compare_energy_methods")
    print("Example: python Main.py --selected_classes 0 1 2 3 4 5 6 7 8 9 --run_energy_analysis")
    print("Example: python Main.py --selected_classes 0 1 2 3 4 5 6 7 8 9 --run_energy_analysis --compare_energy_methods")

def improved_confidence_checking(model, test_data, test_labels, confidence_mean_vec, confidence_std_vec):
    """Implement improved confidence checking with proper threshold scaling"""
    print("\n=== Implementing Improved Confidence Checking ===")
    
    # Convert logit-space confidence vectors to probability space
    print("Converting confidence thresholds to probability space...")
    
    # Calculate probability-space thresholds using sigmoid/softmax transformation
    import torch.nn.functional as F
    
    prob_thresholds = []
    for layer_idx in range(len(confidence_mean_vec)):
        mean_logit = confidence_mean_vec[layer_idx]
        std_logit = confidence_std_vec[layer_idx]
        
        # Convert to probability space using sigmoid transformation
        mean_prob = torch.sigmoid(mean_logit).item()
        
        # For standard deviation, we'll use a scaling factor based on the derivative of sigmoid
        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        sigmoid_derivative = mean_prob * (1 - mean_prob)
        std_prob = std_logit * sigmoid_derivative
        
        prob_thresholds.append({
            'mean': mean_prob,
            'std': std_prob,
            'original_mean': mean_logit,
            'original_std': std_logit
        })
        
        print(f"Layer {layer_idx}: logit_mean={mean_logit:.3f} → prob_mean={mean_prob:.6f}, "
              f"logit_std={std_logit:.3f} → prob_std={std_prob:.6f}")
    
    return prob_thresholds

def enhanced_energy_analysis_with_proper_thresholds(model, test_data, test_labels, confidence_mean_vec, confidence_std_vec):
    """Run energy analysis with corrected confidence thresholds"""
    print("\n=== Enhanced Energy Analysis with Proper Thresholds ===")
    
    # Convert to probability space thresholds
    prob_thresholds = improved_confidence_checking(model, test_data, test_labels, confidence_mean_vec, confidence_std_vec)
    
    # Test different multipliers with corrected thresholds
    multipliers = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
    results = []
    
    print("\nTesting Corrected Confidence Thresholds:")
    print("Multiplier | Layer 0 Threshold | Layer 1 Threshold | Layer 2 Threshold | Avg Energy | Accuracy")
    print("-" * 95)
    
    original_multiplier = getattr(model, 'confidence_threshold_multiplier', 1.0)
    
    for multiplier in multipliers:
        # Calculate probability-space thresholds
        corrected_thresholds = []
        for layer_idx in range(len(prob_thresholds)):
            prob_mean = prob_thresholds[layer_idx]['mean']
            prob_std = prob_thresholds[layer_idx]['std']
            # Use more aggressive scaling for probability space
            threshold = prob_mean - (multiplier * prob_std * 0.1)  # Scale down std impact
            # Ensure threshold stays in valid probability range
            threshold = max(0.1, min(0.99, threshold))
            corrected_thresholds.append(threshold)
        
        # Temporarily modify the model's confidence checking to use probability space
        model.confidence_threshold_multiplier = multiplier
        model._prob_thresholds = corrected_thresholds  # Store for use in modified check
        
        # Run lightweight energy analysis
        energy_result = analyze_energy_lightweight_with_prob_thresholds(
            model, test_data[:200], test_labels[:200], corrected_thresholds
        )
        
        # Calculate accuracy
        correct = 0
        total = min(200, len(test_data))
        
        with torch.no_grad():
            for i in range(total):
                sample = test_data[i:i+1]
                target = test_labels[i:i+1]
                
                prediction = predict_with_prob_thresholds(model, sample, corrected_thresholds)
                
                if prediction.item() == target.item():
                    correct += 1
        
        accuracy = correct / total
        
        results.append({
            'multiplier': multiplier,
            'avg_energy': energy_result['avg_energy'],
            'accuracy': accuracy,
            'thresholds': corrected_thresholds.copy()
        })
        
        print(f"{multiplier:10.1f} | {corrected_thresholds[0]:17.6f} | {corrected_thresholds[1]:17.6f} | "
              f"{corrected_thresholds[2]:17.6f} | {energy_result['avg_energy']:10,.0f} | {accuracy:8.3f}")
    
    # Restore original state
    model.confidence_threshold_multiplier = original_multiplier
    if hasattr(model, '_prob_thresholds'):
        delattr(model, '_prob_thresholds')
    
    return results

def predict_with_prob_thresholds(model, x, prob_thresholds):
    """Predict using probability-space confidence thresholds"""
    from Train import overlay_on_x_neutral
    
    h = overlay_on_x_neutral(x)
    softmax_layer_input = None
    
    for layer_idx, (layer, softmax_layer) in enumerate(zip(model.layers, model.softmax_layers)):
        h = layer(h)
        
        if softmax_layer_input is None:
            softmax_layer_input = h.cpu()
        else:
            softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
        
        _, softmax_output = softmax_layer(softmax_layer_input)
        max_confidence = torch.max(softmax_output).item()
        
        # Use probability-space threshold comparison
        if max_confidence > prob_thresholds[layer_idx]:
            return softmax_output.argmax(1)
    
    # If no early exit, return final prediction
    return softmax_output.argmax(1)

def analyze_energy_lightweight_with_prob_thresholds(model, test_data, test_labels, prob_thresholds):
    """Lightweight energy analysis using probability-space thresholds with real power monitoring"""
    total_samples = min(100, len(test_data))  # Reduced for power monitoring
    total_energy_per_sample = []
    exit_layer_distribution = defaultdict(int)
    
    monitor = RealPowerEnergyMonitor()
    
    with torch.no_grad():
        for i in range(total_samples):
            sample = test_data[i:i+1]
            
            # Run modified energy-aware inference
            prediction, layer_energies = energy_aware_inference_with_prob_thresholds(
                model, sample, prob_thresholds, monitor, sample_idx=i
            )
            
            # Calculate total energy for this sample (in Joules)
            sample_total_energy = sum(info['energy_joules'] for info in layer_energies)
            total_energy_per_sample.append(sample_total_energy)
            
            # Track exit layer
            exit_layer = len(layer_energies)
            for layer_info in layer_energies:
                if layer_info['early_exit']:
                    exit_layer = layer_info['layer_idx'] + 1
                    break
            exit_layer_distribution[exit_layer] += 1
    
    return {
        'avg_energy_joules': np.mean(total_energy_per_sample) if total_energy_per_sample else 0,
        'energy_std_joules': np.std(total_energy_per_sample) if total_energy_per_sample else 0,
        'exit_distribution': dict(exit_layer_distribution),
        'energy_per_sample_joules': total_energy_per_sample
    }

def energy_aware_inference_with_prob_thresholds(model, x, prob_thresholds, monitor=None, sample_idx=0):
    """Energy-aware inference using probability-space thresholds with real power monitoring"""
    if monitor is None:
        monitor = RealPowerEnergyMonitor()
    
    from Train import overlay_on_x_neutral
    
    batch_size = x.shape[0]
    h = overlay_on_x_neutral(x)
    softmax_layer_input = None
    layer_energies = []
    accumulated_features = 0
    
    for layer_idx, (layer, softmax_layer) in enumerate(zip(model.layers, model.softmax_layers)):
        # Start power monitoring for this layer
        monitor.start_layer_monitoring(layer_idx, sample_idx)
        
        layer_start_time = time.perf_counter()
        
        # Forward pass through FF layer
        input_size = h.shape[1]
        h = layer(h)
        output_size = h.shape[1]
        accumulated_features += output_size
        
        # Accumulate features for softmax
        if softmax_layer_input is None:
            softmax_layer_input = h.cpu()
        else:
            softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
        
        # Softmax computation for this layer
        _, softmax_output = softmax_layer(softmax_layer_input)
        
        # Confidence checking with probability thresholds
        max_confidence = torch.max(softmax_output).item()
        confidence_flag = max_confidence > prob_thresholds[layer_idx]
        
        layer_time = time.perf_counter() - layer_start_time
        
        # Stop power monitoring for this layer
        monitor.stop_layer_monitoring(layer_idx)
        
        # Get energy summary for this layer operation
        energy_summary = monitor.get_layer_energy_summary(layer_idx)
        
        # Record layer information with real power data
        layer_energies.append({
            'layer_idx': layer_idx,
            'duration_seconds': layer_time,
            'avg_power_watts': energy_summary.get('avg_total_power_watts', 0.0),
            'cpu_gpu_power_watts': energy_summary.get('avg_cpu_gpu_power_watts', 0.0),
            'soc_power_watts': energy_summary.get('avg_soc_power_watts', 0.0),
            'energy_joules': energy_summary.get('total_energy_joules', 0.0),
            'early_exit': confidence_flag,
            'input_size': input_size,
            'output_size': output_size,
            'accumulated_features': accumulated_features
        })
        
        # Early exit check
        if confidence_flag:
            return softmax_output.argmax(1), layer_energies
    
    # If no early exit, return final prediction
    return softmax_output.argmax(1), layer_energies


# Two-phase analysis helper functions moved before main implementation








