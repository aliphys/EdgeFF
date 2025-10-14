import re
import json
from typing import Dict, List, Any

def parse_mnist_log(log_content: str) -> Dict[str, Any]:
    """Parse MNIST Forward-Forward log file into structured JSON."""
    
    result = {
        "model_info": {},
        "configuration": {},
        "dataset_results": {},
        "layer_analysis": {},
        "specialization": {},
        "energy_analysis": {}
    }
    
    # Parse model architecture
    arch_match = re.search(r'layers \(hidden depth\): (\d+)\n\[([^\]]+)\]', log_content)
    if arch_match:
        result["model_info"]["hidden_layers"] = int(arch_match.group(1))
        result["model_info"]["architecture"] = [int(x.strip()) for x in arch_match.group(2).split(',')]
    
    # Parse configuration
    config_section = re.search(r'Configuration:(.*?)Moving model', log_content, re.DOTALL)
    if config_section:
        config_text = config_section.group(1)
        result["configuration"] = {
            "selected_classes": extract_value(config_text, r'Selected classes:\s*(.+)'),
            "filter_by_layer": extract_value(config_text, r'Filter by layer:\s*(.+)'),
            "target_layer": int(extract_value(config_text, r'Target layer:\s*(\d+)', "0")),
            "correct_only": extract_value(config_text, r'Correct only:\s*(\w+)') == 'True',
            "run_specialization": extract_value(config_text, r'Run specialization:\s*(\w+)') == 'True',
            "run_energy_analysis": extract_value(config_text, r'Run energy analysis:\s*(\w+)') == 'True',
            "goodness_threshold": float(extract_value(config_text, r'Goodness threshold:\s*([\d.]+)', "0.0")),
            "confidence_threshold_multiplier": float(extract_value(config_text, r'Confidence threshold multiplier:\s*([\d.]+)', "1.0")),
        }
    
    # Parse dataset results (TEST set)
    test_match = re.search(r'Results for the .*TEST.*set:.*?F1-score:\s*([\d.]+).*?Accuracy:\s*([\d.]+).*?Error:\s*([\d.]+)', log_content, re.DOTALL)
    if test_match:
        result["dataset_results"]["test"] = {
            "f1_score": float(test_match.group(1)),
            "accuracy": float(test_match.group(2)),
            "error": float(test_match.group(3))
        }
    
    # Parse validation results
    val_match = re.search(r'Results for the .*VALIDATION.*set:.*?F1-score:\s*([\d.]+).*?Accuracy:\s*([\d.]+).*?Error:\s*([\d.]+)', log_content, re.DOTALL)
    if val_match:
        result["dataset_results"]["validation"] = {
            "f1_score": float(val_match.group(1)),
            "accuracy": float(val_match.group(2)),
            "error": float(val_match.group(3))
        }
    
    # Parse fixed layer evaluation
    layer_acc = re.findall(r'Layer (\d+) only - Accuracy: ([\d.]+)', log_content)
    if layer_acc:
        result["layer_analysis"]["fixed_layer_accuracy"] = {
            f"layer_{layer}": float(acc) for layer, acc in layer_acc
        }
    
    # Parse confidence vector statistics
    conf_stats = re.findall(r'Layer (\d+): mean=([\d.]+), std=([\d.]+), std/mean ratio=([\d.]+)', log_content)
    if conf_stats:
        result["layer_analysis"]["confidence_statistics"] = []
        for layer, mean, std, ratio in conf_stats:
            result["layer_analysis"]["confidence_statistics"].append({
                "layer": int(layer),
                "mean": float(mean),
                "std": float(std),
                "std_mean_ratio": float(ratio)
            })
    
    # Parse layer specialization
    spec_scores = re.findall(r'Layer (\d+): ([\d.]+)', 
                              re.search(r'Layer specialization scores.*?:(.*?)---', log_content, re.DOTALL).group(1) 
                              if re.search(r'Layer specialization scores.*?:(.*?)---', log_content, re.DOTALL) else "")
    if spec_scores:
        result["specialization"]["scores"] = {
            f"layer_{layer}": float(score) for layer, score in spec_scores
        }
    
    # Parse best layer for each class
    class_best = re.findall(r'Class (\d+): Layer (\d+) \(accuracy: ([\d.]+)\)', log_content)
    if class_best:
        result["specialization"]["best_layer_per_class"] = {}
        for cls, layer, acc in class_best:
            result["specialization"]["best_layer_per_class"][f"class_{cls}"] = {
                "best_layer": int(layer),
                "accuracy": float(acc)
            }
    
    # Parse baseline energy
    baseline_match = re.search(r'--- Baseline Energy Analysis.*?Average energy per sample:\s*([\d.]+)\s*mJ.*?Total energy consumed:\s*([\d.]+)\s*J', log_content, re.DOTALL)
    if baseline_match:
        result["energy_analysis"]["baseline"] = {
            "per_sample_mj": float(baseline_match.group(1)),
            "total_energy_j": float(baseline_match.group(2)),
            "total_samples": 10000
        }
    
    # Parse multiplier-specific energy results with detailed breakdown
    result["energy_analysis"]["multiplier_results"] = []
    
    multiplier_sections = re.finditer(r'--- Processing multiplier ([\d.]+) ---.*?=== Energy Breakdown by Exit Layer ===.*?Layer \| Samples.*?\n-+\n(.*?)(?=---|$)', log_content, re.DOTALL)
    
    for match in multiplier_sections:
        multiplier = float(match.group(1))
        section_text = match.group(0)
        breakdown_text = match.group(2)
        
        # Parse the detailed energy breakdown table
        # Format: Layer | Samples | Accuracy | Per-Sample (mJ) | Total (mJ) | Avg Power (W) | CPU+GPU (W) | SOC (W) | Cum Acc | Cum Energy (mJ) | Cum Time (ms)
        exit_pattern = re.findall(
            r'\s+(\d+)\s+\|\s+(\d+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)',
            breakdown_text
        )
        
        exit_distribution = []
        for layer, samples, accuracy, per_sample_mj, total_mj, avg_power, cpu_gpu_w, soc_w, cum_acc, cum_energy_mj, cum_time_ms in exit_pattern:
            exit_distribution.append({
                "layer": int(layer),
                "samples": int(samples),
                "accuracy": float(accuracy),
                "per_sample_mj": float(per_sample_mj),
                "total_mj": float(total_mj),
                "avg_power_w": float(avg_power),
                "cpu_gpu_power_w": float(cpu_gpu_w),
                "soc_power_w": float(soc_w),
                "cumulative_accuracy": float(cum_acc),
                "cumulative_energy_mj": float(cum_energy_mj),
                "cumulative_time_ms": float(cum_time_ms)
            })
        
        # Extract overall energy statistics
        total_energy_match = re.search(r'Total energy consumed:\s*([\d.]+)\s*J', section_text)
        per_sample_match = re.search(r'Average energy per sample:\s*([\d.]+)\s*J', section_text)
        
        total_energy_j = float(total_energy_match.group(1)) if total_energy_match else 0.0
        per_sample_j = float(per_sample_match.group(1)) if per_sample_match else 0.0
        per_sample_mj = per_sample_j * 1000
        
        # Extract overall accuracy for this multiplier
        overall_acc_match = re.search(r'\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)', section_text)
        overall_accuracy = float(overall_acc_match.group(6)) if overall_acc_match else None
        
        result["energy_analysis"]["multiplier_results"].append({
            "multiplier": multiplier,
            "total_energy_j": total_energy_j,
            "per_sample_mj": per_sample_mj,
            "overall_accuracy": overall_accuracy,
            "exit_distribution": exit_distribution
        })
    
    # Calculate energy savings
    if result["energy_analysis"].get("baseline") and result["energy_analysis"].get("multiplier_results"):
        baseline_energy = result["energy_analysis"]["baseline"]["per_sample_mj"]
        for mult_result in result["energy_analysis"]["multiplier_results"]:
            energy = mult_result["per_sample_mj"]
            savings_pct = ((baseline_energy - energy) / baseline_energy) * 100
            mult_result["energy_savings_pct"] = round(savings_pct, 2)
    
    return result

def extract_value(text: str, pattern: str, default: str = "None") -> str:
    """Extract a value using regex pattern."""
    match = re.search(pattern, text)
    return match.group(1).strip() if match else default

def save_to_json(log_file_path: str, output_json_path: str, enable_comm_power: bool = False):
    """
    Read log file and save parsed data as JSON.
    
    Args:
        log_file_path: Path to input log file
        output_json_path: Path to output JSON file
        enable_comm_power: Whether to include communication power calculations
    """
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    parsed_data = parse_mnist_log(log_content)
    
    # Add communication energy calculations if enabled
    if enable_comm_power:
        parsed_data = add_communication_energy_to_results(parsed_data, enable_comm_power)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=2)
    
    print(f"✅ JSON saved to: {output_json_path}")
    if enable_comm_power:
        print(f"   📡 Communication power calculations included")
    return parsed_data

def parse_all_logs_in_directory(log_dir: str, output_dir: str = None, enable_comm_power: bool = False):
    """Parse all .log files in a directory."""
    import os
    
    if output_dir is None:
        output_dir = log_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    
    results = {}
    for log_file in log_files:
        log_path = os.path.join(log_dir, log_file)
        json_file = log_file.replace('.log', '.json')
        json_path = os.path.join(output_dir, json_file)
        
        try:
            print(f"📄 Processing {log_file}...")
            data = save_to_json(log_path, json_path, enable_comm_power)
            results[log_file] = data
        except Exception as e:
            print(f"❌ Error processing {log_file}: {e}")
    
    print(f"\n✅ Processed {len(results)}/{len(log_files)} files successfully")
    return results

def calculate_communication_energy(width: int, layer: int, enable_comm_power: bool = False) -> float:
    """
    Calculate communication energy for transmitting layer activations.
    
    Args:
        width: Network width (100, 200, or 300)
        layer: Layer number (1, 2, or 3)
        enable_comm_power: Whether to include communication power
    
    Returns:
        Communication energy in mJ (0 if layer 1 or disabled)
    """
    if not enable_comm_power or layer == 1:
        return 0.0
    
    # Constants
    P_TX_W = 0.15081  # Transmit power in Watts (45.7 mA @ 3.3V)
    DATA_RATE_MBPS = 1.0474  # Effective data rate in Mbps
    BYTES_PER_ACTIVATION = 4  # float32
    
    # Calculate activation data size
    activation_size_bytes = width * BYTES_PER_ACTIVATION
    activation_size_bits = activation_size_bytes * 8
    
    # Calculate transmission time (in seconds)
    transmission_time_s = activation_size_bits / (DATA_RATE_MBPS * 1e6)
    
    # Calculate energy (in mJ)
    communication_energy_mj = P_TX_W * transmission_time_s * 1000
    
    return communication_energy_mj

def add_communication_energy_to_results(data: Dict[str, Any], enable_comm_power: bool = False) -> Dict[str, Any]:
    """
    Add communication energy calculations to parsed JSON data.
    
    Args:
        data: Parsed JSON data
        enable_comm_power: Whether to include communication power
    
    Returns:
        Updated data with communication energy fields
    """
    if not enable_comm_power:
        return data
    
    # Extract network width from architecture
    if 'model_info' in data and 'architecture' in data['model_info']:
        width = data['model_info']['architecture'][1]  # Hidden layer width
    else:
        return data
    
    # Add communication energy to energy analysis
    if 'energy_analysis' in data:
        # Note: Multiplier 0.1 is the baseline (samples go through all layers)
        baseline_comm_energy = 0.0
        
        # Add to each multiplier result
        if 'multiplier_results' in data['energy_analysis']:
            for result in data['energy_analysis']['multiplier_results']:
                total_comm_energy = 0.0
                
                # Calculate weighted communication energy based on exit distribution
                if 'exit_distribution' in result:
                    total_samples = sum(exit_info.get('samples', 0) for exit_info in result['exit_distribution'])
                    
                    for exit_info in result['exit_distribution']:
                        layer = exit_info.get('layer', 1)
                        samples = exit_info.get('samples', 0)
                        percentage = (samples / total_samples * 100.0) if total_samples > 0 else 0.0
                        
                        # Communication energy for this layer
                        comm_energy = calculate_communication_energy(width, layer, True)
                        exit_info['communication_energy_mj'] = comm_energy
                        
                        # Weight by percentage of samples exiting at this layer
                        total_comm_energy += comm_energy * (percentage / 100.0)
                
                # Add weighted communication energy to per-sample energy
                result['communication_energy_mj'] = total_comm_energy
                if 'per_sample_mj' in result:
                    result['per_sample_mj_with_comm'] = result['per_sample_mj'] + total_comm_energy
                
                # Save baseline (multiplier 0.1) communication energy
                if result.get('multiplier') == 0.1:
                    baseline_comm_energy = total_comm_energy
        
        # Add communication energy to baseline
        if 'baseline' in data['energy_analysis']:
            data['energy_analysis']['baseline']['communication_energy_mj'] = baseline_comm_energy
            data['energy_analysis']['baseline']['per_sample_mj_with_comm'] = (
                data['energy_analysis']['baseline']['per_sample_mj'] + baseline_comm_energy
            )
    
    # Add communication parameters to metadata
    if enable_comm_power:
        data['communication_config'] = {
            'enabled': True,
            'transmit_power_w': 0.15081,
            'transmit_current_ma': 45.7,
            'voltage_v': 3.3,
            'data_rate_mbps': 1.0474,
            'bytes_per_activation': 4,
            'activation_sizes_bytes': {
                'width_100': {'layer_1': 400, 'layer_2': 400, 'layer_3': 400},
                'width_200': {'layer_1': 800, 'layer_2': 800, 'layer_3': 800},
                'width_300': {'layer_1': 1200, 'layer_2': 1200, 'layer_3': 1200}
            },
            'note': 'Layer 1 is on-device (no communication). Layer 2+ requires data transmission.'
        }
    
    return data

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse MNIST Forward-Forward log files to JSON')
    parser.add_argument('log_file', nargs='?', help='Path to log file (optional, will parse all logs in ./logs/ if not provided)')
    parser.add_argument('output_file', nargs='?', help='Path to output JSON file (optional)')
    parser.add_argument('--enable-comm-power', action='store_true', 
                       help='Include communication power calculations for edge-cloud scenarios')
    
    args = parser.parse_args()
    
    if args.log_file:
        # Parse specific file
        output_path = args.output_file or args.log_file.replace('.log', '.json')
        data = save_to_json(args.log_file, output_path, args.enable_comm_power)
        if not args.enable_comm_power:
            print("\n💡 Tip: Use --enable-comm-power to include communication energy calculations")
    else:
        # Parse all logs in the logs directory
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, 'logs')
        
        if os.path.exists(logs_dir):
            print("=" * 70)
            print("PARSING LOG FILES")
            print("=" * 70)
            if args.enable_comm_power:
                print("📡 Communication power calculations: ENABLED")
            else:
                print("💡 Communication power calculations: DISABLED (use --enable-comm-power to enable)")
            print()
            parse_all_logs_in_directory(logs_dir, enable_comm_power=args.enable_comm_power)
        else:
            print(f"❌ Logs directory not found: {logs_dir}")
            print("\nUsage:")
            print("  python parse_log_to_json.py                           # Parse all logs in ./logs/")
            print("  python parse_log_to_json.py --enable-comm-power       # Parse all with communication power")
            print("  python parse_log_to_json.py <input.log>               # Parse specific log file")
            print("  python parse_log_to_json.py <input.log> --enable-comm-power  # Parse with communication power")
            print("  python parse_log_to_json.py <input.log> <output.json> # Parse with custom output")

