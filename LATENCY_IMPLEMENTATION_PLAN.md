# Latency Implementation Plan

## Overview
Add network latency modeling to account for round-trip time (RTT) when transmitting data to the cloud for inference. This provides end-to-end latency analysis considering both compute time and network delays.

## Current State Analysis

### Available Data in JSON
- **`cumulative_time_ms`**: Per-layer cumulative compute time
  - Example (Width 100, Multiplier 0.1x):
    - Layer 1: 0.645 ms (on-device)
    - Layer 2: 0.664 ms (cumulative)
    - Layer 3: 0.802 ms (cumulative)

### Network Model Parameters
- **Ping (RTT)**: 100 ms (given requirement)
- **One-way latency**: 50 ms
- **Upload time**: Already calculated via data_rate (1.0474 Mbps)
  - Width 100: 400 bytes → 3.055 ms transmission time
  - Width 200: 800 bytes → 6.110 ms transmission time
  - Width 300: 1200 bytes → 9.165 ms transmission time

## Latency Model

### Layer-Specific Latency Calculation

#### **Layer 1 (On-Device)**
```
Total Latency = Compute Time (Layer 1)
              = cumulative_time_ms[layer_1]
              ≈ 0.645 ms (width 100)
```
- No network overhead (local inference)

#### **Layer 2 (Edge → Cloud)**
```
Total Latency = Compute Time (Layer 1)           [local compute]
              + Upload Time                       [send to cloud]
              + One-way Network Latency (50 ms)   [propagation]
              + Compute Time (Layer 2 only)       [cloud compute]
              + Download Time (negligible)        [result back]
              + One-way Network Latency (50 ms)   [propagation]
```

**Simplified:**
```
Latency_L2 = time_ms[L1] + upload_ms + RTT + (time_ms[L2] - time_ms[L1])
           = time_ms[L1] + upload_ms + 100 + (time_ms[L2] - time_ms[L1])
           = time_ms[L2] + upload_ms + 100
```

**Example (Width 100, Multiplier 0.1x):**
```
Latency_L2 = 0.664 ms + 3.055 ms + 100 ms = 103.719 ms
```

#### **Layer 3 (Edge → Cloud)**
```
Latency_L3 = time_ms[L3] + upload_ms + 100
           = 0.802 ms + 3.055 ms + 100 ms = 103.857 ms
```

### Weighted Average Latency

Based on exit distribution:
```
Avg Latency = Σ (exit_percentage_i × latency_i)
```

**Example (Width 100, Multiplier 0.1x):**
- 65.16% exit at L1: 0.6516 × 0.645 ms = 0.420 ms
- 5.44% exit at L2: 0.0544 × 103.719 ms = 5.642 ms
- 29.40% exit at L3: 0.2940 × 103.857 ms = 30.534 ms
- **Total: 36.596 ms**

**Example (Width 100, Multiplier 3.0x):**
- 82.45% exit at L1: 0.8245 × 0.645 ms = 0.532 ms
- 3.45% exit at L2: 0.0345 × 103.719 ms = 3.578 ms
- 14.10% exit at L3: 0.1410 × 103.857 ms = 14.644 ms
- **Total: 18.754 ms**

**Latency Savings: 48.7%** (similar to communication energy savings!)

## Implementation Steps

### Phase 1: Add Latency Calculations to Parser
**File: `parse_log_to_json.py`**

1. **Create `calculate_upload_time_ms()` function**
   ```python
   def calculate_upload_time_ms(width: int, layer: int) -> float:
       """Calculate upload time in milliseconds."""
       if layer == 1:
           return 0.0  # On-device, no upload
       
       data_rate_mbps = 1.0474
       activation_size_bytes = width * 4  # float32
       upload_time_ms = (activation_size_bytes * 8) / (data_rate_mbps * 1000)
       return upload_time_ms
   ```

2. **Create `calculate_layer_latency_ms()` function**
   ```python
   def calculate_layer_latency_ms(
       width: int,
       layer: int,
       cumulative_time_ms: float,
       enable_latency: bool = False,
       rtt_ms: float = 100.0
   ) -> float:
       """Calculate total latency including network delays."""
       if not enable_latency or layer == 1:
           return cumulative_time_ms
       
       upload_time = calculate_upload_time_ms(width, layer)
       total_latency = cumulative_time_ms + upload_time + rtt_ms
       return total_latency
   ```

3. **Create `add_latency_to_results()` function**
   ```python
   def add_latency_to_results(
       data: Dict[str, Any],
       enable_latency: bool = False,
       rtt_ms: float = 100.0
   ) -> Dict[str, Any]:
       """Add latency calculations to parsed JSON data."""
       if not enable_latency:
           return data
       
       # Extract width
       width = data['model_info']['architecture'][1]
       
       # Process each multiplier result
       for result in data['energy_analysis']['multiplier_results']:
           total_latency_ms = 0.0
           
           # Calculate per-layer latency and weighted average
           if 'exit_distribution' in result:
               total_samples = sum(exit_info.get('samples', 0) 
                                 for exit_info in result['exit_distribution'])
               
               for exit_info in result['exit_distribution']:
                   layer = exit_info.get('layer', 1)
                   samples = exit_info.get('samples', 0)
                   percentage = (samples / total_samples * 100.0) if total_samples > 0 else 0.0
                   cumulative_time = exit_info.get('cumulative_time_ms', 0.0)
                   
                   # Calculate total latency for this layer
                   layer_latency = calculate_layer_latency_ms(
                       width, layer, cumulative_time, True, rtt_ms
                   )
                   
                   # Store in exit_info
                   exit_info['upload_time_ms'] = calculate_upload_time_ms(width, layer)
                   exit_info['network_latency_ms'] = rtt_ms if layer > 1 else 0.0
                   exit_info['total_latency_ms'] = layer_latency
                   
                   # Weight by exit percentage
                   total_latency_ms += layer_latency * (percentage / 100.0)
           
           # Add weighted average latency
           result['average_latency_ms'] = total_latency_ms
       
       # Add latency config to metadata
       data['latency_config'] = {
           'enabled': True,
           'rtt_ms': rtt_ms,
           'one_way_latency_ms': rtt_ms / 2,
           'data_rate_mbps': 1.0474,
           'note': 'Layer 1 is on-device (no network latency). Layer 2+ includes upload time + RTT.'
       }
       
       return data
   ```

4. **Update command-line arguments**
   ```python
   parser.add_argument('--enable-latency', action='store_true',
                      help='Include network latency calculations')
   parser.add_argument('--rtt-ms', type=float, default=100.0,
                      help='Round-trip time in milliseconds (default: 100)')
   ```

5. **Integrate into parsing pipeline**
   ```python
   if enable_latency:
       parsed_data = add_latency_to_results(parsed_data, enable_latency, args.rtt_ms)
   ```

### Phase 2: Update Visualization
**File: `visualize_energy_analysis.py`**

1. **Update `aggregate_runs()` to handle latency**
   ```python
   # Check if latency is included
   has_latency = "average_latency_ms" in runs[0]["energy_analysis"]["multiplier_results"][0]
   
   if has_latency:
       latency_values = []
       for run in runs:
           for result in run["energy_analysis"]["multiplier_results"]:
               if result["multiplier"] == mult:
                   latency_values.append(result["average_latency_ms"])
                   break
       
       mult_data["average_latency_ms_mean"] = np.mean(latency_values)
       mult_data["average_latency_ms_std"] = np.std(latency_values)
   ```

2. **Modify `plot_energy_savings_highlight()` to show latency**

   **Option A: Triple Y-axis** (Energy + Accuracy + Latency)
   ```python
   # Left axis: Energy (bars)
   # Right axis 1: Accuracy (blue lines)
   # Right axis 2: Latency (orange lines)
   ```

   **Option B: Separate annotation** (Recommended)
   ```python
   # Keep current dual-axis (Energy + Accuracy)
   # Add latency info in the text annotation:
   ax1.text(i, y_pos,
           f'Energy: −{energy_change:.1f}%\n'
           f'Accuracy: +{accuracy_change:.2f}%\n'
           f'Latency: −{latency_change:.1f}%',
           ha='center', va='bottom', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                    alpha=0.9, edgecolor='gray'))
   ```

   **Option C: New dedicated plot** (Most clear)
   ```python
   def plot_latency_savings(data: Dict[int, Dict], output_dir: str = "output"):
       """Plot latency comparison between baseline and early exit."""
       # Similar structure to energy_savings_highlight
       # Shows latency bars with percentage reduction
   ```

### Phase 3: Verification Script
**File: `verify_latency.py`**

```python
#!/usr/bin/env python3
"""Verify latency calculations with network delays."""

import json

with open('logs/width100run1.json') as f:
    data = json.load(f)

# Check if latency is enabled
if 'latency_config' not in data:
    print("⚠ Latency calculations not enabled in JSON file")
    print("Run: python parse_log_to_json.py --enable-comm-power --enable-latency")
    exit(1)

config = data['latency_config']
print('='*70)
print('LATENCY VERIFICATION')
print('='*70)
print(f"\nConfiguration:")
print(f"  RTT: {config['rtt_ms']} ms")
print(f"  One-way latency: {config['one_way_latency_ms']} ms")
print(f"  Data rate: {config['data_rate_mbps']} Mbps")

for result in data['energy_analysis']['multiplier_results']:
    mult = result['multiplier']
    avg_latency = result.get('average_latency_ms', 0)
    
    print(f"\nMultiplier {mult}x:")
    print(f"  Average Latency: {avg_latency:.2f} ms")
    print(f"  Exit Distribution:")
    
    for exit_info in result['exit_distribution']:
        layer = exit_info['layer']
        samples = exit_info['samples']
        compute_time = exit_info['cumulative_time_ms']
        upload_time = exit_info.get('upload_time_ms', 0)
        network_lat = exit_info.get('network_latency_ms', 0)
        total_lat = exit_info.get('total_latency_ms', compute_time)
        
        print(f"    Layer {layer}: {samples} samples")
        print(f"      Compute: {compute_time:.3f} ms")
        print(f"      Upload:  {upload_time:.3f} ms")
        print(f"      Network: {network_lat:.1f} ms")
        print(f"      Total:   {total_lat:.2f} ms")

# Calculate latency savings
baseline = [r for r in data['energy_analysis']['multiplier_results'] 
            if r['multiplier'] == 0.1][0]
mult3 = [r for r in data['energy_analysis']['multiplier_results'] 
         if r['multiplier'] == 3.0][0]

baseline_lat = baseline.get('average_latency_ms', 0)
mult3_lat = mult3.get('average_latency_ms', 0)
latency_savings = (baseline_lat - mult3_lat) / baseline_lat * 100

print('\n' + '='*70)
print('LATENCY SAVINGS')
print('='*70)
print(f"Baseline (0.1x): {baseline_lat:.2f} ms")
print(f"Early Exit (3.0x): {mult3_lat:.2f} ms")
print(f"Savings: {latency_savings:.2f}%")
print('='*70)
```

## Expected Results

### Width 100, Baseline (0.1x)
- **Compute only**: 0.802 ms (all layers)
- **With network**: ~36.6 ms average
- **Network overhead**: 45.7× increase

### Width 100, Early Exit (3.0x)
- **Compute only**: Variable (mostly Layer 1: 0.645 ms)
- **With network**: ~18.8 ms average
- **Latency savings**: 48.7% vs baseline

### Key Insights
1. **Network latency dominates** when cloud inference is needed
2. **Early exit dramatically reduces latency** by avoiding cloud round-trips
3. **Layer 1 exits are critical** for low-latency inference (<1 ms)

## Visualization Updates

### Enhanced `energy_savings_highlight.png`
Add latency percentage to annotation:
```
Energy: −21.7%
Accuracy: +1.38%
Latency: −48.7%  ← NEW
```

### New Plot: `latency_breakdown.png`
- Stacked bars showing: Compute | Upload | Network RTT
- Grouped by width and multiplier
- Highlights network latency dominance

### New Plot: `latency_savings_highlight.png`
- Similar to energy_savings_highlight
- Shows latency (ms) instead of energy (mJ)
- Demonstrates edge inference advantage

## Command-Line Usage

```bash
# Parse with both communication power and latency
python parse_log_to_json.py --enable-comm-power --enable-latency

# Custom RTT
python parse_log_to_json.py --enable-comm-power --enable-latency --rtt-ms 50

# Verify latency calculations
python verify_latency.py

# Generate visualizations (auto-detects latency data)
python visualize_energy_analysis.py
```

## Benefits of This Approach

1. **Realistic deployment modeling**: Accounts for both energy AND latency
2. **Demonstrates edge advantage**: Shows why on-device inference matters
3. **Configurable RTT**: Can model different network conditions
4. **Unified analysis**: Energy, accuracy, and latency in one framework
5. **Backward compatible**: Works with existing JSON files (optional feature)

## Next Steps

1. ✅ Review and approve plan
2. Implement Phase 1 (parser updates)
3. Re-parse JSON files with latency
4. Implement Phase 2 (visualization updates)
5. Create verification script
6. Generate updated plots
7. Document results

## Notes

- **Upload time is already calculated** in communication power model
- **Download time assumed negligible** (classification result is tiny)
- **RTT includes both propagation delays** (50 ms each way)
- **Compute time already available** in JSON as `cumulative_time_ms`
- **No additional hardware measurements needed** - pure modeling
