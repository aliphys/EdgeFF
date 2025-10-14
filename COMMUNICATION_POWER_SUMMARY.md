# Communication Power Implementation Summary

## Overview
Added communication power modeling to account for the energy cost of transmitting layer activations in edge-cloud deployment scenarios. This provides a more realistic energy analysis when early exit does NOT happen and data must be transmitted to the cloud.

## Communication Power Model

### Hardware Parameters
- **Transmit Power (P_TX)**: 0.15081 W
  - Current: 45.7 mA @ 3.3 V
- **Data Rate**: 1.0474 Mbps (effective)
- **Data Type**: float32 (4 bytes per activation)

### Energy Calculation
```
Energy (mJ) = P_TX × (data_size_bits / data_rate_bps) × 1000
```

### Layer-Specific Communication
- **Layer 1**: On-device (0 mJ communication)
  - Samples that exit at Layer 1 require no data transmission
- **Layer 2+**: Requires transmission
  - Width 100: 400 bytes → 0.4608 mJ per transmission
  - Width 200: 800 bytes → 0.9215 mJ per transmission
  - Width 300: 1200 bytes → 1.3823 mJ per transmission

## Implementation Details

### Weighted Communication Energy
Communication energy is calculated as a weighted average based on the exit distribution:

```
Total Comm Energy = Σ (exit_percentage_i × comm_energy_i)
```

**Example (Width 100, Multiplier 0.1x):**
- 65.16% exit at Layer 1: 0.6516 × 0.0000 mJ = 0.0000 mJ
- 5.44% exit at Layer 2: 0.0544 × 0.4608 mJ = 0.0251 mJ
- 29.40% exit at Layer 3: 0.2940 × 0.4608 mJ = 0.1355 mJ
- **Total**: 0.1605 mJ

### Energy Savings with Communication Power

**Width 100 Example:**
| Metric | Baseline (0.1x) | Early Exit (3.0x) | Savings |
|--------|----------------|-------------------|---------|
| Compute Energy | 16.577 mJ | 12.980 mJ | 21.70% |
| Communication Energy | 0.1605 mJ | 0.0809 mJ | 49.59% |
| **Total Energy** | **16.738 mJ** | **13.061 mJ** | **21.97%** |
| Accuracy | 6.49% | 5.10% | -1.38% |

**Key Insight**: Early exit saves both compute AND communication energy because:
1. Fewer layers need to be computed
2. More samples exit at Layer 1 (on-device), avoiding transmission entirely

## Usage

### Parsing Logs with Communication Power
```bash
python parse_log_to_json.py --enable-comm-power
```

This will:
- Calculate communication energy for each layer
- Add `communication_energy_mj` to each exit layer
- Add `per_sample_mj_with_comm` to each multiplier result
- Add `per_sample_mj_with_comm` to baseline
- Add `communication_config` metadata to JSON

### Visualization
The `visualize_energy_analysis.py` script automatically detects and uses communication power data when available. The `energy_savings_highlight.png` plot now shows:
- Total energy including communication (bars)
- Accuracy comparison (blue lines)
- Energy savings percentage (accounting for communication)

## Files Modified
1. **parse_log_to_json.py**
   - Added `calculate_communication_energy()` function
   - Added `add_communication_energy_to_results()` function
   - Added `--enable-comm-power` command-line flag
   - Added baseline communication energy calculation

2. **visualize_energy_analysis.py**
   - Modified `aggregate_runs()` to detect and use `per_sample_mj_with_comm`
   - Updated energy calculations to include communication power when available
   - All plots now automatically account for communication energy

3. **verify_comm_power.py**
   - Verification script to inspect communication power calculations
   - Shows per-layer communication energy and weighted totals

## Impact on Results

Communication power adds a small but non-negligible overhead:
- **Width 100**: ~0.16 mJ baseline, ~0.08 mJ with early exit
- **Width 200**: ~0.32 mJ baseline, ~0.16 mJ with early exit (estimated)
- **Width 300**: ~0.48 mJ baseline, ~0.24 mJ with early exit (estimated)

This represents approximately **0.5-1%** additional energy overhead for the baseline, with proportionally less overhead for early exit configurations (due to more Layer 1 exits).

## Future Enhancements
- Support for configurable communication parameters (P_TX, data rate)
- Layer-specific data rates (e.g., adaptive bitrate)
- Bi-directional communication modeling (uplink + downlink)
- Power consumption during idle/waiting states
