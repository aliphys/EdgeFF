# Inference Energy and Latency Measurement Guide

## Overview
This implementation measures inference energy consumption (mJ) and latency (ms) per sample using the tegrastats hardware monitor on Jetson devices. The measurements are automatically logged to Weights & Biases (wandb).

## Implementation (Option B: Async with Timestamps)

### How It Works

1. **Background Monitoring**: The `TegratsMonitor` continuously collects power samples in the background at regular intervals (default: 1000ms)

2. **Timestamped Power History**: Recent power samples are stored with timestamps in a rolling buffer (last 1000 samples)

3. **Inference Window Capture**: 
   - When inference starts, a timestamp is recorded
   - During inference, power samples continue to be collected
   - When inference ends, another timestamp is recorded
   - Power samples within the time window are extracted

4. **Metric Calculation**:
   - **Latency**: `end_time - start_time` (converted to ms)
   - **Average Power**: Mean of power samples during inference window (mW)
   - **Energy**: `Average Power × Latency` (mJ)
   - **Per-sample metrics**: Divide by batch size

### Components Modified

#### 1. `tegrats_monitor.py`

**New `InferenceMetrics` class**:
- Stores power samples with timestamps
- Calculates energy and latency metrics
- Handles edge cases (no samples in window, interpolation)

**Enhanced `TegratsMonitor` class**:
- `power_history`: Rolling buffer of recent power samples
- `start_inference_measurement()`: Begin tracking inference
- `stop_inference_measurement(batch_size)`: End tracking and calculate metrics

#### 2. `Evaluation.py`

**New `eval_with_inference_measurement()` function**:
- Wraps model evaluation with energy/latency measurement
- Processes data in batches
- Collects metrics for each batch
- Averages metrics across all batches
- Returns comprehensive results dictionary

**Metrics returned**:
```python
{
    '{set_name}/accuracy': float,
    '{set_name}/f1_score': float,
    '{set_name}/error': float,
    '{set_name}/inference_latency_per_sample_ms': float,      # Average per sample
    '{set_name}/inference_energy_per_sample_mj': float,       # Average per sample
    '{set_name}/inference_avg_power_mw': float,               # Average during inference
    '{set_name}/inference_total_latency_ms': float,           # Total for entire set
    '{set_name}/inference_total_energy_mj': float,            # Total for entire set
}
```

#### 3. `Main.py`

**Modified evaluation section**:
- Checks if `hw_monitor` is available
- Uses `eval_with_inference_measurement()` when available
- Falls back to original evaluation functions otherwise
- Logs all metrics to wandb automatically

### Usage

#### Running with Hardware Monitoring

```bash
# Hardware monitoring is enabled by default
python Main.py --layers 784,100,100,100

# Explicitly enable with custom interval
python Main.py --enable-hw-monitor --hw-interval-ms 500

# Disable hardware monitoring
python Main.py --enable-hw-monitor=False
```

#### Metrics in Weights & Biases

The following metrics will be logged to wandb:

**Training Set**:
- `train/inference_latency_per_sample_ms`
- `train/inference_energy_per_sample_mj`
- `train/inference_avg_power_mw`
- `train/inference_total_latency_ms`
- `train/inference_total_energy_mj`

**Test Set**:
- `test/inference_latency_per_sample_ms`
- `test/inference_energy_per_sample_mj`
- `test/inference_avg_power_mw`
- `test/inference_total_latency_ms`
- `test/inference_total_energy_mj`

**Validation Set**:
- `validation/inference_latency_per_sample_ms`
- `validation/inference_energy_per_sample_mj`
- `validation/inference_avg_power_mw`
- `validation/inference_total_latency_ms`
- `validation/inference_total_energy_mj`

### Example Output

```
Results for the TEST set:
	F1-score: 0.9234
	Accuracy: 0.9245
	Error: 0.0755

Inference Metrics for TEST set:
	Latency per sample: 0.0234 ms
	Energy per sample: 0.0567 mJ
	Average power: 2423.45 mW
```

### Key Features

✅ **Automatic Integration**: No code changes needed in model inference logic
✅ **Batch Processing**: Handles batches of any size efficiently  
✅ **Timestamp-based**: Uses async monitoring with timestamp correlation
✅ **Wandb Logging**: All metrics automatically logged
✅ **Fallback Support**: Works even if hw_monitor is not available
✅ **Per-sample Metrics**: Calculates both total and per-sample values
✅ **Edge Case Handling**: Interpolates when no samples fall in exact window

### Technical Notes

**Sampling Rate Considerations**:
- Default sampling interval: 1000ms (1 Hz)
- For more accurate measurements on fast inference, reduce `--hw-interval-ms`
- Minimum recommended: 100ms (10 Hz)

**Energy Calculation**:
- Energy (mJ) = Average Power (mW) × Time (s)
- Uses VDD_IN channel (total module power) from INA3221
- Averages power samples within inference time window

**Accuracy**:
- Accuracy depends on sampling frequency
- Async approach is slightly less precise than synchronized measurement
- Trade-off: simpler implementation, no interference with inference timing
- Best for batch inference (multiple samples at once)

### Limitations

1. **Sampling Granularity**: If inference is very fast (<< sampling interval), may only capture 1-2 power samples
2. **Background Power**: Includes system baseline power (not subtracted)
3. **Jetson Only**: Requires tegrastats and INA3221 (Jetson Orin devices)
4. **Total Power**: Measures entire module power, not just inference-specific

### Future Enhancements

- Add baseline power subtraction
- Support for per-layer measurements
- GPU-only power isolation
- Adaptive sampling rate during inference
- Warm-up pass filtering
