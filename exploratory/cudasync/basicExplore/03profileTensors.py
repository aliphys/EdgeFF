import torch
import time
import numpy as np
import sys

# Optional: PyTorch Profiler for CUDA timeline
from torch.profiler import profile, ProfilerActivity
PROFILER_AVAILABLE = True

# Ensure CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available.")
    exit()

sizes = [
    (100, 100),
    (250, 250),
    (500, 500),
    (750, 750),
    (1000, 1000),
    (1500, 1500),
    (2000, 2000),
    (3000, 3000),
    (4000, 4000),
    (5000, 5000),
    (7500, 7500),
    (10000, 10000),
    (15000, 15000),
#    (20000, 20000),
]

num_replicates = 53  # 50 for stats, 3 to disregard
num_disregard = 3

print(f"{'Size':>12} | {'NoSync Mean (s)':>15} | {'NoSync Std (s)':>14} | {'Sync Mean (s)':>13} | {'Sync Std (s)':>12}")
print("-" * 75)
for shape in sizes:
    no_sync_times = []
    sync_times = []
    for rep in range(num_replicates):
        a = torch.randn(*shape, device='cuda')
        b = torch.randn(*shape, device='cuda')

        # Time without synchronization
        start_no_sync = time.time()
        c = a + b
        end_no_sync = time.time()
        no_sync_times.append(end_no_sync - start_no_sync)

        # Time with synchronization
        start_sync = time.time()
        c = a + b
        torch.cuda.synchronize()
        end_sync = time.time()
        sync_times.append(end_sync - start_sync)

        del a, b, c  # Free memory

    # Disregard the first 3 timings
    no_sync_valid = no_sync_times[num_disregard:]
    sync_valid = sync_times[num_disregard:]

    no_sync_mean = np.mean(no_sync_valid)
    no_sync_std = np.std(no_sync_valid)
    sync_mean = np.mean(sync_valid)
    sync_std = np.std(sync_valid)

    print(f"{str(shape):>12} | {no_sync_mean:15.6e} | {no_sync_std:14.6e} | {sync_mean:13.6e} | {sync_std:12.6e}")

# Example: Use PyTorch profiler for the largest size (unchanged)
print("\nPyTorch Profiler output for the largest size:")
shape = sizes[-1]
a = torch.randn(200000, device='cuda')
b = torch.randn(200000, device='cuda')
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    c = a + b
    #torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

sys.exit(0)
