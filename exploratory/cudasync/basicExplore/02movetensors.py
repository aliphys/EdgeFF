import torch
import time
import sys

from torch.profiler import profile, ProfilerActivity
PROFILER_AVAILABLE = True

# Ensure CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available.")
    exit()


sizes = [
    (100, 100),
    (500, 500),
    (1000, 1000),
    (2000, 2000),
    (5000, 5000),
    (10000, 10000),
    #(20000, 20000),# crashes
    
]

print(f"{'Size':>12} | {'No Sync (s)':>12} | {'Sync (s)':>12}")
print("-" * 42)
for shape in sizes:
    a = torch.randn(*shape, device='cuda')
    b = torch.randn(*shape, device='cuda')

    # Time without synchronization
    start_no_sync = time.time()
    c = a + b
    end_no_sync = time.time()
    no_sync_time = end_no_sync - start_no_sync

    # Time with synchronization
    start_sync = time.time()
    c = a + b
    torch.cuda.synchronize()
    end_sync = time.time()
    sync_time = end_sync - start_sync

    del a, b, c  # Free memory

    print(f"{str(shape):>12} | {no_sync_time:12.6f} | {sync_time:12.6f}")

custom_sizes = (16000, 16000)

# Example: Use PyTorch profiler for the largest size without synchronization
print("\nPyTorch Profiler output for the largest size (no synchronization):")
print("\nPyTorch Profiler output for the largest size:")
a = torch.randn(custom_sizes, device='cuda')
b = torch.randn(custom_sizes, device='cuda')
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],record_shapes=True, profile_memory=True) as prof:
    for _ in range(1):
        c = a + b
        #torch.cuda.synchronize()
        del c  # Free memory immediately after each addition
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

del a, b  # Free memory

# Example: Use PyTorch profiler for the largest size with synchronization
print("\nPyTorch Profiler output for the largest size (with synchronization):")
print("\nPyTorch Profiler output for the largest size:")
shape = sizes[-1]
a = torch.randn(custom_sizes, device='cuda')
b = torch.randn(custom_sizes, device='cuda')
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    for _ in range(1):
        c = a + b
        torch.cuda.synchronize()
        del c  # Free memory immediately after each addition
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

del a, b  # Free memory

sys.exit(0)
