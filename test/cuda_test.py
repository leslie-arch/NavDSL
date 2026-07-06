import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")

    # Get device name and properties
    device_id = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_id)
    print(f"Device Name: {torch.cuda.get_device_name(device_id)}")
    print(f"Total Memory: {properties.total_memory / 1e9:.2f} GB")

    # Memory allocated and cached
    print(f"Allocated Memory: {torch.cuda.memory_allocated(device_id) / 1e9:.2f} GB")
    print(f"Reserved Memory: {torch.cuda.memory_reserved(device_id) / 1e9:.2f} GB")
else:
    print("CUDA is not available.")
