def calculate_memory_gb(parameters: int, bits_per_parameter: int) -> float:
    """
    Calculate the memory required for a model in gigabytes.

    Args:
        parameters (int): The number of parameters in the model.
        bits_per_parameter (int): The number of bits per parameter.

    Returns:
        float: The memory required in gigabytes.
    """
    bytes_per_parameter = 4  # 4 bytes per parameter
    overhead = 1.2  # Overhead factor

    # Calculate memory in bytes
    memory_bytes = (parameters * bytes_per_parameter) / (bits_per_parameter / 8) * overhead

    # Convert bytes to gigabytes
    memory_gb = memory_bytes / (1024 ** 3)

    return memory_gb

# Example usage:
parameters = 405_000_000_000  # Example number of parameters
bits_per_parameter = 16  # Example bits per parameter (e.g., float32)

memory_gb = calculate_memory_gb(parameters, bits_per_parameter)
print(f"Memory required: {memory_gb:.2f} GB")