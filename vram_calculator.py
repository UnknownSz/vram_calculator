import numpy as np
import json
import argparse

BYTES_PER_GB = 1_073_741_824  # 1 GB in bytes

# Function to calculate Model VRAM
def calculate_model_vram(num_params, bytes_per_param):
    """
    Calculate the Model VRAM required.
    """
    try:
        model_vram = num_params * bytes_per_param
        model_vram_gb = model_vram / BYTES_PER_GB
        return model_vram_gb
    except Exception as e:
        raise ValueError(f"Error calculating Model VRAM: {e}")

# Function to calculate Context VRAM
def calculate_context_vram(context_length, num_hidden_layers, hidden_size, kv_cache_bpw):
    """
    Calculate the Context VRAM required.
    """
    try:
        context_vram = context_length * num_hidden_layers * hidden_size * kv_cache_bpw
        context_vram_gb = context_vram / BYTES_PER_GB
        return context_vram_gb
    except Exception as e:
        raise ValueError(f"Error calculating Context VRAM: {e}")

# Function to calculate total VRAM
def calculate_total_vram(num_params, bytes_per_param, context_length, num_hidden_layers, hidden_size, kv_cache_bpw):
    """
    Calculate the total VRAM required for the LLM.
    """
    model_vram_gb = calculate_model_vram(num_params, bytes_per_param)
    context_vram_gb = calculate_context_vram(context_length, num_hidden_layers, hidden_size, kv_cache_bpw)
    total_vram_gb = model_vram_gb + context_vram_gb
    return total_vram_gb

# Enhanced CLI for User Input
def main():
    parser = argparse.ArgumentParser(description="Calculate VRAM needed for an LLM.")
    
    # Adding CLI arguments
    parser.add_argument("--num_params", type=float, required=True, help="Number of parameters in the model (e.g., 8e9 for 8B).")
    parser.add_argument("--bytes_per_param", type=int, default=1, help="Bytes per parameter (1 for 8-bit quantized models).")
    parser.add_argument("--context_length", type=int, default=4096, help="Context length (number of tokens).")
    parser.add_argument("--num_hidden_layers", type=int, default=32, help="Number of hidden layers.")
    parser.add_argument("--hidden_size", type=int, default=4608, help="Size of the hidden layers.")
    parser.add_argument("--kv_cache_bpw", type=int, default=4, help="Key-value cache bytes per word (usually 4 bytes for float16).")
    parser.add_argument("--output_format", choices=["json", "console"], default="console", help="Output format (default: console).")
    
    args = parser.parse_args()

    # Calculating total VRAM
    total_vram_needed = calculate_total_vram(
        args.num_params, args.bytes_per_param, args.context_length, 
        args.num_hidden_layers, args.hidden_size, args.kv_cache_bpw
    )

    # Output the results
    if args.output_format == "json":
        output_data = {
            "num_params": args.num_params,
            "bytes_per_param": args.bytes_per_param,
            "context_length": args.context_length,
            "num_hidden_layers": args.num_hidden_layers,
            "hidden_size": args.hidden_size,
            "kv_cache_bpw": args.kv_cache_bpw,
            "total_vram_needed_gb": round(total_vram_needed, 2)
        }
        print(json.dumps(output_data, indent=4))
    else:
        print(f"Total VRAM needed: {total_vram_needed:.2f} GB")

# Run the CLI interface if the script is executed directly
if __name__ == "__main__":
    main()
