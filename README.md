
# VRAM Calculator for Large Language Models (LLM)

This Python script calculates the total VRAM required to run a specific Large Language Model (LLM) based on several parameters such as model size, context length, and model architecture. It supports user input via a command-line interface (CLI) and provides flexible output formats (console or JSON).

## Features

- Calculate **Model VRAM** based on the number of parameters and quantization (e.g., 8-bit).
- Calculate **Context VRAM** based on context length, hidden layers, hidden size, and key-value cache bytes.
- Command-line interface for easy input.
- Option to output results in either **console** or **JSON** format.
- Supports different LLM configurations.

## Requirements

- Python 3.6+
- Libraries: `numpy`

You can install the required libraries using the following command:

```bash
pip install numpy
```

## Usage

### Running the Script

You can run the script directly from the command line by providing the necessary parameters. Below is the general command:

```bash
python vram_calculator.py --num_params <NUMBER_OF_PARAMS> --bytes_per_param <BYTES_PER_PARAM> --context_length <CONTEXT_LENGTH> --num_hidden_layers <HIDDEN_LAYERS> --hidden_size <HIDDEN_SIZE> --kv_cache_bpw <KV_CACHE_BYTES_PER_WORD> --output_format <FORMAT>
```

### Parameters

- `--num_params` (float, required): The number of parameters in the model. Example: `8e9` for 8 billion parameters.
- `--bytes_per_param` (int, optional): The number of bytes per parameter (default: `1` for 8-bit quantized models). Adjust for different precisions like float16 or float32.
- `--context_length` (int, optional): The context length (number of tokens) for the model (default: `4096`).
- `--num_hidden_layers` (int, optional): The number of hidden layers in the model (default: `32`).
- `--hidden_size` (int, optional): The size of each hidden layer (default: `4608`).
- `--kv_cache_bpw` (int, optional): The number of bytes per word in the key-value cache (default: `4` bytes for float16 precision).
- `--output_format` (str, optional): The output format, either `console` (default) or `json`.

### Example Command

```bash
python vram_calculator.py --num_params 8e9 --bytes_per_param 1 --context_length 4096 --num_hidden_layers 32 --hidden_size 4608 --kv_cache_bpw 4 --output_format json
```

In this example, the script calculates the VRAM required for a **8B model** with **8-bit quantization**, a **context length** of **4096 tokens**, **32 hidden layers**, a **hidden size** of **4608**, and **4 bytes per word** for the KV cache. The output is displayed in **JSON format**.

### Sample Output (Console)

```
Total VRAM needed: 12.03 GB
```

### Sample Output (JSON)

```json
{
    "num_params": 8000000000.0,
    "bytes_per_param": 1,
    "context_length": 4096,
    "num_hidden_layers": 32,
    "hidden_size": 4608,
    "kv_cache_bpw": 4,
    "total_vram_needed_gb": 12.03
}
```

## Additional Notes

- The script automatically converts memory usage to **GB** from bytes.
- You can adjust the parameters based on the specific configuration of the model you're working with.
- The performance and VRAM estimates are general; actual results may vary depending on model implementation and system optimizations.

## License

This project is licensed under the MIT License.
