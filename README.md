# MNIST Machine Learning with CUDA

A C++ implementation of a neural network for MNIST digit classification using CUDA for accelerated backpropagation.

## Features

- **CUDA-accelerated backpropagation**: All gradient computations run on GPU
- **Flexible data loading**: Python script supports both PyTorch and TensorFlow datasets
- **MAT file format**: Data stored in .mat format for easy C++ consumption
- **Multi-layer neural network**: Configurable architecture with ReLU activations and softmax output

## Requirements

### C++ Dependencies
- **CUDA Toolkit** (version 10.0 or higher)
- **CMake** (version 3.10 or higher)
- **MatIO library** (`libmatio-dev` on Ubuntu/Debian)
- **C++17 compatible compiler** (GCC 7+ or Clang 5+)

### Python Dependencies (for data download)
- Python 3.6+
- NumPy
- SciPy
- PyTorch OR TensorFlow (at least one)

## Installation

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libmatio-dev
```

**Other Linux distributions:**
Install equivalent packages for your distribution.

### 2. Install CUDA

Follow NVIDIA's official CUDA installation guide for your system:
https://developer.nvidia.com/cuda-downloads

### 3. Install Python Dependencies

```bash
pip install numpy scipy torch torchvision
# OR
pip install numpy scipy tensorflow
```

## Building the Project

1. **Download MNIST data:**
```bash
python download_mnist.py
```

This will create a `data/` directory with:
- `mnist_train.mat`
- `mnist_test.mat`

2. **Build the C++ project:**
```bash
mkdir build
cd build
cmake ..
make
```

3. **Run training:**
```bash
./MNIST_ML_CUDA ../data/mnist_train.mat ../data/mnist_test.mat [epochs] [learning_rate] [batch_size]
```

Example:
```bash
./MNIST_ML_CUDA ../data/mnist_train.mat ../data/mnist_test.mat 10 0.01 32
```

## Project Structure

```
ml_cpp/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── download_mnist.py       # Python script to download and convert MNIST
├── include/                # Header files
│   ├── neural_network.h    # Neural network class definition
│   ├── mat_reader.h        # MAT file reader
│   └── cuda_kernels.h      # CUDA kernel declarations
└── src/                    # Source files
    ├── main.cpp            # Main training loop
    ├── neural_network.cpp  # Neural network implementation
    ├── mat_reader.cpp      # MAT file reader implementation
    └── cuda_kernels.cu     # CUDA kernel implementations
```

## Architecture

The default neural network architecture is:
- **Input layer**: 784 neurons (28×28 MNIST images)
- **Hidden layer 1**: 128 neurons with ReLU activation
- **Hidden layer 2**: 64 neurons with ReLU activation
- **Output layer**: 10 neurons with softmax activation (one per digit class)

You can modify the architecture in `src/main.cpp` by changing the `layer_sizes` vector.

## CUDA Implementation

The following operations are accelerated on GPU:
- Matrix multiplications (forward pass)
- Activation functions (ReLU, softmax)
- Gradient computations (backward pass)
- Weight and bias updates

All CUDA kernels are implemented in `src/cuda_kernels.cu`.

## Performance Notes

- The current implementation processes samples one at a time for simplicity
- For better performance, consider implementing true batch processing on GPU
- Adjust `CUDA_NVCC_FLAGS` in `CMakeLists.txt` for your GPU architecture (sm_75 = compute capability 7.5)

## Troubleshooting

### CUDA not found
- Ensure CUDA is installed and `nvcc` is in your PATH
- Set `CUDA_PATH` environment variable if needed

### MatIO library not found
- Install `libmatio-dev` package
- Or build from source: https://github.com/tbeu/matio

### Out of memory errors
- Reduce batch size
- Use a smaller network architecture

## License

This project is provided as-is for educational purposes.
