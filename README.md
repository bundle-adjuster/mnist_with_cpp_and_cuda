# MNIST & Fashion-MNIST Machine Learning with CUDA

A C++ implementation of a neural network for MNIST digit classification and Fashion-MNIST clothing classification using CUDA for accelerated backpropagation.

## Features

- **CUDA-accelerated backpropagation**: All gradient computations run on GPU
- **Multiple datasets**: Supports both MNIST and Fashion-MNIST datasets
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

1. **Download dataset data:**

For MNIST:
```bash
python download_mnist.py --dataset mnist
```

For Fashion-MNIST:
```bash
python download_mnist.py --dataset fashion
```

Or download both:
```bash
python download_mnist.py --dataset mnist
python download_mnist.py --dataset fashion
```

This will create a `data/` directory with:
- `mnist_train.mat` and `mnist_test.mat` (for MNIST)
- `fashion_mnist_train.mat` and `fashion_mnist_test.mat` (for Fashion-MNIST)

You can also specify the source (PyTorch or TensorFlow) and output directory:
```bash
python download_mnist.py --dataset fashion --source torch --output-dir data
```

2. **Build the C++ project:**
```bash
mkdir build
cd build
cmake ..
make
```

3. **Run training:**

For MNIST:
```bash
./MNIST_ML_CUDA ../data/mnist_train.mat ../data/mnist_test.mat [epochs] [learning_rate] [batch_size]
```

For Fashion-MNIST:
```bash
./MNIST_ML_CUDA ../data/fashion_mnist_train.mat ../data/fashion_mnist_test.mat [epochs] [learning_rate] [batch_size]
```

Examples:
```bash
# Train on MNIST
./MNIST_ML_CUDA ../data/mnist_train.mat ../data/mnist_test.mat 10 0.01 32

# Train on Fashion-MNIST
./MNIST_ML_CUDA ../data/fashion_mnist_train.mat ../data/fashion_mnist_test.mat 10 0.01 32
```

4. **Test predictions:**

For MNIST:
```bash
./test_predictions ../data/mnist_test.mat mnist_model.bin
```

For Fashion-MNIST:
```bash
./test_predictions ../data/fashion_mnist_test.mat fashion_mnist_model.bin
```

## Project Structure

```
ml_cpp/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── download_mnist.py       # Python script to download and convert MNIST/Fashion-MNIST
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
- **Input layer**: 784 neurons (28×28 images for both MNIST and Fashion-MNIST)
- **Hidden layer 1**: 128 neurons with ReLU activation
- **Hidden layer 2**: 64 neurons with ReLU activation
- **Output layer**: 10 neurons with softmax activation (one per class)
  - For MNIST: 10 digit classes (0-9)
  - For Fashion-MNIST: 10 clothing classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

You can modify the architecture in `src/main.cpp` by changing the `layer_sizes` vector.

## Fashion-MNIST Classes

Fashion-MNIST has 10 classes:
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

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
