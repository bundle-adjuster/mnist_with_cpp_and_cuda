#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Matrix multiplication kernel (simple version)
__global__ void matrix_multiply_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void cuda_matrix_multiply(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    
    matrix_multiply_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Add bias kernel
__global__ void add_bias_kernel(
    float* output, const float* bias,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int row = idx / cols;
        output[idx] += bias[row];
    }
}

void cuda_add_bias(
    float* output, const float* bias,
    int rows, int cols
) {
    int total = rows * cols;
    int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    add_bias_kernel<<<numBlocks, BLOCK_SIZE>>>(output, bias, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ReLU activation kernel
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

void cuda_relu_activation(float* data, int size) {
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_kernel<<<numBlocks, BLOCK_SIZE>>>(data, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Leaky ReLU activation kernel
__global__ void leaky_relu_kernel(float* data, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (data[idx] > 0.0f) ? data[idx] : alpha * data[idx];
    }
}

void cuda_leaky_relu_activation(float* data, int size, float alpha) {
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    leaky_relu_kernel<<<numBlocks, BLOCK_SIZE>>>(data, size, alpha);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Softmax kernel
__global__ void softmax_kernel(float* data, int size, int batch_size) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    extern __shared__ float sdata[];
    float* batch_data = data + batch * size;
    
    // Find max for numerical stability
    float max_val = batch_data[0];
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (batch_data[i] > max_val) {
            max_val = batch_data[i];
        }
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    
    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        batch_data[i] = expf(batch_data[i] - max_val);
        sum += batch_data[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum = sdata[0];
    
    // Normalize
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        batch_data[i] /= sum;
    }
}

void cuda_softmax(float* data, int size, int batch_size) {
    int numBlocks = batch_size;
    int sharedMem = sizeof(float) * BLOCK_SIZE;
    softmax_kernel<<<numBlocks, BLOCK_SIZE, sharedMem>>>(data, size, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Compute output delta (for cross-entropy + softmax)
// Using pre-computed one-hot encoded targets for better performance
__global__ void compute_output_delta_kernel(
    const float* output, const float* target_onehot,
    float* delta, int size, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * size;
    
    if (idx < total) {
        // For softmax + cross-entropy, delta = output - target_one_hot
        // No conditional needed - target_onehot is already one-hot encoded
        delta[idx] = output[idx] - target_onehot[idx];
    }
}

void cuda_compute_output_delta(
    const float* output, const float* target_onehot,
    float* delta, int size, int batch_size
) {
    int total = batch_size * size;
    int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_output_delta_kernel<<<numBlocks, BLOCK_SIZE>>>(
        output, target_onehot, delta, size, batch_size
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Compute hidden layer delta
// next_weights connects current layer to next layer, stored as (current_size x next_size)
// For backpropagation: delta[current] = (next_weights^T * delta[next]) .* relu_derivative(activations[current])
// Since weights[neuron * next_size + j] is the weight from neuron to j, we compute:
// delta[current][neuron] = sum_j (weights[neuron * next_size + j] * delta[next][j]) * relu_derivative
__global__ void compute_hidden_delta_kernel(
    const float* next_delta, const float* next_weights,
    const float* activations, float* delta,
    int current_size, int next_size, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * current_size;
    
    if (idx < total) {
        int batch = idx / current_size;
        int neuron = idx % current_size;
        
        // Backpropagate through weights: sum over all neurons in next layer
        float sum = 0.0f;
        for (int j = 0; j < next_size; j++) {
            sum += next_delta[batch * next_size + j] * next_weights[neuron * next_size + j];
        }
        
        // Apply Leaky ReLU derivative: if activation <= 0, gradient is alpha (0.01), else 1.0
        float activation = activations[idx];
        delta[idx] = (activation > 0.0f) ? sum : 0.01f * sum;
    }
}

void cuda_compute_hidden_delta(
    const float* next_delta, const float* next_weights,
    const float* activations, float* delta,
    int current_size, int next_size, int batch_size
) {
    int total = batch_size * current_size;
    int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_hidden_delta_kernel<<<numBlocks, BLOCK_SIZE>>>(
        next_delta, next_weights, activations, delta,
        current_size, next_size, batch_size
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Update weights
// weights is stored as (input_size x output_size), where weights[input_idx * output_size + output_idx]
// Gradient: dW = (1/batch) * prev_activations^T * delta
// prev_activations is (batch_size x input_size), delta is (batch_size x output_size)
// So dW is (input_size x output_size)
__global__ void update_weights_kernel(
    float* weights, const float* delta,
    const float* prev_activations,
    float learning_rate,
    int input_size, int output_size, int batch_size
) {
    int input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (input_idx < input_size && output_idx < output_size) {
        float grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad += prev_activations[b * input_size + input_idx] * delta[b * output_size + output_idx];
        }
        // Gradient is already averaged over batch, so we multiply by learning_rate
        float update = learning_rate * grad / batch_size;
        weights[input_idx * output_size + output_idx] -= update;
    }
}

void cuda_update_weights(
    float* weights, const float* delta,
    const float* prev_activations,
    float learning_rate,
    int input_size, int output_size, int batch_size
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x,
                  (input_size + blockSize.y - 1) / blockSize.y);
    
    update_weights_kernel<<<gridSize, blockSize>>>(
        weights, delta, prev_activations, learning_rate,
        input_size, output_size, batch_size
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Update biases
__global__ void update_biases_kernel(
    float* biases, const float* delta,
    float learning_rate,
    int size, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad += delta[b * size + idx];
        }
        biases[idx] -= learning_rate * grad / batch_size;
    }
}

void cuda_update_biases(
    float* biases, const float* delta,
    float learning_rate,
    int size, int batch_size
) {
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    update_biases_kernel<<<numBlocks, BLOCK_SIZE>>>(
        biases, delta, learning_rate, size, batch_size
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ReLU derivative
__global__ void relu_derivative_kernel(
    const float* activations, float* delta,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        delta[idx] *= (activations[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

void cuda_relu_derivative(
    const float* activations, float* delta,
    int size
) {
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_derivative_kernel<<<numBlocks, BLOCK_SIZE>>>(activations, delta, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Leaky ReLU derivative kernel
__global__ void leaky_relu_derivative_kernel(
    const float* pre_activations, float* delta,
    int size, float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        delta[idx] *= (pre_activations[idx] > 0.0f) ? 1.0f : alpha;
    }
}

void cuda_leaky_relu_derivative(
    const float* pre_activations, float* delta,
    int size, float alpha
) {
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    leaky_relu_derivative_kernel<<<numBlocks, BLOCK_SIZE>>>(pre_activations, delta, size, alpha);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Cross-entropy loss
__global__ void cross_entropy_loss_kernel(
    const float* output, const int* target,
    float* loss, int num_classes, int batch_size
) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch < batch_size) {
        int target_class = target[batch];
        float prob = output[batch * num_classes + target_class];
        loss[batch] = -logf(fmaxf(prob, 1e-10f));
    }
}

void cuda_cross_entropy_loss(
    const float* output, const int* target,
    float* loss, int num_classes, int batch_size
) {
    int numBlocks = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cross_entropy_loss_kernel<<<numBlocks, BLOCK_SIZE>>>(
        output, target, loss, num_classes, batch_size
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
