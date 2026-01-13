#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

// CUDA kernel declarations for neural network operations

// Forward propagation kernels
void cuda_matrix_multiply(
    const float* A, const float* B, float* C,
    int M, int N, int K
);

void cuda_add_bias(
    float* output, const float* bias,
    int rows, int cols
);

void cuda_relu_activation(
    float* data, int size
);

void cuda_leaky_relu_activation(
    float* data, int size, float alpha = 0.01f
);

void cuda_softmax(
    float* data, int size, int batch_size
);

// Backward propagation kernels
void cuda_compute_output_delta(
    const float* output, const float* target_onehot,
    float* delta, int size, int batch_size
);

void cuda_compute_hidden_delta(
    const float* next_delta, const float* next_weights,
    const float* activations, float* delta,
    int current_size, int next_size, int batch_size
);

void cuda_update_weights(
    float* weights, const float* delta,
    const float* prev_activations,
    float learning_rate,
    int input_size, int output_size, int batch_size
);

void cuda_update_biases(
    float* biases, const float* delta,
    float learning_rate,
    int size, int batch_size
);

void cuda_relu_derivative(
    const float* activations, float* delta,
    int size
);

void cuda_leaky_relu_derivative(
    const float* pre_activations, float* delta,
    int size, float alpha = 0.01f
);

// Utility kernels
void cuda_cross_entropy_loss(
    const float* output, const int* target,
    float* loss, int num_classes, int batch_size
);

#endif // CUDA_KERNELS_H
