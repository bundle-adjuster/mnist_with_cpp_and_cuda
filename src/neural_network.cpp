#include "neural_network.h"
#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes, float learning_rate, int max_batch_size)
    : layer_sizes_(layer_sizes), num_layers_(layer_sizes.size()), learning_rate_(learning_rate), max_batch_size_(max_batch_size) {
    
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("Neural network must have at least 2 layers");
    }
    
    allocate_gpu_memory();
    initialize_weights();
}

NeuralNetwork::~NeuralNetwork() {
    free_gpu_memory();
}

void NeuralNetwork::allocate_gpu_memory() {
    // Allocate arrays of pointers
    d_weights_ = new float*[num_layers_ - 1];
    d_biases_ = new float*[num_layers_ - 1];
    d_activations_ = new float*[num_layers_];
    d_pre_activations_ = new float*[num_layers_];
    d_deltas_ = new float*[num_layers_ - 1];
    d_weight_grads_ = new float*[num_layers_ - 1];
    d_bias_grads_ = new float*[num_layers_ - 1];
    
    // Initialize temporary buffer pointers
    d_input_buffer_ = nullptr;
    d_target_buffer_ = nullptr;
    d_target_onehot_ = nullptr;
    d_loss_buffer_ = nullptr;
    
    // Allocate memory for each layer
    for (int i = 0; i < num_layers_ - 1; i++) {
        int weight_size = layer_sizes_[i] * layer_sizes_[i + 1];
        int bias_size = layer_sizes_[i + 1];
        
        CUDA_CHECK(cudaMalloc(&d_weights_[i], weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_biases_[i], bias_size * sizeof(float)));
        // Allocate deltas for maximum batch size
        CUDA_CHECK(cudaMalloc(&d_deltas_[i], max_batch_size_ * bias_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weight_grads_[i], weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias_grads_[i], bias_size * sizeof(float)));
    }
    
    // Allocate activations and pre-activations for maximum batch size
    for (int i = 1; i < num_layers_; i++) {
        CUDA_CHECK(cudaMalloc(&d_activations_[i], max_batch_size_ * layer_sizes_[i] * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pre_activations_[i], max_batch_size_ * layer_sizes_[i] * sizeof(float)));
    }
    
    // Pre-allocate temporary buffers for maximum batch size
    CUDA_CHECK(cudaMalloc(&d_input_buffer_, max_batch_size_ * layer_sizes_[0] * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_target_buffer_, max_batch_size_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_target_onehot_, max_batch_size_ * layer_sizes_[num_layers_ - 1] * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss_buffer_, max_batch_size_ * sizeof(float)));
    
    // Initialize host memory for weights
    h_weights_.resize(num_layers_ - 1);
    h_biases_.resize(num_layers_ - 1);
    for (int i = 0; i < num_layers_ - 1; i++) {
        h_weights_[i].resize(layer_sizes_[i] * layer_sizes_[i + 1]);
        h_biases_[i].resize(layer_sizes_[i + 1]);
    }
}

void NeuralNetwork::free_gpu_memory() {
    if (d_weights_) {
        for (int i = 0; i < num_layers_ - 1; i++) {
            if (d_weights_[i]) cudaFree(d_weights_[i]);
            if (d_biases_[i]) cudaFree(d_biases_[i]);
            if (d_deltas_[i]) cudaFree(d_deltas_[i]);
            if (d_weight_grads_[i]) cudaFree(d_weight_grads_[i]);
            if (d_bias_grads_[i]) cudaFree(d_bias_grads_[i]);
        }
        delete[] d_weights_;
        delete[] d_biases_;
        delete[] d_deltas_;
        delete[] d_weight_grads_;
        delete[] d_bias_grads_;
    }
    
    if (d_activations_) {
        for (int i = 1; i < num_layers_; i++) {
            if (d_activations_[i]) cudaFree(d_activations_[i]);
        }
        delete[] d_activations_;
    }
    
    if (d_pre_activations_) {
        for (int i = 1; i < num_layers_; i++) {
            if (d_pre_activations_[i]) cudaFree(d_pre_activations_[i]);
        }
        delete[] d_pre_activations_;
    }
    
    // Free pre-allocated buffers
    if (d_input_buffer_) cudaFree(d_input_buffer_);
    if (d_target_buffer_) cudaFree(d_target_buffer_);
    if (d_target_onehot_) cudaFree(d_target_onehot_);
    if (d_loss_buffer_) cudaFree(d_loss_buffer_);
}

void NeuralNetwork::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // He initialization (better for ReLU)
    // For uniform distribution U[-a, a]: variance = a²/3
    // We want variance = 2/fan_in, so: a²/3 = 2/fan_in => a = sqrt(6/fan_in)
    for (int i = 0; i < num_layers_ - 1; i++) {
        float fan_in = static_cast<float>(layer_sizes_[i]);
        float limit = sqrtf(6.0f / fan_in);
        std::uniform_real_distribution<float> dis(-limit, limit);
        
        for (size_t j = 0; j < h_weights_[i].size(); j++) {
            h_weights_[i][j] = dis(gen);
        }
        
        // Initialize biases to small positive values to avoid dead neurons
        std::uniform_real_distribution<float> bias_dis(0.0f, 0.01f);
        for (size_t j = 0; j < h_biases_[i].size(); j++) {
            h_biases_[i][j] = bias_dis(gen);
        }
    }
    
    copy_weights_to_device();
}

void NeuralNetwork::copy_weights_to_device() {
    for (int i = 0; i < num_layers_ - 1; i++) {
        CUDA_CHECK(cudaMemcpy(d_weights_[i], h_weights_[i].data(),
                             h_weights_[i].size() * sizeof(float),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_biases_[i], h_biases_[i].data(),
                             h_biases_[i].size() * sizeof(float),
                             cudaMemcpyHostToDevice));
    }
}

void NeuralNetwork::copy_weights_to_host() {
    for (int i = 0; i < num_layers_ - 1; i++) {
        CUDA_CHECK(cudaMemcpy(h_weights_[i].data(), d_weights_[i],
                             h_weights_[i].size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_biases_[i].data(), d_biases_[i],
                             h_biases_[i].size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
    }
}

void NeuralNetwork::forward(const float* input, float* output) {
    // Copy input to pre-allocated device buffer
    CUDA_CHECK(cudaMemcpy(d_input_buffer_, input, layer_sizes_[0] * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    const float* current_input = d_input_buffer_;
    
    // Forward pass through all layers
    for (int i = 0; i < num_layers_ - 1; i++) {
        int input_size = layer_sizes_[i];
        int output_size = layer_sizes_[i + 1];
        
        // Matrix multiplication: output = input * weights
        // input is (1 x input_size), weights is (input_size x output_size)
        cuda_matrix_multiply(
            current_input, d_weights_[i], d_activations_[i + 1],
            1, output_size, input_size
        );
        
        // Add bias
        cuda_add_bias(d_activations_[i + 1], d_biases_[i], 1, output_size);
        
        // Apply activation (ReLU for hidden layers, softmax for output)
        if (i < num_layers_ - 2) {
            cuda_leaky_relu_activation(d_activations_[i + 1], 1 * output_size, 0.01f);
        } else {
            cuda_softmax(d_activations_[i + 1], output_size, 1);
        }
        
        current_input = d_activations_[i + 1];
    }
    
    // Copy output to host
    int output_size = layer_sizes_[num_layers_ - 1];
    CUDA_CHECK(cudaMemcpy(output, d_activations_[num_layers_ - 1],
                         output_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

void NeuralNetwork::backward(const float* input, const int* target, float* loss) {
    int batch_size = 1; // For single sample
    int output_size = layer_sizes_[num_layers_ - 1];
    
    // Copy input to pre-allocated device buffer
    CUDA_CHECK(cudaMemcpy(d_input_buffer_, input, layer_sizes_[0] * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Forward pass to get activations - store pre-activation values
    const float* current_input = d_input_buffer_;
    for (int i = 0; i < num_layers_ - 1; i++) {
        int input_size = layer_sizes_[i];
        int output_size = layer_sizes_[i + 1];
        
        // Compute pre-activation: z = input * weights + bias
        cuda_matrix_multiply(
            current_input, d_weights_[i], d_pre_activations_[i + 1],
            batch_size, output_size, input_size
        );
        cuda_add_bias(d_pre_activations_[i + 1], d_biases_[i], batch_size, output_size);
        
        // Copy pre-activation to activation before applying non-linearity
        CUDA_CHECK(cudaMemcpy(d_activations_[i + 1], d_pre_activations_[i + 1],
                             batch_size * output_size * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        
        // Apply activation function
        if (i < num_layers_ - 2) {
            cuda_leaky_relu_activation(d_activations_[i + 1], batch_size * output_size, 0.01f);
        } else {
            cuda_softmax(d_activations_[i + 1], output_size, batch_size);
        }
        
        current_input = d_activations_[i + 1];
    }
    
    // Compute loss using pre-allocated buffers
    CUDA_CHECK(cudaMemcpy(d_target_buffer_, target, sizeof(int), cudaMemcpyHostToDevice));
    
    cuda_cross_entropy_loss(d_activations_[num_layers_ - 1], d_target_buffer_, d_loss_buffer_,
                           output_size, batch_size);
    CUDA_CHECK(cudaMemcpy(loss, d_loss_buffer_, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Convert integer label to one-hot for delta computation
    float target_onehot[10] = {0.0f};  // Assuming max 10 classes
    target_onehot[*target] = 1.0f;
    float* d_target_onehot_single;
    CUDA_CHECK(cudaMalloc(&d_target_onehot_single, output_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_target_onehot_single, target_onehot, output_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Backward pass: compute deltas
    // Output layer delta - use one-hot encoded target
    cuda_compute_output_delta(
        d_activations_[num_layers_ - 1], d_target_onehot_single,
        d_deltas_[num_layers_ - 2], output_size, batch_size
    );
    
    cudaFree(d_target_onehot_single);
    
    // Hidden layers delta - use pre-activations for ReLU derivative
    for (int i = num_layers_ - 3; i >= 0; i--) {
        cuda_compute_hidden_delta(
            d_deltas_[i + 1], d_weights_[i + 1],
            d_pre_activations_[i + 1], d_deltas_[i],  // Use pre-activation for ReLU derivative
            layer_sizes_[i + 1], layer_sizes_[i + 2], batch_size
        );
    }
    
    // Update weights and biases
    // Store activations before ReLU for gradient computation
    // For layer i, we need the input to that layer (output of previous layer after activation)
    const float* prev_layer_output = d_input_buffer_;
    for (int i = 0; i < num_layers_ - 1; i++) {
        int input_size = layer_sizes_[i];
        int output_size = layer_sizes_[i + 1];
        
        cuda_update_weights(
            d_weights_[i], d_deltas_[i],
            prev_layer_output, learning_rate_,
            input_size, output_size, batch_size
        );
        cuda_update_biases(
            d_biases_[i], d_deltas_[i],
            learning_rate_, output_size, batch_size
        );
        
        // For next layer, use the activations after ReLU (or before softmax for output layer)
        prev_layer_output = d_activations_[i + 1];
    }
}

void NeuralNetwork::train_step(const float* input, const int* target, float* loss) {
    backward(input, target, loss);
}

void NeuralNetwork::train_batch(const float* inputs, const int* targets, int batch_size, float* avg_loss) {
    if (batch_size > max_batch_size_) {
        throw std::invalid_argument("Batch size exceeds maximum batch size");
    }
    
    int output_size = layer_sizes_[num_layers_ - 1];
    int input_size = layer_sizes_[0];
    
    // Copy batch to device
    CUDA_CHECK(cudaMemcpy(d_input_buffer_, inputs, batch_size * input_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target_buffer_, targets, batch_size * sizeof(int),
                         cudaMemcpyHostToDevice));
    
    // Convert integer labels to one-hot encoding on CPU and copy to GPU
    std::vector<float> target_onehot(batch_size * output_size, 0.0f);
    for (int i = 0; i < batch_size; i++) {
        int target_class = targets[i];
        if (target_class >= 0 && target_class < output_size) {
            target_onehot[i * output_size + target_class] = 1.0f;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_target_onehot_, target_onehot.data(),
                         batch_size * output_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Forward pass - store pre-activation values for gradient computation
    const float* current_input = d_input_buffer_;
    for (int i = 0; i < num_layers_ - 1; i++) {
        int layer_input_size = layer_sizes_[i];
        int layer_output_size = layer_sizes_[i + 1];
        
        // Compute pre-activation: z = input * weights + bias
        cuda_matrix_multiply(
            current_input, d_weights_[i], d_pre_activations_[i + 1],
            batch_size, layer_output_size, layer_input_size
        );
        cuda_add_bias(d_pre_activations_[i + 1], d_biases_[i], batch_size, layer_output_size);
        
        // Copy pre-activation to activation before applying non-linearity
        CUDA_CHECK(cudaMemcpy(d_activations_[i + 1], d_pre_activations_[i + 1],
                             batch_size * layer_output_size * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        
        // Apply activation function
        if (i < num_layers_ - 2) {
            // Use Leaky ReLU to avoid dead neurons
            cuda_leaky_relu_activation(d_activations_[i + 1], batch_size * layer_output_size, 0.01f);
        } else {
            cuda_softmax(d_activations_[i + 1], layer_output_size, batch_size);
        }
        
        current_input = d_activations_[i + 1];
    }
    
    // Compute loss
    cuda_cross_entropy_loss(d_activations_[num_layers_ - 1], d_target_buffer_, d_loss_buffer_,
                           output_size, batch_size);
    
    // Compute average loss
    float* h_losses = new float[batch_size];
    CUDA_CHECK(cudaMemcpy(h_losses, d_loss_buffer_, batch_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    *avg_loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        *avg_loss += h_losses[i];
    }
    *avg_loss /= batch_size;
    delete[] h_losses;
    
    // Backward pass: compute deltas
    // Output layer delta - use pre-computed one-hot encoded targets
    cuda_compute_output_delta(
        d_activations_[num_layers_ - 1], d_target_onehot_,
        d_deltas_[num_layers_ - 2], output_size, batch_size
    );
    
    // Diagnostic: Print output delta stats (first batch only, to avoid spam)
    static int batch_count = 0;
    if (batch_count == 0) {
        std::cout << "\n=== Gradient Diagnostics (First Batch) ===" << std::endl;
        print_gradient_stats(num_layers_ - 2, batch_size);
    }
    
    // Hidden layers delta - use pre-activations for Leaky ReLU derivative
    for (int i = num_layers_ - 3; i >= 0; i--) {
        cuda_compute_hidden_delta(
            d_deltas_[i + 1], d_weights_[i + 1],
            d_pre_activations_[i + 1], d_deltas_[i],  // Use pre-activation for Leaky ReLU derivative
            layer_sizes_[i + 1], layer_sizes_[i + 2], batch_size
        );
        
        // Diagnostic: Print hidden delta stats (first batch only)
        if (batch_count == 0) {
            print_gradient_stats(i, batch_size);
        }
    }
    
    // Update weights and biases
    // For layer i, we need the input to that layer (output of previous layer after activation)
    const float* prev_layer_output = d_input_buffer_;  // Input layer (no activation)
    
    // Store weights before update for verification (first batch only)
    std::vector<std::vector<float>> weights_before;
    if (batch_count == 0) {
        copy_weights_to_host();
        for (int i = 0; i < num_layers_ - 1; i++) {
            weights_before.push_back(h_weights_[i]);
        }
    }
    
    for (int i = 0; i < num_layers_ - 1; i++) {
        int layer_input_size = layer_sizes_[i];
        int layer_output_size = layer_sizes_[i + 1];
        
        cuda_update_weights(
            d_weights_[i], d_deltas_[i],
            prev_layer_output, learning_rate_,
            layer_input_size, layer_output_size, batch_size
        );
        cuda_update_biases(
            d_biases_[i], d_deltas_[i],
            learning_rate_, layer_output_size, batch_size
        );
        
        // For next layer, use the activations after Leaky ReLU (input to next layer)
        prev_layer_output = d_activations_[i + 1];
    }
    
    // Verify weights were updated (first batch only)
    if (batch_count == 0) {
        copy_weights_to_host();
        std::cout << "\n=== Weight Update Verification ===" << std::endl;
        for (int i = 0; i < num_layers_ - 1; i++) {
            float max_diff = 0.0f;
            int changed_count = 0;
            for (size_t j = 0; j < h_weights_[i].size(); j++) {
                float diff = fabsf(h_weights_[i][j] - weights_before[i][j]);
                max_diff = fmaxf(max_diff, diff);
                if (diff > 1e-6f) changed_count++;
            }
            std::cout << "Layer " << i << ": Max weight change: " << max_diff 
                      << ", Changed: " << changed_count << "/" << h_weights_[i].size() << std::endl;
        }
        std::cout << "====================================\n" << std::endl;
        batch_count++;
    }
}

int NeuralNetwork::predict(const float* input) {
    float* output = new float[layer_sizes_[num_layers_ - 1]];
    forward(input, output);
    
    int predicted = 0;
    float max_prob = output[0];
    for (int i = 1; i < layer_sizes_[num_layers_ - 1]; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted = i;
        }
    }
    
    delete[] output;
    return predicted;
}

float NeuralNetwork::evaluate(const float* inputs, const int* labels, int num_samples) {
    int correct = 0;
    int input_size = layer_sizes_[0];
    
    for (int i = 0; i < num_samples; i++) {
        int predicted = predict(inputs + i * input_size);
        if (predicted == labels[i]) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / num_samples;
}

void NeuralNetwork::save_weights(const std::string& filename) {
    copy_weights_to_host();
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    // Write number of layers
    file.write(reinterpret_cast<const char*>(&num_layers_), sizeof(int));
    
    // Write layer sizes
    file.write(reinterpret_cast<const char*>(layer_sizes_.data()),
              num_layers_ * sizeof(int));
    
    // Write weights and biases
    for (int i = 0; i < num_layers_ - 1; i++) {
        size_t weight_size = h_weights_[i].size();
        file.write(reinterpret_cast<const char*>(&weight_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(h_weights_[i].data()),
                  weight_size * sizeof(float));
        
        size_t bias_size = h_biases_[i].size();
        file.write(reinterpret_cast<const char*>(&bias_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(h_biases_[i].data()),
                  bias_size * sizeof(float));
    }
    
    file.close();
}

void NeuralNetwork::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    
    // Read number of layers
    int saved_num_layers;
    file.read(reinterpret_cast<char*>(&saved_num_layers), sizeof(int));
    
    if (saved_num_layers != num_layers_) {
        throw std::runtime_error("Number of layers mismatch");
    }
    
    // Read layer sizes
    std::vector<int> saved_layer_sizes(num_layers_);
    file.read(reinterpret_cast<char*>(saved_layer_sizes.data()),
             num_layers_ * sizeof(int));
    
    if (saved_layer_sizes != layer_sizes_) {
        throw std::runtime_error("Layer sizes mismatch");
    }
    
    // Read weights and biases
    for (int i = 0; i < num_layers_ - 1; i++) {
        size_t weight_size;
        file.read(reinterpret_cast<char*>(&weight_size), sizeof(size_t));
        h_weights_[i].resize(weight_size);
        file.read(reinterpret_cast<char*>(h_weights_[i].data()),
                 weight_size * sizeof(float));
        
        size_t bias_size;
        file.read(reinterpret_cast<char*>(&bias_size), sizeof(size_t));
        h_biases_[i].resize(bias_size);
        file.read(reinterpret_cast<char*>(h_biases_[i].data()),
                 bias_size * sizeof(float));
    }
    
    file.close();
    copy_weights_to_device();
}

// Diagnostic function to print gradient statistics
void NeuralNetwork::print_gradient_stats(int layer_idx, int batch_size) {
    if (layer_idx < 0 || layer_idx >= num_layers_ - 1) {
        std::cerr << "Invalid layer index: " << layer_idx << std::endl;
        return;
    }
    
    int delta_size = layer_sizes_[layer_idx + 1] * batch_size;
    float* h_delta = new float[delta_size];
    CUDA_CHECK(cudaMemcpy(h_delta, d_deltas_[layer_idx], delta_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    float sum = 0.0f;
    float max_val = h_delta[0];
    float min_val = h_delta[0];
    int zero_count = 0;
    
    for (int i = 0; i < delta_size; i++) {
        sum += h_delta[i];
        max_val = fmaxf(max_val, h_delta[i]);
        min_val = fminf(min_val, h_delta[i]);
        if (fabsf(h_delta[i]) < 1e-6f) zero_count++;
    }
    
    std::cout << "Layer " << layer_idx << " Delta Stats:" << std::endl;
    std::cout << "  Mean: " << (sum / delta_size) << std::endl;
    std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;
    std::cout << "  Zero count: " << zero_count << "/" << delta_size 
              << " (" << (100.0f * zero_count / delta_size) << "%)" << std::endl;
    
    delete[] h_delta;
}

// Diagnostic function to print weight statistics
void NeuralNetwork::print_weight_stats(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= num_layers_ - 1) {
        std::cerr << "Invalid layer index: " << layer_idx << std::endl;
        return;
    }
    
    copy_weights_to_host();
    
    float sum = 0.0f;
    float max_val = h_weights_[layer_idx][0];
    float min_val = h_weights_[layer_idx][0];
    
    for (size_t i = 0; i < h_weights_[layer_idx].size(); i++) {
        sum += h_weights_[layer_idx][i];
        max_val = fmaxf(max_val, h_weights_[layer_idx][i]);
        min_val = fminf(min_val, h_weights_[layer_idx][i]);
    }
    
    std::cout << "Layer " << layer_idx << " Weight Stats:" << std::endl;
    std::cout << "  Mean: " << (sum / h_weights_[layer_idx].size()) << std::endl;
    std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;
    std::cout << "  Total weights: " << h_weights_[layer_idx].size() << std::endl;
}

// Diagnostic function to verify weights are being updated
void NeuralNetwork::verify_weight_updates(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= num_layers_ - 1) {
        std::cerr << "Invalid layer index: " << layer_idx << std::endl;
        return;
    }
    
    // Copy weights before update
    copy_weights_to_host();
    std::vector<float> weights_before = h_weights_[layer_idx];
    
    // Copy weights after a small delay (weights should be on GPU)
    // Note: This is a simplified check - in practice, you'd compare before/after a training step
    std::cout << "Weight update verification for layer " << layer_idx << ":" << std::endl;
    std::cout << "  Weights are stored on GPU and updated in-place." << std::endl;
    std::cout << "  To verify updates, compare weights before/after training step." << std::endl;
}
