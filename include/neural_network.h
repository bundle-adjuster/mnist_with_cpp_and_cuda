#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layer_sizes, float learning_rate = 0.01f, int max_batch_size = 128);
    ~NeuralNetwork();
    
    // Disable copy constructor and assignment operator
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    
    // Forward pass
    void forward(const float* input, float* output);
    
    // Backward pass (CUDA implementation)
    void backward(const float* input, const int* target, float* loss);
    
    // Training step (single sample)
    void train_step(const float* input, const int* target, float* loss);
    
    // Batch training step (process multiple samples at once)
    void train_batch(const float* inputs, const int* targets, int batch_size, float* avg_loss);
    
    // Prediction
    int predict(const float* input);
    
    // Save/Load weights
    void save_weights(const std::string& filename);
    void load_weights(const std::string& filename);
    
    // Get accuracy on dataset
    float evaluate(const float* inputs, const int* labels, int num_samples);
    
    // Diagnostic functions
    void print_gradient_stats(int layer_idx, int batch_size);
    void print_weight_stats(int layer_idx);
    void verify_weight_updates(int layer_idx);
    
private:
    std::vector<int> layer_sizes_;
    int num_layers_;
    float learning_rate_;
    
    // Device (GPU) memory pointers
    float** d_weights_;      // Weights for each layer
    float** d_biases_;       // Biases for each layer
    float** d_activations_;  // Activations for each layer (post-activation)
    float** d_pre_activations_; // Pre-activation values (before ReLU/softmax) for gradient computation
    float** d_deltas_;       // Deltas for backpropagation
    float** d_weight_grads_; // Weight gradients
    float** d_bias_grads_;   // Bias gradients
    
    // Pre-allocated temporary buffers (to avoid per-sample allocation)
    float* d_input_buffer_;   // Input buffer (reused, supports batches)
    int* d_target_buffer_;    // Target buffer (reused, supports batches) - integer labels
    float* d_target_onehot_;  // Target buffer in one-hot format (reused, supports batches)
    float* d_loss_buffer_;   // Loss buffer (reused, supports batches)
    int max_batch_size_;      // Maximum batch size for buffer allocation
    
    // Host (CPU) memory for weights (for saving/loading)
    std::vector<std::vector<float>> h_weights_;
    std::vector<std::vector<float>> h_biases_;
    
    // Helper functions
    void allocate_gpu_memory();
    void free_gpu_memory();
    void initialize_weights();
    void copy_weights_to_host();
    void copy_weights_to_device();
};

#endif // NEURAL_NETWORK_H
