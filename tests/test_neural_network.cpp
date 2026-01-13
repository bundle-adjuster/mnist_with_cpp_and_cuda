#include <gtest/gtest.h>
#include "neural_network.h"
#include <vector>
#include <cmath>

// Test basic neural network creation
TEST(NeuralNetworkTest, CreateNetwork) {
    std::vector<int> layer_sizes = {784, 128, 64, 10};
    float learning_rate = 0.01f;
    int batch_size = 32;
    
    EXPECT_NO_THROW({
        NeuralNetwork nn(layer_sizes, learning_rate, batch_size);
    });
}

// Test forward pass produces valid output
TEST(NeuralNetworkTest, ForwardPass) {
    std::vector<int> layer_sizes = {784, 128, 64, 10};
    NeuralNetwork nn(layer_sizes, 0.01f, 32);
    
    // Create dummy input (all zeros for simplicity)
    std::vector<float> input(784, 0.0f);
    std::vector<float> output(10);
    
    EXPECT_NO_THROW({
        nn.forward(input.data(), output.data());
    });
    
    // Check that output is a valid probability distribution (softmax)
    float sum = 0.0f;
    for (float val : output) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
        sum += val;
    }
    
    // Sum should be approximately 1.0 (softmax normalization)
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// Test prediction returns valid class index
TEST(NeuralNetworkTest, Prediction) {
    std::vector<int> layer_sizes = {784, 128, 64, 10};
    NeuralNetwork nn(layer_sizes, 0.01f, 32);
    
    std::vector<float> input(784, 0.0f);
    
    int predicted = nn.predict(input.data());
    
    // Should return a valid class index (0-9 for MNIST/Fashion-MNIST)
    EXPECT_GE(predicted, 0);
    EXPECT_LT(predicted, 10);
}

// Test batch training step
TEST(NeuralNetworkTest, BatchTraining) {
    std::vector<int> layer_sizes = {784, 128, 64, 10};
    NeuralNetwork nn(layer_sizes, 0.01f, 32);
    
    int batch_size = 4;
    std::vector<float> inputs(batch_size * 784, 0.0f);
    std::vector<int> targets(batch_size, 0);
    
    float loss;
    EXPECT_NO_THROW({
        nn.train_batch(inputs.data(), targets.data(), batch_size, &loss);
    });
    
    // Loss should be a valid float (not NaN, not Inf)
    EXPECT_FALSE(std::isnan(loss));
    EXPECT_FALSE(std::isinf(loss));
    EXPECT_GE(loss, 0.0f);
}

// Test save and load weights
TEST(NeuralNetworkTest, SaveLoadWeights) {
    std::vector<int> layer_sizes = {784, 128, 64, 10};
    NeuralNetwork nn1(layer_sizes, 0.01f, 32);
    NeuralNetwork nn2(layer_sizes, 0.01f, 32);
    
    // Train nn1 a bit
    std::vector<float> inputs(32 * 784, 0.0f);
    std::vector<int> targets(32, 0);
    float loss;
    nn1.train_batch(inputs.data(), targets.data(), 32, &loss);
    
    // Save weights
    std::string test_file = "/tmp/test_weights.bin";
    EXPECT_NO_THROW({
        nn1.save_weights(test_file);
    });
    
    // Load weights into nn2
    EXPECT_NO_THROW({
        nn2.load_weights(test_file);
    });
    
    // Test that both networks produce same output
    std::vector<float> input(784, 0.0f);
    std::vector<float> output1(10);
    std::vector<float> output2(10);
    
    nn1.forward(input.data(), output1.data());
    nn2.forward(input.data(), output2.data());
    
    // Outputs should be very similar (allowing for floating point precision)
    for (size_t i = 0; i < output1.size(); i++) {
        EXPECT_NEAR(output1[i], output2[i], 1e-5f);
    }
}

// Test evaluation function
TEST(NeuralNetworkTest, Evaluate) {
    std::vector<int> layer_sizes = {784, 128, 64, 10};
    NeuralNetwork nn(layer_sizes, 0.01f, 32);
    
    int num_samples = 10;
    std::vector<float> inputs(num_samples * 784, 0.0f);
    std::vector<int> labels(num_samples, 0);
    
    float accuracy = nn.evaluate(inputs.data(), labels.data(), num_samples);
    
    // Accuracy should be between 0 and 1
    EXPECT_GE(accuracy, 0.0f);
    EXPECT_LE(accuracy, 1.0f);
}
