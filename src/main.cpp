#include "neural_network.h"
#include "mat_reader.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <train_mat_file> <test_mat_file> [epochs] [learning_rate] [batch_size]" << std::endl;
        std::cerr << "Example: " << argv[0] << " data/mnist_train.mat data/mnist_test.mat 10 0.01 32" << std::endl;
        return 1;
    }
    
    std::string train_file = argv[1];
    std::string test_file = argv[2];
    int epochs = (argc > 3) ? std::stoi(argv[3]) : 10;
    float learning_rate = (argc > 4) ? std::stof(argv[4]) : 0.01f;
    int batch_size = (argc > 5) ? std::stoi(argv[5]) : 32;
    
    std::cout << "Loading dataset..." << std::endl;
    ImageData train_data = MatReader::load_dataset(train_file);
    ImageData test_data = MatReader::load_dataset(test_file);
    
    MatReader::print_info(train_data);
    MatReader::print_info(test_data);
    
    // Create neural network: 784 -> 128 -> 64 -> 10
    std::vector<int> layer_sizes = {784, 128, 64, 10};
    std::cout << "\nCreating neural network with layers: ";
    for (size_t i = 0; i < layer_sizes.size(); i++) {
        std::cout << layer_sizes[i];
        if (i < layer_sizes.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;
    
    NeuralNetwork nn(layer_sizes, learning_rate, batch_size);
    
    std::cout << "\nStarting training..." << std::endl;
    std::cout << "Epochs: " << epochs << ", Learning rate: " << learning_rate 
              << ", Batch size: " << batch_size << std::endl;
    
    // Training loop
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> indices(train_data.num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    int image_size = train_data.image_height * train_data.image_width;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(), gen);
        
        float epoch_loss = 0.0f;
        int num_batches = (train_data.num_samples + batch_size - 1) / batch_size;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, train_data.num_samples);
            int current_batch_size = end_idx - start_idx;
            
            // Prepare batch data
            std::vector<float> batch_images(current_batch_size * 784);
            std::vector<int> batch_labels(current_batch_size);
            
            for (int i = 0; i < current_batch_size; i++) {
                int idx = indices[start_idx + i];
                const float* image = train_data.get_image(idx);
                std::copy(image, image + 784, batch_images.data() + i * 784);
                batch_labels[i] = train_data.labels[idx];
            }
            
            // Train on batch
            float batch_loss;
            nn.train_batch(batch_images.data(), batch_labels.data(), current_batch_size, &batch_loss);
            epoch_loss += batch_loss * current_batch_size;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        epoch_loss /= train_data.num_samples;
        
        // Evaluate on test set
        float test_accuracy = nn.evaluate(
            test_data.images.data(),
            test_data.labels.data(),
            test_data.num_samples
        );
        
        std::cout << "Epoch " << std::setw(3) << epoch + 1 << "/" << epochs
                  << " | Loss: " << std::fixed << std::setprecision(4) << epoch_loss
                  << " | Test Accuracy: " << std::setprecision(2) << (test_accuracy * 100) << "%"
                  << " | Time: " << duration << "ms" << std::endl;
    }
    
    // Final evaluation
    std::cout << "\nFinal evaluation:" << std::endl;
    float train_accuracy = nn.evaluate(
        train_data.images.data(),
        train_data.labels.data(),
        train_data.num_samples
    );
    float test_accuracy = nn.evaluate(
        test_data.images.data(),
        test_data.labels.data(),
        test_data.num_samples
    );
    
    std::cout << "Train Accuracy: " << std::fixed << std::setprecision(2) 
              << (train_accuracy * 100) << "%" << std::endl;
    std::cout << "Test Accuracy: " << std::setprecision(2) 
              << (test_accuracy * 100) << "%" << std::endl;
    
    // Save model (extract base name from train_file for model name)
    std::string model_file = "model.bin";
    // Try to infer dataset name from file path
    size_t last_slash = train_file.find_last_of("/\\");
    std::string filename = (last_slash != std::string::npos) ? train_file.substr(last_slash + 1) : train_file;
    if (filename.find("mnist") != std::string::npos) {
        model_file = "mnist_model.bin";
    } else if (filename.find("fashion") != std::string::npos) {
        model_file = "fashion_mnist_model.bin";
    }
    std::cout << "\nSaving model to " << model_file << std::endl;
    nn.save_weights(model_file);
    
    return 0;
}
