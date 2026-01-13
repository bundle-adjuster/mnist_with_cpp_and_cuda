#include "mat_reader.h"
#include "neural_network.h"
#include <iomanip>
#include <iostream>

int main() {
  std::cout << "Loading test data..." << std::endl;
  MNISTData test_data = MatReader::load_mnist("../data/mnist_test.mat");

  // Create network and try to load trained model
  std::vector<int> layer_sizes = {784, 128, 64, 10};
  NeuralNetwork nn(layer_sizes, 0.1f, 32);

  try {
    nn.load_weights("mnist_model.bin");
    std::cout << "Loaded trained model" << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Could not load model: " << e.what() << std::endl;
    return 1;
  }

  // Test first 20 samples
  std::cout << "\nPredictions for first 100 test samples:" << std::endl;
  std::cout << std::setw(5) << "Index" << std::setw(10) << "True"
            << std::setw(10) << "Pred" << std::endl;
  std::cout << std::string(25, '-') << std::endl;

  int correct = 0;
  for (int i = 0; i < 100; i++) {
    const float *image = test_data.get_image(i);
    int predicted = nn.predict(image);
    int true_label = test_data.labels[i];

    std::cout << std::setw(5) << i << std::setw(10) << true_label
              << std::setw(10) << predicted;

    if (predicted == true_label) {
      std::cout << "  ✓";
      correct++;
    }
    std::cout << std::endl;
  }

  std::cout << "\nAccuracy on first 100: " << (correct * 100.0 / 100) << "%"
            << std::endl;

  // Print some sample pixel values
  std::cout << "\nFirst 10 pixels of sample 0:" << std::endl;
  const float *img0 = test_data.get_image(0);
  for (int i = 0; i < 10; i++) {
    std::cout << img0[i] << " ";
  }
  std::cout << std::endl;

  // Print sum of all pixels for sample 0
  float sum = 0;
  for (int i = 0; i < 784; i++) {
    sum += img0[i];
  }
  std::cout << "Sum of all pixels in sample 0: " << sum << std::endl;

  return 0;
}
