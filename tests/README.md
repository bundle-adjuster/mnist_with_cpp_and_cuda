# Tests

This directory contains unit tests for the MNIST ML CUDA project using Google Test framework.

## Running Tests

### Build and run all tests:
```bash
cd build
cmake ..
make
ctest
```

### Run tests directly:
```bash
./run_tests
```

### Run with verbose output:
```bash
./run_tests --gtest_color=yes
```

### Run specific test:
```bash
./run_tests --gtest_filter=NeuralNetworkTest.CreateNetwork
```

## Test Structure

- `test_main.cpp` - Main entry point for tests
- `test_neural_network.cpp` - Tests for the NeuralNetwork class

## Adding New Tests

1. Create a new test file in the `tests/` directory
2. Include `gtest/gtest.h`
3. Use `TEST(TestSuiteName, TestName)` macro to define tests
4. Add the test file to `CMakeLists.txt` in the `run_tests` executable

Example:
```cpp
#include <gtest/gtest.h>
#include "neural_network.h"

TEST(MyTestSuite, MyTest) {
    // Your test code here
    EXPECT_EQ(1, 1);
}
```
