cmake_minimum_required(VERSION 3.10)
project(MNIST_ML_CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Get CUDA include directories
if(NOT CUDA_INCLUDE_DIRS)
    if(CUDA_TOOLKIT_ROOT_DIR)
        set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
    endif()
endif()

if(CUDA_INCLUDE_DIRS)
    message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")
else()
    message(WARNING "CUDA_INCLUDE_DIRS not set, trying default locations")
    set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
endif()

# Find MatIO library
find_library(MATIO_LIB matio PATHS /usr/lib /usr/local/lib)
find_path(MATIO_INCLUDE_DIR matio.h PATHS /usr/include /usr/local/include)

if(NOT MATIO_LIB OR NOT MATIO_INCLUDE_DIR)
    message(FATAL_ERROR "MatIO library not found. Please install libmatio-dev (Ubuntu/Debian) or libmatio (other distros)")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${MATIO_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

# CUDA settings
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=sm_75")
set(CUDA_SEPARABLE_COMPILATION ON)

# Source files
set(CPP_SOURCES
    src/main.cpp
    src/neural_network.cpp
    src/mat_reader.cpp
)

set(CUDA_SOURCES
    src/cuda_kernels.cu
)

# Create executable
add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

# Add CUDA include directories to target
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${MATIO_LIB} ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})

# CUDA properties
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Print configuration
message(STATUS "CUDA found: ${CUDA_FOUND}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "MatIO library: ${MATIO_LIB}")
message(STATUS "MatIO include: ${MATIO_INCLUDE_DIR}")

# Test predictions executable
add_executable(test_predictions 
    test_predictions.cpp
    src/neural_network.cpp
    src/mat_reader.cpp
    ${CUDA_SOURCES}
)
target_include_directories(test_predictions PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(test_predictions ${MATIO_LIB} ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
set_property(TARGET test_predictions PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Google Test setup
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For CMake 3.14+, we can use FetchContent_MakeAvailable
# For older versions, use FetchContent_GetProperties
if(CMAKE_VERSION VERSION_LESS "3.14")
    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
    endif()
else()
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_MakeAvailable(googletest)
endif()

# Enable testing
enable_testing()

# Test executable
# Using custom main for flexibility (can add custom initialization if needed)
add_executable(run_tests
    tests/test_main.cpp
    tests/test_neural_network.cpp
    src/neural_network.cpp
    src/mat_reader.cpp
    ${CUDA_SOURCES}
)

target_include_directories(run_tests PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${MATIO_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(run_tests
    GTest::gtest
    GTest::gmock
    ${MATIO_LIB}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

set_property(TARGET run_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Register tests with CTest
add_test(NAME NeuralNetworkTests COMMAND run_tests)
