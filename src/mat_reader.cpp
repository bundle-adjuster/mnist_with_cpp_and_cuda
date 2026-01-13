#include "mat_reader.h"
#include <matio.h>
#include <iostream>
#include <stdexcept>

MNISTData MatReader::load_mnist(const std::string& mat_file) {
    mat_t* matfp = Mat_Open(mat_file.c_str(), MAT_ACC_RDONLY);
    if (matfp == nullptr) {
        throw std::runtime_error("Failed to open .mat file: " + mat_file);
    }
    
    MNISTData data;
    
    // Read images
    matvar_t* images_var = Mat_VarRead(matfp, "images");
    if (images_var == nullptr) {
        Mat_Close(matfp);
        throw std::runtime_error("Failed to read 'images' variable from .mat file");
    }
    
    if (images_var->data_type != MAT_T_SINGLE && images_var->data_type != MAT_T_DOUBLE) {
        Mat_VarFree(images_var);
        Mat_Close(matfp);
        throw std::runtime_error("Images must be float or double type");
    }
    
    // Get dimensions
    size_t num_dims = images_var->rank;
    if (num_dims != 3) {
        Mat_VarFree(images_var);
        Mat_Close(matfp);
        throw std::runtime_error("Images must be 3D array (num_samples, height, width)");
    }
    
    data.num_samples = images_var->dims[0];
    data.image_height = images_var->dims[1];
    data.image_width = images_var->dims[2];
    size_t total_pixels = data.num_samples * data.image_height * data.image_width;
    
    // Convert to float
    data.images.resize(total_pixels);
    if (images_var->data_type == MAT_T_SINGLE) {
        float* src = static_cast<float*>(images_var->data);
        std::copy(src, src + total_pixels, data.images.begin());
    } else { // MAT_T_DOUBLE
        double* src = static_cast<double*>(images_var->data);
        for (size_t i = 0; i < total_pixels; i++) {
            data.images[i] = static_cast<float>(src[i]);
        }
    }
    Mat_VarFree(images_var);
    
    // Read labels
    matvar_t* labels_var = Mat_VarRead(matfp, "labels");
    if (labels_var == nullptr) {
        Mat_Close(matfp);
        throw std::runtime_error("Failed to read 'labels' variable from .mat file");
    }
    
    if (labels_var->data_type != MAT_T_INT32 && labels_var->data_type != MAT_T_UINT8) {
        Mat_VarFree(labels_var);
        Mat_Close(matfp);
        throw std::runtime_error("Labels must be int32 or uint8 type");
    }
    
    data.labels.resize(data.num_samples);
    if (labels_var->data_type == MAT_T_INT32) {
        int32_t* src = static_cast<int32_t*>(labels_var->data);
        std::copy(src, src + data.num_samples, data.labels.begin());
    } else { // MAT_T_UINT8
        uint8_t* src = static_cast<uint8_t*>(labels_var->data);
        for (size_t i = 0; i < data.num_samples; i++) {
            data.labels[i] = static_cast<int>(src[i]);
        }
    }
    Mat_VarFree(labels_var);
    
    Mat_Close(matfp);
    
    return data;
}

void MatReader::print_info(const MNISTData& data) {
    std::cout << "MNIST Data Info:" << std::endl;
    std::cout << "  Number of samples: " << data.num_samples << std::endl;
    std::cout << "  Image size: " << data.image_height << "x" << data.image_width << std::endl;
    std::cout << "  Total pixels per image: " << data.image_height * data.image_width << std::endl;
}
