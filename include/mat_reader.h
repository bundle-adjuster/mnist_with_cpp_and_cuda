#ifndef MAT_READER_H
#define MAT_READER_H

#include <string>
#include <vector>

struct ImageData {
    std::vector<float> images;  // Flattened images: [num_samples * height * width]
    std::vector<int> labels;     // Labels: [num_samples]
    int num_samples;
    int image_height;
    int image_width;
    
    // Get pointer to specific image
    const float* get_image(int index) const {
        return images.data() + index * image_height * image_width;
    }
};

class MatReader {
public:
    static ImageData load_dataset(const std::string& mat_file);
    static void print_info(const ImageData& data);
    
    // Legacy function name for backward compatibility
    static ImageData load_mnist(const std::string& mat_file) {
        return load_dataset(mat_file);
    }
};

#endif // MAT_READER_H
