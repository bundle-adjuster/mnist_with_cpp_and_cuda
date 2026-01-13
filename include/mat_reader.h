#ifndef MAT_READER_H
#define MAT_READER_H

#include <string>
#include <vector>

struct MNISTData {
    std::vector<float> images;  // Flattened images: [num_samples * 28 * 28]
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
    static MNISTData load_mnist(const std::string& mat_file);
    static void print_info(const MNISTData& data);
};

#endif // MAT_READER_H
