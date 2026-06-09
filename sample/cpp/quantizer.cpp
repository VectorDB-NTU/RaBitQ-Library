#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"

int main() {
    // generate random data
    size_t dim = 128;
    size_t bit = 4;

    float* data = new float[dim];
    for (size_t i = 0; i < dim; i++) {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // choose rotator
    rabitqlib::Rotator<float>* rotator = rabitqlib::choose_rotator<float>(dim);

    size_t padded_dim = rotator->size();
    float* rotated_data = new float[padded_dim];
    rotator->rotate(data, rotated_data);

    // print rotated_data
    for (size_t i = 0; i < padded_dim; i++) {
        std::cout << rotated_data[i] << " ";
    }
    std::cout << '\n';

    // quantize
    uint8_t* code = new uint8_t[padded_dim];
    float delta = 0;
    float vl = 0;
    rabitqlib::quant::quantize_scalar(rotated_data, padded_dim, bit, code, delta, vl);

    // [Note: we don't need to store vl as vl = - delta * (2^bit - 1) / 2]

    // reconstruct
    float* reconstructed_data = new float[padded_dim];
    rabitqlib::quant::reconstruct_vec(code, delta, vl, padded_dim, reconstructed_data);

    // print reconstructed_data
    for (size_t i = 0; i < padded_dim; i++) {
        std::cout << reconstructed_data[i] << " ";
    }
    std::cout << '\n';

    delete rotator;
    delete[] data;
    delete[] rotated_data;
    delete[] code;
    delete[] reconstructed_data;
}
