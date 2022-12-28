#ifndef DLIB_UTILS_H
#define DLIB_UTILS_H

#include <cstdint>

#include <dlib/image_io.h>
#include <opencv2/opencv.hpp>

#include <iostream>

namespace detection {

/**
 * Perhaps, a temporary method until I do
 * a permanent solution.
 */
static dlib::array2d<dlib::rgb_pixel> AsRGBOpenCVMatrix(const cv::Mat& mat) {
    cv::Mat clone = mat.clone();
    dlib::array2d<dlib::rgb_pixel> dlib_mat(clone.rows, clone.cols);
    cv::cvtColor(clone, clone, cv::COLOR_GRAY2BGR);

//    std::cout << clone.rows << "/" << clone.cols << std::endl;

    for (size_t i = 0; i < clone.rows; i++) {
        for (size_t j = 0; j < clone.cols; j++) {
            cv::Vec3b color_vector = clone.at<cv::Vec3b>(i, j);

            unsigned char b = static_cast<unsigned char>(color_vector[0]);
            unsigned char g = static_cast<unsigned char>(color_vector[1]);
            unsigned char r = static_cast<unsigned char>(color_vector[2]);

            dlib_mat[i][j] = dlib::rgb_pixel(r, g, b);
        }
    }

    return dlib_mat;
}

} // namespace detection

#endif //DLIB_UTILS_H
