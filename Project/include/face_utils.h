#ifndef FACE_UTILS_H
#define FACE_UTILS_H

#include <vector>

#include <opencv2/opencv.hpp>

namespace detection {

std::vector<cv::Mat> extractFaces(cv::Mat& image,
                                  const std::string& face_cascade_file,
                                  const std::string& right_eye_cascade_file,
                                  const std::string& left_eye_cascade_file,
                                  bool is_debug);

}

#endif // FACE_UTILS_H
