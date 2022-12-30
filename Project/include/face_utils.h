#ifndef FACE_UTILS_H
#define FACE_UTILS_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "face_detection_model.h"
#include "rect.h"

namespace detection {

void drawFaces(cv::Mat& image,
               const std::vector<Rect>& faces_origins,
               const std::vector<std::string>& labels,
               const cv::Scalar& color = cv::Scalar(0, 0, 255));

void drawFaces(cv::Mat& image,
               const std::vector<Face>& faces);

void drawFaces(cv::Mat& image,
               const std::vector<Face>& faces,
               const std::vector<std::string>& labels);

}

#endif // FACE_UTILS_H
