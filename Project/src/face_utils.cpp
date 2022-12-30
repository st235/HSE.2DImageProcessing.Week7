#include "face_utils.h"

namespace detection {

void drawFaces(cv::Mat& image,
               const std::vector<Rect>& faces_origins,
               const std::vector<std::string>& labels,
               const cv::Scalar& color) {
    if (labels.size() != faces_origins.size()) {
        throw std::runtime_error("Labels size is different from faces rects size");
    }

    for (size_t i = 0; i < faces_origins.size(); i++) {
        const auto& origin = faces_origins[i];
        const auto& label = labels[i];

        cv::rectangle(image,
                      cv::Point2f(origin.x, origin.y), cv::Point2f(origin.x + origin.width, origin.y + origin.height),
                      color, 6, 1, 0);

        cv::putText(image, label,
                    cv::Point(origin.x, origin.y - 15), cv::FONT_HERSHEY_COMPLEX,
                    1, color, 2, cv::LINE_8);
    }
}

void drawFaces(cv::Mat& image,
               const std::vector<Face>& faces,
               const std::vector<std::string>& labels) {
    for (size_t i = 0; i < faces.size(); i++) {
        Face face = faces[i];
        const auto& origin = face.origin;

        cv::rectangle(image,
                      cv::Point2f(origin.x, origin.y), cv::Point2f(origin.x + origin.width, origin.y + origin.height),
                      cv::Scalar(0, 0, 255), 6, 1, 0);

        if (face.eyesDetected()) {
            const auto& eyes = face.eyesEscapedFromFaceBasis();
            const auto& left_eye = eyes.left;
            const auto& right_eye = eyes.right;

            cv::rectangle(image,
                          cv::Point2f(left_eye.x, left_eye.y), cv::Point2f(left_eye.x + left_eye.width, left_eye.y + left_eye.height),
                          cv::Scalar(0, 255, 0), 6, 1, 0);

            cv::rectangle(image,
                          cv::Point2f(right_eye.x, right_eye.y), cv::Point2f(right_eye.x + right_eye.width, right_eye.y + right_eye.height),
                          cv::Scalar(0, 255, 0), 6, 1, 0);
        }

        if (labels.size() == faces.size()) {
            const auto& label = labels[i];

            if (label.empty()) {
                continue;
            }

            cv::putText(image, label,
                        cv::Point(origin.x, origin.y - 15), cv::FONT_HERSHEY_COMPLEX,
                        1, cv::Scalar(0, 0, 255), 2, cv::LINE_8);
        }
    }
}

void drawFaces(cv::Mat& image,
               const std::vector<Face>& faces) {
    std::vector<std::string> labels;
    drawFaces(image, faces, labels);
}


} // namespace detection
