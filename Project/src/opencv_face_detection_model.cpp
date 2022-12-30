#include "opencv_face_detection_model.h"

namespace detection {

OpenCVFaceDetectionModel::OpenCVFaceDetectionModel(const std::string& face_cascade_file_path,
                                                   const std::string& right_eye_cascade_file_path,
                                                   const std::string& left_eye_cascade_file_path):
    _face_cascade(),
    _right_eye_cascade(),
    _left_eye_cascade(left_eye_cascade_file_path) {
    _face_cascade.load(face_cascade_file_path);
    _right_eye_cascade.load(right_eye_cascade_file_path);
    _left_eye_cascade.load(left_eye_cascade_file_path);
}

OpenCVFaceDetectionModel::OpenCVFaceDetectionModel(const OpenCVFaceDetectionModel& that):
        _face_cascade(that._face_cascade),
        _right_eye_cascade(that._right_eye_cascade),
        _left_eye_cascade(that._left_eye_cascade) {
    // empty on purpose
}

OpenCVFaceDetectionModel& OpenCVFaceDetectionModel::operator=(const OpenCVFaceDetectionModel& that) {
    if (this != &that) {
        this->_face_cascade = that._face_cascade;
        this->_right_eye_cascade = that._right_eye_cascade;
        this->_left_eye_cascade = that._left_eye_cascade;
    }

    return *this;
}

std::vector<Face> OpenCVFaceDetectionModel::extractFaces(cv::Mat& image) {
    std::vector<Face> result_faces;

    cv::Mat greyscale_image;
    cv::cvtColor(image, greyscale_image, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    _face_cascade.detectMultiScale(greyscale_image, faces, 1.1, 6);

    for(size_t i = 0; i < faces.size(); i++) {
        cv::Rect face = faces[i];
        cv::Mat face_area = greyscale_image(face);

        std::vector<cv::Rect> left_eyes;
        std::vector<cv::Rect> right_eyes;

        _left_eye_cascade.detectMultiScale(face_area, left_eyes, 1.1, 6);
        _right_eye_cascade.detectMultiScale(face_area, right_eyes, 1.1, 6);

        cv::Rect left_eye;
        cv::Rect right_eye;
        cv::Mat output_image = face_area;

        if (left_eyes.size() == 1 && right_eyes.size() == 1) {
            float fx = static_cast<float>(face.x),
                    fy = static_cast<float>(face.y),
                    fw = static_cast<float>(face.width),
                    fh = static_cast<float>(face.height);

            left_eye = left_eyes[0];
            right_eye = right_eyes[0];

            float rx = static_cast<float>(right_eye.x),
                    ry = static_cast<float>(right_eye.y),
                    rw = static_cast<float>(right_eye.width),
                    rh = static_cast<float>(right_eye.height);

            float lx = static_cast<float>(left_eye.x),
                    ly = static_cast<float>(left_eye.y),
                    lw = static_cast<float>(left_eye.width),
                    lh = static_cast<float>(left_eye.height);

            float dx = (lx + fw / 2 + lw / 2) - (rx + rw / 2),
                    dy = (ly + lh / 2) - (ry + rh / 2);

            // tricky way to calculate pi
            float pi = atan(1) * 4;

            float angle_rad = atan2(dy, dx);
            float angle_degree = angle_rad * 180 / pi;

            float face_center_x = fx + fw / 2,
                    face_center_y = fy + fh / 2;

            cv::Mat rotation_mat = cv::getRotationMatrix2D(cv::Point2f(face_center_x, face_center_y), angle_degree,
                                                           1.0 /* scale */);

            cv::warpAffine(face_area, output_image, rotation_mat, cv::Size2i(face_area.cols, face_area.rows));
        }

        result_faces.push_back(Face(
                output_image,
                Rect::from(face),
                Eyes::from(left_eye, right_eye)));
    }

    return result_faces;
}

} // namespace detection
