#include "dlib_face_detection_model.h"

#include "dlib_utils.h"
#include "rect.h"

namespace detection {

DLibFaceDetectionModel::DLibFaceDetectionModel():
    _detector(dlib::get_frontal_face_detector()) {
    // empty on purpose
}

DLibFaceDetectionModel::DLibFaceDetectionModel(const DLibFaceDetectionModel& that):
    _detector(that._detector) {
    // empty on purpose
}

DLibFaceDetectionModel& DLibFaceDetectionModel::operator=(const DLibFaceDetectionModel& that) {
    if (this != &that) {
        this->_detector = that._detector;
    }

    return *this;
}

std::vector<Face> DLibFaceDetectionModel::extractFaces(const Rect& viewport, cv::Mat& raw_image) {
    dlib::array2d<dlib::rgb_pixel> image = AsRGBOpenCVMatrix(raw_image);

    std::vector<Face> result_faces;
    std::vector<dlib::rectangle> faces = _detector(image);
    for(size_t i = 0; i < faces.size(); i++) {
        const auto& face = faces[i];
        Rect face_origin(face.left(), face.top(), (face.right() - face.left()), (face.bottom() - face.top()));

        if (shouldClip(viewport, face_origin)) {
            continue;
        }

        // dlib can detect faces that are outside the viewport
        // however opencv cannot extract such areas, therefore
        // we need to find intersection with a viewport
        Rect face_origin_within_viewport = face_origin.intersection(viewport);

        cv::Mat face_area = raw_image(Rect::toCVRect(face_origin_within_viewport));

        result_faces.push_back(Face(
                face_area,
                face_origin,
                Eyes()));
    }

    return result_faces;
}
    
} // namespace detection
