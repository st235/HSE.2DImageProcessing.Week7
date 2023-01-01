#include "face_detection_model.h"

#include <iostream>
#include "face_utils.h"

namespace detection {

bool FaceDetectionModel::shouldClip(const Rect& viewport, const Rect& face_origin) const {
    if (!face_origin.intersects(viewport)) {
        return true;
    }

    const auto& intersection = face_origin.intersection(viewport);
    double visible_area = static_cast<double>(intersection.area()) / static_cast<double>(face_origin.area());
    // at least 70% of the face should be visible
    // otherwise we need to clip
    return visible_area < 0.3;
}

} // namespace detection
