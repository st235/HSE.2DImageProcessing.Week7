#include "metrics_tracker.h"

#include <unordered_map>

namespace detection {

ConfusionMatrix::ConfusionMatrix():
    tp(0),
    tn(0),
    fp(0),
    fn(0) {
    // empty on purpose
}

ConfusionMatrix::ConfusionMatrix(uint32_t tp,
                                 uint32_t tn,
                                 uint32_t fp,
                                 uint32_t fn):
        tp(tp),
        tn(tn),
        fp(fp),
        fn(fn) {
    // empty on purpose
}

ConfusionMatrix::ConfusionMatrix(const ConfusionMatrix& that):
        tp(that.tp),
        tn(that.tn),
        fp(that.fp),
        fn(that.fn) {
    // empty on purpose
}

ConfusionMatrix& ConfusionMatrix::operator=(const ConfusionMatrix& that) {
    if (this != &that) {
        this->tp = that.tp;
        this->tn = that.tn;
        this->fp = that.fp;
        this->fn = that.fn;
    }

    return *this;
}

ConfusionMatrix ConfusionMatrix::merge(const ConfusionMatrix& that) {
    return ConfusionMatrix(tp + that.tp,
                           tn + that.tn,
                           fp + that.fp,
                           fn + that.fn);
}

ConfusionMatrix ConfusionMatrix::operator+(const ConfusionMatrix& that) {
    return merge(that);
}

ConfusionMatrix& ConfusionMatrix::operator+=(const ConfusionMatrix& that) {
    this->tp += that.tp;
    this->tn += that.tn;
    this->fp += that.fp;
    this->fn += that.fn;
    return *this;
}

void MetricsTracker::trackDetection(const FrameInfo& frame_info,
                                    const std::vector<Rect>& detected_face_origins) {
    std::vector<Rect> annotated_faces_origins = frame_info.face_origins();
    bool matched_annotated[annotated_faces_origins.size()];
    bool matched_detected[detected_face_origins.size()];

    uint32_t tp = 0,
             fp = 0,
             fn = 0;


    for (size_t i = 0; i < annotated_faces_origins.size(); i++) {
        matched_annotated[i] = false;
    }

    for (size_t i = 0; i < detected_face_origins.size(); i++) {
        matched_detected[i] = false;
    }

    for (size_t i = 0; i < annotated_faces_origins.size(); i++) {
        if (matched_annotated[i]) {
            continue;
        }

        for (size_t j = 0; j < detected_face_origins.size(); j++) {
            if (matched_detected[j]) {
                continue;
            }

            if (Rect::iou(annotated_faces_origins[i], detected_face_origins[j]) > 0.6) {
                matched_annotated[i] = true;
                matched_detected[j] = true;
                tp += 1;
                break;
            }
        }
    }

    _detections_per_frame_lookup.insert({ frame_info.id(), ConfusionMatrix(tp, 0,
                                                                           detected_face_origins.size() - tp /* fp */,
                                                                           annotated_faces_origins.size() - tp /* fn */) });
}

MetricsTracker::MetricsTracker():
    _detections_per_frame_lookup() {
    // empty on purpose
}

MetricsTracker::MetricsTracker(const MetricsTracker& that):
    _detections_per_frame_lookup(that._detections_per_frame_lookup) {
    // empty on purpose
}

MetricsTracker& MetricsTracker::operator=(const MetricsTracker& that) {
    if (this != &that) {
        this->_detections_per_frame_lookup = that._detections_per_frame_lookup;
    }

    return *this;
}

ConfusionMatrix MetricsTracker::overallDetectionMetrics() const {
    ConfusionMatrix matrix;

    for (const auto& entry: _detections_per_frame_lookup) {
        matrix += entry.second;
    }

    return matrix;
}

void MetricsTracker::keepTrackOf(const FrameInfo& frame_info,
                                 const std::vector<std::string>& labels,
                                 const std::vector<Rect>& face_origins) {
    trackDetection(frame_info, face_origins);
}

} // namespace detection
