#include "metrics_tracker.h"

#include <unordered_map>

namespace {

const static std::string UNKNOWN_LABEL = "unknown";

}

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

double ConfusionMatrix::tpr() const {
    return recall();
}

double ConfusionMatrix::fnr() const {
    return static_cast<double>(fn) / (static_cast<double>(tp) + static_cast<double>(fn));
}

double ConfusionMatrix::tnr() const {
    return static_cast<double>(tn) / (static_cast<double>(tn) + static_cast<double>(fp));
}

double ConfusionMatrix::fpr() const {
    return static_cast<double>(fp) / (static_cast<double>(fp) + static_cast<double>(tn));
}

double ConfusionMatrix::precision() const {
    return static_cast<double>(tp) / (static_cast<double>(tp) + static_cast<double>(fp));
}

double ConfusionMatrix::recall() const {
    return static_cast<double>(tp) / (static_cast<double>(tp) + static_cast<double>(fn));
}

double ConfusionMatrix::f1() const {
    double p = precision();
    double r = recall();
    return 2.0 * (p * r) / (p + r);
}

double ConfusionMatrix::accuracy() const {
    double n = static_cast<double>(tp) +
               static_cast<double>(tn);
    double d = static_cast<double>(tp) +
               static_cast<double>(tn) +
               static_cast<double>(fp) +
               static_cast<double>(fn);
    return n / d;
}

bool ConfusionMatrix::empty() const {
    return tp == 0 && tn == 0 && fp == 0 && fn == 0;
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

    uint32_t matched = 0;

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
                matched += 1;
                break;
            }
        }
    }

    _detections_per_frame_lookup.insert({ frame_info.id(),
                  ConfusionMatrix(
                          matched,
                          0 /* tn, not applicable for the project */,
                          detected_face_origins.size() - matched /* fp */,
                          annotated_faces_origins.size() - matched /* fn */) });
}

void MetricsTracker::trackRecognition(const FrameInfo& frame_info,
                                      const std::vector<std::string>& labels,
                                      const std::vector<Rect>& detected_face_origins) {
    std::vector<std::string> annotated_faces_labels = frame_info.labels();
    std::vector<Rect> annotated_faces_origins = frame_info.face_origins();
    int32_t matches[annotated_faces_origins.size()];
    bool matched_annotated[annotated_faces_origins.size()];
    bool matched_detected[detected_face_origins.size()];

    for (size_t i = 0; i < annotated_faces_origins.size(); i++) {
        matches[i] = -1;
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
                matches[i] = j;
                matched_annotated[i] = true;
                matched_detected[j] = true;
                break;
            }
        }
    }

    ConfusionMatrix overall_known_metric;

    for (const auto& class_label: _labels) {
        bool is_unknown = (class_label == UNKNOWN_LABEL);

        // true positive
        bool detected_correctly = 0,
             // true negative
             not_a_class_everywhere = 0,
             // false positive
             detected_by_us_but_annotated_differently = 0,
             // false negative
             annotated_but_not_detected = 0;

        for (size_t i = 0; i < annotated_faces_origins.size(); i++) {
            // matched by detection algorithm
            bool detection_matched = matched_annotated[i];

            if (!detection_matched) {
                // do not consider anything else as
                // it will reflect the quality of detection
                // rather recognition.
                continue;
            }

            if (annotated_faces_labels[i] == class_label &&
                    labels[matches[i]] == class_label) {
                // detected correctly,
                // classes match and within
                // the current group.
                detected_correctly += 1;
            } else if (annotated_faces_labels[i] == class_label &&
                        labels[matches[i]] != class_label) {
                // we detected something different here
                // but was expecting a specific class
                annotated_but_not_detected += 1;
            } else if (annotated_faces_labels[i] != class_label &&
                       labels[matches[i]] == class_label) {
                // we detected a class but annotation
                // says there is no such a class
                detected_by_us_but_annotated_differently += 1;
            } else {
                // not a class and we agree on it with
                // annotations
                not_a_class_everywhere += 1;
            }
        }

        ConfusionMatrix metrics(detected_correctly /* tp */,
                                not_a_class_everywhere /* tn */,
                                detected_by_us_but_annotated_differently /* fp */,
                                annotated_but_not_detected /* fn */);

        if (is_unknown) {
            _unknown_recognitions_per_frame_lookup.insert({ frame_info.id(), metrics });
        } else {
            overall_known_metric += metrics;
        }
    }

    _known_recognitions_per_frame_lookup.insert({ frame_info.id(), overall_known_metric });
}

MetricsTracker::MetricsTracker(std::vector<std::string> labels):
    _labels(labels),
    _detections_per_frame_lookup(),
    _known_recognitions_per_frame_lookup(),
    _unknown_recognitions_per_frame_lookup() {
    // empty on purpose
}

MetricsTracker::MetricsTracker(const MetricsTracker& that):
    _labels(that._labels),
    _detections_per_frame_lookup(that._detections_per_frame_lookup),
    _known_recognitions_per_frame_lookup(that._known_recognitions_per_frame_lookup),
    _unknown_recognitions_per_frame_lookup(that._unknown_recognitions_per_frame_lookup) {
    // empty on purpose
}

MetricsTracker& MetricsTracker::operator=(const MetricsTracker& that) {
    if (this != &that) {
        this->_labels = that._labels;
        this->_detections_per_frame_lookup = that._detections_per_frame_lookup;
        this->_known_recognitions_per_frame_lookup = that._known_recognitions_per_frame_lookup;
        this->_unknown_recognitions_per_frame_lookup = that._unknown_recognitions_per_frame_lookup;
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

ConfusionMatrix MetricsTracker::overallKnownRecognitionMetrics() const {
    ConfusionMatrix matrix;

    for (const auto& entry: _known_recognitions_per_frame_lookup) {
        matrix += entry.second;
    }

    return matrix;
}

ConfusionMatrix MetricsTracker::overallUnknownRecognitionMetrics() const {
    ConfusionMatrix matrix;

    for (const auto& entry: _unknown_recognitions_per_frame_lookup) {
        matrix += entry.second;
    }

    return matrix;
}

ConfusionMatrix MetricsTracker::overallRecognitionMetrics() const {
    ConfusionMatrix matrix;
    matrix += overallKnownRecognitionMetrics();
    matrix += overallUnknownRecognitionMetrics();
    return matrix;
}

void MetricsTracker::keepTrackOf(const FrameInfo& frame_info,
                                 const std::vector<std::string>& labels,
                                 const std::vector<Rect>& face_origins) {
    trackDetection(frame_info, face_origins);
    trackRecognition(frame_info, labels, face_origins);
}

} // namespace detection
