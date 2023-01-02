#include "metrics_tracker.h"

#include <unordered_map>

#include "strings.h"

namespace {

const static std::string UNKNOWN_LABEL = "unknown";

}

namespace detection {

MultiClassificationMatrix::MultiClassificationMatrix(uint32_t number_of_classes):
    _number_of_classes(number_of_classes) {
    _classification_matrix = new uint32_t[_number_of_classes * _number_of_classes];

    for (uint32_t i = 0; i < _number_of_classes; i++) {
        for (uint32_t j = 0; j < _number_of_classes; j++) {
            _classification_matrix[i * _number_of_classes + j] = 0;
        }
    }
}

MultiClassificationMatrix::MultiClassificationMatrix(const MultiClassificationMatrix& that):
    _number_of_classes(that._number_of_classes) {
    _classification_matrix = new uint32_t[_number_of_classes * _number_of_classes];

    for (uint32_t i = 0; i < _number_of_classes; i++) {
        for (uint32_t j = 0; j < _number_of_classes; j++) {
            _classification_matrix[i * _number_of_classes + j] =
                    that._classification_matrix[i * _number_of_classes + j];
        }
    }
}

MultiClassificationMatrix& MultiClassificationMatrix::operator=(const MultiClassificationMatrix& that) {
    if (this != &that) {
        this->_number_of_classes = that._number_of_classes;

        delete[] _classification_matrix;
        this->_classification_matrix = new uint32_t[_number_of_classes * _number_of_classes];
        for (uint32_t i = 0; i < _number_of_classes; i++) {
            for (uint32_t j = 0; j < _number_of_classes; j++) {
                this->_classification_matrix[i * _number_of_classes + j] =
                        that._classification_matrix[i * _number_of_classes + j];
            }
        }
    }

    return *this;
}

uint32_t MultiClassificationMatrix::classesSize() const {
    return _number_of_classes;
}

uint32_t MultiClassificationMatrix::metricAt(uint32_t i, uint32_t j) const {
    return _classification_matrix[i * _number_of_classes + j];
}

double MultiClassificationMatrix::accuracy() const {
    double n = 0;
    double d = 0;

    for (uint32_t i = 0; i < _number_of_classes; i++) {
        for (uint32_t j = 0; j < _number_of_classes; j++) {
            double value = static_cast<double>(_classification_matrix[i * _number_of_classes + j]);
            if (i == j) {
                n += value;
            }
            d += value;
        }
    }

    return n / d;
}

void MultiClassificationMatrix::track(uint32_t annotated_label_id,
                                      uint32_t detected_label_id) {
    if (annotated_label_id < 0 || annotated_label_id >= _number_of_classes) {
        throw std::runtime_error("Annotated label is invalid: " + std::AsString(annotated_label_id));
    }

    if (detected_label_id < 0 || detected_label_id >= _number_of_classes) {
        throw std::runtime_error("Detected label is invalid: " + std::AsString(detected_label_id));
    }

    _classification_matrix[annotated_label_id * _number_of_classes + detected_label_id] += 1;
}

BinaryClassificationMatrix MultiClassificationMatrix::getBinaryMatrix(uint32_t label_id) const {
    if (label_id < 0 || label_id >= _number_of_classes) {
        throw std::runtime_error("Label is invalid: " + std::AsString(label_id));
    }

    uint32_t tp = 0,
             tn = 0,
             fp = 0,
             fn = 0;

    for (uint32_t i = 0; i < _number_of_classes; i++) {
        for (uint32_t j = 0; j < _number_of_classes; j++) {
            double value = static_cast<double>(_classification_matrix[i * _number_of_classes + j]);

            // true positive
            if (i == label_id && j == label_id) {
                tp += value;
            } if (i == label_id && j != label_id) {
                fn += value;
            } if (i != label_id && j == label_id) {
                fp += value;
            } else {
                tn += value;
            }
        }
    }

    return BinaryClassificationMatrix(tp, tn, fp, fn);
}

MultiClassificationMatrix MultiClassificationMatrix::remove(uint32_t label_id) const {
    if (label_id < 0 || label_id >= _number_of_classes) {
        throw std::runtime_error("Label is invalid: " + std::AsString(label_id));
    }

    MultiClassificationMatrix result(_number_of_classes - 1);

    for (uint32_t i = 0; i < _number_of_classes; i++) {
        int ri = i;

        if (i == label_id) {
            continue;
        }

        if (i > label_id) {
            ri -= 1;
        }

        for (uint32_t j = 0; j < _number_of_classes; j++) {
            int rj = j;

            if (j == label_id) {
                continue;
            }

            if (j > label_id) {
                rj -= 1;
            }

            result._classification_matrix[ri * (_number_of_classes - 1) + rj] =
                _classification_matrix[i * _number_of_classes + j];
        }
    }

    return result;
}

MultiClassificationMatrix MultiClassificationMatrix::operator+(const MultiClassificationMatrix& that) {
    if (_number_of_classes != that._number_of_classes) {
        throw std::runtime_error("Number of classes are different: " +
            std::AsString(_number_of_classes) + " vs. " + std::AsString(that._number_of_classes));
    }

    MultiClassificationMatrix result(_number_of_classes);

    for (uint32_t i = 0; i < _number_of_classes * _number_of_classes; i++) {
        result._classification_matrix[i] = _classification_matrix[i] + that._classification_matrix[i];
    }

    return result;
}

MultiClassificationMatrix& MultiClassificationMatrix::operator+=(const MultiClassificationMatrix& that) {
    if (_number_of_classes != that._number_of_classes) {
        throw std::runtime_error("Number of classes are different: " +
                                 std::AsString(_number_of_classes) + " vs. " + std::AsString(that._number_of_classes));
    }

    for (uint32_t i = 0; i < _number_of_classes * _number_of_classes; i++) {
        _classification_matrix[i] += that._classification_matrix[i];
    }

    return *this;
}

MultiClassificationMatrix::~MultiClassificationMatrix() {
    delete[] _classification_matrix;
}

BinaryClassificationMatrix::BinaryClassificationMatrix():
    tp(0),
    tn(0),
    fp(0),
    fn(0) {
    // empty on purpose
}

BinaryClassificationMatrix::BinaryClassificationMatrix(int32_t tp,
                                                       int32_t tn,
                                                       int32_t fp,
                                                       int32_t fn):
        tp(tp),
        tn(tn),
        fp(fp),
        fn(fn) {
    // empty on purpose
}

BinaryClassificationMatrix::BinaryClassificationMatrix(const BinaryClassificationMatrix& that):
        tp(that.tp),
        tn(that.tn),
        fp(that.fp),
        fn(that.fn) {
    // empty on purpose
}

BinaryClassificationMatrix& BinaryClassificationMatrix::operator=(const BinaryClassificationMatrix& that) {
    if (this != &that) {
        this->tp = that.tp;
        this->tn = that.tn;
        this->fp = that.fp;
        this->fn = that.fn;
    }

    return *this;
}

BinaryClassificationMatrix BinaryClassificationMatrix::merge(const BinaryClassificationMatrix& that) {
    return BinaryClassificationMatrix(tp + that.tp,
                           tn + that.tn,
                           fp + that.fp,
                           fn + that.fn);
}

double BinaryClassificationMatrix::tpr() const {
    return recall();
}

double BinaryClassificationMatrix::fnr() const {
    return static_cast<double>(fn) / (static_cast<double>(tp) + static_cast<double>(fn));
}

double BinaryClassificationMatrix::tnr() const {
    return static_cast<double>(tn) / (static_cast<double>(tn) + static_cast<double>(fp));
}

double BinaryClassificationMatrix::fpr() const {
    return static_cast<double>(fp) / (static_cast<double>(fp) + static_cast<double>(tn));
}

double BinaryClassificationMatrix::precision() const {
    return static_cast<double>(tp) / (static_cast<double>(tp) + static_cast<double>(fp));
}

double BinaryClassificationMatrix::recall() const {
    return static_cast<double>(tp) / (static_cast<double>(tp) + static_cast<double>(fn));
}

double BinaryClassificationMatrix::f1() const {
    double p = precision();
    double r = recall();
    return 2.0 * (p * r) / (p + r);
}

double BinaryClassificationMatrix::accuracy() const {
    double n = static_cast<double>(tp) +
               static_cast<double>(tn);
    double d = static_cast<double>(tp) +
               static_cast<double>(tn) +
               static_cast<double>(fp) +
               static_cast<double>(fn);
    return n / d;
}

bool BinaryClassificationMatrix::empty() const {
    return tp == 0 && tn == 0 && fp == 0 && fn == 0;
}

BinaryClassificationMatrix BinaryClassificationMatrix::operator+(const BinaryClassificationMatrix& that) {
    return merge(that);
}

BinaryClassificationMatrix& BinaryClassificationMatrix::operator+=(const BinaryClassificationMatrix& that) {
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

        double max_iou = 0;
        int32_t max_index = -1;

        for (size_t j = 0; j < detected_face_origins.size(); j++) {
            if (matched_detected[j]) {
                continue;
            }

            double iou = Rect::iou(annotated_faces_origins[i], detected_face_origins[j]);

            if (iou > 1.0) {
                // assert iou always within the [0.0, 1.0]
                throw std::runtime_error("IOU exceed 1.0 that is impossible, iou was:" + std::AsString(iou));
            }

            if (iou >= 0.5 &&
                    iou > max_iou) {
                max_iou = iou;
                max_index = j;
                break;
            }
        }

        if (max_index != -1) {
            matched_annotated[i] = true;
            matched_detected[max_index] = true;
            matched += 1;
        }
    }

    _detections_per_frame_lookup.insert({ frame_info.id(),
                  BinaryClassificationMatrix(
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

        double max_iou = 0;
        int32_t max_index = -1;

        for (size_t j = 0; j < detected_face_origins.size(); j++) {
            if (matched_detected[j]) {
                continue;
            }

            double iou = Rect::iou(annotated_faces_origins[i], detected_face_origins[j]);

            if (iou > 1.0) {
                // assert iou always within the [0.0, 1.0]
                throw std::runtime_error("IOU exceed 1.0 that is impossible, iou was:" + std::AsString(iou));
            }

            if (iou >= 0.5 &&
                iou > max_iou) {
                max_iou = iou;
                max_index = j;
                break;
            }
        }

        if (max_index != -1) {
            matches[i] = max_index;
            matched_annotated[i] = true;
            matched_detected[max_index] = true;
        }
    }

    MultiClassificationMatrix recognition_matrix(_labels.size());

    for (size_t i = 0; i < annotated_faces_origins.size(); i++) {
        bool is_matched = matched_annotated[i];

        if (!is_matched) {
            continue;
        }

        const auto& annotated_id = _labels_to_ids_lookup[annotated_faces_labels[i]];
        const auto& detected_id = _labels_to_ids_lookup[labels[matches[i]]];

        recognition_matrix.track(annotated_id, detected_id);
    }

    _recognitions_per_frame_lookup.insert({ frame_info.id(), recognition_matrix });
}

MetricsTracker::MetricsTracker(std::vector<std::string> labels):
    _labels(labels),
    _labels_to_ids_lookup(),
    _detections_per_frame_lookup(),
    _recognitions_per_frame_lookup(labels.size()) {
    for (uint32_t i = 0; i < labels.size(); i++) {
        _labels_to_ids_lookup.insert({ labels[i], i });
    }
}

MetricsTracker::MetricsTracker(const MetricsTracker& that):
    _labels(that._labels),
    _labels_to_ids_lookup(that._labels_to_ids_lookup),
    _detections_per_frame_lookup(that._detections_per_frame_lookup),
    _recognitions_per_frame_lookup(that._recognitions_per_frame_lookup) {
    // empty on purpose
}

MetricsTracker& MetricsTracker::operator=(const MetricsTracker& that) {
    if (this != &that) {
        this->_labels = that._labels;
        this->_labels_to_ids_lookup = that._labels_to_ids_lookup;
        this->_detections_per_frame_lookup = that._detections_per_frame_lookup;
        this->_recognitions_per_frame_lookup = that._recognitions_per_frame_lookup;
    }

    return *this;
}

BinaryClassificationMatrix MetricsTracker::overallDetectionMetrics() const {
    BinaryClassificationMatrix matrix;

    for (const auto& entry: _detections_per_frame_lookup) {
        matrix += entry.second;
    }

    return matrix;
}

MultiClassificationMatrix MetricsTracker::overallKnownRecognitionMetrics() const {
    MultiClassificationMatrix matrix(_labels.size() - 1);

    for (const auto& entry: _recognitions_per_frame_lookup) {
        const auto& frame_matrix = entry.second;
        uint32_t unknown_label_id = _labels_to_ids_lookup.at(UNKNOWN_LABEL);
        matrix += frame_matrix.remove(unknown_label_id);
    }

    return matrix;
}

BinaryClassificationMatrix MetricsTracker::overallUnknownRecognitionMetrics() const {
    MultiClassificationMatrix matrix(_labels.size());

    for (const auto& entry: _recognitions_per_frame_lookup) {
        matrix += entry.second;
    }

    uint32_t unknown_label_id = _labels_to_ids_lookup.at(UNKNOWN_LABEL);
    return matrix.getBinaryMatrix(unknown_label_id);
}

void MetricsTracker::keepTrackOf(const FrameInfo& frame_info,
                                 const std::vector<std::string>& labels,
                                 const std::vector<Rect>& face_origins) {
    trackDetection(frame_info, face_origins);
    trackRecognition(frame_info, labels, face_origins);
}

} // namespace detection
