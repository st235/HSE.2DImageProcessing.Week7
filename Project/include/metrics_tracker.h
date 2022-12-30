#ifndef METRICS_TRACKER_H
#define METRICS_TRACKER_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

#include "annotations_tracker.h"
#include "rect.h"

namespace detection {

struct ConfusionMatrix {
public:
  uint32_t tp;
  uint32_t tn;
  uint32_t fp;
  uint32_t fn;

  ConfusionMatrix();
  ConfusionMatrix(uint32_t tp,
                  uint32_t tn,
                  uint32_t fp,
                  uint32_t fn);
  ConfusionMatrix(const ConfusionMatrix& that);
  ConfusionMatrix& operator=(const ConfusionMatrix& that);

  ConfusionMatrix merge(const ConfusionMatrix& that);

  ConfusionMatrix operator+(const ConfusionMatrix& that);
  ConfusionMatrix& operator+=(const ConfusionMatrix& that);

  ~ConfusionMatrix() = default;
};

class MetricsTracker {
private:
    std::unordered_map<uint32_t, ConfusionMatrix> _detections_per_frame_lookup;

    void trackDetection(const FrameInfo& frame_info,
                        const std::vector<Rect>& detected_face_origins);

public:
    MetricsTracker();
    MetricsTracker(const MetricsTracker& that);
    MetricsTracker& operator=(const MetricsTracker& that);

    void keepTrackOf(const FrameInfo& frame_info,
                     const std::vector<std::string>& detected_labels,
                     const std::vector<Rect>& detected_face_origins);

    ConfusionMatrix overallDetectionMetrics() const;

    ~MetricsTracker() = default;
};

} // namespace detection

#endif //METRICS_TRACKER_H
