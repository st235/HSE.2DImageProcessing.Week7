#ifndef METRICS_TRACKER_H
#define METRICS_TRACKER_H

#include <cstdint>
#include <string>
#include <vector>

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
    ConfusionMatrix _confusion_metric;

public:
    MetricsTracker();
    MetricsTracker(const MetricsTracker& that);
    MetricsTracker& operator=(const MetricsTracker& that);

    void keepTrackOf(const FrameInfo& frame_info,
                     const std::vector<std::string>& labels,
                     const std::vector<Rect>& face_origins);

    ~MetricsTracker() = default;
};

} // namespace detection

#endif //METRICS_TRACKER_H
