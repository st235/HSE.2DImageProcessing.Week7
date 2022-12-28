#ifndef RECT_H
#define RECT_H

#include <cstdint>

#include <opencv2/opencv.hpp>

namespace detection {

struct Rect {
public:
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;

  static Rect from(const cv::Rect& that);
  static cv::Rect toCVRect(const Rect& that);
  static double iou(const Rect& one, const Rect& another);

  Rect();
  Rect(uint32_t x, uint32_t y, uint32_t width, uint32_t height);
  Rect(const Rect& that);
  Rect& operator=(const Rect& that);

  bool intersects(const Rect& that) const;
  Rect intersection(const Rect& that) const;

  Rect escapeFromOldBasis(const Rect basis) const;

  uint64_t area() const;

  bool empty() const;

  ~Rect() = default;
};

} // namespace detection

#endif //RECT_H
