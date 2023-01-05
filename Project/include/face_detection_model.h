#ifndef FACE_DETECTION_MODEL_H
#define FACE_DETECTION_MODEL_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "rect.h"

namespace detection {

struct Eyes {
public:
  const Rect left;
  const Rect right;

  static Eyes from(cv::Rect left, cv::Rect right) {
      return Eyes(Rect::from(left), Rect::from(right));
  }

  Eyes():
      left(),
      right() {
      // empty on purpose
  }

  Eyes(Rect left, Rect right):
      left(left),
      right(right) {
        // empty on purpose
    }

  Eyes(const Eyes& that):
      left(that.left),
      right(that.right) {
      // empty on purpose
  }

  bool empty() const {
      return left.empty() && right.empty();
  }

  ~Eyes() = default;
};

struct Face {
public:
  const cv::Mat image;
  const Rect origin;
  const Eyes eyes;

  Face(const cv::Mat& image, const Rect& origin, const Eyes& eyes = Eyes()):
      image(image),
      origin(origin),
      eyes(eyes) {
      // empty on purpose
  }

  Face(const Face& face):
          image(face.image),
          origin(face.origin),
          eyes(face.eyes) {
      // empty on purpose
  }

  Eyes eyesEscapedFromFaceBasis() const {
      return Eyes(eyes.left.escapeFromOldBasis(origin),
                  eyes.right.escapeFromOldBasis(origin));
  }

  bool eyesDetected() const {
      return !eyes.empty();
  }

  ~Face() = default;
};

class FaceDetectionModel {
protected:
  bool shouldClip(const Rect& viewport, const Rect& face_origin) const;

public:
  virtual std::vector<Face> extractFaces(const Rect& viewport, cv::Mat& image) = 0;

  virtual ~FaceDetectionModel() = default;
};

} // namespace detection

#endif //FACE_DETECTION_MODEL_H
