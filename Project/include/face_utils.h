#ifndef FACE_UTILS_H
#define FACE_UTILS_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "rect.h"

namespace detection {
    
struct Eyes {
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

std::vector<Face> extractFaces(cv::Mat& image,
                               const std::string& face_cascade_file,
                               const std::string& right_eye_cascade_file,
                               const std::string& left_eye_cascade_file);

void drawFaces(cv::Mat& image,
               const std::vector<Face>& faces);

void drawFaces(cv::Mat& image,
               const std::vector<Face>& faces,
               const std::vector<std::string>& labels);

}

#endif // FACE_UTILS_H
