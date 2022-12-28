#ifndef VIDEO_PLAYER_H
#define VIDEO_PLAYER_H

#include <cstdint>
#include <string>

#include <opencv2/opencv.hpp>

namespace detection {

class VideoPlayer {
private:
  std::string _video_file;
  cv::VideoCapture _video_capture;
  uint32_t _current_frame;
  uint32_t _playback_group_size;

public:
  enum class PlaybackGroupState {
      STARTING_NEW_GROUP,
      PLAYING_EXISTING_GROUP
  };

  VideoPlayer(const std::string& video_file,
              uint32_t playback_group_size = 10);
  VideoPlayer(const VideoPlayer& that);
  VideoPlayer& operator=(const VideoPlayer& that);

  bool isOpened() const;

  size_t framesCount() const;

  uint32_t currentFrame() const;

  bool hasNextFrame() const;
  VideoPlayer::PlaybackGroupState nextFrame(cv::Mat& frame);

  ~VideoPlayer() = default;
};
    
} // namespace detection

#endif //VIDEO_PLAYER_H
