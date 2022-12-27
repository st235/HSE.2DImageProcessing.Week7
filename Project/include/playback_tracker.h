#ifndef PLAYBACK_TRACKER_H
#define PLAYBACK_TRACKER_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "rect.h"

namespace detection {

struct FrameInfo {
private:
  static const std::string UNKNOWN_LABEL;

  uint32_t id;
  std::vector<std::string> labels;
  std::vector<Rect> face_origins;
public:
    FrameInfo(uint32_t id,
              const std::vector<std::string>& labels,
              const std::vector<Rect>& face_origins);
    FrameInfo(const FrameInfo& that);
    FrameInfo& operator=(const FrameInfo& that);

    ~FrameInfo() = default;
};

class PlaybackTracker {
private:
  std::string _video_file;
  std::map<uint32_t, FrameInfo> _playback_info;

public:
    explicit PlaybackTracker(const std::string video_file);
    PlaybackTracker(const PlaybackTracker& that);
    PlaybackTracker& operator=(const PlaybackTracker& that);

    inline std::string id() const { return _video_file; }

    /**
     * Reads playback information for
     * the given file.
     * File should use the following structure:
     * frame_index_1
     * label1 rect1.x,rect1.y,rect1.width,rect1.height
     * label2 rect2.x,rect2.y,rect2.width,rect2.height
     * ...
     * frame_index_n
     * labeln rectn.x,rectn.y,rectn.width,rectn.height
     * ...
     *
     * Please, do keep in mind that labels not from
     * the dataset should be marked as 'unknown'
     * if you want to calculate the score for them.
     */
    void read(const std::string& file);

    /**
     * Finds frame with the given {@code frame_id} or
     * the closest to it lower bound. If frame was not
     * found - which usually should not be the case -
     * empty frame info will be returned.
     * Works for O(log n) to support lower bound search.
     */
    FrameInfo describeFrame(uint32_t frame_id);

    ~PlaybackTracker() = default;
};

} // namespace detection

#endif //PLAYBACK_TRACKER_H
