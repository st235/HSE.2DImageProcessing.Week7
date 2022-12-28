#include "video_player.h"

namespace detection {

VideoPlayer::VideoPlayer(const std::string& video_file,
                         uint32_t playback_group_size):
    _video_file(video_file),
    _video_capture(video_file),
    _playback_group_size(playback_group_size),
    _current_frame(0) {
    // empty on purpose
}

VideoPlayer::VideoPlayer(const VideoPlayer& that):
        _video_file(that._video_file),
        _video_capture(that._video_capture),
        _playback_group_size(that._playback_group_size),
        _current_frame(that._current_frame) {
    // empty on purpose
}

VideoPlayer& VideoPlayer::operator=(const VideoPlayer& that) {
    if (this != &that) {
        this->_video_file = that._video_file;
        this->_video_capture = that._video_capture;
        this->_playback_group_size = that._playback_group_size;
        this->_current_frame = that._current_frame;
    }

    return *this;
}

bool VideoPlayer::isOpened() const {
    return _video_capture.isOpened();
}

size_t VideoPlayer::framesCount() const {
    return static_cast<size_t>(_video_capture.get(cv::CAP_PROP_FRAME_COUNT));
}

uint32_t VideoPlayer::currentFrame() const {
    return _current_frame;
}

bool VideoPlayer::hasNextFrame() const {
    bool has_frame_to_advance_further = (_current_frame + 1) < framesCount();
    return _video_capture.isOpened() && has_frame_to_advance_further;
}

VideoPlayer::PlaybackGroupState VideoPlayer::nextFrame(cv::Mat& frame) {
    _video_capture >> frame;
    uint32_t position_within_playback_group = _current_frame % _playback_group_size;
    // advancing our player
    _current_frame += 1;

    if (position_within_playback_group == 0) {
        // beginning of the playback group
        return PlaybackGroupState::STARTING_NEW_GROUP;
    }

    return PlaybackGroupState::PLAYING_EXISTING_GROUP;
}

} // namespace detection
