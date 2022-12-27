#include "playback_tracker.h"

#include <fstream>
#include <unordered_set>

#include "strings.h"

namespace detection {

const std::string FrameInfo::UNKNOWN_LABEL = "";

FrameInfo::FrameInfo(uint32_t id,
                     const std::vector<std::string>& labels,
                     const std::vector<Rect>& face_origins):
    id(id),
    labels(labels),
    face_origins(face_origins) {
    // empty on purpose
}

FrameInfo::FrameInfo(const FrameInfo& that):
    id(that.id),
    labels(that.labels),
    face_origins(that.face_origins) {
    // empty on purpose
}

FrameInfo& FrameInfo::operator=(const FrameInfo& that) {
    if (this != &that) {
        this->id = that.id;
        this->labels = that.labels;
        this->face_origins = that.face_origins;
    }

    return *this;
}

PlaybackTracker::PlaybackTracker(const std::string video_file):
    _video_file(video_file),
    _playback_info() {
    // empty on purpose
}

PlaybackTracker::PlaybackTracker(const PlaybackTracker& that):
        _video_file(that._video_file),
        _playback_info(that._playback_info) {
    // empty on purpose
}

PlaybackTracker& PlaybackTracker::operator=(const PlaybackTracker& that) {
    if (this != &that) {
        this->_video_file = that._video_file;
        this->_playback_info = that._playback_info;
    }

    return *this;
}

void PlaybackTracker::read(const std::string& file) {
    std::ifstream file_storage;
    file_storage.open(file, std::ios::in);

    uint32_t frame_id;
    std::unordered_set<std::string> labels;
    std::vector<Rect> origins;

    std::string line;
    while (std::getline(file_storage, line)) {
        if (line.empty()) {
            break;
        }

        frame_id = static_cast<uint32_t>(stoi(line));
        if (_playback_info.find(frame_id) != _playback_info.end()) {
            throw std::runtime_error("Frame " + std::to_string(frame_id)
                + "has been declared multiple times.");
        }

        while (std::getline(file_storage, line)) {
            if (line.empty()) {
                std::vector <std::string> frame_labels(labels.begin(), labels.end());
                std::vector <Rect> frame_origins(origins.begin(), origins.end());

                labels.clear();
                origins.clear();

                _playback_info.insert({frame_id, FrameInfo(frame_id, frame_labels, frame_origins)});
                break;
            }

            std::vector<std::string> tokens = std::Split(line, ' ' /* delimiter */);
            const auto& label = tokens[0];

            if (labels.find(label) != labels.end()) {
                throw std::runtime_error("Label " + std::to_string(label)
                    + " has been declared multiple times in frame " + std::to_string(frame_id));
            }

            labels.insert(label);

            std::vector<std::string> raw_rect = std::Split(tokens[1], ',' /* delimiter */);
            origins.push_back(Rect(static_cast<uint32_t>(stoi(raw_rect[0])) /* x */,
                                   static_cast<uint32_t>(stoi(raw_rect[1])) /* y */,
                                   static_cast<uint32_t>(stoi(raw_rect[2])) /* width */,
                                   static_cast<uint32_t>(stoi(raw_rect[3])) /* height */));
        }
    }

    std::vector<std::string> frame_labels(labels.begin(), labels.end());
    std::vector<Rect> frame_origins(origins.begin(), origins.end());

    labels.clear();
    origins.clear();

    _playback_info.insert({frame_id, FrameInfo(frame_id, frame_labels, frame_origins)});
}

FrameInfo PlaybackTracker::describeFrame(uint32_t frame_id) {
    if (_playback_info.find(frame_id) != _playback_info.end()) {
        return _playback_info.find(frame_id)->second;
    }

    const auto& lower_bound = _playback_info.lower_bound(frame_id);

    if (lower_bound != _playback_info.end()) {
        return lower_bound->second;
    }

    // no value or lower bound
    return FrameInfo(-1, {} /* labels */, {} /* labels */);
}

} // namespace detection
