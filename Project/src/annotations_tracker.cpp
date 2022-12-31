#include "annotations_tracker.h"

#include <fstream>
#include <unordered_set>

#include "file_utils.h"
#include "strings.h"

namespace detection {

const std::string FrameInfo::UNKNOWN_LABEL = "unknown";

FrameInfo::FrameInfo():
    _id(-1),
    _labels(),
    _face_origins() {
    // empty on purpose
}

FrameInfo::FrameInfo(uint32_t id,
                     const std::vector<std::string>& labels,
                     const std::vector<Rect>& face_origins):
    _id(id),
    _labels(labels),
    _face_origins(face_origins) {
    // empty on purpose
}

FrameInfo::FrameInfo(const FrameInfo& that):
    _id(that._id),
    _labels(that._labels),
    _face_origins(that._face_origins) {
    // empty on purpose
}

FrameInfo& FrameInfo::operator=(const FrameInfo& that) {
    if (this != &that) {
        this->_id = that._id;
        this->_labels = that._labels;
        this->_face_origins = that._face_origins;
    }

    return *this;
}

std::unique_ptr<AnnotationsTracker> AnnotationsTracker::LoadForVideo(const std::string& video_file) {
    std::string video_file_name = utils::GetFileName(video_file);
    std::string config_file = utils::ReplaceFilenameWithExtension(video_file, video_file_name + ".txt");

    std::unique_ptr<AnnotationsTracker> tracker = std::make_unique<AnnotationsTracker>(video_file);
    tracker->read(config_file);
    return std::move(tracker);
}

AnnotationsTracker::AnnotationsTracker(const std::string video_file):
    _video_file(video_file),
    _playback_info() {
    // empty on purpose
}

AnnotationsTracker::AnnotationsTracker(const AnnotationsTracker& that):
        _video_file(that._video_file),
        _playback_info(that._playback_info) {
    // empty on purpose
}

AnnotationsTracker& AnnotationsTracker::operator=(const AnnotationsTracker& that) {
    if (this != &that) {
        this->_video_file = that._video_file;
        this->_playback_info = that._playback_info;
    }

    return *this;
}

void AnnotationsTracker::read(const std::string& file) {
    if (!utils::Exists(file)) {
        throw std::runtime_error("File does not exists: " + file);
    }

    std::ifstream file_storage;
    file_storage.open(file, std::ios::in);

    uint32_t frame_id;
    std::unordered_set<std::string> seen_labels;
    std::vector<std::string> labels;
    std::vector<Rect> origins;

    std::string line;
    while (std::getline(file_storage, line)) {
        if (line.empty()) {
            break;
        }

        frame_id = static_cast<uint32_t>(stoi(line));
        if (_playback_info.find(frame_id) != _playback_info.end()) {
            throw std::runtime_error("Frame " + std::AsString(frame_id)
                + "has been declared multiple times.");
        }

        while (std::getline(file_storage, line)) {
            if (line.empty()) {
                std::vector<std::string> frame_labels(labels.begin(), labels.end());
                std::vector<Rect> frame_origins(origins.begin(), origins.end());

                seen_labels.clear();
                labels.clear();
                origins.clear();

                _playback_info.insert({frame_id, FrameInfo(frame_id, frame_labels, frame_origins)});
                break;
            }

            std::vector<std::string> tokens = std::Split(line, ' ' /* delimiter */);
            const auto& label = tokens[0];

            if (label != FrameInfo::UNKNOWN_LABEL && seen_labels.find(label) != seen_labels.end()) {
                throw std::runtime_error("Label " + std::AsString(label)
                    + " has been declared multiple times in frame " + std::AsString(frame_id));
            }

            seen_labels.insert(label);
            labels.push_back(label);

            std::vector<std::string> raw_rect = std::Split(tokens[1], ',' /* delimiter */);
            origins.push_back(Rect(static_cast<int32_t>(stoi(raw_rect[0])) /* x */,
                                   static_cast<int32_t>(stoi(raw_rect[1])) /* y */,
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

bool AnnotationsTracker::hasInfo(uint32_t frame_id) const {
    return _playback_info.find(frame_id) != _playback_info.end();
}

FrameInfo AnnotationsTracker::describeFrame(uint32_t frame_id) const {
    if (_playback_info.find(frame_id) != _playback_info.end()) {
        return _playback_info.at(frame_id);
    }

    // no value
    return FrameInfo();
}

} // namespace detection
