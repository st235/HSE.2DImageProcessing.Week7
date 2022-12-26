#include "labels_resolver.h"

namespace detection {

LabelsResolver::LabelsResolver():
    _next_available_id(0),
    _label_to_id_lookup_table(),
    _id_to_label_lookup_table() {
    // empty on purpose
}

LabelsResolver::LabelsResolver(const LabelsResolver& that):
    _next_available_id(that._next_available_id),
    _label_to_id_lookup_table(that._label_to_id_lookup_table),
    _id_to_label_lookup_table(that._id_to_label_lookup_table) {
    // empty on purpose
}

LabelsResolver& LabelsResolver::operator=(const LabelsResolver& that) {
    if (this != &that) {
        this->_next_available_id = that._next_available_id;
        this->_label_to_id_lookup_table = that._label_to_id_lookup_table;
        this->_id_to_label_lookup_table = that._id_to_label_lookup_table;
    }

    return *this;
}

uint32_t LabelsResolver::obtainIdByLabel(const std::string& label) {
    if (_label_to_id_lookup_table.find(label) == _label_to_id_lookup_table.end()) {
        _label_to_id_lookup_table[label] = _next_available_id;
        _next_available_id += 1;
    }

    return _label_to_id_lookup_table[label];
}

std::string LabelsResolver::obtainLabelById(uint32_t id) {
    if (_id_to_label_lookup_table.find(id) == _id_to_label_lookup_table.end()) {
        throw std::runtime_error("Cannot find the given id.");
    }

    return _id_to_label_lookup_table[id];
}

uint32_t LabelsResolver::operator[](const std::string& label) {
    return this->obtainIdByLabel(label);
}

} // namespace detection
