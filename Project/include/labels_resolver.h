#ifndef PROJECT_LABELS_RESOLVER_H
#define PROJECT_LABELS_RESOLVER_H

#include <cstdint>
#include <string>
#include <unordered_map>

namespace detection {

class LabelsResolver {
private:
    uint32_t _next_available_id;
    std::unordered_map<std::string, uint32_t> _label_to_id_lookup_table;
    std::unordered_map<uint32_t, std::string> _id_to_label_lookup_table;

public:
    LabelsResolver();
    LabelsResolver(const LabelsResolver& that);
    LabelsResolver& operator=(const LabelsResolver& that);

    uint32_t obtainIdByLabel(const std::string& label);
    std::string obtainLabelById(uint32_t id);
    uint32_t operator[](const std::string& label);

    ~LabelsResolver() = default;
};

} // namespace detection

#endif // PROJECT_LABELS_RESOLVER_H