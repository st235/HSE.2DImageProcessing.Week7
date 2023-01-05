#ifndef LABELS_RESOLVER_H
#define LABELS_RESOLVER_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace detection {

class LabelsResolver {
private:
  static const int UNKNOWN_LABEL_ID;
  static const std::string UNKNOWN_LABEL;

  int32_t _next_available_id;
  std::unordered_map<std::string, int32_t> _label_to_id_lookup_table;
  std::unordered_map<int32_t, std::string> _id_to_label_lookup_table;

public:
  LabelsResolver();
  LabelsResolver(const LabelsResolver& that);
  LabelsResolver& operator=(const LabelsResolver& that);

  bool hasId(int32_t id) const;

  std::vector<std::string> getLabels() const;

  uint32_t size() const;

  int32_t obtainIdByLabel(const std::string& label);
  std::string obtainLabelById(int32_t id);
  int32_t operator[](const std::string& label);

  void write(const std::string& file) const;
  void read(const std::string& file);

  ~LabelsResolver() = default;
};

} // namespace detection

#endif // LABELS_RESOLVER_H
