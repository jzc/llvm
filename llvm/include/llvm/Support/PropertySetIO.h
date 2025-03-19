//==-- PropertySetIO.h -- models a sequence of property sets and their I/O -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Models a sequence of property sets and their input and output operations.
// TODO use Yaml as I/O engine.
// PropertyValue set format:
//   '['<PropertyValue set name>']'
//   <property name>=<property type>'|'<property value>
//   <property name>=<property type>'|'<property value>
//   ...
//   '['<PropertyValue set name>']'
//   <property name>=<property type>'|'<property value>
// where
//   <PropertyValue set name>, <property name> are strings
//   <property type> - string representation of the property type
//   <property value> - string representation of the property value.
//
// For example:
// [Staff/Ages]
// person1=1|20
// person2=1|25
// [Staff/Experience]
// person1=1|1
// person2=1|2
// person3=1|12
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PROPERTYSETIO_H
#define LLVM_SUPPORT_PROPERTYSETIO_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include <variant>

namespace llvm {
namespace util {

class PropertySetRegistry;

// Represents a property value. PropertyValue name is stored in the encompassing
// container.
class PropertyValue {
public:
  using SizeTy = uint64_t;
  using ByteArrayTy = SmallVector<char, 8>;

  // Defines supported property types
  enum Type { first = 0, NONE = first, UINT32, BYTE_ARRAY, last = BYTE_ARRAY };

  PropertyValue() = default;

  PropertyValue(uint32_t Val) : Ty(UINT32), Value(Val) {}
  PropertyValue(StringRef Data)
      : Ty(BYTE_ARRAY), Value(ByteArrayTy(Data.begin(), Data.end())) {}

  template <typename C, typename T = typename C::value_type>
  PropertyValue(const C &Data)
      : PropertyValue({reinterpret_cast<const char *>(Data.data()),
                       Data.size() * sizeof(T)}) {}

  // get property value as unsigned 32-bit integer
  uint32_t asUint32() const {
    assert(Ty == UINT32 && "must be UINT32 value");
    return std::get<uint32_t>(Value);
  }

  StringRef asByteArray() const {
    assert(Ty == BYTE_ARRAY && "must be BYTE_ARRAY value");
    const auto &ByteArrayRef = std::get<ByteArrayTy>(Value);
    return {ByteArrayRef.data(), ByteArrayRef.size()};
  }

  Type getType() const { return Ty; }

private:
  Type Ty = NONE;
  std::variant<std::monostate, uint32_t, ByteArrayTy> Value = {};
};

/// Structure for specialization of DenseMap in PropertySetRegistry.
struct PropertySetKeyInfo {
  static unsigned getHashValue(const SmallString<16> &K) { return xxHash64(K); }

  static SmallString<16> getEmptyKey() { return SmallString<16>(""); }

  static SmallString<16> getTombstoneKey() { return SmallString<16>("_"); }

  static bool isEqual(StringRef L, StringRef R) { return L == R; }
};

using PropertyMapTy = DenseMap<SmallString<16>, unsigned, PropertySetKeyInfo>;
/// A property set. Preserves insertion order when iterating elements.
using PropertySet = MapVector<SmallString<16>, PropertyValue, PropertyMapTy>;

/// A registry of property sets. Maps a property set name to its
/// content.
///
/// The order of keys is preserved and corresponds to the order of insertion.
class PropertySetRegistry {
public:
  using MapTy = MapVector<SmallString<16>, PropertySet, PropertyMapTy>;

  // Specific property category names used by tools.
  static constexpr char SYCL_SPECIALIZATION_CONSTANTS[] =
      "SYCL/specialization constants";
  static constexpr char SYCL_SPEC_CONSTANTS_DEFAULT_VALUES[] =
      "SYCL/specialization constants default values";
  // TODO: remove SYCL_DEVICELIB_REQ_MASK when devicelib online linking path
  // is totally removed.
  static constexpr char SYCL_DEVICELIB_REQ_MASK[] = "SYCL/devicelib req mask";
  static constexpr char SYCL_DEVICELIB_METADATA[] = "SYCL/devicelib metadata";
  static constexpr char SYCL_KERNEL_PARAM_OPT_INFO[] = "SYCL/kernel param opt";
  static constexpr char SYCL_PROGRAM_METADATA[] = "SYCL/program metadata";
  static constexpr char SYCL_MISC_PROP[] = "SYCL/misc properties";
  static constexpr char SYCL_ASSERT_USED[] = "SYCL/assert used";
  static constexpr char SYCL_EXPORTED_SYMBOLS[] = "SYCL/exported symbols";
  static constexpr char SYCL_IMPORTED_SYMBOLS[] = "SYCL/imported symbols";
  static constexpr char SYCL_DEVICE_GLOBALS[] = "SYCL/device globals";
  static constexpr char SYCL_DEVICE_REQUIREMENTS[] = "SYCL/device requirements";
  static constexpr char SYCL_HOST_PIPES[] = "SYCL/host pipes";
  static constexpr char SYCL_VIRTUAL_FUNCTIONS[] = "SYCL/virtual functions";
  static constexpr char SYCL_IMPLICIT_LOCAL_ARG[] = "SYCL/implicit local arg";
  static constexpr char SYCL_REGISTERED_KERNELS[] = "SYCL/registered kernels";

  static constexpr char PROPERTY_REQD_WORK_GROUP_SIZE[] =
      "reqd_work_group_size_uint64_t";

  /// Function for bulk addition of an entire property set in the given
  /// \p Category .
  template <typename MapTy> void add(StringRef Category, const MapTy &Props) {
    assert(PropSetMap.find(Category) == PropSetMap.end() &&
           "category already added");
    auto &PropSet = PropSetMap[Category];

    for (const auto &Prop : Props)
      PropSet.insert_or_assign(Prop.first, PropertyValue(Prop.second));
  }

  /// Adds the given \p PropVal with the given \p PropName into the given \p
  /// Category .
  template <typename T>
  void add(StringRef Category, StringRef PropName, const T &PropVal) {
    auto &PropSet = PropSetMap[Category];
    PropSet.insert({PropName, PropertyValue(PropVal)});
  }

  void remove(StringRef Category, StringRef PropName) {
    auto PropertySetIt = PropSetMap.find(Category);
    if (PropertySetIt == PropSetMap.end())
      return;
    auto &PropertySet = PropertySetIt->second;
    auto PropIt = PropertySet.find(PropName);
    if (PropIt == PropertySet.end())
      return;
    PropertySet.erase(PropIt);
  }

  /// Parses from the given \p Buf a property set registry.
  static Expected<std::unique_ptr<PropertySetRegistry>>
  read(const MemoryBuffer *Buf);
  static Expected<std::unique_ptr<PropertySetRegistry>>
  readJSON(const MemoryBuffer *Buf);

  /// Dumps the property set registry to the given \p Out stream.
  void write(raw_ostream &Out) const;
  void writeJSON(raw_ostream &Out) const;

  MapTy::const_iterator begin() const { return PropSetMap.begin(); }
  MapTy::const_iterator end() const { return PropSetMap.end(); }

  /// Retrieves a property set with given \p Name .
  PropertySet &operator[](StringRef Name) { return PropSetMap[Name]; }
  /// Constant access to the underlying map.
  const MapTy &getPropSets() const { return PropSetMap; }

private:
  MapTy PropSetMap;
};

} // namespace util
} // namespace llvm

#endif // #define LLVM_SUPPORT_PROPERTYSETIO_H
