//==- PropertySetIO.cpp - models a sequence of property sets and their I/O -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PropertySetIO.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <string>

using namespace llvm::util;
using namespace llvm;

namespace {

::llvm::Error makeError(const Twine &Msg) {
  return createStringError(std::error_code{}, Msg);
}

} // anonymous namespace

Expected<std::unique_ptr<PropertySetRegistry>>
PropertySetRegistry::readJSON(const MemoryBuffer *Buf) {
  auto Res = std::make_unique<PropertySetRegistry>();
  Expected<json::Value> V = json::parse(Buf->getBuffer());
  if (!V)
    return V.takeError();
  const json::Object *O = V->getAsObject();
  if (!O)
    return makeError("expected JSON object");
  for (const auto &[CategoryName, Value] : *O) {
    const json::Array *PropsArray = Value.getAsArray();
    if (!PropsArray)
      return makeError("expected JSON array for properties");
    PropertySet &PropSet = Res->PropSetMap[StringRef(CategoryName)];
    for (const auto &PropPair : *PropsArray) {
      const json::Array *PropArray = PropPair.getAsArray();
      if (!PropArray || PropArray->size() != 2)
        return makeError(
            "expected property as [PropertyName, PropertyValue] pair");

      const json::Value &PropNameVal = (*PropArray)[0];
      const json::Value &PropValueVal = (*PropArray)[1];

      std::optional<StringRef> PropName = PropNameVal.getAsString();
      if (!PropName)
        return makeError("expected property name as string");

      PropertyValue Prop;
      if (std::optional<uint64_t> Val = PropValueVal.getAsUINT64()) {
        Prop = PropertyValue(static_cast<uint32_t>(*Val));
      } else if (const json::Array *Val = PropValueVal.getAsArray()) {
        SmallVector<unsigned char, 8> Vec;
        for (const auto &V : *Val) {
          std::optional<uint64_t> Byte = V.getAsUINT64();
          if (!Byte)
            return makeError("invalid byte array value");
          if (*Byte > std::numeric_limits<unsigned char>::max())
            return makeError("byte array value out of range");
          Vec.push_back(static_cast<unsigned char>(*Byte));
        }
        Prop = PropertyValue(Vec);
      } else {
        return makeError("unsupported property type");
      }

      if (PropSet.find(*PropName) != PropSet.end())
        return makeError("duplicate property name");
      PropSet.insert({*PropName, Prop});
    }
  }
  return Res;
}

namespace llvm {
// output a property to a stream
raw_ostream &operator<<(raw_ostream &Out, const PropertyValue &Prop) {
  Out << static_cast<int>(Prop.getType()) << "|";
  switch (Prop.getType()) {
  case PropertyValue::Type::UINT32:
    Out << Prop.asUint32();
    break;
  case PropertyValue::Type::BYTE_ARRAY: {
    json::OStream J(Out);
    J.array([&] {
      auto ByteArrayRef = Prop.asByteArray();
      for (const auto &Byte : ByteArrayRef) {
        J.value(Byte);
      }
    });
    break;
  }
  default:
    llvm_unreachable(
        ("unsupported property type: " + utostr(Prop.getType())).c_str());
  }
  return Out;
}
} // namespace llvm

void PropertySetRegistry::write(raw_ostream &Out) const {
  for (const auto &PropSet : PropSetMap) {
    Out << "[" << PropSet.first << "]\n";

    for (const auto &Props : PropSet.second) {
      Out << Props.first << "=" << Props.second << "\n";
    }
  }
}

void PropertySetRegistry::writeJSON(raw_ostream &Out) const {
  json::OStream J(Out);
  J.object([&] {
    for (const auto &PropSet : PropSetMap) {
      J.attributeArray(PropSet.first, [&] {
        for (const auto &Props : PropSet.second) {
          J.array([&] {
            J.value(Props.first);
            switch (Props.second.getType()) {
            case PropertyValue::Type::UINT32:
              J.value(Props.second.asUint32());
              break;
            case PropertyValue::Type::BYTE_ARRAY: {
              auto ByteArrayRef = Props.second.asByteArray();
              J.value(json::Array(ByteArrayRef.bytes()));
              break;
            }
            default:
              llvm_unreachable(("unsupported property type: " +
                                utostr(Props.second.getType()))
                                   .c_str());
            }
          });
        }
      });
    }
  });
}
