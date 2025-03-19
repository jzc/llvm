//==- PropertySetIO.cpp - models a sequence of property sets and their I/O -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PropertySetIO.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LineIterator.h"

#include <memory>
#include <string>

using namespace llvm::util;
using namespace llvm;

namespace {

using byte = Base64::byte;

::llvm::Error makeError(const Twine &Msg) {
  return createStringError(std::error_code{}, Msg);
}

} // anonymous namespace

// Casts from int value to a type tag.
static Expected<PropertyValue::Type> getTypeTag(int T) {
  if (T < PropertyValue::first || T > PropertyValue::last)
    return createStringError(std::error_code(), "bad property type ", T);
  return static_cast<PropertyValue::Type>(T);
}

Expected<std::unique_ptr<PropertySetRegistry>>
PropertySetRegistry::read(const MemoryBuffer *Buf) {
  auto Res = std::make_unique<PropertySetRegistry>();
  PropertySet *CurPropSet = nullptr;

  for (line_iterator LI(*Buf); !LI.is_at_end(); LI++) {
    // see if this line starts a new property set
    if (LI->starts_with("[")) {
      // yes - parse the category (property name)
      auto EndPos = LI->rfind(']');
      if (EndPos == StringRef::npos)
        return makeError("invalid line: " + *LI);
      StringRef Category = LI->substr(1, EndPos - 1);
      CurPropSet = &(*Res)[Category];
      continue;
    }
    if (!CurPropSet)
      return makeError("property category missing");
    // parse name and type+value
    auto Parts = LI->split('=');

    if (Parts.first.empty() || Parts.second.empty())
      return makeError("invalid property line: " + *LI);
    auto TypeVal = Parts.second.split('|');

    if (TypeVal.first.empty() || TypeVal.second.empty())
      return makeError("invalid property value: " + Parts.second);
    APInt Tint;

    // parse type
    if (TypeVal.first.getAsInteger(10, Tint))
      return makeError("invalid property type: " + TypeVal.first);
    Expected<PropertyValue::Type> Ttag =
        getTypeTag(static_cast<int>(Tint.getSExtValue()));
    StringRef Val = TypeVal.second;

    if (!Ttag)
      return Ttag.takeError();
    PropertyValue Prop;

    // parse value depending on its type
    switch (Ttag.get()) {
    case PropertyValue::Type::UINT32: {
      APInt ValV;
      if (Val.getAsInteger(10, ValV))
        return createStringError(std::error_code{},
                                 "invalid property value: ", Val.data());
      Prop = PropertyValue(static_cast<uint32_t>(ValV.getZExtValue()));
      break;
    }
    case PropertyValue::Type::BYTE_ARRAY: {
      Expected<std::unique_ptr<byte[]>> DecArr =
          Base64::decode(Val.data(), Val.size());
      if (!DecArr)
        return DecArr.takeError();
      char *ThePtr = reinterpret_cast<char *>((*DecArr).get());
      auto DecodedSize = Base64::getDecodedSize(Val.size());
      StringRef Res(ThePtr, DecodedSize);
      Prop = PropertyValue(Res);
      break;
    }
    default:
      return createStringError(std::error_code{},
                               "unsupported property type: ", Ttag.get());
    }
    (*CurPropSet)[Parts.first] = std::move(Prop);
  }
  if (!CurPropSet)
    return makeError("invalid property set registry");

  return Expected<std::unique_ptr<PropertySetRegistry>>(std::move(Res));
}

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
    auto ByteArrayRef = Prop.asByteArray();
    Base64::encode(ByteArrayRef.bytes_begin(), Out, ByteArrayRef.size());
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
