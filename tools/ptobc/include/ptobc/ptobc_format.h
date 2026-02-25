#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ptobc {

// PTOBC v0 constants.
static constexpr uint8_t kSectionStrings = 0x01;
static constexpr uint8_t kSectionTypes = 0x02;
static constexpr uint8_t kSectionAttrs = 0x03;
static constexpr uint8_t kSectionConstPool = 0x04;
static constexpr uint8_t kSectionOpcodeSchemaExt = 0x05;
static constexpr uint8_t kSectionModule = 0x06;
static constexpr uint8_t kSectionDebugInfo = 0x07;
static constexpr uint8_t kSectionExtra = 0x7F;

static constexpr uint16_t kVersionV0 = 0x0000;
static constexpr uint16_t kFlagsV0 = 0x0000;
static constexpr uint16_t kOpcodeGeneric = 0xFFFF;

struct Buffer {
  std::vector<uint8_t> bytes;
  void append(const void* p, size_t n);
  void appendU8(uint8_t v);
  void appendU16LE(uint16_t v);
  void appendU32LE(uint32_t v);
};

struct StringTable {
  std::unordered_map<std::string, uint64_t> toId;
  std::vector<std::string> fromId;

  uint64_t intern(const std::string& s);
};

struct PTOBCFile {
  // Tables
  StringTable strings;
  std::vector<std::string> typeAsm; // 1-based IDs; 0 means none
  std::vector<std::string> attrAsm; // 1-based IDs; 0 means none

  // Sections payloads
  std::vector<uint8_t> moduleBytes;

  std::vector<uint8_t> buildStringsSection() const;
  std::vector<uint8_t> buildTypesSection() const;
  std::vector<uint8_t> buildAttrsSection() const;
  std::vector<uint8_t> buildConstPoolSection() const;

  std::vector<uint8_t> serialize() const;
};

// Helpers to read a PTOBC file from disk.
std::vector<uint8_t> readFile(const std::string& path);
void writeFile(const std::string& path, const std::vector<uint8_t>& data);

} // namespace ptobc
