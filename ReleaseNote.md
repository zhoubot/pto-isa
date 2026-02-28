<p align="center">
  <img src="docs/figures/pto_logo.svg" alt="PTO Tile Lib" width="100" />
</p>

# Release Notes — PTO Tile Library

<div align="center">

[![License](https://img.shields.io/badge/License-CANN%20Open%20Software%20License%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-9.0.0-green.svg)](version.cmake)

</div>

This file summarizes changes in PTO Tile Library.

> 📝 **Format**: Keep a Changelog style (Added/Changed/Fixed/Deprecated/Removed/Security).

---

## Unreleased

*No unreleased changes yet.*

### Added

- Initial public release of PTO Tile Library.

### Changed

*No changes yet.*

### Fixed

*No fixes yet.*

### Deprecated

*No deprecated features yet.*

### Removed

*No removed features yet.*

### Security

- See [`SECURITY.md`](SECURITY.md) for the vulnerability reporting process.

---

## Compatibility Notes

| Platform | Requirement |
|----------|-------------|
| **Ascend (NPU / simulator)** | Ascend CANN toolkit `>= 8.3` (see `version.info`); exact supported SoCs and toolchains depend on your installed CANN distribution |
| **CPU simulator** | macOS / Linux / Windows with C++20 toolchain and Python 3.8+ |

For detailed setup instructions, see the [Getting Started Guide](docs/getting-started.md).

---

## Supported Hardware

| Hardware | Architecture Code | Status |
|----------|-------------------|--------|
| Ascend A2 (910B) | `a2a3` | ✅ Supported |
| Ascend A3 (910C) | `a2a3` | ✅ Supported |
| Ascend A5 (950) | `a5` | ✅ Supported |
| CPU (x86_64 / AArch64) | `cpu` | ✅ Supported |

---

## Installation & Upgrade

### From Source

```bash
# Clone the repository
git clone https://gitcode.com/cann/pto-isa.git
cd pto-isa

# Build
chmod +x build.sh
./build.sh
```

### Using PTO-DSL (Python)

```bash
pip install -e ./PTODSL
```

---

## Migration Guides

### From Previous Versions

*No migration guides available yet (first public release).*

---

## Known Issues

*No known issues at this time.*

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

---

## Resources

- 📖 [Documentation](https://pto-isa.gitcode.com)
- 💬 [Community](https://gitcode.com/cann/community)
- 🐛 [Issue Tracker](https://gitcode.com/cann/pto-isa/issues)
- 🐍 [PyPTO](https://gitcode.com/cann/pypto/) - Formal Pythonic programming interface
- 🐍 [PTO-DSL](PTODSL/README.md) - In-core level Python DSL (included)
- 📚 [TileLang Ascend](https://github.com/tile-ai/tilelang-ascend/) - High-level DSL

