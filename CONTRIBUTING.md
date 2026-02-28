# Contributing to PTO Tile Library

Welcome to PTO Tile Library! We appreciate your interest in contributing to this project. This guide will help you get started with the contribution process.

## Code of Conduct

Please follow our [Community Code of Conduct](https://gitcode.com/cann/community) to ensure a welcoming and respectful environment for all contributors.

## Getting Started

### Prerequisites

Before contributing, make sure you have:

- A GitHub/GitCode account
- Basic knowledge of C++20 and Python
- Understanding of tile-based parallel programming (helpful but not required)
- Familiarity with Ascend CANN (for NPU-related contributions)

### Finding Issues to Work On

1. Browse existing [issues](https://gitcode.com/cann/pto-isa/issues) to find something to work on
2. Look for issues labeled `good first issue` for beginner-friendly tasks
3. Create a new issue if you've found a bug or have a feature request

## Contribution Workflow

### 1. Claim an Issue

Before starting work on any issue:

1. Browse the [issue list](https://gitcode.com/cann/pto-isa/issues)
2. Select an issue and comment `/assign` or `/assign @yourself` to claim it
3. Wait for maintainers to acknowledge before starting

### 2. Fork and Clone

```bash
# Fork the repository on GitHub/GitCode
# Then clone your fork
git clone https://gitcode.com/YOUR_USERNAME/pto-isa.git
cd pto-isa
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 4. Make Your Changes

Follow these guidelines:

- **Code Style**: Use `clang-format` for C++ and `ruff format` for Python
- **Testing**: Add tests for new features or bug fixes
- **Documentation**: Update docs for any API changes

```bash
# Format C++ code
clang-format -i -style=file <your_file.cpp>

# Format Python code
ruff format <your_file.py>
```

### 5. Submit a Pull Request

1. Push your changes to your fork:
   ```bash
   git push origin your-branch-name
   ```

2. Open a Pull Request using our [template](.gitcode/PULL_REQUEST_TEMPLATE.md)

3. Fill in all required information:
   - **Description**: What does this PR do?
   - **Motivation**: Why is this change needed?
   - **Testing**: How did you test your changes?

4. Link the related issue using keywords like `Fixes #123` or `Closes #456`

5. Comment on the issue to notify maintainers:
   > "This PR addresses the issue. Please review."

### 6. Code Review Process

1. Maintainers will review your PR
2. Address any feedback by pushing additional commits
3. Once approved, a maintainer will merge your PR

## Contribution Types

### 🐛 Bug Fixes

If you find a bug:

1. Create a `Bug Report` issue
2. Describe the bug with reproduction steps
3. Submit a PR with the fix

### ✨ New Features

For new features:

1. Create a `Feature Request` issue first
2. Wait for design discussion and approval
3. Implement the feature with tests and documentation

### 📝 Documentation

Help improve our docs:

1. Find docs issues or suggest improvements
2. Submit PRs for typo fixes, clarifications, or new content

### 🚀 New Operators

To contribute new operators:

#### Step 1: Create an Issue

Create a `Requirement` issue with:
- **Background**: What problem does this solve?
- **Value**: Why is it useful?
- **Design**: Your proposed implementation

#### Step 2: Design Review

SIG members will review and provide feedback. Address comments and request re-review.

#### Step 3: Implementation

Minimum deliverable structure:
```
docs/
├── isa/
│   └── ${op_name}.md           # Operator documentation
include/
├── pto/
│   ├── common/
│   │   ├── pto_instr_impl.hpp  # Implementation aggregation
│   │   └── pto_instr.hpp       # Public API
│   └── ${op_class}/
│       └── ${op_name}.hpp       # Operator implementation
tests/
├── ${op_class}/src/st/
│   ├── ${op_name}/
│   │   ├── ${op_name}.cpp      # Test harness
│   │   ├── main.cpp            # Entry point
│   │   ├── gen_data.py         # Test data generation
│   │   └── CMakeLists.txt      # Build config
│   └── CMakeLists.txt
└── run_st.sh                    # Test runner script
```

#### Step 4: Submit PR

Ensure:
- [ ] Code passes `clang-format` and `ruff format`
- [ ] Tests are included
- [ ] Documentation is complete
- [ ] PR links to the related issue

Comment `compile` to trigger CI checks.

## Development Setup

### CPU Simulation (Recommended for Starters)

```bash
# Clone and setup
git clone https://gitcode.com/cann/pto-isa.git
cd pto-isa

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy pytest

# Run tests
python3 tests/run_cpu.py --clean --verbose
```

### NPU Development (Linux Only)

See [Getting Started Guide](docs/getting-started.md) for:
- CANN toolkit installation
- NPU driver setup
- Building and testing

## Style Guides

### C++ Code

- Follow C++20 standard
- Use `clang-format` with `.clang-format` config
- Add comments for complex logic

### Python Code

- Follow PEP 8
- Use `ruff format` for formatting
- Add docstrings to functions

### Documentation

- Use clear, concise language
- Include code examples where appropriate
- Keep markdown tables properly aligned

## License

By contributing to PTO Tile Library, you agree that your contributions will be licensed under the [CANN Open Software License Agreement Version 2.0](LICENSE).

## Questions?

- Open an issue for bugs or feature requests
- Join our community discussions
- Contact maintainers directly for sensitive topics

Thank you for contributing to PTO Tile Library! 🎉
