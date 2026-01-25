"""
Tests for AICPU kernel compilation via BinaryCompiler.

This test module verifies the AICPU real compilation pipeline:
1. Real compilation integration tests with actual ASCEND cross-compiler toolchain
"""

import os
import sys
import pytest
from pathlib import Path

# Add runtime/python to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from binary_compiler import BinaryCompiler


class TestRealEnvironment:
    """Integration tests using real environment (if available)."""

    @pytest.mark.skipif(not os.getenv("ASCEND_HOME_PATH"), reason="ASCEND_HOME_PATH not set")
    def test_compile_real_environment(self):
        """Test real AICPU compilation with src/aicpu using graph as source and include dir"""
        # Get directories
        runtime_dir = Path(__file__).parent.parent
        aicpu_dir = runtime_dir / "src" / "aicpu"
        graph_dir = runtime_dir / "tests" / "example_graph_impl"
        aicpu_dir = runtime_dir / "tests" / "example_aicpu_impl"

        assert aicpu_dir.exists(), f"AICPU directory not found: {aicpu_dir}"
        assert graph_dir.exists(), f"Graph directory not found: {graph_dir}"

        # Initialize compiler
        compiler = BinaryCompiler()

        # Use graph as both include and source directory
        include_dirs = [str(graph_dir)]
        source_dirs = [str(graph_dir), str(aicpu_dir)]

        # Compile
        binary_data = compiler.compile(
            target_platform="aicpu",
            include_dirs=include_dirs,
            source_dirs=source_dirs
        )

        # Verify binary data
        assert len(binary_data) > 0, "Binary data should not be empty"
        assert isinstance(binary_data, bytes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
