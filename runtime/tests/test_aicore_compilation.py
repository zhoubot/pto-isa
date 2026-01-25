"""
Tests for AICore kernel compilation via BinaryCompiler.

This test module verifies the AICore real compilation pipeline:
1. Real compilation integration tests with actual ASCEND toolchain
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
        """Test real compilation with src/aicore using graph as source and include dir"""
        # Get directories
        runtime_dir = Path(__file__).parent.parent
        aicore_dir = runtime_dir / "src" / "aicore"
        graph_dir = runtime_dir / "tests" / "example_graph_impl"

        assert aicore_dir.exists(), f"AICore directory not found: {aicore_dir}"
        assert graph_dir.exists(), f"Graph directory not found: {graph_dir}"

        # Initialize compiler
        compiler = BinaryCompiler()

        # Use graph as both include and source directory
        include_dirs = [str(graph_dir)]
        source_dirs = [str(graph_dir)]

        # Compile
        binary_data = compiler.compile(
            target_platform="aicore",
            include_dirs=include_dirs,
            source_dirs=source_dirs
        )

        # Verify binary data
        assert len(binary_data) > 0, "Binary data should not be empty"
        assert isinstance(binary_data, bytes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
