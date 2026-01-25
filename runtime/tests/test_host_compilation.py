"""
Tests for Host executable compilation via BinaryCompiler.

This test module verifies the Host real compilation pipeline:
1. Real compilation integration tests with standard host compilers
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
        """Test real Host compilation with src/host using graph as source and include dir"""
        # Get directories
        runtime_dir = Path(__file__).parent.parent
        host_dir = runtime_dir / "src" / "host"
        graph_dir = runtime_dir / "tests" / "example_graph_impl"
        host_dir = runtime_dir / "tests" / "example_host_impl"

        assert host_dir.exists(), f"Host directory not found: {host_dir}"
        assert graph_dir.exists(), f"Graph directory not found: {graph_dir}"

        # Initialize compiler
        compiler = BinaryCompiler()

        # Use graph as both include and source directory
        include_dirs = [str(graph_dir)]
        source_dirs = [str(graph_dir), str(host_dir)]

        # Compile
        binary_data = compiler.compile(
            target_platform="host",
            include_dirs=include_dirs,
            source_dirs=source_dirs
        )

        # Verify binary data
        assert isinstance(binary_data, bytes), "Binary data should be bytes"
        assert len(binary_data) > 0, "Binary data should not be empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
