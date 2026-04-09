import os
import tempfile
import unittest

import numpy as np
import torch

from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file as save_file_pt

HAS_KVIKIO = False
try:
    import kvikio

    HAS_KVIKIO = True
except ImportError:
    pass

HAS_CUDA = torch.cuda.is_available()


class TestGdsMetadata(unittest.TestCase):
    """Tests that GDS mode correctly parses metadata without mmapping the full file."""

    @unittest.skipUnless(HAS_KVIKIO, "kvikio not installed")
    def test_gds_open_reads_metadata(self):
        """GDS mode should parse the header and expose keys/metadata."""
        tensors = {
            "weight": torch.randn(64, 128),
            "bias": torch.randn(64),
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            fname = f.name
        try:
            save_file_pt(tensors, fname)
            device = "cuda:0" if HAS_CUDA else "cpu"
            with safe_open(fname, framework="pt", device=device, use_gds=True) as f:
                keys = f.keys()
                self.assertIn("weight", keys)
                self.assertIn("bias", keys)
                self.assertEqual(len(keys), 2)
        finally:
            os.unlink(fname)

    @unittest.skipUnless(HAS_KVIKIO and HAS_CUDA, "kvikio and CUDA required")
    def test_gds_get_tensor_pytorch(self):
        """GDS mode should load tensors directly onto GPU via kvikio."""
        tensors = {
            "weight": torch.randn(32, 64),
            "bias": torch.randn(32),
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            fname = f.name
        try:
            save_file_pt(tensors, fname)
            with safe_open(fname, framework="pt", device="cuda:0", use_gds=True) as f:
                w = f.get_tensor("weight")
                b = f.get_tensor("bias")

            # Verify device
            self.assertTrue(w.is_cuda)
            self.assertTrue(b.is_cuda)

            # Verify values match
            torch.testing.assert_close(w.cpu(), tensors["weight"])
            torch.testing.assert_close(b.cpu(), tensors["bias"])
        finally:
            os.unlink(fname)

    @unittest.skipUnless(HAS_KVIKIO and HAS_CUDA, "kvikio and CUDA required")
    def test_gds_get_slice_pytorch(self):
        """GDS mode should support sliced tensor access."""
        tensors = {
            "matrix": torch.randn(16, 32),
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            fname = f.name
        try:
            save_file_pt(tensors, fname)
            with safe_open(fname, framework="pt", device="cuda:0", use_gds=True) as f:
                sliced = f.get_slice("matrix")[:8]

            self.assertTrue(sliced.is_cuda)
            self.assertEqual(list(sliced.shape), [8, 32])
            torch.testing.assert_close(sliced.cpu(), tensors["matrix"][:8])
        finally:
            os.unlink(fname)

    @unittest.skipUnless(HAS_KVIKIO and HAS_CUDA, "kvikio and CUDA required")
    def test_gds_load_file_convenience(self):
        """load_file with use_gds=True should work end-to-end."""
        from safetensors.torch import load_file

        tensors = {
            "a": torch.randn(10, 20),
            "b": torch.zeros(5),
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            fname = f.name
        try:
            save_file_pt(tensors, fname)
            loaded = load_file(fname, device="cuda:0", use_gds=True)

            self.assertIn("a", loaded)
            self.assertIn("b", loaded)
            self.assertTrue(loaded["a"].is_cuda)
            torch.testing.assert_close(loaded["a"].cpu(), tensors["a"])
            torch.testing.assert_close(loaded["b"].cpu(), tensors["b"])
        finally:
            os.unlink(fname)

    @unittest.skipUnless(HAS_KVIKIO and HAS_CUDA, "kvikio and CUDA required")
    def test_gds_multiple_dtypes(self):
        """GDS should handle various tensor dtypes correctly."""
        tensors = {
            "f32": torch.randn(8, 8, dtype=torch.float32),
            "f16": torch.randn(4, 4, dtype=torch.float16),
            "i32": torch.randint(0, 100, (4, 4), dtype=torch.int32),
            "bf16": torch.randn(4, 4, dtype=torch.bfloat16),
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            fname = f.name
        try:
            save_file_pt(tensors, fname)
            with safe_open(fname, framework="pt", device="cuda:0", use_gds=True) as f:
                for name, original in tensors.items():
                    loaded = f.get_tensor(name)
                    self.assertTrue(loaded.is_cuda, f"{name} should be on CUDA")
                    self.assertEqual(loaded.dtype, original.dtype, f"{name} dtype mismatch")
                    torch.testing.assert_close(loaded.cpu(), original)
        finally:
            os.unlink(fname)

    def test_gds_without_kvikio_raises(self):
        """Requesting GDS without kvikio installed should raise an ImportError."""
        if HAS_KVIKIO:
            self.skipTest("kvikio is installed; cannot test missing import")

        tensors = {"test": torch.randn(4, 4)}
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            fname = f.name
        try:
            save_file_pt(tensors, fname)
            with self.assertRaises(Exception):
                # Should fail during kvikio import
                safe_open(fname, framework="pt", device="cpu", use_gds=True)
        finally:
            os.unlink(fname)


if __name__ == "__main__":
    unittest.main()
