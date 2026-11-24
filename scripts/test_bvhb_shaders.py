import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "re"))
import bvhb_shaders


SCRIPT = Path(__file__).resolve().parent / "re" / "bvhb_shaders.py"


def build_synthetic_dxbc(chunk_type=b"RTS0", chunk_data=b""):
    chunk = chunk_type + struct.pack("<I", len(chunk_data)) + chunk_data
    blob = struct.pack("<4s16sIII", b"DXBC", bytes(16), 1, 36 + len(chunk), 1)
    blob += struct.pack("<I", 36) + chunk
    return blob


class RootDescriptorTests(unittest.TestCase):
    def test_root_descriptors_print_their_register_classes(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                descriptors = ((2, 7), (3, 11), (4, 13))
                payload_size = 12 if version == 2 else 8
                payload_offset = 24 + 12 * len(descriptors)
                root = bytearray(struct.pack("<6I", version, len(descriptors), 24, 0, 0, 0))
                for index, (kind, register) in enumerate(descriptors):
                    root += struct.pack("<3I", kind, 0, payload_offset + index * payload_size)
                for kind, register in descriptors:
                    root += struct.pack("<2I", register, 0)
                    if version == 2:
                        root += struct.pack("<I", 0)
                blob = build_synthetic_dxbc(b"RTS0", bytes(root))
                source = Path(tmp) / "synthetic.dll"
                source.write_bytes(blob)
                result = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), str(source), str(Path(tmp) / "out")],
                    capture_output=True, text=True, check=True,
                )
                self.assertIn("CBV b7 | SRV t11 | UAV u13", result.stdout)

    def test_shaders_without_rts0_prints_hyphen(self):
        with tempfile.TemporaryDirectory() as tmp:
            blob = build_synthetic_dxbc(b"SHEX", b"\0" * 16)
            source = Path(tmp) / "synthetic.dll"
            source.write_bytes(blob)
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(source), str(Path(tmp) / "out")],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("RS: -", result.stdout)

    def test_extract_shaders_direct_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            blob = build_synthetic_dxbc(b"SHEX", b"\0" * 16)
            out_dir = Path(tmp) / "out"
            results = list(bvhb_shaders.extract_shaders(blob, out_dir=str(out_dir)))
            self.assertEqual(len(results), 1)
            k, offset, size, names, rs_desc, extracted_blob = results[0]
            self.assertEqual(k, 0)
            self.assertEqual(offset, 0)
            self.assertEqual(size, len(blob))
            self.assertEqual(names, ["SHEX"])
            self.assertIsNone(rs_desc)
            self.assertTrue((out_dir / "00.dxbc").is_file())


class CliHardeningTests(unittest.TestCase):
    def test_help_flag_prints_usage_and_exits_zero(self):
        for flag in ("-h", "--help"):
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), flag],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("usage: python bvhb_shaders.py", result.stdout)

    def test_missing_arguments_prints_usage_and_exits_two(self):
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage: python bvhb_shaders.py", result.stdout)

    def test_missing_file_reports_error_and_exits_one(self):
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "nonexistent_bvhb.dll"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("error: file not found: 'nonexistent_bvhb.dll'", result.stderr)

    def test_main_programmatic_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            blob = build_synthetic_dxbc(b"SHEX", b"\0" * 16)
            source = Path(tmp) / "synthetic.dll"
            source.write_bytes(blob)
            out_dir = Path(tmp) / "out"

            stdout = io.StringIO()
            ret = bvhb_shaders.main([str(source), str(out_dir)], out=stdout)
            self.assertEqual(ret, 0)
            self.assertIn("00 @0x0", stdout.getvalue())
            self.assertIn("RS: -", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
