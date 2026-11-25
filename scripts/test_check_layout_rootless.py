from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parent / "re" / "check_layout.py"


class CheckLayoutRootlessTests(unittest.TestCase):
    def test_reports_each_shader_without_an_embedded_root_signature(self):
        chunk = struct.pack("<4sI", b"STAT", 4) + bytes(4)
        container = struct.pack("<4s16sIIII", b"DXBC", bytes(16), 1, 36 + len(chunk), 1, 36)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("00.dxbc", "01.dxbc"):
                (Path(tmp) / name).write_bytes(container + chunk)
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), tmp], capture_output=True, text=True,
            )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            [
                "00.dxbc no embedded root signature (RTS0); skipped",
                "01.dxbc no embedded root signature (RTS0); skipped",
                "loads on parameter boundaries: 0, off: 0",
            ],
            result.stdout.splitlines(),
        )


if __name__ == "__main__":
    unittest.main()
