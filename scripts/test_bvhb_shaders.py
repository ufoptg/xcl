from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parent / "re" / "bvhb_shaders.py"


def summarize_root(root, folder):
    chunk = b"RTS0" + struct.pack("<I", len(root)) + root
    blob = struct.pack("<4s16sIII", b"DXBC", bytes(16), 1, 36 + len(chunk), 1)
    blob += struct.pack("<I", 36) + chunk
    source = folder / "synthetic.dll"
    source.write_bytes(blob)
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT), str(source), str(folder / "out")],
        capture_output=True, text=True, check=True,
    ).stdout


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
                self.assertIn("CBV b7 | SRV t11 | UAV u13", summarize_root(root, Path(tmp)))

    def test_distinguishes_bindings_in_different_register_spaces(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                constants = struct.pack("<3I", 0, 2, 4)
                descriptor = struct.pack("<2I", 0, 3)
                if version == 2:
                    descriptor += struct.pack("<I", 0)
                constants_at = 24 + 3 * 12
                descriptor_at = constants_at + len(constants)
                table_at = descriptor_at + len(descriptor)
                root = bytearray(struct.pack("<6I", version, 3, 24, 0, 0, 0))
                for kind, offset in ((1, constants_at), (2, descriptor_at), (0, table_at)):
                    root += struct.pack("<3I", kind, 0, offset)
                root += constants + descriptor + struct.pack("<2I", 2, table_at + 8)
                for space, table_offset in ((0, 0), (5, 1)):
                    root += struct.pack("<4I", 0, 1, 0, space)
                    if version == 2:
                        root += struct.pack("<I", 0)
                    root += struct.pack("<I", table_offset)
                text = summarize_root(root, Path(tmp))
                self.assertIn("consts b0 space2 x4 | CBV b0 space3 | table[SRVx1@0,SRVx1@0 space5]", text)


if __name__ == "__main__":
    unittest.main()
