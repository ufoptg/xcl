import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "re"))
import pe_exports


SCRIPT = Path(__file__).resolve().parent / "re" / "pe_exports.py"


def build_synthetic_pe(magic=0x20B, with_exports=True):
    image = bytearray(0x400)
    image[:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, 0x80)
    image[0x80:0x84] = b"PE\0\0"
    optional_size = 0xF0 if magic == 0x20B else 0xE0
    machine = 0x8664 if magic == 0x20B else 0x14C
    struct.pack_into("<HHIIIHH", image, 0x84, machine, 1, 0, 0, 0, optional_size, 0x2002)
    optional = 0x98
    struct.pack_into("<H", image, optional, magic)
    struct.pack_into("<II", image, optional + 32, 0x1000, 0x200)
    struct.pack_into("<II", image, optional + 56, 0x2000, 0x200)
    directory = optional + (112 if magic == 0x20B else 96)
    struct.pack_into("<I", image, directory - 4, 16)
    if with_exports:
        struct.pack_into("<II", image, directory, 0x1000, 0x100)
    else:
        struct.pack_into("<II", image, directory, 0, 0)
    section = optional + optional_size
    image[section:section + 8] = b".edata\0\0"
    struct.pack_into("<IIII", image, section + 8, 0x200, 0x1000, 0x200, 0x200)
    struct.pack_into("<I", image, section + 36, 0x40000040)
    if with_exports:
        struct.pack_into("<7I", image, 0x20C, 0x1080, 7, 3, 3, 0x1040, 0x1050, 0x1060)
        struct.pack_into("<3I", image, 0x240, 0x1100, 0x1110, 0x10C0)
        struct.pack_into("<3I", image, 0x250, 0x1090, 0x1098, 0x10A0)
        struct.pack_into("<3H", image, 0x260, 0, 0, 2)
        for offset, text in ((0x280, b"synthetic.dll\0"), (0x290, b"Alpha\0"),
                             (0x298, b"Beta\0"), (0x2A0, b"Forwarded\0"),
                             (0x2C0, b"other.Target\0")):
            image[offset:offset + len(text)] = text
    return bytes(image)


class ExportAliasTests(unittest.TestCase):
    def test_all_names_for_an_ordinal_are_printed(self):
        for magic in (0x10B, 0x20B):
            with self.subTest(magic=hex(magic)), tempfile.TemporaryDirectory() as tmp:
                image = build_synthetic_pe(magic=magic, with_exports=True)
                source = Path(tmp) / "synthetic.dll"
                source.write_bytes(image)
                result = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), str(source)],
                    capture_output=True, text=True, check=True,
                )
                self.assertRegex(result.stdout, r"(?m)^  #7\s+Alpha, Beta @0x1100$")
                self.assertRegex(result.stdout, r"(?m)^  #8\s+\(ordinal only\) @0x1110$")
                self.assertRegex(result.stdout, r"(?m)^  #9\s+Forwarded @0x10c0 -> other.Target$")

    def test_pe_without_exports_prints_no_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = build_synthetic_pe(with_exports=False)
            source = Path(tmp) / "no_exports.dll"
            source.write_bytes(image)
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(source)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("== " + str(source), result.stdout)
            self.assertIn("(no exports)", result.stdout)

    def test_parse_exports_direct_api(self):
        image = build_synthetic_pe(with_exports=True)
        res = pe_exports.parse_exports(image)
        self.assertIsNotNone(res)
        dll_name, base, nfun, nnames, exports = res
        self.assertEqual(dll_name, "synthetic.dll")
        self.assertEqual(base, 7)
        self.assertEqual(nfun, 3)
        self.assertEqual(nnames, 3)
        self.assertEqual(len(exports), 3)

        ord7, names7, rva7, fwd7 = exports[0]
        self.assertEqual(ord7, 7)
        self.assertEqual(names7, ["Alpha", "Beta"])
        self.assertEqual(rva7, 0x1100)
        self.assertIsNone(fwd7)

        ord8, names8, rva8, fwd8 = exports[1]
        self.assertEqual(ord8, 8)
        self.assertEqual(names8, [])
        self.assertEqual(rva8, 0x1110)
        self.assertIsNone(fwd8)

        ord9, names9, rva9, fwd9 = exports[2]
        self.assertEqual(ord9, 9)
        self.assertEqual(names9, ["Forwarded"])
        self.assertEqual(rva9, 0x10C0)
        self.assertEqual(fwd9, "other.Target")


class CliHardeningTests(unittest.TestCase):
    def test_help_flag_prints_usage_and_exits_zero(self):
        for flag in ("-h", "--help"):
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), flag],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("usage: python pe_exports.py", result.stdout)

    def test_missing_arguments_prints_usage_and_exits_two(self):
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage: python pe_exports.py", result.stdout)

    def test_missing_file_reports_error_and_exits_one(self):
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "nonexistent_file.dll"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("error: file not found: 'nonexistent_file.dll'", result.stderr)

    def test_invalid_pe_header_reports_error_and_exits_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "corrupt.dll"
            bad.write_bytes(b"NOT_A_PE_BINARY")
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(bad)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("not a valid PE", result.stderr)

    def test_main_function_programmatic(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = build_synthetic_pe(with_exports=True)
            source = Path(tmp) / "synthetic.dll"
            source.write_bytes(image)

            stdout = io.StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = stdout
                ret = pe_exports.main([str(source)])
            finally:
                sys.stdout = old_stdout

            self.assertEqual(ret, 0)
            self.assertIn("synthetic.dll: 3 functions, 3 names, ordinal base 7", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
