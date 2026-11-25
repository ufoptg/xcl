"""Print a PE's exports: ordinal, all names for that ordinal, RVA (forwarders shown).

usage: python pe_exports.py [-h] <pe_file> [<pe_file> ...]
"""
import os
import struct
import sys


def parse_exports(b):
    """Parse export table from PE binary bytes.

    Returns (dll_name, base, nfun, nnames, [(ordinal, names, rva, forwarder), ...])
    or None if no exports directory is present.
    Raises ValueError on malformed PE data.
    """
    if len(b) < 0x40 or b[:2] != b"MZ":
        raise ValueError("not a valid PE: missing MZ signature or truncated header")
    pe = struct.unpack_from("<I", b, 0x3C)[0]
    if len(b) < pe + 24 or b[pe:pe + 4] != b"PE\0\0":
        raise ValueError("not a valid PE: missing PE signature")
    nsec = struct.unpack_from("<H", b, pe + 6)[0]
    opt = pe + 24
    if len(b) < opt + 2:
        raise ValueError("truncated optional header")
    magic = struct.unpack_from("<H", b, opt)[0]
    if magic not in (0x10B, 0x20B):
        raise ValueError(f"unsupported PE optional header magic: {hex(magic)}")
    dd = opt + (112 if magic == 0x20B else 96)
    if len(b) < dd + 8:
        raise ValueError("truncated data directories")
    exp_rva, exp_size = struct.unpack_from("<II", b, dd)
    secs = []
    so = opt + struct.unpack_from("<H", b, pe + 20)[0]
    for i in range(nsec):
        sec_off = so + 40 * i
        if len(b) < sec_off + 40:
            raise ValueError(f"truncated section header {i}")
        vs, va, raw_sz, raw = struct.unpack_from("<IIII", b, sec_off + 8)
        secs.append((va, max(vs, raw_sz), raw))

    def off(rva):
        for va, sz, raw in secs:
            if va <= rva < va + sz:
                file_off = rva - va + raw
                if file_off >= len(b):
                    raise ValueError(f"RVA {hex(rva)} mapped to offset {hex(file_off)} beyond file size")
                return file_off
        raise ValueError(f"RVA {hex(rva)} outside any section")

    def cstr(rva):
        o = off(rva)
        null_pos = b.find(b"\0", o)
        if null_pos == -1:
            raise ValueError(f"unterminated string at offset {hex(o)}")
        return b[o:null_pos].decode("latin1")

    if not exp_rva:
        return None

    e = off(exp_rva)
    if len(b) < e + 40:
        raise ValueError("truncated export directory")
    name_rva, base, nfun, nnames, afun, anames, aords = struct.unpack_from("<IIIIIII", b, e + 12)
    dll_name = cstr(name_rva)
    names = {}
    for i in range(nnames):
        nr = struct.unpack_from("<I", b, off(anames) + 4 * i)[0]
        o = struct.unpack_from("<H", b, off(aords) + 2 * i)[0]
        names.setdefault(o, []).append(cstr(nr))

    exports = []
    for i in range(nfun):
        rva = struct.unpack_from("<I", b, off(afun) + 4 * i)[0]
        if not rva:
            continue
        fwd = cstr(rva) if exp_rva <= rva < exp_rva + exp_size else None
        exports.append((base + i, names.get(i, []), rva, fwd))

    return dll_name, base, nfun, nnames, exports


def dump_pe_exports(path, file=None):
    """Read a PE file and print its exports summary."""
    if file is None:
        file = sys.stdout
    with open(path, "rb") as f:
        b = f.read()
    res = parse_exports(b)
    print(f"== {path}", file=file)
    if res is None:
        print("  (no exports)", file=file)
        return
    dll_name, base, nfun, nnames, exports = res
    print(f"  {dll_name}: {nfun} functions, {nnames} names, ordinal base {base}", file=file)
    for ord_num, ord_names, rva, fwd in exports:
        name_str = ", ".join(ord_names) if ord_names else "(ordinal only)"
        fwd_str = f" -> {fwd}" if fwd else ""
        print(f"  #{ord_num:<4} {name_str} @{rva:#x}{fwd_str}", file=file)


def main(argv=None, out=None):
    if argv is None:
        argv = sys.argv[1:]
    if out is None:
        out = sys.stdout
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip(), file=out)
        return 0 if (argv and argv[0] in ("-h", "--help")) else 2

    has_error = False
    for path in argv:
        if not os.path.isfile(path):
            print(f"error: file not found: '{path}'", file=sys.stderr)
            has_error = True
            continue
        try:
            dump_pe_exports(path, file=out)
        except ValueError as e:
            print(f"error: '{path}': {e}", file=sys.stderr)
            has_error = True
        except Exception as e:
            print(f"error: failed to read '{path}': {e}", file=sys.stderr)
            has_error = True
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
