"""Extract the DXBC containers embedded in XboxBvhBuilder.dll and print each one's root
signature (RTS0) parameters, with b/t/u registers for root CBV/SRV/UAV descriptors.

usage: python bvhb_shaders.py [-h] <XboxBvhBuilder.dll> [out dir, default ./bvhb_shaders]
"""
import os
import struct
import sys

TYPES = {0: 'table', 1: 'consts', 2: 'CBV', 3: 'SRV', 4: 'UAV'}
RANGE = {0: 'SRV', 1: 'UAV', 2: 'CBV', 3: 'Sampler'}


def parts(blob):
    if len(blob) < 32:
        return
    n = struct.unpack_from('<I', blob, 28)[0]
    for k in range(n):
        if len(blob) < 32 + 4 * (k + 1):
            return
        off = struct.unpack_from('<I', blob, 32 + 4 * k)[0]
        if len(blob) < off + 8:
            continue
        cc = blob[off:off + 4].decode('latin1', errors='replace')
        size = struct.unpack_from('<I', blob, off + 4)[0]
        if len(blob) < off + 8 + size:
            continue
        yield cc, blob[off + 8:off + 8 + size]


def rts0(r):
    if len(r) < 24:
        return None
    ver, n, poff, ns, soff, flags = struct.unpack_from('<6I', r, 0)
    out = []
    for k in range(n):
        param_off = poff + 12 * k
        if len(r) < param_off + 12:
            break
        ty, vis, pay = struct.unpack_from('<3I', r, param_off)
        if ty == 0:
            if len(r) < pay + 8:
                break
            nr, roff = struct.unpack_from('<2I', r, pay)
            rs = []
            stride = 24 if ver >= 2 else 20
            for j in range(nr):
                range_off = roff + stride * j
                if len(r) < range_off + (24 if ver >= 2 else 20):
                    break
                w = struct.unpack_from('<6I' if ver >= 2 else '<5I', r, range_off)
                range_name = RANGE.get(w[0], f'?{w[0]}')
                count_str = f'{w[1]}' if w[1] != 0xFFFFFFFF else '∞'
                rs.append(f'{range_name}x{count_str}@{w[2]}')
            out.append('table[' + ','.join(rs) + ']')
        elif ty == 1:
            if len(r) < pay + 12:
                break
            reg, space, num = struct.unpack_from('<3I', r, pay)
            out.append(f'consts b{reg} x{num}')
        else:
            if len(r) < pay + 8:
                break
            reg, space = struct.unpack_from('<2I', r, pay)
            type_name = TYPES.get(ty, f'?{ty}')
            reg_class = "btu"[ty - 2] if 2 <= ty < 5 else '?'
            out.append(f'{type_name} {reg_class}{reg}')
    return f'v{ver} flags {flags:#x} samplers {ns}: ' + ' | '.join(out)


def extract_shaders(b, out_dir=None):
    """Find and extract DXBC shader blobs from raw binary data.

    Yields (index, offset, size, names, rts0_desc, blob).
    If out_dir is specified, saves each blob to '{index:02d}.dxbc'.
    """
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    i = b.find(b'DXBC')
    k = 0
    while i >= 0:
        if len(b) >= i + 28:
            size = struct.unpack_from('<I', b, i + 24)[0]
            if 32 < size < 4 << 20 and i + size <= len(b):
                blob = b[i:i + size]
                names = [cc for cc, _ in parts(blob)]
                rs = next((p for cc, p in parts(blob) if cc == 'RTS0'), None)
                rs_desc = rts0(rs) if rs else None
                if out_dir is not None:
                    path = os.path.join(out_dir, f'{k:02d}.dxbc')
                    with open(path, 'wb') as f:
                        f.write(blob)
                yield k, i, size, names, rs_desc, blob
                k += 1
        i = b.find(b'DXBC', i + 4)


def main(argv=None, out=None):
    if argv is None:
        argv = sys.argv[1:]
    if out is None:
        out = sys.stdout

    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__.strip(), file=out)
        return 0 if (argv and argv[0] in ('-h', '--help')) else 2

    dll_path = argv[0]
    out_dir = argv[1] if len(argv) > 1 else 'bvhb_shaders'

    if not os.path.isfile(dll_path):
        print(f"error: file not found: '{dll_path}'", file=sys.stderr)
        return 1

    try:
        with open(dll_path, 'rb') as f:
            b = f.read()
    except Exception as e:
        print(f"error: failed to read '{dll_path}': {e}", file=sys.stderr)
        return 1

    for k, i, size, names, rs_desc, _ in extract_shaders(b, out_dir=out_dir):
        rs_str = rs_desc if rs_desc else '-'
        print(f'{k:02d} @{i:#x} {size:6} B {",".join(names)}  RS: {rs_str}', file=out)

    return 0


if __name__ == '__main__':
    sys.exit(main())
