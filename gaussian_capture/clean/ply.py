"""Reading and writing splat PLY files."""

import os
import numpy as np


# ----------------------------------------------------------------------
# Clean Splat (1.2.0): remove the splats in empty space from a trained
# splat. Every pixel of the model depth seen by a dataset camera says: up to
# the surface the ray is empty. Idea of the maintainer (26.09.2026);
# prototype and measurement: _CLAUDE_/260926_priors_test (carve.py).
# ----------------------------------------------------------------------
def _gcapture_ply_read(path):
    """Read a standard 3DGS PLY: binary little endian, one element 'vertex',
    float properties only (LichtFeld, Postshot, ...). Returns a dict with
    header (lines without 'ply' and 'end_header'), names, rows (float32 N x P).
    Anything else -> ValueError with the reason."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("%s is not a PLY file" % os.path.basename(path))
        header, names, n, fmt = [], [], None, None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header has no end_header")
            text = line.decode("ascii", "surrogateescape").strip()
            if text == "end_header":
                break
            header.append(text)
            parts = text.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1] if len(parts) > 1 else None
            elif parts[0] == "element":
                if parts[1] != "vertex" or n is not None:
                    raise ValueError("PLY has an element '%s' - only splats "
                                     "(one element 'vertex') can be cleaned"
                                     % parts[1])
                n = int(parts[2])
            elif parts[0] == "property":
                if parts[1] not in ("float", "float32"):
                    raise ValueError("PLY property '%s' is %s - only float "
                                     "properties are supported"
                                     % (parts[-1], parts[1]))
                names.append(parts[-1])
        if fmt != "binary_little_endian":
            raise ValueError("PLY format '%s' - only binary_little_endian is "
                             "supported" % fmt)
        if n is None or not names:
            raise ValueError("PLY has no splats (element 'vertex')")
        for need in ("x", "y", "z"):
            if need not in names:
                raise ValueError("PLY has no property '%s'" % need)
        rows = np.fromfile(f, dtype="<f4", count=n * len(names))
    if rows.size != n * len(names):
        raise ValueError("PLY is truncated (%d of %d values)"
                         % (rows.size, n * len(names)))
    return {"header": header, "names": names,
            "rows": rows.reshape(n, len(names))}


def _gcapture_ply_write(path, ply, keep, comment):
    """Write only the rows with keep=True; the header stays unchanged except
    for the new count and one comment line. Writes <path>.part first, then
    replaces -- if writing fails, no half-written file is left."""
    rows = ply["rows"][keep]
    out = ["ply"]
    for text in ply["header"]:
        if text.startswith("element vertex"):
            out.append("comment " + comment)
            out.append("element vertex %d" % len(rows))
        else:
            out.append(text)
    out.append("end_header")
    tmp = path + ".part"
    try:
        with open(tmp, "wb") as f:
            f.write(("\n".join(out) + "\n").encode("ascii", "surrogateescape"))
            f.write(np.ascontiguousarray(rows, dtype="<f4").tobytes())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
