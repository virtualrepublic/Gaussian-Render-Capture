"""Scene versions, output paths, msgbus, load/save handlers, migration of old scenes."""

import bpy
import os
import re
from bpy.app.handlers import persistent

from ..core import progress


# ----------------------------------------------------------------------
# Beginner aids (v90): selection into a collection, scene versions
# ----------------------------------------------------------------------
def _gcapture_version_files(folder, name):
    """Existing version numbers of <name>_vNNN*.blend in the folder."""
    found = []
    if not folder or not os.path.isdir(folder):
        return found
    rx = re.compile(r"^%s_[vV](\d+)(?:[_\-.].*)?\.blend$" % re.escape(name))
    for fn in os.listdir(folder):
        m = rx.match(fn)
        if m:
            found.append(int(m.group(1)))
    return found


def _gcapture_next_version_path(folder, name, width=3):
    """Path of the next free version <name>_vNNN.blend in the folder."""
    nums = _gcapture_version_files(folder, name)
    nxt = (max(nums) + 1) if nums else 1
    return os.path.join(folder, "%s_v%0*d.blend" % (name, width, nxt))


def _gcapture_clean_name(raw):
    """Scene name without extension, version token and invalid characters."""
    stem = os.path.splitext(os.path.basename(raw or ""))[0]
    name, _ = _exp_split_name_version(stem)
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("_.- ")
    return name or "Scene"


def _gcapture_managed_render_path(name, vtag):
    """Relative render path directly into the dataset:
    //<Name>_COLMAP/<vNNN>/images/<Name>_<vNNN>_"""
    return "//%s_COLMAP/%s/images/%s_%s_" % (name, vtag, name, vtag)

_GCAPTURE_MANAGED_RX = re.compile(
    r"^//(?P<n>[^/]+)_COLMAP/(?P<v>[vV]\d+)/images/(?P=n)_(?P=v)_$")


def _gcapture_sync_render_version(filepath):
    """Sets a render path managed by the addon to the file's name and
    version (v119). Returns: number of changed scenes."""
    stem = os.path.splitext(os.path.basename(filepath or ""))[0]
    name, vtag = _exp_split_name_version(stem)
    if not vtag:
        return 0
    name = _gcapture_clean_name(name)
    changed = 0
    for scene in bpy.data.scenes:
        rp = scene.render.filepath.replace("\\", "/")
        m = _GCAPTURE_MANAGED_RX.match(rp)
        if not m or (m.group("n"), m.group("v")) == (name, vtag):
            continue
        scene.render.filepath = _gcapture_managed_render_path(name, vtag)
        changed += 1
        print("[Gaussian Render Capture] render output follows %s: %s"
              % (os.path.basename(filepath), scene.render.filepath))
    return changed


_GCAPTURE_OUT_RX = re.compile(r"^//[^/]+_COLMAP/[vV]\d+/?$")


def _gcapture_render_path_is_default_or_managed(scene):
    """True if the render path is Blender's default or is managed by the
    addon -- then Save Scene Version may set it (v129)."""
    rp = (scene.render.filepath or "").replace("\\", "/").strip()
    return (rp in ("", "/tmp", "/tmp/", "//")
            or bool(_GCAPTURE_MANAGED_RX.match(rp)))


def _gcapture_auto_output_dir(filepath):
    """//<Name>_COLMAP/<vNNN>/ for the file (like _exp_resolve_output_dir)."""
    if not filepath:
        return ""
    stem = os.path.splitext(os.path.basename(filepath))[0]
    name, vtag = _exp_split_name_version(stem)
    return "//%s_COLMAP/%s/" % (name or "Scene", vtag or "v001")


def _gcapture_auto_image_dir(scene):
    """Folder of the render path as it is set in the scene (relative stays
    relative)."""
    rp = (scene.render.filepath or "").replace("\\", "/")
    if not rp.strip():
        return ""
    return rp if rp.endswith("/") else rp.rsplit("/", 1)[0] + "/"


def _gcapture_sync_path_fields(scene, filepath):
    """Output Folder and Image Folder show the actual paths and
    follow version and render path as long as they hold the value entered
    by the addon (v129). Unsaved scenes stay empty (= automatic)."""
    s = getattr(scene, "gcapture_settings", None)
    if s is None or not filepath:
        return
    out_auto = _gcapture_auto_output_dir(filepath)
    cur = (s.exp_output_dir or "").replace("\\", "/").strip()
    if not cur or cur == s.exp_output_dir_auto or _GCAPTURE_OUT_RX.match(cur):
        if s.exp_output_dir != out_auto:
            s.exp_output_dir = out_auto
        if s.exp_output_dir_auto != out_auto:
            s.exp_output_dir_auto = out_auto
    img_auto = _gcapture_auto_image_dir(scene)
    cur = (s.exp_image_dir or "").replace("\\", "/").strip()
    if img_auto and (not cur or cur == s.exp_image_dir_auto):
        if s.exp_image_dir != img_auto:
            s.exp_image_dir = img_auto
        if s.exp_image_dir_auto != img_auto:
            s.exp_image_dir_auto = img_auto


def _gcapture_sync_all(filepath):
    _gcapture_sync_render_version(filepath)
    for scene in bpy.data.scenes:
        _gcapture_sync_path_fields(scene, filepath)


_GCAPTURE_MSGBUS_OWNER = object()


def _gcapture_on_render_path_change(*args):
    for scene in bpy.data.scenes:
        _gcapture_sync_path_fields(scene, bpy.data.filepath)


def _gcapture_msgbus_subscribe():
    """Apply changes to the render path to Image Folder immediately (v129).
    Must be re-subscribed after every load."""
    bpy.msgbus.clear_by_owner(_GCAPTURE_MSGBUS_OWNER)
    bpy.msgbus.subscribe_rna(key=(bpy.types.RenderSettings, "filepath"),
                             owner=_GCAPTURE_MSGBUS_OWNER, args=(),
                             notify=_gcapture_on_render_path_change)


# Migration from "Gaussian Render Scan" (up to 1.0.2, 23.09.2026): the same
# settings and markers under the new name. Runs after every
# load and on activation; without legacy data it changes nothing.
_GCAPTURE_LEGACY_KEYS = (("gscan_prepared", "gcapture_prepared"),
                         ("gscan_res_ok", "gcapture_res_ok"),
                         ("gscan_base_loc", "gcapture_base_loc"),
                         ("gscan_center", "gcapture_center"))
_GCAPTURE_LEGACY_ATTRS = ("fit", "shift", "adj", "adj_pos", "adj_tgt")
_GCAPTURE_LEGACY_NAMES = (("GScan_Group", "GCapture_Group", "objects"),
                          ("Scan_Rig", "Capture_Rig", "collections"))


def _gcapture_migrate_legacy():
    """Migrate scenes from Gaussian Render Scan. From Blender 5.0 the settings
    live in the system properties, before that in the scene's dict.
    Returns: number of migrated entries."""
    n = 0
    for scene in bpy.data.scenes:
        sysp = getattr(scene, "bl_system_properties_get", None)
        g = sysp() if sysp else None
        dst = g if g is not None else scene
        for store in (g, scene):
            if store is None or "fco_settings" not in store:
                continue
            old = store["fco_settings"]
            if "gcapture_settings" not in dst and hasattr(old, "to_dict"):
                dst["gcapture_settings"] = old.to_dict()
                n += 1
            del store["fco_settings"]
    for data in (bpy.data.scenes, bpy.data.objects):
        for idb in data:
            for a, b in _GCAPTURE_LEGACY_KEYS:
                if a in idb:
                    if b not in idb:
                        idb[b] = idb[a]
                    del idb[a]
                    n += 1
    for me in bpy.data.meshes:
        for suffix in _GCAPTURE_LEGACY_ATTRS:
            attr = me.attributes.get("gscan_" + suffix)
            if attr is not None and me.attributes.get("gcapture_" + suffix) is None:
                attr.name = "gcapture_" + suffix
                n += 1
    # Object and collection names only if the file comes from the old add-on.
    if n:
        for a, b, kind in _GCAPTURE_LEGACY_NAMES:
            data = getattr(bpy.data, kind)
            idb = data.get(a)
            if idb is not None and data.get(b) is None:
                idb.name = b
                n += 1
        print("[Gaussian Render Capture] scene of Gaussian Render Scan taken "
              "over (%d entries)" % n)
    return n


def _gcapture_migrate_timer():
    try:
        _gcapture_migrate_legacy()
    except Exception as exc:
        print("[Gaussian Render Capture] migration:", exc)
    return None


@persistent
def _gcapture_on_load_post(*args):
    progress._GCAPTURE_PROGRESS.clear()   # a new file has no running operation
    try:
        _gcapture_migrate_legacy()
    except Exception as exc:
        print("[Gaussian Render Capture] migration:", exc)
    _gcapture_sync_all(bpy.data.filepath)
    _gcapture_msgbus_subscribe()


@persistent
def _gcapture_on_save_pre(*args):
    # Blender passes the target path (Save As); otherwise the current one.
    target = next((a for a in args if isinstance(a, str) and a), bpy.data.filepath)
    _gcapture_sync_all(target)


def _exp_blend_basename():
    path = bpy.data.filepath
    if not path:
        return None
    stem, _ = os.path.splitext(os.path.basename(path))
    return stem


def _exp_split_name_version(stem):
    """Splits the .blend base name into (name_without_version, version token).
    Example: 'Scene_v001' -> ('Scene', 'v001'); 'Bugatti_Chiron_v002_Addon'
    -> ('Bugatti_Chiron', 'v002'). Takes the LAST vN token. Without a version
    -> (stem, None). Separators before the version are stripped."""
    matches = list(re.finditer(r'[vV]\d+', stem))
    if not matches:
        return stem, None
    m = matches[-1]
    name = stem[:m.start()].rstrip('_-. ')
    version = stem[m.start():m.end()]
    return name, version


def _exp_resolve_output_dir(s):
    """Returns the target folder for the COLMAP model.

    If the 'Output Folder' field is EMPTY (default), a structure is built
    automatically next to the .blend file:
        <Name>_COLMAP / <vNNN> /
    i.e. a folder named after the scene (up to the version token) plus
    '_COLMAP', and inside it a subfolder with the version token. Example:
    'Scene_v001.blend' -> 'Scene_COLMAP/v001/'. Without a version,
    'v001' is used; if the .blend is not saved yet, 'Scene_COLMAP'.

    If the field is filled in, the path is used as-is (// = relative to
    the .blend is resolved)."""
    od = (s.exp_output_dir or "").strip()
    if od:
        return bpy.path.abspath(od)
    # Empty field -> automatic structure next to the .blend.
    base = bpy.path.abspath("//") if bpy.data.filepath else os.getcwd()
    stem = _exp_blend_basename()
    if stem:
        name, version = _exp_split_name_version(stem)
    else:
        name, version = "Scene", None
    folder = "%s_COLMAP" % (name if name else "Scene")
    sub = version if version else "v001"
    return os.path.join(base, folder, sub)


def _exp_output_has_export(out_dir):
    """True if the target folder already contains a COLMAP export (one of
    the sparse/0 files). Images alone do not count: since v90 the scene
    renders directly to images/, which is not an export yet."""
    if not out_dir or not os.path.isdir(out_dir):
        return False
    sparse = os.path.join(out_dir, "sparse", "0")
    for fn in ("cameras.txt", "images.txt", "points3D.txt"):
        if os.path.isfile(os.path.join(sparse, fn)):
            return True
    return False


def _exp_same_dir(a, b):
    """True if both paths point to the same folder."""
    if not a or not b:
        return False
    return (os.path.normcase(os.path.normpath(os.path.abspath(a))) ==
            os.path.normcase(os.path.normpath(os.path.abspath(b))))
