"""See-through (glass) detection of materials."""


_GCAPTURE_SEE_THROUGH_BSDF = {'BSDF_GLASS', 'BSDF_TRANSPARENT', 'BSDF_REFRACTION'}


def _gcapture_tree_see_through(tree, starts, depth=0):
    """True if a node reachable backwards from starts is glass-like: a Glass,
    Transparent or Refraction BSDF, or a Principled BSDF with Transmission
    >= 0.5 (not driven by a link). Node groups are followed (Mecabricks keeps
    its glass in mb_base_transparent). Unconnected nodes do not count."""
    if depth > 6:
        return False
    stack, seen = list(starts), set()
    while stack:
        n = stack.pop()
        if n.name in seen:
            continue
        seen.add(n.name)
        if n.type in _GCAPTURE_SEE_THROUGH_BSDF:
            return True
        if n.type == 'BSDF_PRINCIPLED':
            sock = n.inputs.get("Transmission Weight") or n.inputs.get("Transmission")
            if sock is not None and not sock.is_linked and sock.default_value >= 0.5:
                return True
        if n.type == 'GROUP' and n.node_tree is not None:
            outs = [g for g in n.node_tree.nodes if g.type == 'GROUP_OUTPUT']
            if _gcapture_tree_see_through(n.node_tree, outs, depth + 1):
                return True
        for sock in n.inputs:
            for link in sock.links:
                stack.append(link.from_node)
    return False


def _gcapture_is_see_through(mat):
    """Glass-like material (1.3.0): the cameras see through it. The
    visibility filter of the start points then keeps what lies behind it,
    e.g. the interior of a car behind its windows."""
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return False
    nodes = mat.node_tree.nodes
    outs = [n for n in nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output]
    outs = outs or [n for n in nodes if n.type == 'OUTPUT_MATERIAL']
    return _gcapture_tree_see_through(mat.node_tree, outs)


def _gcapture_see_through_slots(obj, cache):
    """Per material slot of obj: see-through? None if no slot is (fast path)."""
    flags = []
    for sl in obj.material_slots:
        mat = sl.material
        key = mat.name if mat is not None else ""
        if key not in cache:
            cache[key] = _gcapture_is_see_through(mat)
        flags.append(cache[key])
    return flags if any(flags) else None
