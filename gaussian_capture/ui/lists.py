"""UILists of the guide and look-target entries."""

import bpy


# ----------------------------------------------------------------------
# UIList for the guide entries (editable label + object count)
# ----------------------------------------------------------------------
class GCAPTURE_UL_guides(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        n = len(item.objects)
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            # Editable label (double-click to rename).
            layout.prop(item, "label", text="", emboss=False,
                        icon='MESH_ICOSPHERE')
            sub = layout.row()
            sub.alignment = 'RIGHT'
            sub.label(text=("%d objs" % n) if n != 1 else "1 obj")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.label)


class GCAPTURE_UL_colls(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            if item.coll is not None:
                layout.label(text=item.coll.name, icon='OUTLINER_COLLECTION')
            else:
                layout.label(text="(empty)", icon='ERROR')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.coll.name if item.coll else "(empty)")
