bl_info = {
    "name": "IFC Trim/Extend Tool",
    "author": "Meelis Sinijarv, sinijarv.meelis@gmail.com",
    "version": (0, 0, 1),
    "blender": (5, 2, 1),
    "location": "View3D > Sidebar > IFC Tools",
    "description": (
        "Trim or extend IFC extrusions (and their Axes) to a snapped point or by offset. "
        "Click a destination first (optional), then click one or multiple elements near the end to modify. "
        "Draws a dashed overlay of the trimmed/extended portion with live dimension. "
        "Openings and attached products keep their world position; openings left outside the "
        "extrusion are deleted. Bonsai / IfcOpenShell 0.8.5. Undo/Redo supported."
    ),
    "category": "Import-Export",
}

import bpy
import gpu
import blf
import math
import traceback
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
import bpy_extras
import ifcopenshell
import ifcopenshell.api
import ifcopenshell.util.representation
import ifcopenshell.util.unit
import ifcopenshell.util.placement

try:
    import bonsai.tool as tool
    import bonsai.core.geometry as core_geometry
    from bonsai.bim.ifc import IfcStore
except ImportError:
    import blenderbim.tool as tool  # older Bonsai (BlenderBIM) naming
    core_geometry = None
    from blenderbim.bim.ifc import IfcStore


SNAP_RADIUS_ENDPOINT = 50  # AutoCAD-like magnet: endpoints snap from farther away
SNAP_RADIUS_MIDPOINT = 50  # same generous grab distance for edge midpoints
SNAP_RADIUS_NEAREST = 20   # nearest-on-edge stays tighter so END/MID win first
SNAP_PIXEL_RADIUS = SNAP_RADIUS_ENDPOINT  # used in "too far" warnings (max magnet)

# AutoCAD AutoSnap marker (yellow, original 7 px size = 100%)
COLOR_OSNAP = (1.0, 1.0, 0.0, 1.0)
OSNAP_SIZE = 7.0
OSNAP_WIDTH = 2.0
OSNAP_LABELS = {
    'ENDPOINT': 'Endpoint',
    'VERTEX': 'Endpoint',
    'MIDPOINT': 'Midpoint',
    'NEAREST': 'Nearest',
    'EDGE': 'Nearest',
}

# Module-level handoff so the undoable apply operator can reuse the modal's helpers
# without storing Blender ID pointers in RNA (those dangle after undo).
_PENDING_APPLY = {}


def _is_undo_hotkey(event):
    if event.value != 'PRESS':
        return False
    if event.type == 'Z' and (event.ctrl or event.oskey) and not event.shift:
        return True
    return False


def _is_redo_hotkey(event):
    if event.value != 'PRESS':
        return False
    if event.type == 'Z' and (event.ctrl or event.oskey) and event.shift:
        return True
    if event.type == 'Y' and (event.ctrl or event.oskey) and not event.shift:
        return True
    return False


# Overlay styling (screen-space pixels, CAD-like)
DASH_PX = 9.0
GAP_PX = 6.0
TICK_PX = 9.0
LINE_W_REMAIN = 3.0
LINE_W_DASH = 2.4
COLOR_REMAIN = (0.82, 0.88, 0.95, 0.88)
COLOR_EXTEND = (0.22, 0.78, 0.92, 1.0)
COLOR_TRIM = (0.98, 0.48, 0.22, 1.0)
COLOR_INVALID = (0.95, 0.22, 0.22, 1.0)
COLOR_CONSTRUCT = (0.70, 0.74, 0.78, 0.42)
COLOR_LABEL_BG = (0.06, 0.08, 0.10, 0.82)
COLOR_HINT_BG = (0.07, 0.09, 0.12, 0.78)


# ---------------------------
# Draw helpers (3D + 2D overlay)
# ---------------------------
def draw_points(points, color=(0, 1, 0, 1), size=10):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'POINTS', {"pos": points})
    shader.bind()
    shader.uniform_float("color", color)
    gpu.state.point_size_set(size)
    batch.draw(shader)


def closest_point_on_segment_2d(p, a, b):
    """Closest point on 2D segment a-b to point p. Returns (point, t)."""
    ab = b - a
    length_sq = ab.length_squared
    if length_sq < 1e-9:
        return a, 0.0
    t = (p - a).dot(ab) / length_sq
    t = max(0.0, min(1.0, t))
    return a + ab * t, t


def _ui_scale():
    try:
        return float(bpy.context.preferences.system.ui_scale) or 1.0
    except Exception:
        return 1.0


def _v2(p):
    return Vector((float(p[0]), float(p[1])))


def _v3xy(p):
    return (float(p[0]), float(p[1]), 0.0)


def _line_quad(p0, p1, width):
    """Two triangles forming a screen-space quad of `width` pixels."""
    d = p1 - p0
    length = d.length
    if length < 1e-6:
        return []
    n = Vector((-d.y, d.x)) / length * (width * 0.5)
    a = p0 + n
    b = p0 - n
    c = p1 - n
    d1 = p1 + n
    return [_v3xy(a), _v3xy(b), _v3xy(c), _v3xy(a), _v3xy(c), _v3xy(d1)]


def _draw_tris(verts, color):
    if len(verts) < 3:
        return
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRIS', {"pos": verts})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def draw_solid_px(a, b, color, width=2.0):
    a = _v2(a)
    b = _v2(b)
    _draw_tris(_line_quad(a, b, width), color)


def draw_dashed_px(a, b, color, width=2.0, dash=DASH_PX, gap=GAP_PX):
    """Screen-space dashed line. Dash/gap are in pixels so they stay even at any zoom."""
    a = _v2(a)
    b = _v2(b)
    vec = b - a
    length = vec.length
    if length < 0.5:
        return
    direction = vec / length
    verts = []
    t = 0.0
    while t < length:
        t1 = min(t + dash, length)
        p0 = a + direction * t
        p1 = a + direction * t1
        verts.extend(_line_quad(p0, p1, width))
        t += dash + gap
    _draw_tris(verts, color)


def draw_end_ticks_px(a, b, color, tick_len=TICK_PX, width=1.6):
    a = _v2(a)
    b = _v2(b)
    d = b - a
    if d.length < 1e-6:
        return
    n = Vector((-d.y, d.x)).normalized() * (tick_len * 0.5)
    verts = []
    verts.extend(_line_quad(a - n, a + n, width))
    verts.extend(_line_quad(b - n, b + n, width))
    _draw_tris(verts, color)


def draw_circle_px(center, radius, color, fill=True, segments=22, ring_width=1.6):
    c = _v2(center)
    pts = []
    for i in range(segments):
        ang = 2.0 * math.pi * i / segments
        pts.append(c + Vector((math.cos(ang), math.sin(ang))) * radius)
    if fill:
        verts = []
        cz = _v3xy(c)
        for i in range(segments):
            verts.append(cz)
            verts.append(_v3xy(pts[i]))
            verts.append(_v3xy(pts[(i + 1) % segments]))
        _draw_tris(verts, color)
        return
    verts = []
    for i in range(segments):
        verts.extend(_line_quad(pts[i], pts[(i + 1) % segments], ring_width))
    _draw_tris(verts, color)


def draw_diamond_px(center, size, color, ring_width=1.6):
    c = _v2(center)
    pts = [
        c + Vector((0.0, size)),
        c + Vector((size, 0.0)),
        c + Vector((0.0, -size)),
        c + Vector((-size, 0.0)),
    ]
    verts = []
    for i in range(4):
        verts.extend(_line_quad(pts[i], pts[(i + 1) % 4], ring_width))
    _draw_tris(verts, color)


def draw_square_px(center, size, color, ring_width=1.6):
    """AutoCAD Endpoint marker: axis-aligned square."""
    c = _v2(center)
    pts = [
        c + Vector((-size, -size)),
        c + Vector((size, -size)),
        c + Vector((size, size)),
        c + Vector((-size, size)),
    ]
    verts = []
    for i in range(4):
        verts.extend(_line_quad(pts[i], pts[(i + 1) % 4], ring_width))
    _draw_tris(verts, color)


def draw_triangle_px(center, size, color, ring_width=1.6):
    """AutoCAD Midpoint marker: equilateral triangle pointing up, centroid at center."""
    c = _v2(center)
    h = size * math.sqrt(3.0)
    pts = [
        c + Vector((0.0, (2.0 / 3.0) * h)),
        c + Vector((size, -(1.0 / 3.0) * h)),
        c + Vector((-size, -(1.0 / 3.0) * h)),
    ]
    verts = []
    for i in range(3):
        verts.extend(_line_quad(pts[i], pts[(i + 1) % 3], ring_width))
    _draw_tris(verts, color)


def draw_hourglass_px(center, size, color, ring_width=1.6):
    """AutoCAD Nearest marker: vertical hourglass (two triangles meeting at center)."""
    c = _v2(center)
    tl = c + Vector((-size, size))
    tr = c + Vector((size, size))
    bl = c + Vector((-size, -size))
    br = c + Vector((size, -size))
    verts = []
    verts.extend(_line_quad(tl, tr, ring_width))
    verts.extend(_line_quad(tr, c, ring_width))
    verts.extend(_line_quad(c, tl, ring_width))
    verts.extend(_line_quad(bl, br, ring_width))
    verts.extend(_line_quad(br, c, ring_width))
    verts.extend(_line_quad(c, bl, ring_width))
    _draw_tris(verts, color)


def _normalize_snap_kind(kind):
    if kind in {'VERTEX', 'ENDPOINT'}:
        return 'ENDPOINT'
    if kind in {'MIDPOINT', 'MID'}:
        return 'MIDPOINT'
    return 'NEAREST'


def draw_osnap_label_px(center, text, ui):
    """AutoCAD-style AutoSnap tooltip, offset to the lower-right of the marker."""
    font_id = 0
    size = 11.0 * ui
    try:
        blf.size(font_id, size)
    except TypeError:
        blf.size(font_id, size, 72)
    tw, th = blf.dimensions(font_id, text)
    c = _v2(center)
    pad = 3.0 * ui
    x = c.x + (OSNAP_SIZE * ui) + 8.0 * ui
    y = c.y - (OSNAP_SIZE * ui) - th - 2.0 * ui
    region = getattr(bpy.context, "region", None)
    if region is not None:
        x = max(4.0, min(x, region.width - tw - 8.0))
        y = max(4.0, min(y, region.height - th - 8.0))
    draw_filled_rect_px(
        x - pad, y - pad * 0.4,
        tw + 2.0 * pad, th + pad,
        (0.08, 0.08, 0.08, 0.72),
    )
    blf.position(font_id, x, y, 0)
    try:
        blf.color(font_id, COLOR_OSNAP[0], COLOR_OSNAP[1], COLOR_OSNAP[2], 1.0)
    except Exception:
        pass
    blf.draw(font_id, text)


def draw_osnap_marker(center, kind, ui):
    """Draw the AutoCAD AutoSnap glyph for the given snap kind."""
    kind = _normalize_snap_kind(kind)
    size = OSNAP_SIZE * ui
    width = OSNAP_WIDTH * ui
    color = COLOR_OSNAP
    if kind == 'ENDPOINT':
        draw_square_px(center, size, color, ring_width=width)
    elif kind == 'MIDPOINT':
        draw_triangle_px(center, size, color, ring_width=width)
    else:
        draw_hourglass_px(center, size, color, ring_width=width)
    label = OSNAP_LABELS.get(kind)
    if label:
        draw_osnap_label_px(center, label, ui)


def draw_filled_rect_px(x, y, w, h, color):
    x0, y0, x1, y1 = float(x), float(y), float(x + w), float(y + h)
    verts = [
        (x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0),
        (x0, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0),
    ]
    _draw_tris(verts, color)


def format_length_m(meters, unit_scale):
    """Pretty-print a length using the IFC file unit when it is a common SI/imperial unit."""
    meters = abs(float(meters))
    if unit_scale and unit_scale > 0:
        file_val = meters / unit_scale
        if abs(unit_scale - 0.001) < 1e-6:
            if file_val >= 10000:
                return f"{meters:.3f} m"
            rounded = round(file_val)
            if abs(file_val - rounded) < 0.05:
                return f"{int(rounded)} mm"
            return f"{file_val:.1f} mm"
        if abs(unit_scale - 0.01) < 1e-6:
            return f"{file_val:.2f} cm"
        if abs(unit_scale - 1.0) < 1e-6:
            return f"{meters:.3f} m"
        if abs(unit_scale - 0.0254) < 1e-6:
            return f"{file_val:.3f} in"
        if abs(unit_scale - 0.3048) < 1e-6:
            return f"{file_val:.3f} ft"
    if meters < 1.0:
        mm = meters * 1000.0
        rounded = round(mm)
        if abs(mm - rounded) < 0.05:
            return f"{int(rounded)} mm"
        return f"{mm:.1f} mm"
    return f"{meters:.3f} m"


def draw_dim_label_px(mid, a, b, text, color):
    """Dimension text, offset perpendicular to the dashed segment, with a dark pill."""
    mid = _v2(mid)
    a = _v2(a)
    b = _v2(b)
    d = b - a
    if d.length < 1e-3:
        n = Vector((0.0, 1.0))
    else:
        n = Vector((-d.y, d.x)).normalized()
    if n.y < 0:
        n = -n
    ui = _ui_scale()
    pos = mid + n * (16.0 * ui)

    font_id = 0
    size = 13.0 * ui
    try:
        blf.size(font_id, size)
    except TypeError:
        blf.size(font_id, size, 72)
    tw, th = blf.dimensions(font_id, text)
    pad_x = 7.0 * ui
    pad_y = 4.0 * ui
    x = pos.x - tw * 0.5
    y = pos.y - th * 0.5

    region = bpy.context.region
    if region is not None:
        x = max(6.0, min(x, region.width - tw - 6.0))
        y = max(6.0, min(y, region.height - th - 6.0))

    draw_filled_rect_px(x - pad_x, y - pad_y, tw + 2.0 * pad_x, th + 2.0 * pad_y, COLOR_LABEL_BG)

    blf.position(font_id, x, y, 0)
    try:
        blf.color(font_id, color[0], color[1], color[2], 1.0)
    except Exception:
        pass
    try:
        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 3, 0.0, 0.0, 0.0, 0.85)
        blf.shadow_offset(font_id, 1, -1)
    except Exception:
        pass
    blf.draw(font_id, text)
    try:
        blf.disable(font_id, blf.SHADOW)
    except Exception:
        pass


def draw_hint_px(region, text):
    font_id = 0
    ui = _ui_scale()
    size = 12.0 * ui
    try:
        blf.size(font_id, size)
    except TypeError:
        blf.size(font_id, size, 72)
    tw, th = blf.dimensions(font_id, text)
    pad_x, pad_y = 10.0 * ui, 6.0 * ui
    x = 14.0 * ui
    y = 14.0 * ui
    draw_filled_rect_px(x - pad_x * 0.4, y - pad_y * 0.5, tw + pad_x, th + pad_y, COLOR_HINT_BG)
    blf.position(font_id, x, y, 0)
    try:
        blf.color(font_id, 0.86, 0.89, 0.93, 1.0)
    except Exception:
        pass
    blf.draw(font_id, text)


# ---------------------------
# Property Group for panel
# ---------------------------
class IFCTrimExtendProperties(bpy.types.PropertyGroup):
    offset: bpy.props.FloatProperty(
        name="Offset",
        description="Offset added to the extension distance or subtracted from the trim distance (if destination pick is on). If pick is off, this is the trim/extend amount",
        default=0.0,
        unit='LENGTH',
    )
    pick_destination: bpy.props.BoolProperty(
        name="Pick Destination",
        description="If enabled, user must pick a destination point; if disabled, offset is used directly as trim/extend amount",
        default=True,
    )


# ---------------------------
# Main Operator
# ---------------------------
class IFC_OT_trim_extend(bpy.types.Operator):
    bl_idname = "ifc.trim_extend"
    bl_label = "IFC Trim/Extend Tool"
    # No UNDO here: each apply is a nested operator so Ctrl+Z is a real Blender+IFC step.
    bl_options = {'REGISTER'}
    bl_description = (
        "Trim or extend IFC extrusions (and their Axes) to a snapped point or by offset. "
        "Click a destination first (optional), then click multiple elements near the end to modify. "
        "Dashed preview shows the change. Openings stay in place; voids left outside are deleted. "
        "Esc to finish."
    )

    # ---------------------------
    def invoke(self, context, event):
        props = context.scene.ifc_trim_extend_props
        self.offset = props.offset
        self.pick_destination = props.pick_destination

        self.state = 'DEST' if self.pick_destination else 'ELEMENT'
        self.target_obj = None
        self.snap_point = None
        self.snap_kind = None
        self.selected_end = None
        self._handle = None
        self._snap_candidates = []
        self._extrude = None
        self._preview = None
        self._applied_count = 0

        if context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "3D View only")
            return {'CANCELLED'}

        if self.pick_destination:
            self.build_snap_candidates(context)

        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_callback_px, (), 'WINDOW', 'POST_PIXEL'
        )
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()

        if self.pick_destination:
            self.report({'INFO'}, "Click Endpoint, Midpoint or Nearest as the destination")
        else:
            self.report({'INFO'}, "Click an IFC extrusion near the end to trim/extend (Esc to finish)")
        return {'RUNNING_MODAL'}

    # ---------------------------
    def modal(self, context, event):
        if _is_undo_hotkey(event):
            return self.undo_last(context)
        if _is_redo_hotkey(event):
            return self.redo_last(context)

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.finish(context)
            return {'FINISHED'} if self._applied_count else {'CANCELLED'}

        if context.area is None or context.area.type != 'VIEW_3D' or context.region_data is None:
            return {'PASS_THROUGH'}

        context.area.tag_redraw()

        if self.state == 'DEST':
            if event.type == 'MOUSEMOVE':
                self.snap_point, self.snap_kind = self.find_snap(context, event)
                return {'PASS_THROUGH'}

            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                if self.snap_point is None:
                    nearest = self.find_snap(context, event, ignore_radius=True)
                    if nearest[0] is not None:
                        dist = nearest[2]
                        kind = _normalize_snap_kind(nearest[1])
                        tol = {
                            'ENDPOINT': SNAP_RADIUS_ENDPOINT,
                            'MIDPOINT': SNAP_RADIUS_MIDPOINT,
                            'NEAREST': SNAP_RADIUS_NEAREST,
                        }.get(kind, SNAP_PIXEL_RADIUS)
                        self.report({'WARNING'}, f"Nearest snap is {dist:.0f}px away (tolerance {tol}px)")
                    else:
                        self.report({'WARNING'}, "No snap candidates at all under this view")
                    return {'RUNNING_MODAL'}

                self.state = 'ELEMENT'
                self._preview = None
                self.report({'INFO'}, "Now click the element near the end to trim/extend (Esc to finish)")
                return {'RUNNING_MODAL'}

            return {'PASS_THROUGH'}

        if self.state == 'ELEMENT':
            if event.type == 'MOUSEMOVE':
                obj, hit_location = self.pick_object(context, event)
                if obj is not None:
                    element = tool.Ifc.get_entity(obj)
                    if element is not None and self.get_extrusion_item(element) is not None:
                        self.target_obj = obj
                        if not self.cache_extrusion(obj, element):
                            self.report({'WARNING'}, "Could not read extrusion axis — overlay disabled")
                        self.selected_end = self.determine_nearer_end(obj, element, hit_location)
                        self.update_preview()
                    else:
                        self.target_obj = None
                        self._extrude = None
                        self._preview = None
                else:
                    self.target_obj = None
                    self._extrude = None
                    self._preview = None
                return {'PASS_THROUGH'}

            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                obj, hit_location = self.pick_object(context, event)
                if obj is None:
                    self.report({'WARNING'}, "No object under cursor")
                    return {'RUNNING_MODAL'}

                element = tool.Ifc.get_entity(obj)
                if element is None or self.get_extrusion_item(element) is None:
                    self.report({'WARNING'}, "Selected object has no IfcExtrudedAreaSolid body")
                    return {'RUNNING_MODAL'}

                self.target_obj = obj
                if not self.cache_extrusion(obj, element):
                    self.report({'WARNING'}, "Could not read extrusion axis — overlay disabled")
                self.selected_end = self.determine_nearer_end(obj, element, hit_location)
                self.update_preview()

                if self._preview is not None and not self._preview.get("valid", False):
                    self.report({'INFO'}, "Invalid depth — operation cancelled")
                    self.finish(context)
                    return {'CANCELLED'}

                try:
                    self.apply_current(context)
                    self.report({'INFO'}, "Trim/Extend applied. Click another element, Ctrl+Z to undo, Esc to finish.")
                    self.target_obj = None
                    self._extrude = None
                    self._preview = None
                    return {'RUNNING_MODAL'}
                except Exception as e:
                    traceback.print_exc()
                    self.report({'ERROR'}, f"Trim/Extend failed: {e}")
                    self.finish(context)
                    return {'CANCELLED'}

            return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}

    def apply_current(self, context):
        """Run one undoable IFC trim/extend via a nested operator (Ctrl+Z-safe)."""
        _PENDING_APPLY["op"] = self
        try:
            result = bpy.ops.ifc.trim_extend_apply()
        finally:
            _PENDING_APPLY.pop("op", None)
        if result is None or 'FINISHED' not in result:
            raise RuntimeError("Trim/Extend apply did not finish")
        self._applied_count += 1
        # Nested bpy.ops from a live modal often skip Blender's undo push.
        # Force a matching undo step so scene.last_transaction and IFC history stay 1:1.
        try:
            bpy.ops.ed.undo_push(message="IFC Trim/Extend")
        except Exception:
            pass

    def _remove_draw_handler(self):
        if self._handle:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            except Exception:
                pass
            self._handle = None

    def _add_draw_handler(self):
        if self._handle is None:
            self._handle = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_callback_px, (), 'WINDOW', 'POST_PIXEL'
            )

    def _refresh_after_history(self, context):
        self.target_obj = None
        self._extrude = None
        self._preview = None
        if self.pick_destination:
            try:
                self.build_snap_candidates(context)
            except Exception:
                self._snap_candidates = []
        if context.area:
            context.area.tag_redraw()

    def undo_last(self, context):
        """Ctrl+Z while the tool is live: drop the overlay, undo, restore overlay."""
        if not self._applied_count:
            self.report({'INFO'}, "Nothing to undo")
            return {'RUNNING_MODAL'}
        self._remove_draw_handler()
        try:
            bpy.ops.ed.undo()
            self._applied_count = max(0, self._applied_count - 1)
            self.report({'INFO'}, "Undid last Trim/Extend")
        except Exception as e:
            self.report({'WARNING'}, f"Undo failed: {e}")
        self._add_draw_handler()
        self._refresh_after_history(context)
        return {'RUNNING_MODAL'}

    def redo_last(self, context):
        self._remove_draw_handler()
        try:
            bpy.ops.ed.redo()
            self._applied_count += 1
            self.report({'INFO'}, "Redid Trim/Extend")
        except Exception as e:
            self.report({'WARNING'}, f"Redo failed: {e}")
        self._add_draw_handler()
        self._refresh_after_history(context)
        return {'RUNNING_MODAL'}

    # ---------------------------
    def pick_object(self, context, event):
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = bpy_extras.view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = bpy_extras.view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

        depsgraph = context.evaluated_depsgraph_get()
        result, location, normal, index, obj, matrix = context.scene.ray_cast(
            depsgraph, origin, direction
        )
        if not result:
            return None, None
        return obj, location

    # ---------------------------
    def get_extrusion_item(self, element):
        try:
            representation = ifcopenshell.util.representation.get_representation(
                element, "Model", "Body", "MODEL_VIEW"
            )
            if representation is None:
                representation = ifcopenshell.util.representation.get_representation(
                    element, "Model", "Body"
                )
            if representation is None:
                return None
            for item in representation.Items:
                if item.is_a("IfcExtrudedAreaSolid"):
                    return item
        except Exception:
            return None
        return None

    def get_axis_representation(self, element):
        representation = ifcopenshell.util.representation.get_representation(
            element, "Model", "Axis", "GRAPH_VIEW"
        )
        if representation is None:
            representation = ifcopenshell.util.representation.get_representation(
                element, "Model", "Axis"
            )
        return representation

    def get_axis_endpoints(self, element):
        representation = self.get_axis_representation(element)
        if representation is None:
            return None
        for item in representation.Items:
            if item.is_a("IfcIndexedPolyCurve"):
                coords = [Vector(p) for p in item.Points.CoordList]
                if len(coords) >= 2:
                    return coords[0], coords[-1]
            elif item.is_a("IfcPolyline") and item.Points and len(item.Points) >= 2:
                return Vector(item.Points[0].Coordinates), Vector(item.Points[-1].Coordinates)
        return None

    def set_axis_endpoints(self, element, p0, p1):
        p0t = (float(p0[0]), float(p0[1]), float(p0[2]))
        p1t = (float(p1[0]), float(p1[1]), float(p1[2]))
        ifc = tool.Ifc.get()
        representation = self.get_axis_representation(element)

        if representation is None:
            axis_context = ifcopenshell.util.representation.get_context(
                ifc, "Model", "Axis", "GRAPH_VIEW"
            )
            if axis_context is None:
                print("[trim_extend] No Axis context — cannot write Axis representation")
                return False
            representation = tool.Ifc.run(
                "geometry.add_axis_representation", context=axis_context, axis=[p0t, p1t]
            )
            tool.Ifc.run(
                "geometry.assign_representation", product=element, representation=representation
            )
            print(f"[trim_extend] created Axis representation #{representation.id()}")
            return True

        for item in representation.Items:
            if item.is_a("IfcIndexedPolyCurve"):
                tool.Ifc.run(
                    "attribute.edit_attributes",
                    product=item.Points,
                    attributes={"CoordList": [p0t, p1t]},
                )
                print(f"[trim_extend] Axis IfcIndexedPolyCurve #{item.id()} -> {p0t} .. {p1t}")
                return True
            if item.is_a("IfcPolyline") and item.Points:
                tool.Ifc.run(
                    "attribute.edit_attributes",
                    product=item.Points[0],
                    attributes={"Coordinates": p0t},
                )
                tool.Ifc.run(
                    "attribute.edit_attributes",
                    product=item.Points[-1],
                    attributes={"Coordinates": p1t},
                )
                print(f"[trim_extend] Axis IfcPolyline #{item.id()} -> {p0t} .. {p1t}")
                return True

        print("[trim_extend] Axis representation has no IfcIndexedPolyCurve / IfcPolyline to edit")
        return False

    def get_extrusion_endpoints_world(self, obj, element):
        item = self.get_extrusion_item(element)
        if item is None or item.Position is None:
            return None
        axis_entity = item.Position.Axis
        axis_dir = Vector(axis_entity.DirectionRatios) if axis_entity else Vector((0, 0, 1))
        if axis_dir.length < 1e-9:
            axis_dir = Vector((0, 0, 1))
        axis_dir.normalize()
        ifc_file = tool.Ifc.get()
        unit_scale = ifcopenshell.util.unit.calculate_unit_scale(ifc_file)
        p0_local = Vector(item.Position.Location.Coordinates) * unit_scale
        depth = float(item.Depth) * unit_scale
        p1_local = p0_local + axis_dir * depth
        mw = obj.matrix_world
        return mw @ p0_local, mw @ p1_local

    def determine_nearer_end(self, obj, element, hit_world):
        endpoints = self.get_extrusion_endpoints_world(obj, element)
        if endpoints is None or hit_world is None:
            return 'END'
        p0w, p1w = endpoints
        d0 = (hit_world - p0w).length
        d1 = (hit_world - p1w).length
        return 'START' if d0 <= d1 else 'END'

    def cache_extrusion(self, obj, element):
        item = self.get_extrusion_item(element)
        if item is None or item.Position is None:
            self._extrude = None
            return False
        axis_entity = item.Position.Axis
        axis_dir = Vector(axis_entity.DirectionRatios) if axis_entity else Vector((0, 0, 1))
        if axis_dir.length < 1e-9:
            axis_dir = Vector((0, 0, 1))
        axis_dir.normalize()
        ifc_file = tool.Ifc.get()
        unit_scale = ifcopenshell.util.unit.calculate_unit_scale(ifc_file)
        p0 = Vector(item.Position.Location.Coordinates) * unit_scale
        depth = float(item.Depth) * unit_scale
        self._extrude = {
            "p0_local": p0,
            "axis_local": axis_dir,
            "depth": depth,
            "unit_scale": unit_scale,
        }
        return True

    def update_preview(self):
        self._preview = None
        if self._extrude is None or self.target_obj is None:
            return
        obj = self.target_obj
        try:
            inv = obj.matrix_world.inverted()
        except ValueError:
            return
        ex = self._extrude
        p0 = ex["p0_local"]
        axis = ex["axis_local"]
        depth = ex["depth"]
        mw = obj.matrix_world
        p0w = mw @ p0
        p1w = mw @ (p0 + axis * depth)

        if self.pick_destination:
            if self.snap_point is None:
                return
            target_local = inv @ self.snap_point
            s = (target_local - p0).dot(axis)
            # Apply offset: positive offset adds to extension, subtracts from trim
            if self.selected_end == 'START':
                s_eff = s - self.offset
            else:
                s_eff = s + self.offset

            if self.selected_end == 'START':
                new_depth = depth - s_eff
                new_p0_local = p0 + axis * s_eff
                new_p1_local = p0 + axis * depth
                moving_old = p0w
                moving_new = mw @ new_p0_local
                op = "EXTEND" if s_eff < 0.0 else "TRIM"
            else:
                new_depth = s_eff
                new_p0_local = p0
                new_p1_local = p0 + axis * new_depth
                moving_old = p1w
                moving_new = mw @ new_p1_local
                op = "EXTEND" if s_eff > depth else "TRIM"

            delta = (moving_new - moving_old).length
            proj = moving_new

            if op == "EXTEND":
                remain_a, remain_b = p0w, p1w
            elif self.selected_end == 'START':
                remain_a, remain_b = moving_new, p1w
            else:
                remain_a, remain_b = p0w, moving_new

            self._preview = {
                "op": op,
                "end": self.selected_end,
                "delta": delta,
                "new_depth": new_depth,
                "old_depth": depth,
                "dash_a": moving_old,
                "dash_b": moving_new,
                "remain_a": remain_a,
                "remain_b": remain_b,
                "proj": moving_new,
                "snap": self.snap_point.copy(),
                "valid": new_depth > 1e-6,
                "off_axis": (self.snap_point - moving_new).length > 1e-4,
            }
        else:
            # No destination: offset is direct trim/extend amount
            if self.selected_end == 'START':
                new_depth = depth + self.offset
                new_p0_local = p0 - axis * self.offset
                new_p1_local = p0 + axis * depth
                moving_old = p0w
                moving_new = mw @ new_p0_local
                op = "EXTEND" if self.offset > 0 else "TRIM"
                remain_a, remain_b = moving_new, p1w
            else:
                new_depth = depth + self.offset
                new_p0_local = p0
                new_p1_local = p0 + axis * new_depth
                moving_old = p1w
                moving_new = mw @ new_p1_local
                op = "EXTEND" if self.offset > 0 else "TRIM"
                remain_a, remain_b = p0w, moving_new

            delta = abs(self.offset)
            proj = moving_new

            self._preview = {
                "op": op,
                "end": self.selected_end,
                "delta": delta,
                "new_depth": new_depth,
                "old_depth": depth,
                "dash_a": moving_old,
                "dash_b": moving_new,
                "remain_a": remain_a,
                "remain_b": remain_b,
                "proj": moving_new,
                "snap": None,
                "valid": new_depth > 1e-6,
                "off_axis": False,
            }

    def build_snap_candidates(self, context):
        self._snap_candidates = []
        for obj in context.visible_objects:
            if obj.type != 'MESH':
                continue
            mat = obj.matrix_world
            mesh = obj.data
            world_verts = [mat @ v.co for v in mesh.vertices]
            for v in world_verts:
                self._snap_candidates.append(('ENDPOINT', v))
            for e in mesh.edges:
                a = world_verts[e.vertices[0]]
                b = world_verts[e.vertices[1]]
                self._snap_candidates.append(('MIDPOINT', a.lerp(b, 0.5)))
                self._snap_candidates.append(('NEAREST', a, b))

    def find_snap(self, context, event, ignore_radius=False):
        """AutoCAD-like osnap: Endpoint (square), Midpoint (triangle), Nearest (hourglass).

        Endpoints and midpoints use a larger magnet so they acquire from farther
        away than nearest-on-edge. Geometric snaps win unless nearest is clearly
        closer (under ~45% of the geometric distance).
        """
        region = context.region
        rv3d = context.region_data
        mouse = Vector((event.mouse_region_x, event.mouse_region_y))

        best = {
            'ENDPOINT': (float('inf'), None),
            'MIDPOINT': (float('inf'), None),
            'NEAREST': (float('inf'), None),
        }

        for cand in self._snap_candidates:
            kind = cand[0]
            if kind in {'ENDPOINT', 'MIDPOINT', 'VERTEX'}:
                if kind == 'VERTEX':
                    kind = 'ENDPOINT'
                world_v = cand[1]
                screen = bpy_extras.view3d_utils.location_3d_to_region_2d(region, rv3d, world_v)
                if screen is None:
                    continue
                dist = (screen - mouse).length
                if dist < best[kind][0]:
                    best[kind] = (dist, world_v)
            else:
                _, a, b = cand
                sa = bpy_extras.view3d_utils.location_3d_to_region_2d(region, rv3d, a)
                sb = bpy_extras.view3d_utils.location_3d_to_region_2d(region, rv3d, b)
                if sa is None or sb is None:
                    continue
                closest_2d, t = closest_point_on_segment_2d(mouse, sa, sb)
                dist = (closest_2d - mouse).length
                if dist < best['NEAREST'][0]:
                    best['NEAREST'] = (dist, a.lerp(b, t))

        if ignore_radius:
            closest_kind = None
            closest_dist = float('inf')
            closest_pt = None
            for kind, (dist, pt) in best.items():
                if pt is not None and dist < closest_dist:
                    closest_dist = dist
                    closest_pt = pt
                    closest_kind = kind
            return closest_pt, closest_kind, closest_dist

        radii = {
            'ENDPOINT': SNAP_RADIUS_ENDPOINT,
            'MIDPOINT': SNAP_RADIUS_MIDPOINT,
            'NEAREST': SNAP_RADIUS_NEAREST,
        }

        geo = []
        for kind in ('ENDPOINT', 'MIDPOINT'):
            dist, pt = best[kind]
            if pt is not None and dist <= radii[kind]:
                geo.append((dist, kind, pt))

        near_dist, near_pt = best['NEAREST']
        near_ok = near_pt is not None and near_dist <= radii['NEAREST']

        if geo:
            geo.sort(key=lambda g: g[0])
            best_geo_dist, best_geo_kind, best_geo_pt = geo[0]
            # Prefer END/MID from farther away; nearest only wins if clearly closer.
            if (not near_ok) or near_dist >= best_geo_dist * 0.45:
                return best_geo_pt, best_geo_kind

        if near_ok:
            return near_pt, 'NEAREST'
        return None, None

    # ---------------------------
    def apply_trim_extend(self, context):
        obj = self.target_obj
        element = tool.Ifc.get_entity(obj)
        item = self.get_extrusion_item(element)
        if item is None:
            raise RuntimeError("Extrusion item not found")
        if item.Position is None:
            raise RuntimeError("Extrusion has no Position placement")

        axis_entity = item.Position.Axis
        axis_dir = Vector(axis_entity.DirectionRatios) if axis_entity else Vector((0, 0, 1))
        axis_dir.normalize()

        p0_raw = Vector(item.Position.Location.Coordinates)
        depth_raw = item.Depth
        ifc_file = tool.Ifc.get()
        unit_scale = ifcopenshell.util.unit.calculate_unit_scale(ifc_file)
        p0 = p0_raw * unit_scale
        depth = depth_raw * unit_scale

        origin_delta_local = Vector((0.0, 0.0, 0.0))

        if self.pick_destination:
            target_local = obj.matrix_world.inverted() @ self.snap_point
            s = (target_local - p0).dot(axis_dir)
            # Apply offset: positive offset adds to extension, subtracts from trim
            if self.selected_end == 'START':
                s_eff = s - self.offset
            else:
                s_eff = s + self.offset

            if self.selected_end == 'START':
                new_depth = depth - s_eff
                if new_depth <= 1e-6:
                    raise RuntimeError(f"Resulting depth would be zero or negative ({new_depth})")
                new_p0 = p0
                origin_delta_local = axis_dir * s_eff
                print(f"[trim_extend] moved START origin along axis by {s_eff} m, "
                      f"new Depth={new_depth / unit_scale}")
            else:
                new_depth = s_eff
                if new_depth <= 1e-6:
                    raise RuntimeError(f"Resulting depth would be zero or negative ({new_depth})")
                new_p0 = p0
                print(f"[trim_extend] moved END, origin unchanged, new Depth={new_depth / unit_scale}")
        else:
            if self.selected_end == 'START':
                new_depth = depth + self.offset
                if new_depth <= 1e-6:
                    raise RuntimeError(f"Resulting depth would be zero or negative ({new_depth})")
                new_p0 = p0
                origin_delta_local = -axis_dir * self.offset
                print(f"[trim_extend] offset {self.offset} m, moved START origin by {-self.offset} m, "
                      f"new Depth={new_depth / unit_scale}")
            else:
                new_depth = depth + self.offset
                if new_depth <= 1e-6:
                    raise RuntimeError(f"Resulting depth would be zero or negative ({new_depth})")
                new_p0 = p0
                print(f"[trim_extend] offset {self.offset} m, moved END, new Depth={new_depth / unit_scale}")

        if origin_delta_local.length > 1e-12:
            attached = self.collect_attached_products(element)
            ifc_worlds = self.snapshot_ifc_world_matrices(attached)
            blender_worlds = self.snapshot_blender_worlds(attached)
            self.move_product_origin(context, obj, origin_delta_local)
            self.restore_attached_world_positions(
                ifc_worlds, blender_worlds, origin_delta_local, unit_scale
            )
            print(
                f"[trim_extend] kept {len(attached)} attached product(s) in world place "
                f"after origin shift {tuple(round(c, 6) for c in origin_delta_local)}"
            )

        tool.Ifc.run(
            "attribute.edit_attributes",
            product=item.Position.Location,
            attributes={"Coordinates": tuple(new_p0 / unit_scale)},
        )
        tool.Ifc.run(
            "attribute.edit_attributes",
            product=item,
            attributes={"Depth": new_depth / unit_scale},
        )
        new_p1 = new_p0 + axis_dir * new_depth

        print(f"[trim_extend] AFTER (re-read from entity, file units): Depth={item.Depth}, "
              f"Position.Location={tuple(item.Position.Location.Coordinates)}")

        self.trim_axis_representation(
            element, new_p0, new_p1, axis_dir, unit_scale,
            0.0, depth, new_depth
        )

        self.delete_openings_outside_extrusion(
            element, new_p0, axis_dir, new_depth, unit_scale
        )

        self.reload_body_in_viewport(context)

    def move_product_origin(self, context, obj, delta_local):
        delta_world = obj.matrix_world.to_3x3() @ delta_local
        obj.matrix_world.translation += delta_world
        context.view_layer.update()
        print(f"[trim_extend] origin += world {tuple(delta_world)}")
        try:
            if core_geometry is not None:
                core_geometry.edit_object_placement(
                    tool.Ifc, tool.Geometry, tool.Surveyor, obj=obj
                )
                print("[trim_extend] IfcLocalPlacement updated via edit_object_placement()")
            else:
                bpy.ops.bim.edit_object_placement()
        except Exception as e:
            print(f"[trim_extend] edit_object_placement failed: {e}")

    def collect_attached_products(self, element):
        """Products that live in this element's placement tree, nest, or voids.

        IfcDistributionPort stays glued to the host (connection points on a
        trimmed end). Openings and every other attached product must keep their
        world position when the host origin moves.
        """
        results = []
        seen = set()

        def add(product):
            if product is None:
                return
            try:
                pid = product.id()
            except Exception:
                return
            if pid in seen or product == element:
                return
            if product.is_a("IfcDistributionPort"):
                return
            seen.add(pid)
            results.append(product)

        placement = getattr(element, "ObjectPlacement", None)
        if placement is not None:
            for ref in getattr(placement, "ReferencedByPlacements", None) or []:
                for product in getattr(ref, "PlacesObject", None) or []:
                    add(product)

        for rel in getattr(element, "HasOpenings", None) or []:
            add(getattr(rel, "RelatedOpeningElement", None))

        for rel in getattr(element, "IsNestedBy", None) or []:
            for nested in getattr(rel, "RelatedObjects", None) or []:
                add(nested)

        return results

    def snapshot_ifc_world_matrices(self, products):
        snapshots = []
        for product in products:
            matrix = self._ifc_world_matrix_raw(product)
            if matrix is not None:
                snapshots.append((product, matrix))
        return snapshots

    def snapshot_blender_worlds(self, products):
        snapshots = []
        for product in products:
            try:
                obj = tool.Ifc.get_object(product)
            except Exception:
                obj = None
            if obj is None:
                continue
            try:
                snapshots.append((obj, obj.matrix_world.copy()))
            except Exception:
                continue
        return snapshots

    def restore_attached_world_positions(self, ifc_snapshots, blender_snapshots, delta_local, unit_scale):
        """Rewrite child placements so they stay put after the host origin moved.

        Restoring the pre-move world matrix is idempotent: if IfcOpenShell already
        compensated a child, this is a no-op; openings (IfcFeatureElement) are
        not compensated by edit_object_placement and get fixed here.
        """
        for product, matrix in ifc_snapshots:
            restored = False
            try:
                tool.Ifc.run(
                    "geometry.edit_object_placement",
                    product=product,
                    matrix=matrix,
                    is_si=False,
                    should_transform_children=True,
                )
                restored = True
            except TypeError:
                try:
                    tool.Ifc.run(
                        "geometry.edit_object_placement",
                        product=product,
                        matrix=matrix,
                        is_si=False,
                    )
                    restored = True
                except Exception as e:
                    print(f"[trim_extend] edit_object_placement restore #{product.id()} failed: {e}")
            except Exception as e:
                print(f"[trim_extend] edit_object_placement restore #{product.id()} failed: {e}")
            if not restored:
                self._nudge_relative_location(product, delta_local, unit_scale)
            else:
                print(f"[trim_extend] restored world placement of #{product.id()} ({product.is_a()})")
        for obj, mw in blender_snapshots:
            try:
                obj.matrix_world = mw
            except Exception:
                pass

    def _nudge_relative_location(self, product, delta_local_m, unit_scale):
        """Fallback: subtract the host origin shift from a relative Location."""
        placement = getattr(product, "ObjectPlacement", None)
        if placement is None or not placement.is_a("IfcLocalPlacement"):
            return
        rel = placement.RelativePlacement
        loc = getattr(rel, "Location", None) if rel is not None else None
        if loc is None:
            return
        coords = tuple(loc.Coordinates)
        old = Vector((
            float(coords[0]),
            float(coords[1]) if len(coords) > 1 else 0.0,
            float(coords[2]) if len(coords) > 2 else 0.0,
        ))
        new = old - Vector(delta_local_m) / unit_scale
        new_coords = tuple(float(new[i]) for i in range(len(coords)))
        try:
            tool.Ifc.run(
                "attribute.edit_attributes",
                product=loc,
                attributes={"Coordinates": new_coords},
            )
            print(f"[trim_extend] nudged #{product.id()} location {coords} -> {new_coords}")
        except Exception as e:
            print(f"[trim_extend] nudge location #{product.id()} failed: {e}")

    def _ifc_world_matrix_raw(self, product):
        placement = getattr(product, "ObjectPlacement", None)
        if placement is None:
            return None
        try:
            return ifcopenshell.util.placement.get_local_placement(placement)
        except Exception as e:
            print(f"[trim_extend] get_local_placement #{product.id()} failed: {e}")
            return None

    def _ifc_world_matrix(self, product):
        raw = self._ifc_world_matrix_raw(product)
        if raw is None:
            return None
        return Matrix([[float(raw[i][j]) for j in range(4)] for i in range(4)])

    def _first_extruded_solid(self, product):
        try:
            rep = ifcopenshell.util.representation.get_representation(
                product, "Model", "Body", "MODEL_VIEW"
            )
            if rep is None:
                rep = ifcopenshell.util.representation.get_representation(
                    product, "Model", "Body"
                )
        except Exception:
            return None
        if rep is None:
            return None
        return self._find_extrusion_in_items(getattr(rep, "Items", None) or [])

    def _find_extrusion_in_items(self, items):
        for item in items:
            if item is None:
                continue
            try:
                if item.is_a("IfcExtrudedAreaSolid"):
                    return item
                if item.is_a("IfcMappedItem"):
                    source = item.MappingSource.MappedRepresentation
                    found = self._find_extrusion_in_items(getattr(source, "Items", None) or [])
                    if found is not None:
                        return found
                if item.is_a("IfcBooleanResult"):
                    found = self._find_extrusion_in_items(
                        [item.FirstOperand, item.SecondOperand]
                    )
                    if found is not None:
                        return found
            except Exception:
                continue
        return None

    def _profile_radius(self, solid, unit_scale):
        if solid is None:
            return 0.0
        area = getattr(solid, "SweptArea", None)
        if area is None:
            return 0.0
        try:
            if area.is_a("IfcCircleProfileDef") or area.is_a("IfcCircleHollowProfileDef"):
                return abs(float(area.Radius)) * unit_scale
            if area.is_a("IfcEllipseProfileDef"):
                return max(abs(float(area.SemiAxis1)), abs(float(area.SemiAxis2))) * unit_scale
            if area.is_a("IfcRectangleProfileDef"):
                return 0.5 * max(abs(float(area.XDim)), abs(float(area.YDim))) * unit_scale
            if area.is_a("IfcIShapeProfileDef"):
                return 0.5 * max(abs(float(area.OverallWidth)), abs(float(area.OverallDepth))) * unit_scale
        except Exception:
            return 0.0
        return 0.0

    def _opening_span_along_axis(self, host, opening, p0, axis, unit_scale):
        """Return (s_min, s_max) in metres along the host extrusion axis, or None."""
        host_m = self._ifc_world_matrix(host)
        open_m = self._ifc_world_matrix(opening)
        if host_m is None or open_m is None:
            return None
        host_si = host_m.copy()
        open_si = open_m.copy()
        host_si.translation *= unit_scale
        open_si.translation *= unit_scale
        try:
            origin_host = host_si.inverted() @ open_si.translation
        except ValueError:
            return None
        s_values = [(origin_host - p0).dot(axis)]
        solid = self._first_extruded_solid(opening)
        radius = self._profile_radius(solid, unit_scale)
        if solid is not None:
            try:
                depth = float(solid.Depth) * unit_scale
                d = Vector(solid.ExtrudedDirection.DirectionRatios)
                if d.length < 1e-12:
                    d = Vector((0.0, 0.0, 1.0))
                d.normalize()
                loc = Vector((0.0, 0.0, 0.0))
                if solid.Position is not None:
                    loc = Vector(solid.Position.Location.Coordinates) * unit_scale
                    z = (
                        Vector(solid.Position.Axis.DirectionRatios)
                        if solid.Position.Axis
                        else Vector((0.0, 0.0, 1.0))
                    )
                    x = (
                        Vector(solid.Position.RefDirection.DirectionRatios)
                        if solid.Position.RefDirection
                        else Vector((1.0, 0.0, 0.0))
                    )
                    if z.length > 1e-12:
                        z.normalize()
                    if x.length > 1e-12:
                        x.normalize()
                    y = z.cross(x)
                    if y.length > 1e-12:
                        y.normalize()
                        x = y.cross(z)
                        if x.length > 1e-12:
                            x.normalize()
                    rot = Matrix((
                        (x.x, y.x, z.x),
                        (x.y, y.y, z.y),
                        (x.z, y.z, z.z),
                    ))
                    d = rot @ d
                rot_to_host = host_si.to_3x3().inverted() @ open_si.to_3x3()
                for pt in (loc, loc + d * depth):
                    pt_host = origin_host + rot_to_host @ pt
                    s_values.append((pt_host - p0).dot(axis))
            except Exception:
                pass
        s_min = min(s_values) - radius
        s_max = max(s_values) + radius
        return s_min, s_max

    def delete_openings_outside_extrusion(self, element, p0, axis, depth, unit_scale):
        """Delete openings that no longer overlap the host extrusion along its axis."""
        rels = list(getattr(element, "HasOpenings", None) or [])
        removed = 0
        for rel in rels:
            opening = getattr(rel, "RelatedOpeningElement", None)
            if opening is None:
                continue
            span = self._opening_span_along_axis(element, opening, p0, axis, unit_scale)
            if span is None:
                continue
            s_min, s_max = span
            if s_max < -1e-5 or s_min > depth + 1e-5:
                print(
                    f"[trim_extend] opening #{opening.id()} outside extrusion "
                    f"s=[{s_min:.4f}, {s_max:.4f}] m depth={depth:.4f} m — deleting"
                )
                if self.remove_opening_cleanly(opening):
                    removed += 1
        if removed:
            print(f"[trim_extend] deleted {removed} opening(s) outside the trimmed extrusion")
        return removed

    def remove_opening_cleanly(self, opening):
        """Remove the IFC opening and its Blender object so nothing is left behind."""
        opening_obj = None
        try:
            opening_obj = tool.Ifc.get_object(opening)
        except Exception:
            opening_obj = None

        deleted = False
        try:
            tool.Ifc.run("feature.remove_feature", feature=opening)
            deleted = True
        except Exception as e:
            print(f"[trim_extend] feature.remove_feature failed: {e}")
            try:
                tool.Ifc.run("void.remove_opening", opening=opening)
                deleted = True
            except Exception as e2:
                print(f"[trim_extend] void.remove_opening failed: {e2}")
                try:
                    tool.Ifc.run("root.remove_product", product=opening)
                    deleted = True
                except Exception as e3:
                    print(f"[trim_extend] root.remove_product failed: {e3}")
                    return False

        if opening_obj is not None:
            try:
                tool.Ifc.unlink(element=opening)
            except Exception:
                pass
            try:
                bpy.data.objects.remove(opening_obj, do_unlink=True)
            except Exception:
                pass
        return deleted

    def trim_axis_representation(self, element, new_p0_m, new_p1_m, extrude_dir, unit_scale, s, old_depth, new_depth):
        existing = self.get_axis_endpoints(element)
        if existing is None:
            ok = self.set_axis_endpoints(
                element, new_p0_m / unit_scale, new_p1_m / unit_scale
            )
            print(f"[trim_extend] Axis was missing, created={ok}")
            return

        a0 = existing[0] * unit_scale
        a1 = existing[1] * unit_scale
        a_vec = a1 - a0
        if a_vec.length < 1e-9:
            a_dir = Vector(extrude_dir)
        else:
            a_dir = a_vec.normalized()

        new_a0 = a0
        new_a1 = a0 + a_dir * new_depth

        self.set_axis_endpoints(element, new_a0 / unit_scale, new_a1 / unit_scale)
        print(
            f"[trim_extend] Axis BEFORE (m) {tuple(a0)} -> {tuple(a1)}; "
            f"AFTER (m) {tuple(new_a0)} -> {tuple(new_a1)}"
        )

    def reload_body_in_viewport(self, context):
        obj = self.target_obj
        element = tool.Ifc.get_entity(obj)
        body = ifcopenshell.util.representation.get_representation(
            element, "Model", "Body", "MODEL_VIEW"
        )
        if body is None:
            body = ifcopenshell.util.representation.get_representation(
                element, "Model", "Body"
            )
        if body is None:
            print("[trim_extend] no Body representation to reload")
            return
        try:
            if core_geometry is not None:
                core_geometry.switch_representation(
                    tool.Ifc, tool.Geometry, obj=obj, representation=body
                )
                print("[trim_extend] Viewport refreshed via switch_representation()")
            else:
                bpy.ops.bim.switch_representation()
                print("[trim_extend] Viewport refreshed via bim.switch_representation()")
        except Exception as reload_error:
            print(f"[trim_extend] representation reload failed: {reload_error}")
            try:
                bpy.ops.bim.update_representation()
                print("[trim_extend] fell back to bim.update_representation()")
            except Exception as e2:
                print(f"[trim_extend] update_representation failed: {e2}")
                self.report({'WARNING'}, "Applied, but viewport reload failed - see console")

    # ---------------------------
    def draw_callback_px(self):
        try:
            self._draw_callback_px()
        except ReferenceError:
            return
        except Exception:
            return

    def _draw_callback_px(self):
        context = bpy.context
        region = getattr(context, "region", None)
        rv3d = getattr(context, "region_data", None)
        if region is None or rv3d is None:
            return
        if self.state not in {'DEST', 'ELEMENT'}:
            return

        gpu.state.blend_set('ALPHA')
        try:
            gpu.state.depth_test_set('NONE')
        except Exception:
            pass

        ui = _ui_scale()
        to2d = lambda w: bpy_extras.view3d_utils.location_3d_to_region_2d(region, rv3d, w)

        snap = None
        pv = self._preview

        if self.state == 'DEST' and self.snap_point is not None:
            snap = to2d(self.snap_point)
            if snap is not None:
                draw_osnap_marker(snap, self.snap_kind, ui)

        elif self.state == 'ELEMENT':
            if pv is not None:
                remain_a = to2d(pv["remain_a"])
                remain_b = to2d(pv["remain_b"])
                dash_a = to2d(pv["dash_a"])
                dash_b = to2d(pv["dash_b"])
                proj = to2d(pv["proj"])
                if pv.get("snap") is not None:
                    snap = to2d(pv["snap"])

                if remain_a is not None and remain_b is not None:
                    draw_solid_px(remain_a, remain_b, COLOR_REMAIN, LINE_W_REMAIN * ui)

                if dash_a is not None and dash_b is not None:
                    if not pv["valid"]:
                        color = COLOR_INVALID
                    elif pv["op"] == "EXTEND":
                        color = COLOR_EXTEND
                    else:
                        color = COLOR_TRIM
                    draw_dashed_px(
                        dash_a, dash_b, color,
                        width=LINE_W_DASH * ui,
                        dash=DASH_PX * ui,
                        gap=GAP_PX * ui,
                    )
                    draw_end_ticks_px(dash_a, dash_b, color, tick_len=TICK_PX * ui, width=1.6 * ui)

                    unit_scale = self._extrude["unit_scale"] if self._extrude else 1.0
                    if not pv["valid"]:
                        label = "INVALID DEPTH"
                    else:
                        label = f"{pv['op']}  {format_length_m(pv['delta'], unit_scale)}"
                    mid = _v2(dash_a).lerp(_v2(dash_b), 0.5)
                    draw_dim_label_px(mid, dash_a, dash_b, label, color)

                if pv.get("off_axis", False) and snap is not None and proj is not None:
                    draw_dashed_px(
                        snap, proj, COLOR_CONSTRUCT,
                        width=1.15 * ui, dash=4.0 * ui, gap=4.0 * ui,
                    )

                if proj is not None:
                    draw_circle_px(proj, 4.0 * ui, (1.0, 1.0, 1.0, 0.95), fill=True)

                if snap is not None:
                    draw_osnap_marker(snap, self.snap_kind, ui)
            else:
                if self.pick_destination and self.snap_point is not None:
                    snap = to2d(self.snap_point)
                    if snap is not None:
                        draw_osnap_marker(snap, self.snap_kind, ui)

        if self.state == 'DEST':
            hint = "Click Endpoint / Midpoint / Nearest  ·  Esc cancel"
        else:
            if pv is not None and pv["valid"]:
                unit_scale = self._extrude["unit_scale"] if self._extrude else 1.0
                hint = (
                    f"{pv['op']} {pv['end']}  {format_length_m(pv['delta'], unit_scale)}"
                    f"  ·  L = {format_length_m(pv['new_depth'], unit_scale)}"
                    "  ·  LMB apply  ·  Ctrl+Z undo  ·  Esc finish"
                )
            else:
                hint = "Hover over an IFC extrusion to preview  ·  LMB apply  ·  Ctrl+Z undo  ·  Esc finish"
        draw_hint_px(region, hint)

        gpu.state.blend_set('NONE')

    def finish(self, context):
        self._remove_draw_handler()


# ---------------------------
# Undoable apply (nested from the modal picker)
# ---------------------------
class IFC_OT_trim_extend_apply(bpy.types.Operator):
    """Internal: one IFC trim/extend, wrapped in IfcStore.execute_ifc_operator so Ctrl+Z works."""
    bl_idname = "ifc.trim_extend_apply"
    bl_label = "Apply IFC Trim/Extend"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}
    bl_description = "Internal undoable apply for IFC Trim/Extend"

    transaction_key = ""
    transaction_data = None

    def execute(self, context):
        IfcStore.execute_ifc_operator(self, context)
        return {'FINISHED'}

    def _execute(self, context):
        source = _PENDING_APPLY.get("op")
        if source is None:
            raise RuntimeError("Internal error: no pending trim/extend")
        source.apply_trim_extend(context)


# ---------------------------
# Panel
# ---------------------------
class IFC_PT_trim_extend_panel(bpy.types.Panel):
    bl_label = "IFC Trim/Extend"
    bl_idname = "IFC_PT_trim_extend_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "IFC Tools"

    def draw(self, context):
        layout = self.layout
        props = context.scene.ifc_trim_extend_props
        layout.prop(props, "pick_destination")
        layout.prop(props, "offset")
        layout.operator("ifc.trim_extend", text="Trim / Extend")
        col = layout.column(align=True)
        col.scale_y = 0.85
        if props.pick_destination:
            col.label(text="1) Click destination (end / mid / nearest)")
            col.label(text="2) Click element near the end")
        else:
            col.label(text="Click element near the end")
            col.label(text="(offset is trim/extend amount)")
        col.label(text="Repeat for multiple elements")
        col.label(text="Uncheck Pick Destination to trim-")
        col.label(text="extend directly with offset value")
        col.label(text="Esc to finish")


# ---------------------------
# Register
# ---------------------------
def register():
    bpy.utils.register_class(IFCTrimExtendProperties)
    bpy.utils.register_class(IFC_OT_trim_extend_apply)
    bpy.utils.register_class(IFC_OT_trim_extend)
    bpy.utils.register_class(IFC_PT_trim_extend_panel)
    bpy.types.Scene.ifc_trim_extend_props = bpy.props.PointerProperty(type=IFCTrimExtendProperties)


def unregister():
    del bpy.types.Scene.ifc_trim_extend_props
    bpy.utils.unregister_class(IFC_PT_trim_extend_panel)
    bpy.utils.unregister_class(IFC_OT_trim_extend)
    bpy.utils.unregister_class(IFC_OT_trim_extend_apply)
    bpy.utils.unregister_class(IFCTrimExtendProperties)


if __name__ == "__main__":
    register()



