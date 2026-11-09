"""Generate the README header graphic (docs/header.png).

Same design language as the other ATaylorAerospace repositories: a navy starfield, a
letter-spaced title, four translucent panels with teal headers, and a central illustration. Here
the illustration is the embodiment > task > primitive tree embedded in the Poincaré disk (Sarkar
construction at K = -1) next to the Lorentz hyperboloid it is isometric to, with an open-loop
rollout drawn as a geodesic path. Everything is drawn with Pillow; the geometry is computed with
the Möbius formulas of the unit ball.

Run:  uv run --no-sync python docs/make_header.py
"""

# ruff: noqa: RUF001  (mathematical glyphs are deliberate in the artwork)
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 2480, 1380
OUT = Path(__file__).with_name("header.png")

# ---------------------------------------------------------------- palette
NAVY_TOP = (9, 22, 48)
NAVY_BOTTOM = (13, 33, 68)
PANEL = (18, 44, 88, 150)
PANEL_EDGE = (58, 96, 150, 220)
TEAL = (86, 214, 196)
TEAL_DIM = (68, 168, 156)
WHITE = (240, 244, 250)
SOFT = (188, 200, 222)
DIM = (140, 158, 190)
AMBER = (245, 195, 90)
ORANGE = (255, 158, 60)
BLUE = (110, 160, 255)
BLUE_SOFT = (150, 190, 255)
GREEN = (120, 220, 150)
RED = (255, 120, 110)

DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono"


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    base = MONO if mono else DEJAVU
    return ImageFont.truetype(base + ("-Bold" if bold else "") + ".ttf", size)


# ---------------------------------------------------------------- canvas
img = Image.new("RGB", (W, H), NAVY_TOP)
px = img.load()
for y in range(H):
    t = y / (H - 1)
    col = tuple(int(NAVY_TOP[i] * (1 - t) + NAVY_BOTTOM[i] * t) for i in range(3))
    for x in range(W):
        px[x, y] = col

rng = random.Random(7)
stars = Image.new("RGBA", (W, H), (0, 0, 0, 0))
sd = ImageDraw.Draw(stars)
for _ in range(900):
    x, y = rng.uniform(0, W), rng.uniform(0, H)
    r = rng.choice([0.6, 0.8, 1.0, 1.3, 1.8])
    a = rng.randint(60, 200)
    sd.ellipse([x - r, y - r, x + r, y + r], fill=(220, 230, 255, a))
img.paste(stars, (0, 0), stars)

overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(overlay)


def text_width(text: str, fnt: ImageFont.FreeTypeFont, spacing: float = 0.0) -> float:
    return sum(fnt.getlength(ch) + spacing for ch in text) - spacing


def spaced(x: float, y: float, text: str, fnt, fill, spacing: float, anchor: str = "l") -> None:
    """Letter-spaced text; ``anchor`` is ``l`` (left) or ``m`` (centred on x)."""
    w = text_width(text, fnt, spacing)
    cx = x - w / 2 if anchor == "m" else x
    for ch in text:
        d.text((cx, y), ch, font=fnt, fill=fill, anchor="lm")
        cx += fnt.getlength(ch) + spacing


def text(x: float, y: float, s: str, fnt, fill, anchor: str = "lm") -> None:
    d.text((x, y), s, font=fnt, fill=fill, anchor=anchor)


def fit(s: str, size: int, max_w: float, **kw) -> ImageFont.FreeTypeFont:
    while size > 10 and font(size, **kw).getlength(s) > max_w:
        size -= 1
    return font(size, **kw)


def panel(x0: int, y0: int, x1: int, y1: int, title: str) -> None:
    d.rounded_rectangle([x0, y0, x1, y1], radius=22, fill=PANEL, outline=PANEL_EDGE, width=2)
    d.rounded_rectangle([x0, y0, x1, y0 + 70], radius=22, fill=(24, 56, 108, 170))
    d.rectangle([x0, y0 + 40, x1, y0 + 70], fill=(24, 56, 108, 170))
    d.line([(x0 + 2, y0 + 70), (x1 - 2, y0 + 70)], fill=PANEL_EDGE, width=2)
    spaced((x0 + x1) / 2, y0 + 36, title, font(30, bold=True), TEAL, 3.5, anchor="m")


def glow_layer() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    return layer, ImageDraw.Draw(layer)


# ---------------------------------------------------------------- title
TITLE = "HYPERBOLIC LATENT EVALUATION FOR WORLD MODELS"
spaced(W / 2, 64, TITLE, font(58, bold=True), WHITE, 6.0, anchor="m")
text(
    W / 2,
    128,
    "Frozen V-JEPA 2-AC and DINO-WM encoders · predictor heads retrained in Euclidean, "
    "Poincaré and Lorentz latent spaces · curvature always swept",
    font(27),
    SOFT,
    anchor="mm",
)

# ---------------------------------------------------------------- top-left: native geometry
panel(40, 178, 1050, 500, "DISTANCES IN EACH MODEL'S NATIVE GEOMETRY")
rows = [
    ("Euclidean", "d(x, y) = ‖x − y‖", "K = 0, flat baseline"),
    ("Poincaré", "d(x, y) = (2/√|K|) artanh(√|K| ‖−x ⊕ y‖)", "Möbius ⊕"),
    ("Lorentz", "d(x, y) = (1/√|K|) arcosh(−K ⟨x, y⟩ₗ)", "Minkowski ⟨,⟩ₗ"),
    ("Normalised", "e = d(ŷ, y) / d(x₀, y)", "= 1 for a static head"),
]
y = 292
for name, formula, note in rows:
    text(80, y, name, font(25, mono=True), TEAL)
    text(250, y, formula, font(25, mono=True), WHITE)
    fx = 250 + font(25, mono=True).getlength(formula) + 30
    if fx + font(21, mono=True).getlength(note) <= 1020:
        text(1020, y, note, font(21, mono=True), DIM, anchor="rm")
    y += 52
text(
    80,
    y + 6,
    "Every metric takes a Manifold and calls manifold.dist — there is no Euclidean fallback.",
    font(22),
    SOFT,
)

# ---------------------------------------------------------------- top-right: tasks
panel(1430, 178, 2440, 500, "FOUR TASKS, FOUR FALSIFICATION CRITERIA")
cols = (1470, 1830, 2400)
text(cols[0], 285, "Task", font(26, bold=True), WHITE)
text(cols[1], 285, "Metric (native geometry)", font(26, bold=True), WHITE)
text(cols[2], 285, "Wins if", font(26, bold=True), WHITE, anchor="rm")
d.line([(1470, 312), (2400, 312)], fill=PANEL_EDGE, width=2)
task_rows = [
    ("Latent rollout", "normalised error vs h", "lower, h > 1"),
    ("Hierarchy reconstruction", "distortion · mAP · ρ(depth)", "better, ρ > 0"),
    ("Long-horizon consistency", "ρ(latent, video) · sat. h", "higher, later"),
    ("Compositional generalisation", "unseen − seen error", "smaller gap"),
]
y = 346
for task, metric, wins in task_rows:
    text(cols[0], y, task, font(24), WHITE)
    text(cols[1], y, metric, font(22, mono=True), SOFT)
    text(cols[2], y, wins, font(22, mono=True), AMBER, anchor="rm")
    y += 44

# ---------------------------------------------------------------- centre label
spaced(
    W / 2,
    540,
    "TREE EMBEDDED IN THE POINCARÉ BALL  ·  OPEN-LOOP ROLLOUT ON THE LORENTZ HYPERBOLOID  ·  K = −1",
    font(30, bold=True),
    TEAL,
    3.5,
    anchor="m",
)

# ---------------------------------------------------------------- Poincaré disk with a tree
CX, CY, R = 520, 790, 235


def mobius_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xy, xx, yy = float(x @ y), float(x @ x), float(y @ y)
    return ((1 + 2 * xy + yy) * x + (1 - xx) * y) / (1 + 2 * xy + xx * yy)


def expmap(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return x
    lam = 2 / (1 - float(x @ x))
    return mobius_add(x, math.tanh(lam * n / 2) * v / n)


def logmap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u = mobius_add(-x, y)
    n = np.linalg.norm(u)
    if n < 1e-12:
        return np.zeros(2)
    lam = 2 / (1 - float(x @ x))
    return (2 / lam) * math.atanh(min(n, 1 - 1e-9)) * u / n


def step(x: np.ndarray, angle: float, length: float) -> np.ndarray:
    """Move ``length`` geodesic units from ``x`` in direction ``angle`` (tangent scaled by 1/λ_x)."""
    lam = 2 / (1 - float(x @ x))
    return expmap(x, (length / lam) * np.array([math.cos(angle), math.sin(angle)]))


def geodesic(x: np.ndarray, y: np.ndarray, n: int = 24) -> list[tuple[float, float]]:
    v = logmap(x, y)
    return [to_px(expmap(x, t * v)) for t in np.linspace(0, 1, n)]


def to_px(p: np.ndarray) -> tuple[float, float]:
    return (CX + R * float(p[0]), CY - R * float(p[1]))


# Sarkar-style tree: root at the origin, children spread around the outward direction.
nodes: list[tuple[np.ndarray, int]] = [(np.zeros(2), 0)]
edges: list[tuple[np.ndarray, np.ndarray, int]] = []


def add_children(parent, heading, depth, fan, spread, length):
    angles = [heading + spread * (i - (fan - 1) / 2) for i in range(fan)]
    for ang in angles:
        child = step(parent, ang, length)
        nodes.append((child, depth))
        edges.append((parent, child, depth))
        yield child, ang


# Sarkar slots: a node with k children and one parent divides the circle into k + 1 directions.
level1 = list(add_children(np.zeros(2), 0.0, 1, 2, math.pi, 1.6))
for p1, a1 in level1:
    for p2, a2 in list(add_children(p1, a1, 2, 4, 2 * math.pi / 5, 1.3)):
        list(add_children(p2, a2, 3, 2, 2 * math.pi / 3, 1.1))

disk_glow, gd = glow_layer()
gd.ellipse([CX - R - 30, CY - R - 30, CX + R + 30, CY + R + 30], fill=(60, 110, 230, 70))
disk_glow = disk_glow.filter(ImageFilter.GaussianBlur(40))
overlay.alpha_composite(disk_glow)

d.ellipse([CX - R, CY - R, CX + R, CY + R], fill=(12, 30, 66, 200), outline=BLUE, width=3)
for rho in (1.0, 2.0, 3.0):  # circles of constant geodesic distance from the origin
    rr = R * math.tanh(rho / 2)
    d.ellipse([CX - rr, CY - rr, CX + rr, CY + rr], outline=(60, 100, 170, 160), width=1)
# a few boundary-orthogonal geodesics as a hint of the tiling
for k in range(6):
    ang = k * math.pi / 3 + 0.25
    a = np.array([math.cos(ang), math.sin(ang)]) * 0.999
    b = np.array([math.cos(ang + 2.2), math.sin(ang + 2.2)]) * 0.999
    d.line(geodesic(a, b, 40), fill=(60, 100, 170, 120), width=1)

edge_col = {1: TEAL, 2: BLUE_SOFT, 3: AMBER}
for a, b, depth in edges:
    d.line(geodesic(a, b), fill=(*edge_col[depth], 230), width=4 - depth + 1)
node_r = {0: 9, 1: 7, 2: 5.5, 3: 4.5}
node_col = {0: WHITE, 1: TEAL, 2: BLUE_SOFT, 3: AMBER}
for p, depth in nodes:
    x, y = to_px(p)
    r = node_r[depth]
    d.ellipse([x - r, y - r, x + r, y + r], fill=(*node_col[depth], 255))

text(CX, CY + R + 40, "Poincaré ball  ·  27 nodes, 16 leaves", font(24), SOFT, anchor="mm")
text(CX, CY + R + 72, "distance from the origin tracks depth", font(21), DIM, anchor="mm")
text(CX, CY - 22, "root", font(19), DIM, anchor="mm")
text(CX + R + 18, CY - 40, "● embodiment", font(20), TEAL, anchor="lm")
text(CX + R + 18, CY - 8, "● task", font(20), BLUE_SOFT, anchor="lm")
text(CX + R + 18, CY + 24, "● primitive", font(20), AMBER, anchor="lm")

# ---------------------------------------------------------------- Lorentz hyperboloid
HX, HY = 1640, 870
SCALE = 105


def hyp_project(x: float, y: float) -> tuple[float, float]:
    """Point (x, y) of the ball's tangent chart -> hyperboloid (x0, x1, x2) -> tilted 2-D view."""
    r = math.hypot(x, y)
    t = math.sinh(r)
    x1, x2 = (t * x / r, t * y / r) if r > 1e-9 else (0.0, 0.0)
    x0 = math.cosh(r)
    # view: x1 to the right, x2 into the screen (foreshortened), x0 upward
    sx = HX + SCALE * (x1 * 0.95 + x2 * 0.40)
    sy = HY - SCALE * (0.62 * (x0 - 1)) + SCALE * 0.26 * x2
    return sx, sy


hyp_glow, hg = glow_layer()
hg.ellipse([HX - 420, HY - 330, HX + 420, HY + 90], fill=(70, 120, 240, 55))
hyp_glow = hyp_glow.filter(ImageFilter.GaussianBlur(50))
overlay.alpha_composite(hyp_glow)

for rho in (0.5, 1.0, 1.5, 1.9, 2.2):  # parallels: constant distance from the apex
    pts = [
        hyp_project(rho * math.cos(a), rho * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 80)
    ]
    d.line(pts, fill=(110, 160, 235, 200), width=2)
for k in range(12):  # meridians
    a = k * math.pi / 6
    pts = [hyp_project(rho * math.cos(a), rho * math.sin(a)) for rho in np.linspace(0, 2.2, 30)]
    d.line(pts, fill=(100, 150, 230, 150), width=1)


def ball_to_chart(p: np.ndarray) -> tuple[float, float]:
    """Ball point -> polar chart (geodesic distance from the origin, direction)."""
    n = float(np.linalg.norm(p))
    if n < 1e-9:
        return 0.0, 0.0
    rho = 2 * math.atanh(min(n, 1 - 1e-9))
    return rho * float(p[0]) / n, rho * float(p[1]) / n


# an open-loop rollout (orange) against the embedded true frames (teal): the latent rollout
# task scores the geodesic distance between the two at every horizon (thin ties)
def walk(seed: int, heading: float, drift: float, length: float, n: int):
    rng_w = random.Random(seed)
    state, pts = np.zeros(2), [hyp_project(0.0, 0.0)]
    keys = list(pts)
    for _ in range(n):
        heading += rng_w.uniform(-0.45, 0.45) + drift
        nxt = step(state, heading, length)
        lv = logmap(state, nxt)
        pts += [hyp_project(*ball_to_chart(expmap(state, t * lv))) for t in np.linspace(0.1, 1, 8)]
        keys.append(pts[-1])
        state = nxt
    return pts, keys


pred_path, pred_keys = walk(5, 3.55, 0.10, 0.34, 5)
true_path, true_keys = walk(5, 3.55, -0.12, 0.37, 5)
roll_glow, rg = glow_layer()
rg.line(pred_path, fill=(255, 160, 60, 150), width=14)
rg.line(true_path, fill=(86, 214, 196, 120), width=12)
roll_glow = roll_glow.filter(ImageFilter.GaussianBlur(10))
overlay.alpha_composite(roll_glow)
for a, b in zip(pred_keys[1:], true_keys[1:], strict=True):
    d.line([a, b], fill=(255, 120, 110, 220), width=2)
d.line(true_path, fill=(*TEAL, 255), width=4)
d.line(pred_path, fill=(*ORANGE, 255), width=4)
for x, y in true_keys:
    d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(*TEAL, 255), outline=(255, 255, 255, 200))
for x, y in pred_keys:
    d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(*ORANGE, 255), outline=(255, 255, 255, 200))
sx, sy = pred_keys[0]
d.ellipse([sx - 8, sy - 8, sx + 8, sy + 8], fill=(*WHITE, 255))
text(sx + 16, sy + 20, "s₀ = exp₀(P z₀)", font(20, mono=True), SOFT, anchor="lm")
px_, py_ = pred_keys[-1]
text(px_ - 6, py_ + 34, "rollout  s₊ = exp_s(P₀→s δ)", font(21, mono=True), ORANGE, anchor="lm")
tx_, ty_ = true_keys[-1]
text(tx_ - 6, ty_ - 30, "embedded true frames", font(21, mono=True), TEAL, anchor="lm")
mx, my = pred_keys[3]
text(mx + 18, my - 4, "geodesic error", font(19, mono=True), RED, anchor="lm")

text(
    HX - 40,
    HY + 82,
    "Lorentz hyperboloid  ·  isometric to the ball to 1e-9",
    font(24),
    SOFT,
    anchor="mm",
)
text(
    HX - 40,
    HY + 112,
    "same head weights, same geodesic units, twin numbers",
    font(21),
    DIM,
    anchor="mm",
)


# dashed isometry link between the two models
def dashed(p0, p1, color, dash=14, gap=10, width=2):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    s = 0.0
    while s < length:
        e = min(s + dash, length)
        d.line(
            [(p0[0] + ux * s, p0[1] + uy * s), (p0[0] + ux * e, p0[1] + uy * e)],
            fill=color,
            width=width,
        )
        s += dash + gap


LX0, LX1, LY = CX + R + 190, HX - 520, 810
dashed((LX0, LY), (LX1, LY), (86, 214, 196, 200))
text((LX0 + LX1) / 2, LY - 34, "isometry", font(22, mono=True), TEAL, anchor="mm")
text((LX0 + LX1) / 2, LY + 30, "Poincaré ⇄ Lorentz", font(20, mono=True), TEAL_DIM, anchor="mm")

# sweep grid caption on the far right of the centre band
spaced(2420 - 240, 620, "CURVATURE SWEEP", font(24, bold=True), TEAL, 3, anchor="m")
text(2420, 664, "K ∈ {−0.1, −0.25, −0.5, −1, −2, −4}", font(23, mono=True), AMBER, anchor="rm")
text(2420, 698, "dim ∈ {8, 16, 32, 64, 128}", font(23, mono=True), AMBER, anchor="rm")
text(2420, 732, "seeds {0, 1, 2}  ·  mean ± std", font(23, mono=True), AMBER, anchor="rm")
text(2420, 800, "Sarkar (2011): trees embed in H²", font(21), DIM, anchor="rm")
text(2420, 828, "with distortion 1 + ε; flat space", font(21), DIM, anchor="rm")
text(2420, 856, "needs dimension growing with the tree", font(21), DIM, anchor="rm")

# ---------------------------------------------------------------- bottom-left: guarantees
panel(40, 1040, 1050, 1340, "EXPERIMENTAL GUARANTEES")
checks = [
    "Encoders frozen: assert_frozen before every optimiser step and checkpoint",
    "Curvature swept, never fixed: the table always shows the whole curve",
    "Identical heads by construction: same seed, same weights, geodesic units",
]
y = 1140
for line in checks:
    text(80, y, "✔", font(26, bold=True), GREEN)
    text(118, y, line, font(25), WHITE)
    y += 46
text(
    80,
    1294,
    "hyperbolic head on the flat manifold == Euclidean head, exactly (tested)",
    font(23, mono=True),
    AMBER,
)

# ---------------------------------------------------------------- bottom-right: verification
panel(1430, 1040, 2440, 1340, "ONE-COMMAND REPORT, VERIFIED ON CPU")
pill_font = font(23, bold=True)
x = 1470
for label in ("Poincaré ball", "Lorentz hyperboloid", "Euclidean baseline"):
    w = pill_font.getlength(label) + 44
    d.rounded_rectangle(
        [x, 1128, x + w, 1188], radius=14, fill=(24, 56, 108, 220), outline=TEAL, width=2
    )
    text(x + w / 2, 1158, label, pill_font, WHITE, anchor="mm")
    x += w + 22
text(1470, 1232, "500 tests · 4 metrics · 4 tasks · 3 geometries", font(24, mono=True), WHITE)
text(
    1470,
    1270,
    "scripts/make_report.sh: byte-identical tables and figures",
    font(22, mono=True),
    DIM,
)
text(1470, 1304, "CI: ruff · pytest · smoke run in every geometry", font(22, mono=True), DIM)

# ---------------------------------------------------------------- compose and save
img = img.convert("RGBA")
img.alpha_composite(overlay)
img.convert("RGB").save(OUT, optimize=True)
print(f"saved {OUT} ({W}x{H})")
