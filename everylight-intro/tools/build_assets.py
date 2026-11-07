#!/usr/bin/env python3
"""Build the intro's image assets from the public-domain source photos.

    python3 everylight-intro/tools/build_assets.py

Reads  everylight-intro/sources/*.jpg   (originals, see CREDITS.md)
Writes everylight-intro/assets/*.jpg    (graded 960x720 frames, for reference)
       everylight-intro/assets.js       (same frames as data URIs, so the page
                                         works from file:// and can be recorded)
Requires Pillow and numpy.
"""
import base64, io, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC, OUT = os.path.join(ROOT, "sources"), os.path.join(ROOT, "assets")
W, H = 960, 720
rs = np.random.RandomState(1962)


def load_gray(name, box):
    im = Image.open(os.path.join(SRC, name)).convert("L").crop(box)
    return im.resize((W, H), Image.LANCZOS)


def sepia(gray, fade=0.10, contrast=1.08):
    """Map a gray image onto a warm, slightly faded 1960s print palette."""
    g = np.asarray(gray, np.float32) / 255.0
    g = np.clip((g - 0.5) * contrast + 0.5, 0, 1)
    g = fade + g * (1 - fade * 1.6)                      # lifted blacks, dull whites
    lo, mid, hi = np.array([38, 24, 14]), np.array([150, 108, 66]), np.array([242, 226, 190])
    g3 = g[..., None]
    rgb = np.where(g3 < 0.5, lo + (mid - lo) * (g3 / 0.5), mid + (hi - mid) * ((g3 - 0.5) / 0.5))
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def save(img, key):
    img.save(os.path.join(OUT, key + ".jpg"), quality=84, optimize=True)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- photos
PHOTOS = {
    # key: (file, crop box in source pixels, fade)
    "p1": ("oakridge_playground_1944.jpg", (0, 40, 1240, 970), 0.12),
    "p2": ("emerson_wight_playground_1916.jpg", (40, 20, 1180, 875), 0.08),
    "p3": ("robert_e_lee_school_1917.jpg", (90, 30, 1220, 878), 0.08),
    "p4": ("glenwood_school_play_1947.jpg", (0, 37, 1280, 997), 0.10),
}

# --------------------------------------------------------------------------- the -1 floor
def noise_blob(size, cx, cy, rx, ry, seed, rough=0.35):
    """Irregular, organic mask (0..1) shaped like a spill."""
    r = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:size[1], 0:size[0]].astype(np.float32)
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    n = Image.fromarray((r.rand(size[1] // 8 + 1, size[0] // 8 + 1) * 255).astype(np.uint8)).resize(size, Image.BICUBIC)
    n = np.asarray(n, np.float32) / 255.0
    return np.clip((1.0 - d + (n - 0.5) * rough * 2) * 6, 0, 1)


def draw_blood(layer_rgba, mask, color=(105, 4, 4), alpha=0.92):
    a = np.asarray(layer_rgba, np.float32)
    m = np.clip(mask, 0, 1)[..., None] * alpha
    a[..., :3] = a[..., :3] * (1 - m) + np.array(color) * m
    a[..., 3:] = np.maximum(a[..., 3:], m * 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def _blur(a, r):
    return np.asarray(Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), np.float32) / 255


def _poly(points):
    m = Image.new("L", (W, H), 0); ImageDraw.Draw(m).polygon(points, fill=255)
    return np.asarray(m, np.float32) / 255


def build_lab():
    """The -1 floor: a real autopsy room at night, blood that sits on its surfaces,
    and a faint green light leaking in from somewhere outside the frame (left)."""
    base = Image.open(os.path.join(SRC, "gorgas_autopsy_room.jpg")).convert("L")
    base = base.crop((22, 60, 1222, 960)).resize((W, H), Image.LANCZOS)
    g = np.asarray(base, np.float32) / 255.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # night grade: crushed, cold
    dark = g ** 2.1 * 0.46
    rgb = np.stack([dark * 0.80, dark * 0.90, dark * 1.0], -1) * 255

    # off-screen green source, low on the left (a door we never see)
    d = np.sqrt(((xx + 260) / 1.0) ** 2 + ((yy - 610) / 0.75) ** 2)
    light = np.clip(1.0 - d / 1150, 0, 1) ** 2.2
    floor_bias = np.clip((yy - 380) / 300, 0, 1)                           # floor / table sides catch most of it
    lf = light * (0.45 + 0.55 * floor_bias)
    green = np.array([70, 190, 60], np.float32)
    rgb += (g ** 1.4)[..., None] * lf[..., None] * green * 1.35
    rgb += lf[..., None] * np.array([4, 16, 6])                           # a little haze in the air

    # ---------------------------------------------------------------- blood mask
    M = np.zeros((H, W), np.float32)
    tray3 = _poly([(740, 392), (852, 396), (902, 516), (660, 497)])       # near table, tray inset
    top2 = _poly([(262, 386), (522, 381), (530, 397), (250, 410)])
    top1 = _poly([(72, 380), (330, 362), (338, 368), (80, 389)])
    M = np.maximum(M, _blur(noise_blob((W, H), 745, 474, 105, 34, 21, rough=0.3), 1.6) * tray3)
    M = np.maximum(M, _blur(noise_blob((W, H), 815, 440, 34, 22, 22, rough=0.3), 1.6) * tray3 * 0.75)
    M = np.maximum(M, _blur(noise_blob((W, H), 400, 398, 95, 11, 23, rough=0.3), 1.2) * top2)
    M = np.maximum(M, _blur(noise_blob((W, H), 205, 376, 60, 8, 24, rough=0.3), 1.2) * top1 * 0.85)

    dm = Image.new("L", (W, H), 0); dd = ImageDraw.Draw(dm)
    def drip(x, y0, L, w, v=None):
        v = v or int(150 + rs.rand() * 105)
        x1 = x + (rs.rand() - 0.5) * 2
        dd.line([(x, y0), (x1, y0 + L)], fill=v, width=w)
        dd.ellipse((x1 - w * 0.9, y0 + L - w * 0.6, x1 + w * 0.9, y0 + L + w * 1.2), fill=v)
    for _ in range(9):                                                      # over table 3's front rim
        x = 675 + rs.rand() * 190; y = 500 + (x - 660) * 0.078
        drip(x, y, 12 + rs.rand() * 48, 2 + int(rs.rand() * 3))
    drip(758, 560, 120, 4); drip(752, 600, 90, 3)                           # down the pipe to the floor
    for _ in range(8):                                                      # table 2 front edge
        x = 270 + rs.rand() * 240; y = 409 - (x - 250) * 0.047
        drip(x, y, 8 + rs.rand() * 40, 2 + int(rs.rand() * 2))
    drip(296, 432, 150, 4)                                                  # along table 2's hose
    for _ in range(5):                                                      # table 1 front edge
        x = 110 + rs.rand() * 200; y = 388 - (x - 80) * 0.075
        drip(x, y, 6 + rs.rand() * 26, 2)
    M = np.maximum(M, _blur(np.asarray(dm, np.float32) / 255, 0.7))

    # floor: pools (flat in perspective), splatter, and a drag mark toward the light
    for cx, cy, rx, ry, sd, al in ((298, 594, 34, 7, 31, 0.95), (760, 706, 70, 12, 32, 0.95),
                                   (430, 566, 60, 10, 33, 0.8), (235, 505, 30, 5, 34, 0.7)):
        M = np.maximum(M, _blur(noise_blob((W, H), cx, cy, rx, ry, sd, rough=0.3), 1.2) * al)
    sm = Image.new("L", (W, H), 0); sd_ = ImageDraw.Draw(sm)
    for k in range(7):
        o = (k - 3) * 6
        pts = [(440 + o * 0.3, 572 + o * 0.4), (330, 610 + o), (190, 652 + o * 1.2), (40, 700 + o * 1.4), (-20, 718 + o * 1.5)]
        sd_.line(pts, fill=int(120 + rs.rand() * 110), width=4 + int(rs.rand() * 6), joint="curve")
    streak = np.asarray(sm.filter(ImageFilter.GaussianBlur(2.2)), np.float32) / 255
    broken = _blur(rs.rand(H, W).astype(np.float32), 3.0)
    streak *= np.clip((broken - 0.35) * 3.2, 0.15, 1)                       # dry, skipping drag texture
    M = np.maximum(M, streak * 0.85)
    for _ in range(70):                                                     # drops thrown off along the drag
        t = rs.rand(); x = 430 - t * 420 + (rs.rand() - 0.5) * 70; y = 575 + t * 140 + (rs.rand() - 0.5) * 30
        r = 0.8 + rs.rand() * 2.6
        M = np.maximum(M, np.clip(1.3 - (((xx - x) / r) ** 2 + ((yy - y) / (r * 0.55)) ** 2), 0, 1))

    # ---------------------------------------------------------------- shade it
    tex = _blur(rs.rand(H, W).astype(np.float32), 1.2)
    M = np.clip(M, 0, 1)
    thick = _blur(M, 3.0) * M                                               # deep in the middle, thin at the rim
    a = rgb
    lum = a.mean(-1, keepdims=True)
    thin = lum * np.array([0.52, 0.07, 0.06]) + np.array([9, 0, 0])
    deep = np.array([20, 1, 1], np.float32) + lf[..., None] * np.array([10, 8, 2])
    k = np.clip(thick ** 0.7, 0, 1)[..., None]
    blood = thin * (1 - k) + deep * k
    rim = np.clip(M - _blur(M, 1.5), 0, 1)[..., None]                       # darker dried edge
    blood *= 1 - rim * 0.5
    m = M[..., None] * 0.95
    a = a * (1 - m) + blood * m
    # wet glints facing the green light (light from the left: surfaces whose slope faces -x)
    gx = np.zeros_like(thick); gx[:, 1:-1] = thick[:, 2:] - thick[:, :-2]
    glint = np.clip(gx * 5, 0, 1) * (tex > 0.6) * (0.15 + lf * 1.6) * (thick > 0.35)
    a += glint[..., None] * np.array([70, 120, 70])

    # vignette
    v = np.clip(1.18 - (((xx - 520) / 700) ** 2 + ((yy - 400) / 560) ** 2), 0.2, 1)[..., None]
    return Image.fromarray(np.clip(a * v, 0, 255).astype(np.uint8))


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for key, (fn, box, fade) in PHOTOS.items():
        data[key] = save(sepia(load_gray(fn, box), fade), key)
    data["lab"] = save(build_lab(), "lab")
    with open(os.path.join(ROOT, "assets.js"), "w") as f:
        f.write("// Generated by tools/build_assets.py from public-domain photos (see CREDITS.md). Do not edit.\n")
        f.write("window.EVERYLIGHT_ASSETS = " + json.dumps(data) + ";\n")
    print("assets.js", os.path.getsize(os.path.join(ROOT, "assets.js")) // 1024, "KB")


if __name__ == "__main__":
    main()
