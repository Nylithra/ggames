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


def build_lab():
    base = Image.open(os.path.join(SRC, "gorgas_autopsy_room.jpg")).convert("L")
    base = base.crop((22, 60, 1222, 960)).resize((W, H), Image.LANCZOS)
    g = np.asarray(base, np.float32) / 255.0
    # window on the back wall becomes the uranium room
    win = (474, 158, 776, 352)
    wm = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = win
    wm[y0:y1, x0:x1] = np.clip((g[y0:y1, x0:x1] - 0.62) / 0.25, 0, 1)
    wm = np.asarray(Image.fromarray((wm * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), np.float32) / 255
    # night: crush everything, cold green cast
    dark = g ** 2.4 * 0.42
    rgb = np.stack([dark * 0.62, dark * 0.95, dark * 0.78], -1) * 255
    # glow + bloom from the window
    bloom = np.asarray(Image.fromarray((wm * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(45)), np.float32) / 255
    bloom2 = np.asarray(Image.fromarray((wm * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(140)), np.float32) / 255
    green = np.array([120, 255, 70], np.float32)
    rgb += wm[..., None] * (0.35 + 0.65 * g[..., None]) * np.array([95, 235, 70]) * 0.85
    rgb += bloom[..., None] * green * 0.55 + bloom2[..., None] * green * 0.35
    # light spilling onto the porcelain tables / floor (bright surfaces catch it)
    rgb += (g ** 3)[..., None] * bloom2[..., None] * green * 1.1
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGBA")

    # the subject, pressed against the glass (soft, as through thick glass)
    sil = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(sil)
    cx = 626
    d.ellipse((cx - 23, 196, cx + 21, 248), fill=255)                      # head
    d.polygon([(cx - 52, 268), (cx + 50, 266), (cx + 70, 352), (cx - 74, 352)], fill=255)
    d.rectangle((cx - 9, 240, cx + 9, 272), fill=255)                      # neck
    d.line([(cx - 40, 276), (cx - 70, 232), (cx - 86, 186)], fill=255, width=20, joint="curve")
    d.line([(cx + 38, 274), (cx + 72, 228), (cx + 88, 182)], fill=255, width=20, joint="curve")
    for hx, hy, s in ((cx - 88, 178, -1), (cx + 90, 174, 1)):              # spread hands
        d.ellipse((hx - 14, hy - 12, hx + 14, hy + 14), fill=255)
        for k in range(5):
            a = np.radians(-150 + k * 28 + (s * 6))
            d.line([(hx, hy), (hx + np.cos(a) * 28, hy + np.sin(a) * 30)], fill=255, width=6)
    sil = sil.filter(ImageFilter.GaussianBlur(3.2))
    clip = Image.new("L", (W, H), 0); ImageDraw.Draw(clip).rectangle(win, fill=255)
    sm = np.asarray(sil, np.float32) / 255 * np.asarray(clip, np.float32) / 255
    a = np.asarray(img, np.float32)
    a[..., :3] *= (1 - sm[..., None] * 0.86)
    img = Image.fromarray(a.astype(np.uint8), "RGBA")
    # eyes catching the light
    eyes = Image.new("L", (W, H), 0); ed = ImageDraw.Draw(eyes)
    ed.ellipse((cx - 11, 217, cx - 6, 221), fill=45); ed.ellipse((cx + 6, 216, cx + 11, 220), fill=45)
    eyes = eyes.filter(ImageFilter.GaussianBlur(1.2))
    a = np.asarray(img, np.float32); em = np.asarray(eyes, np.float32)[..., None] / 255
    a[..., :3] += em * np.array([200, 255, 190]); img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")

    # blood: handprints + smears on the glass, spills on the tables and floor
    bm = np.zeros((H, W), np.float32)
    def hand(hx, hy, rot, sc=1.0):
        m = Image.new("L", (W, H), 0); dd = ImageDraw.Draw(m)
        dd.ellipse((hx - 13 * sc, hy - 12 * sc, hx + 13 * sc, hy + 15 * sc), fill=255)
        for k in range(5):
            a = rot + np.radians(-150 + k * 27)
            L = (20 if k == 0 else 27) * sc
            dd.line([(hx + np.cos(a) * 10 * sc, hy + np.sin(a) * 10 * sc), (hx + np.cos(a) * (10 * sc + L), hy + np.sin(a) * (10 * sc + L))], fill=255, width=int(6 * sc))
        for k in range(5):                                                  # drips
            x = hx + (rs.rand() - 0.5) * 22 * sc; L = 25 + rs.rand() * 110
            w = 2 + int(rs.rand() * 3)
            dd.line([(x, hy + 8), (x + (rs.rand() - 0.5) * 3, hy + L)], fill=230, width=w)
            dd.ellipse((x - w, hy + L - w, x + w, hy + L + w + 1), fill=230)
        for k in range(4):                                                  # finger smears sliding down
            ox = (k - 1.5) * 6 * sc
            dd.line([(hx + ox, hy + 4), (hx + ox + (rs.rand() - 0.5) * 10, hy + 60 + rs.rand() * 70)], fill=int(70 + rs.rand() * 70), width=int(3 + rs.rand() * 3))
        return np.asarray(m.filter(ImageFilter.GaussianBlur(0.9)), np.float32) / 255
    bm = np.maximum(bm, hand(cx - 92, 186, 0.15))
    bm = np.maximum(bm, hand(cx + 96, 181, -0.2))
    bm = np.maximum(bm, hand(cx - 30, 300, 0.4, 0.8))
    bm *= np.asarray(clip, np.float32) / 255
    for _ in range(60):                                                     # spatter on glass
        x, y, r = x0 + rs.rand() * (x1 - x0), y0 + rs.rand() * (y1 - y0), rs.rand() * 2.6
        yy, xx = np.ogrid[0:H, 0:W]; bm = np.maximum(bm, np.clip(1.5 - ((xx - x) ** 2 + (yy - y) ** 2) / max(r * r, 0.5), 0, 1) * 0.9)
    tex = np.asarray(Image.fromarray((rs.rand(H, W) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
    # table tops (pools) + drips over the edges + floor
    spills = [(215, 380, 70, 10, 11, 0.8), (410, 400, 80, 13, 12, 0.85), (775, 445, 70, 34, 13, 0.9),
              (440, 610, 90, 16, 14, 0.6), (225, 525, 45, 8, 15, 0.55), (810, 655, 85, 20, 16, 0.65)]
    for cxs, cys, rx, ry, sd, al in spills:
        bm = np.maximum(bm, noise_blob((W, H), cxs, cys, rx, ry, sd, rough=0.55) * al)
    dm = Image.new("L", (W, H), 0); dd = ImageDraw.Draw(dm)
    for (ex0, ex1, ey, Lmax) in ((140, 260, 392, 90), (330, 470, 420, 120), (700, 850, 470, 130)):
        for _ in range(7):
            x = ex0 + rs.rand() * (ex1 - ex0); L = 10 + rs.rand() * Lmax * 0.7; w = 1 + int(rs.rand() * 3)
            dd.line([(x, ey), (x, ey + L)], fill=240, width=w); dd.ellipse((x - w, ey + L - w, x + w, ey + L + w), fill=240)
    for k in range(5):                                                      # drag marks to the window
        o = (k - 2) * 7
        dd.line([(560 + o, 715), (590 + o, 600), (615 + o * 0.6, 480), (628 + o * 0.3, 380)], fill=int(30 + rs.rand() * 45), width=5 + int(rs.rand() * 5), joint="curve")
    bm = np.maximum(bm, np.asarray(dm.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255)
    bm *= 0.65 + tex * 0.5
    # blood reads almost black in green light, glossy where the glow hits it
    a = np.asarray(img, np.float32)
    lit = np.clip(bloom2 * 1.5 + bloom, 0, 1)[..., None]
    m = np.clip(bm, 0, 1)[..., None] * 0.92
    tint = np.array([0.50, 0.08, 0.05])
    lum = a[..., :3].mean(-1, keepdims=True)
    blood = lum * 0.95 * tint + np.array([10, 0, 0])
    a[..., :3] = a[..., :3] * (1 - m) + blood * m
    spec = (tex > 0.86)[..., None] * m * lit * 60
    a[..., :3] += spec
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA").convert("RGB")

    # steel window frame and wired glass
    fr = Image.new("RGBA", (W, H), (0, 0, 0, 0)); fd = ImageDraw.Draw(fr)
    for k in range(-400, 400, 18):
        fd.line([(x0 + k, y0), (x0 + k + (y1 - y0), y1)], fill=(10, 40, 10, 38), width=1)
        fd.line([(x1 - k, y0), (x1 - k - (y1 - y0), y1)], fill=(10, 40, 10, 38), width=1)
    fr = Image.composite(fr, Image.new("RGBA", (W, H), (0, 0, 0, 0)), clip)
    fd = ImageDraw.Draw(fr)
    fd.rectangle((x0 - 14, y0 - 14, x1 + 14, y1 + 14), outline=(22, 30, 26, 255), width=14)
    fd.rectangle((x0 - 15, y0 - 15, x1 + 15, y1 + 15), outline=(60, 80, 66, 255), width=1)
    fd.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), outline=(5, 10, 6, 255), width=2)
    for bx, by in ((x0 - 7, y0 - 7), (x1 + 7, y0 - 7), (x0 - 7, y1 + 7), (x1 + 7, y1 + 7)):
        fd.ellipse((bx - 3, by - 3, bx + 3, by + 3), fill=(80, 110, 90, 255))
    img = Image.alpha_composite(img.convert("RGBA"), fr.filter(ImageFilter.GaussianBlur(0.6))).convert("RGB")

    # stencil on the wall
    st = Image.new("L", (W, H), 0); sd = ImageDraw.Draw(st)
    try:
        font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", 46); small = ImageFont.truetype("DejaVuSansMono-Bold.ttf", 17)
    except OSError:
        font = small = ImageFont.load_default()
    sd.text((40, 560), "KAT -1", font=font, fill=255); sd.text((42, 612), "LAB-0 / URANYUM", font=small, fill=255)
    st = np.asarray(st.filter(ImageFilter.GaussianBlur(0.7)), np.float32)[..., None] / 255
    a = np.asarray(img, np.float32); a = a * (1 - st * 0.55) + np.array([14, 26, 18]) * st * 0.55
    # vignette
    yy, xx = np.mgrid[0:H, 0:W]
    v = np.clip(1.15 - (((xx - 600) / 640) ** 2 + ((yy - 300) / 520) ** 2), 0.15, 1)[..., None]
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
