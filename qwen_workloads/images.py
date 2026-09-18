"""Image helpers: encode files to data URLs and generate synthetic test images."""
from __future__ import annotations
import base64, io, mimetypes, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def to_data_url(path_or_img, fmt: str = "PNG") -> str:
    if isinstance(path_or_img, Image.Image):
        buf = io.BytesIO(); path_or_img.save(buf, format=fmt)
        return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()
    p = Path(path_or_img)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def _font(size=18):
    for name in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "DejaVuSans.ttf"]:
        try: return ImageFont.truetype(name, size)
        except Exception: pass
    return ImageFont.load_default()


def make_shapes_image(seed: int = 0) -> tuple[Image.Image, dict]:
    """Random colored shapes; returns image + ground truth counts."""
    rnd = random.Random(seed)
    img = Image.new("RGB", (640, 400), "white"); d = ImageDraw.Draw(img)
    colors = ["red", "blue", "green", "orange", "purple"]
    truth = {"circles": 0, "squares": 0, "triangles": 0, "colors": {}}
    for _ in range(rnd.randint(4, 7)):
        kind = rnd.choice(["circle", "square", "triangle"]); c = rnd.choice(colors)
        x, y, s = rnd.randint(20, 540), rnd.randint(20, 300), rnd.randint(50, 90)
        if kind == "circle": d.ellipse([x, y, x + s, y + s], fill=c); truth["circles"] += 1
        elif kind == "square": d.rectangle([x, y, x + s, y + s], fill=c); truth["squares"] += 1
        else: d.polygon([(x, y + s), (x + s, y + s), (x + s // 2, y)], fill=c); truth["triangles"] += 1
        truth["colors"][c] = truth["colors"].get(c, 0) + 1
    return img, truth


def make_text_image(text: str, width: int = 800) -> Image.Image:
    """Render paragraph text like a document/screenshot (for OCR tests)."""
    f = _font(20); lines = text.split("\n")
    img = Image.new("RGB", (width, 40 + 30 * len(lines)), "white"); d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((30, 20 + 30 * i), line, fill="black", font=f)
    return img


def make_bar_chart(values: dict[str, int], title: str = "Monthly sales (units)") -> Image.Image:
    f = _font(16); img = Image.new("RGB", (720, 440), "white"); d = ImageDraw.Draw(img)
    d.text((20, 10), title, fill="black", font=_font(22))
    mx = max(values.values()); n = len(values); bw = 600 // n
    for i, (k, v) in enumerate(values.items()):
        x0 = 60 + i * bw; h = int(300 * v / mx)
        d.rectangle([x0, 380 - h, x0 + bw - 20, 380], fill="steelblue")
        d.text((x0, 385), k, fill="black", font=f)
        d.text((x0, 360 - h), str(v), fill="black", font=f)
    d.line([50, 380, 680, 380], fill="black", width=2)
    return img
