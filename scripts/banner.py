#!/usr/bin/env python3
"""
Build the animated terminal banner for the GitHub profile README.

  python3 scripts/banner.py --photo ~/Downloads/sujay-portfolio/src/assets/sujay-profile.png

Writes assets/banner-dark.svg and assets/banner-light.svg. The portrait is a 1-bit
Floyd-Steinberg dither of the photo (plain light background), drawn as round dots.
Edit ROWS to change the info panel, then re-run and bump ?v= in README.md if GitHub
serves a cached copy.

Requires: numpy, pillow, scipy.
"""
import argparse
import html
import os

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets")

W, H = 1180, 640
MONO = "'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
CW = 0.6  # monospace advance, in em

# portrait panel and its dot grid
PX, PY, PW, PH = 34, 84, 400, 522
GX, GY, GW, GH = PX + 20, PY + 48, PW - 40, PH - 84
CELL = 2
CROP = 0.58  # portrait crop height as a share of the photo; smaller zooms in
HEADROOM = 14  # photo px kept above the hair

PROMPT = "sujay@portfolio ~ $ "
COMMAND = "fetch --live"
ROWS = [
    ("role", "Full-Stack AI Engineer"),
    ("base", "Hyderabad, India"),
    ("focus", "Agentic RAG · Multi-Agent Systems"),
    ("agents", "LangGraph · LangChain · MCP · Claude"),
    ("retrieval", "FAISS · Cross-encoder rerank · RRF"),
    ("ml", "scikit-learn · XGBoost · TensorFlow"),
    ("backend", "Python · FastAPI · Node.js · Flask"),
    ("frontend", "React · TypeScript · Vite"),
    ("data", "PostgreSQL · MongoDB"),
    ("infra", "Docker · GitHub Actions"),
    ("evals", "Gold sets · pytest · Latency benchmarks"),
    ("speaks", "English · Telugu · Hindi · Kannada"),
    ("linkedin", "/in/sujaybichala"),
]

# colours follow the portfolio (src/index.css); light is its inverse for GitHub's light theme
THEMES = {
    "dark": dict(bg="#040406", panel="#0b0b10", panel2="#101017", line="#24242e", text="#e7e4dc",
                 dim="#807e8a", faint="#3a3a44", ac="#b388ff", ok="#5fd38d", live="#ff6b6b"),
    "light": dict(bg="#f6f5f1", panel="#ffffff", panel2="#f1f0ec", line="#e0ded6", text="#1d1c22",
                  dim="#6b6975", faint="#c9c7c0", ac="#6a3fd0", ok="#1f9d55", live="#d64545"),
}


def foreground(rgb, tol=28):
    """Subject mask: everything not reachable from the top/left/right border through
    near-background colour. The bottom edge is excluded so a white shirt survives."""
    border = np.concatenate([rgb[0], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    near = np.linalg.norm(rgb.astype(float) - bg, axis=2) < tol
    lab, _ = ndimage.label(near)
    seeds = np.unique(np.concatenate([lab[0], lab[:, 0], lab[:, -1]]))
    fg = ~np.isin(lab, seeds[seeds > 0])
    fg = ndimage.binary_opening(fg, iterations=2)
    lab, n = ndimage.label(fg)
    if n > 1:
        fg = lab == 1 + int(np.argmax(ndimage.sum(fg, lab, range(1, n + 1))))
    return fg


def dither(gray):
    """Serpentine Floyd-Steinberg; True = lit."""
    g = gray.astype(np.float32).copy()
    h, w = g.shape
    out = np.zeros((h, w), bool)
    for y in range(h):
        rev = y % 2
        for x in (range(w - 1, -1, -1) if rev else range(w)):
            old = g[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new > 0
            e = old - new
            d = -1 if rev else 1
            if 0 <= x + d < w:
                g[y, x + d] += e * 7 / 16
            if y + 1 < h:
                if 0 <= x - d < w:
                    g[y + 1, x - d] += e * 3 / 16
                g[y + 1, x] += e * 5 / 16
                if 0 <= x + d < w:
                    g[y + 1, x + d] += e * 1 / 16
    return out


def portrait_dots(photo):
    img = Image.open(photo).convert("RGB")
    rgb = np.asarray(img)
    fg = foreground(rgb)
    ys, xs = np.nonzero(fg)
    # head-and-shoulders box at the grid's aspect, from the top of the hair
    cx = int(np.median(xs[ys < ys.min() + 300]))
    top = max(0, ys.min() - HEADROOM)
    bh = int(img.height * CROP)
    bw = int(bh * GW / GH)
    box = (cx - bw // 2, top, cx + bw // 2, top + bh)
    cols, rows = GW // CELL, GH // CELL

    gray = ImageOps.grayscale(img.crop(box))
    gray = gray.filter(ImageFilter.UnsharpMask(radius=40, percent=160, threshold=0))  # local contrast
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=1))   # detail
    lum = np.asarray(gray.resize((cols, rows), Image.LANCZOS), np.float32) / 255
    m = Image.fromarray(fg.astype(np.uint8) * 255).crop(box).resize((cols, rows), Image.BILINEAR)
    m = np.asarray(m, np.float32) / 255 > 0.5

    # stretch to the subject's own range
    lo, hi = np.percentile(lum[m], [2, 98])
    lum = np.clip((lum - lo) / (hi - lo), 0, 1)
    rim = m & ~ndimage.binary_erosion(m, iterations=1)

    def dots(density):
        lit = dither(np.clip(np.where(m, density, 0.0) + rim * 0.45, 0, 1))
        return [(GX + x * CELL + CELL / 2, GY + y * CELL + CELL / 2) for y, x in zip(*np.nonzero(lit))]

    # dark page: dots are light. Cap the white shirt so the face leads, lift mid-tones, and
    # keep a floor density so dark hair and blazer stay readable.
    dark = 0.06 + 0.94 * np.minimum(lum, 0.82) ** 0.7
    # light page: dots are ink, so they mark the shadows instead
    light = 0.04 + 0.96 * (1 - lum) ** 1.1
    return {"dark": dots(dark), "light": dots(light)}


def text(x, y, s, fill, size=14, anchor="start", weight=None, extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-family="{MONO}" font-size="{size}"'
            f' text-anchor="{anchor}"{w}{extra}>{html.escape(s)}</text>')


def build(theme, pts):
    t = THEMES[theme]
    pts = pts[theme]
    o = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'role="img" aria-labelledby="t d"><title id="t">Sujay Bichala, Full-Stack AI Engineer</title>'
             f'<desc id="d">Terminal-style profile card with a dithered portrait and a summary of roles and stack.</desc>')
    o.append(f'<defs><clipPath id="reveal"><rect x="{PX}" y="{PY + 34}" width="{PW}" height="0">'
             f'<animate attributeName="height" begin="0.2s" dur="1.6s" values="0;{PH - 34}" fill="freeze" '
             f'calcMode="spline" keySplines=".2 .7 .3 1"/></rect></clipPath>'
             f'<linearGradient id="scan" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{t["ac"]}" stop-opacity="0"/>'
             f'<stop offset="1" stop-color="{t["ac"]}" stop-opacity=".12"/></linearGradient></defs>')

    # window
    o.append(f'<rect width="{W}" height="{H}" rx="16" fill="{t["bg"]}"/>')
    o.append(f'<rect x="12" y="12" width="{W - 24}" height="{H - 24}" rx="12" fill="{t["panel"]}" stroke="{t["line"]}"/>')
    o.append(f'<path d="M12 56H{W - 12}" stroke="{t["line"]}"/>')
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        o.append(f'<circle cx="{36 + i * 20}" cy="34" r="6" fill="{c}"/>')
    o.append(text(W / 2, 39, "sujay@portfolio — profile", t["dim"], 13, "middle"))

    # portrait panel
    o.append(f'<rect x="{PX}" y="{PY}" width="{PW}" height="{PH}" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>')
    o.append(f'<path d="M{PX} {PY + 34}H{PX + PW}" stroke="{t["line"]}"/>')
    o.append(text(PX + 14, PY + 22, "VISUAL.MAP", t["ac"], 12, weight="700", extra=' letter-spacing="1"'))
    o.append(text(PX + PW - 14, PY + 22, f"{GW // CELL}×{GH // CELL} / 1-BIT", t["dim"], 11, "end"))
    # corner ticks
    for (x, y, dx, dy) in [(GX - 6, GY - 6, 1, 1), (GX + GW + 6, GY - 6, -1, 1),
                           (GX - 6, GY + GH + 6, 1, -1), (GX + GW + 6, GY + GH + 6, -1, -1)]:
        o.append(f'<path d="M{x} {y + 12 * dy}V{y}H{x + 12 * dx}" fill="none" stroke="{t["faint"]}" stroke-width="1.5"/>')
    d = "".join(f"M{x:g} {y:g}h0" for x, y in pts)
    o.append(f'<g clip-path="url(#reveal)"><path d="{d}" stroke="{t["ac"]}" stroke-width="1.85" stroke-linecap="round">'
             f'<animate attributeName="opacity" begin="1.9s;flick.end+7s" dur="0.5s" values="1;.35;1;.6;1" id="flick"/></path></g>')
    # recurring scanline
    o.append(f'<rect x="{GX - 10}" y="{GY - 40}" width="{GW + 20}" height="40" fill="url(#scan)" opacity="0">'
             f'<animate attributeName="y" begin="2.2s" dur="4.5s" repeatCount="indefinite" values="{GY - 40};{GY + GH}"/>'
             f'<animate attributeName="opacity" begin="2.2s" dur="4.5s" repeatCount="indefinite" values="0;1;1;0" keyTimes="0;.1;.85;1"/></rect>')
    o.append(text(PX + 14, PY + PH - 14, f"PTS {len(pts)} · FLOYD–STEINBERG", t["dim"], 11))

    # info panel
    IX, IY, IW, IH = PX + PW + 20, PY, W - (PX + PW + 20) - 34, PH
    o.append(f'<rect x="{IX}" y="{IY}" width="{IW}" height="{IH}" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>')
    o.append(f'<path d="M{IX} {IY + 34}H{IX + IW}" stroke="{t["line"]}"/>')
    o.append(text(IX + 14, IY + 22, "SYSTEM.INFO", t["ac"], 12, weight="700", extra=' letter-spacing="1"'))
    o.append(f'<circle cx="{IX + IW - 142}" cy="{IY + 18}" r="4" fill="{t["live"]}">'
             f'<animate attributeName="opacity" dur="1.6s" repeatCount="indefinite" values="1;.25;1"/></circle>')
    o.append(text(IX + IW - 132, IY + 22, "LIVE", t["live"], 11, weight="700"))
    o.append(f'<rect x="{IX + IW - 92}" y="{IY + 8}" width="80" height="20" rx="10" fill="none" stroke="{t["ac"]}" stroke-opacity=".5"/>')
    o.append(text(IX + IW - 52, IY + 22, "@sujayyy", t["ac"], 11, "middle"))

    fs, lh = 14, 24
    x0, y = IX + 22, IY + 66
    # typed command
    o.append(f'<clipPath id="type"><rect x="{x0 + len(PROMPT) * fs * CW}" y="{y - 16}" height="22" width="0">'
             f'<animate attributeName="width" begin="0.4s" dur="{0.07 * len(COMMAND):.2f}s" fill="freeze" calcMode="discrete" '
             f'values="{";".join(f"{i * fs * CW:.1f}" for i in range(len(COMMAND) + 1))}"/></rect></clipPath>')
    o.append(f'<text x="{x0}" y="{y}" font-family="{MONO}" font-size="{fs}"><tspan fill="{t["ok"]}">sujay@portfolio</tspan>'
             f'<tspan fill="{t["dim"]}"> ~ $ </tspan></text>')
    o.append(f'<g clip-path="url(#type)">{text(x0 + len(PROMPT) * fs * CW, y, COMMAND, t["text"], fs)}</g>')

    # name + rule, then rows fading in one by one
    t0 = 0.4 + 0.07 * len(COMMAND) + 0.25
    y += lh + 8
    head = [(text(x0, y, "Sujay Bichala", t["text"], 18, weight="700"), 0)]
    y += 14
    head.append((f'<path d="M{x0} {y}H{x0 + 200}" stroke="{t["ac"]}" stroke-opacity=".6"/>', 1))
    for el, i in head:
        o.append(f'<g opacity="0">{el}<animate attributeName="opacity" begin="{t0 + i * 0.05:.2f}s" dur=".25s" to="1" fill="freeze"/></g>')
    y += 24
    kw = 11
    for i, (k, v) in enumerate(ROWS):
        lead = "." * (kw - len(k))
        el = (f'<text x="{x0}" y="{y}" font-family="{MONO}" font-size="{fs}"><tspan fill="{t["ac"]}">{html.escape(k)}</tspan>'
              f'<tspan fill="{t["faint"]}"> {lead} </tspan><tspan fill="{t["text"]}">{html.escape(v)}</tspan></text>')
        o.append(f'<g opacity="0">{el}<animate attributeName="opacity" begin="{t0 + 0.15 + i * 0.07:.2f}s" dur=".3s" to="1" fill="freeze"/></g>')
        y += lh

    # palette strip and live cursor
    t1 = t0 + 0.15 + len(ROWS) * 0.07
    sw = "".join(f'<rect x="{x0 + i * 26}" y="{y - 8}" width="22" height="12" rx="2" fill="{c}"/>'
                 for i, c in enumerate([t["ac"], t["ok"], "#febc2e", t["live"], t["dim"], t["text"]]))
    o.append(f'<g opacity="0">{sw}<animate attributeName="opacity" begin="{t1:.2f}s" dur=".3s" to="1" fill="freeze"/></g>')

    # prompt with a blinking cursor
    y += lh + 6
    cx = x0 + len(PROMPT) * fs * CW
    prompt = (f'<text x="{x0}" y="{y}" font-family="{MONO}" font-size="{fs}"><tspan fill="{t["ok"]}">sujay@portfolio</tspan>'
              f'<tspan fill="{t["dim"]}"> ~ $ </tspan></text>')
    o.append(f'<g opacity="0">{prompt}<animate attributeName="opacity" begin="{t1 + 0.2:.2f}s" dur=".2s" to="1" fill="freeze"/></g>')
    o.append(f'<rect x="{cx:.1f}" y="{y - 13}" width="9" height="17" fill="{t["ac"]}" opacity="0">'
             f'<animate attributeName="opacity" begin="{t1 + 0.3:.2f}s" dur="1.1s" repeatCount="indefinite" '
             f'values="1;0" keyTimes="0;.5" calcMode="discrete"/></rect>')

    fy = IY + IH - 14
    o.append(f'<circle cx="{IX + 18}" cy="{fy - 4}" r="4" fill="{t["ok"]}"/>')
    o.append(text(IX + 28, fy, "ALL EVALS PASSING", t["ok"], 11, weight="700"))
    o.append(text(IX + IW - 14, fy, "UTC+5:30 · HYD NODE", t["dim"], 11, "end"))

    o.append("</svg>")
    return "".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    args = ap.parse_args()
    pts = portrait_dots(os.path.expanduser(args.photo))
    os.makedirs(OUT, exist_ok=True)
    for theme in THEMES:
        path = os.path.join(OUT, f"banner-{theme}.svg")
        with open(path, "w") as f:
            f.write(build(theme, pts))
        print(f"{path}  {os.path.getsize(path) // 1024} KB  {len(pts[theme])} dots")


if __name__ == "__main__":
    main()
