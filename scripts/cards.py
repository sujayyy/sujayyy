#!/usr/bin/env python3
"""
Build the README cards below the banner, dark and light:

  assets/evals-{theme}.svg      two radar charts of measured eval results
  assets/numbers-{theme}.svg    headline numbers, each from a project README
  assets/languages-{theme}.svg  language mix across public repos (GitHub API)

  python3 scripts/cards.py

Standard library only, so the weekly workflow needs no installs. Set GITHUB_TOKEN to
raise the API rate limit. RADARS and NUMBERS are copied from the project READMEs by
hand: update them when a README's results change.
"""
import html
import json
import os
import urllib.request
from math import cos, pi, sin

from theme import MONO, SANS, THEMES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets")
USER = "sujayyy"
W = 860

# (title, subtitle, [(axis, value 0-1, label)]); \n splits an axis name over two lines
RADARS = [
    ("RAG eval · Nimbus", "24-case gold set, 18 answerable / 6 not", [
        ("Answer accuracy", 0.83, "0.83"),
        ("Numeric\ngrounding", 1.0, "1.00"),
        ("Answer\nrate", 1.0, "18/18"),
        ("Correct refusals", 1.0, "6/6"),
        ("Page\nrecall", 0.87, "0.87"),
        ("Citation\nrecall", 0.815, "0.815"),
    ]),
    ("Agent eval · PriceOps", "13 labelled scenarios · claude-haiku-4-5", [
        ("Action", 1.0, "13/13"),
        ("Event", 1.0, "13/13"),
        ("Guardrail\nprice", 1.0, "13/13"),
        ("Urgency", 1.0, "13/13"),
        ("Market read", 1.0, "13/13"),
        ("Cites a\nnumber", 1.0, "13/13"),
        ("≤ 3\nsentences", 1.0, "13/13"),
    ]),
]

# (value, what, where)
NUMBERS = [
    ("0.980", "ROC-AUC", "FinGuard"),
    ("9.2 ms", "p95 inference latency", "FinGuard"),
    ("0.83", "RAG answer accuracy", "Nimbus"),
    ("13/13", "agent eval scenarios", "PriceOps"),
    ("159", "automated tests", "PriceOps · Medicare · FinGuard"),
    ("33.6/s", "requests under load, 0% failed", "Medicare Excellence"),
]

# linguist colours
LANG_COLORS = {
    "Python": "#3572A5", "TypeScript": "#3178c6", "JavaScript": "#f1e05a", "CSS": "#663399",
    "HTML": "#e34c26", "Dockerfile": "#384d54", "Shell": "#89e051", "Batchfile": "#C1F12E",
    "Jupyter Notebook": "#DA5B0B",
}
# notebooks store outputs inline and would swamp the mix; the profile repo is just this tooling
SKIP_LANGS = {"Jupyter Notebook"}
SKIP_REPOS = {USER}


def esc(s):
    return html.escape(str(s))


def svg(w, h, title, body, t):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{esc(title)}"><title>{esc(title)}</title>'
            f'<rect x=".5" y=".5" width="{w - 1}" height="{h - 1}" rx="10" fill="{t["panel"]}" stroke="{t["line"]}"/>'
            f'{body}</svg>')


def fade_in(el, delay, dur=.4):
    """Starts at 0s and holds at 0 until `delay`, so a viewer without SMIL still shows `el`."""
    end = delay + dur
    return (f'<g>{el}<animate attributeName="opacity" dur="{end:.2f}s" fill="freeze" '
            f'keyTimes="0;{delay / end:.3f};1" values="0;0;1"/></g>')


def radar(cx, cy, r, title, sub, axes, t):
    n = len(axes)
    ang = [-pi / 2 + 2 * pi * i / n for i in range(n)]
    o = [f'<text x="{cx}" y="40" fill="{t["text"]}" font-family="{MONO}" font-size="14" '
         f'font-weight="700" text-anchor="middle">{esc(title)}</text>',
         f'<text x="{cx}" y="58" fill="{t["dim"]}" font-family="{MONO}" font-size="11" '
         f'text-anchor="middle">{esc(sub)}</text>']
    for lv in (.25, .5, .75, 1):
        pts = " ".join(f"{cx + r * lv * cos(a):.1f},{cy + r * lv * sin(a):.1f}" for a in ang)
        o.append(f'<polygon points="{pts}" fill="none" stroke="{t["line"]}"/>')
    for a in ang:
        o.append(f'<line x1="{cx}" y1="{cy}" x2="{cx + r * cos(a):.1f}" y2="{cy + r * sin(a):.1f}" stroke="{t["line"]}"/>')

    # the shape grows out of the centre
    pts = " ".join(f"{r * v * cos(a):.1f},{r * v * sin(a):.1f}" for (_, v, _), a in zip(axes, ang))
    dots = "".join(f'<circle cx="{r * v * cos(a):.1f}" cy="{r * v * sin(a):.1f}" r="3.5" fill="{t["ac"]}"/>'
                   for (_, v, _), a in zip(axes, ang))
    o.append(f'<g transform="translate({cx} {cy})"><g>'
             f'<animateTransform attributeName="transform" type="scale" values="0;0;1" dur="1.4s" '
             f'fill="freeze" calcMode="spline" keyTimes="0;.15;1" keySplines="0 0 1 1;.2 .8 .2 1"/>'
             f'<polygon points="{pts}" fill="{t["ac"]}" fill-opacity=".18" stroke="{t["ac"]}" stroke-width="2" '
             f'stroke-linejoin="round"/>{dots}</g></g>')

    for i, ((name, _, label), a) in enumerate(zip(axes, ang)):
        lines = name.split("\n")
        block = len(lines) * 13  # name lines plus the value line, 13px apart
        lx, ly = cx + (r + 18) * cos(a), cy + (r + 18) * sin(a)
        anchor = "middle" if abs(cos(a)) < .3 else ("start" if cos(a) > 0 else "end")
        ly += -block - 2 if sin(a) < -.9 else (12 if sin(a) > .9 else 4 - block / 2)
        spans = "".join(f'<tspan x="{lx:.1f}" dy="{0 if j == 0 else 13}" fill="{t["text"]}" font-weight="700">{esc(ln)}</tspan>'
                        for j, ln in enumerate(lines))
        el = (f'<text x="{lx:.1f}" y="{ly:.1f}" font-family="{MONO}" font-size="11" text-anchor="{anchor}">{spans}'
              f'<tspan x="{lx:.1f}" dy="13" fill="{t["ac"]}">{esc(label)}</tspan></text>')
        o.append(fade_in(el, .9 + i * .06))
    return "".join(o)


def evals_card(t):
    h = 450
    body = "".join(radar(W / 4 + i * W / 2, 262, 108, *rd, t) for i, rd in enumerate(RADARS))
    body += f'<line x1="{W / 2}" y1="24" x2="{W / 2}" y2="{h - 24}" stroke="{t["line"]}"/>'
    return svg(W, h, "Eval radars: Nimbus RAG eval and PriceOps agent eval", body, t)


def numbers_card(t):
    h = 250
    o = [f'<text x="28" y="40" fill="{t["ac"]}" font-family="{MONO}" font-size="15" font-weight="700">{USER}</text>',
         f'<text x="{W - 28}" y="40" fill="{t["dim"]}" font-family="{MONO}" font-size="12" text-anchor="end">'
         f'measured, not claimed</text>',
         f'<line x1="28" y1="56" x2="{W - 28}" y2="56" stroke="{t["line"]}"/>']
    cw = (W - 56) / 3
    for i, (val, what, where) in enumerate(NUMBERS):
        x, y = 28 + (i % 3) * cw, 104 + (i // 3) * 84
        el = (f'<text x="{x}" y="{y}" fill="{t["text"]}" font-family="{SANS}" font-size="32" font-weight="700">{esc(val)}</text>'
              f'<text x="{x}" y="{y + 22}" fill="{t["dim"]}" font-family="{SANS}" font-size="13">{esc(what)}</text>'
              f'<text x="{x}" y="{y + 39}" fill="{t["ac"]}" font-family="{MONO}" font-size="11">{esc(where)}</text>')
        o.append(fade_in(el, .2 + i * .12))
    return svg(W, h, "Headline numbers from project READMEs", "".join(o), t)


def fetch(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": USER})
    if os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def language_totals():
    repos = [r for r in fetch(f"https://api.github.com/users/{USER}/repos?per_page=100&type=owner")
             if not r["fork"] and r["name"] not in SKIP_REPOS]
    totals = {}
    for r in repos:
        for lang, b in fetch(r["languages_url"]).items():
            if lang not in SKIP_LANGS:
                totals[lang] = totals.get(lang, 0) + b
    return sorted(totals.items(), key=lambda kv: -kv[1]), len(repos)


def size(b):
    return f"{b / 1e6:.2f} MB" if b >= 1e6 else f"{b / 1e3:.1f} kB" if b >= 1e3 else f"{b} B"


def languages_card(langs, n_repos, t):
    total = sum(b for _, b in langs)
    top = [(lang, b) for lang, b in langs if b / total >= .005][:7]
    rest = total - sum(b for _, b in top)
    if rest:
        top.append(("Other", rest))
    rows = (len(top) + 1) // 2
    h = 118 + rows * 28
    o = [f'<text x="28" y="40" fill="{t["text"]}" font-family="{MONO}" font-size="14" font-weight="700">Most used languages</text>',
         f'<text x="{W - 28}" y="40" fill="{t["dim"]}" font-family="{MONO}" font-size="12" text-anchor="end">'
         f'{len(langs)} languages · {n_repos} public repos</text>',
         f'<clipPath id="bar"><rect x="28" y="60" width="{W - 56}" height="10" rx="5">'
         f'<animate attributeName="width" dur="1.3s" values="0;0;{W - 56}" fill="freeze" calcMode="spline" '
         f'keyTimes="0;.15;1" keySplines="0 0 1 1;.2 .8 .2 1"/></rect></clipPath>',
         f'<rect x="28" y="60" width="{W - 56}" height="10" rx="5" fill="{t["line"]}"/>']
    x, segs = 28.0, []
    for lang, b in langs:
        w = (W - 56) * b / total
        segs.append(f'<rect x="{x:.2f}" y="60" width="{w + .5:.2f}" height="10" fill="{LANG_COLORS.get(lang, t["dim"])}"/>')
        x += w
    o.append(f'<g clip-path="url(#bar)">{"".join(segs)}</g>')
    cw = (W - 56) / 2
    for i, (lang, b) in enumerate(top):
        cx, cy = 28 + (i % 2) * cw, 104 + (i // 2) * 28
        el = (f'<circle cx="{cx + 6}" cy="{cy - 4}" r="5" fill="{LANG_COLORS.get(lang, t["dim"])}"/>'
              f'<text x="{cx + 20}" y="{cy}" fill="{t["text"]}" font-family="{SANS}" font-size="14">{esc(lang)}</text>'
              f'<text x="{cx + cw - 90}" y="{cy}" fill="{t["dim"]}" font-family="{MONO}" font-size="12" text-anchor="end">{size(b)}</text>'
              f'<text x="{cx + cw - 24}" y="{cy}" fill="{t["dim"]}" font-family="{MONO}" font-size="12" text-anchor="end">'
              f'{100 * b / total:.1f}%</text>')
        o.append(fade_in(el, .6 + i * .06))
    return svg(W, h, "Most used languages across public repos", "".join(o), t)


def main():
    os.makedirs(OUT, exist_ok=True)
    langs, n_repos = language_totals()
    for name, t in THEMES.items():
        for card, content in [("evals", evals_card(t)), ("numbers", numbers_card(t)),
                              ("languages", languages_card(langs, n_repos, t))]:
            path = os.path.join(OUT, f"{card}-{name}.svg")
            with open(path, "w") as f:
                f.write(content)
    print("languages:", ", ".join(f"{l} {b}" for l, b in langs), f"({n_repos} repos)")


if __name__ == "__main__":
    main()
