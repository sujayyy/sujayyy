"""Shared palette and type for the profile SVGs (banner.py, cards.py).

Colours follow the portfolio (src/index.css); light is its inverse for GitHub's light theme.
"""

MONO = "'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"

THEMES = {
    "dark": dict(bg="#040406", panel="#0b0b10", panel2="#101017", line="#24242e", text="#e7e4dc",
                 dim="#807e8a", faint="#3a3a44", ac="#b388ff", ok="#5fd38d", live="#ff6b6b"),
    "light": dict(bg="#f6f5f1", panel="#ffffff", panel2="#f1f0ec", line="#e0ded6", text="#1d1c22",
                  dim="#6b6975", faint="#c9c7c0", ac="#6a3fd0", ok="#1f9d55", live="#d64545"),
}
