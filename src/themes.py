"""Colour themes for the web UI: (accent for light mode, accent for dark mode)."""
PALETTES = {
    "ม่วงลาเวนเดอร์": ("#6b63d9", "#a9a3ff"),
    "ฟ้าสกาย": ("#2f7fd1", "#7fb8ff"),
    "เขียวมินต์": ("#1f9a7a", "#6fe0c0"),
    "พีชส้ม": ("#e4693a", "#ffa079"),
    "ชมพูกุหลาบ": ("#cf4f86", "#ff9cc4"),
    "กราไฟต์": ("#4b5563", "#c4cad4"),
}
DEFAULT_PALETTE = "ม่วงลาเวนเดอร์"

# Surface colours that depend on light / dark mode (everything else derives from --accent)
_LIGHT = {
    "page": "#f2f3fa", "panel": "rgba(255,255,255,.88)", "card": "#ffffff", "ink": "#14161f",
    "ink-text": "#ffffff", "muted": "rgba(22,26,43,.58)", "line": "rgba(22,26,43,.10)",
    "shadow": "rgba(45,55,130,.28)", "glass-edge": "rgba(255,255,255,.75)", "warm": "#ffc58f",
}
_DARK = {
    "page": "#0c0e15", "panel": "rgba(24,27,38,.86)", "card": "#1b1f2b", "ink": "#eceef6",
    "ink-text": "#12141c", "muted": "rgba(236,238,246,.60)", "line": "rgba(255,255,255,.11)",
    "shadow": "rgba(0,0,0,.55)", "glass-edge": "rgba(255,255,255,.08)", "warm": "#b8693a",
}


def accent(name, dark=False):
    light, dark_variant = PALETTES.get(name, PALETTES[DEFAULT_PALETTE])
    return dark_variant if dark else light


def css_vars(name, dark=False):
    """CSS custom properties that assets/style.css builds on."""
    values = {"accent": accent(name, dark), **(_DARK if dark else _LIGHT)}
    return ":root{" + ";".join(f"--{k}:{v}" for k, v in values.items()) + ";}"
