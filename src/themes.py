"""Colour themes for the web UI: (accent for light mode, accent for dark mode)."""
PALETTES = {
    "ส้มอิฐ": ("#b8480f", "#f39a5f"),
    "เขียวป่า": ("#3f6b2a", "#8cc473"),
    "น้ำเงินกรมท่า": ("#1e4f8a", "#7fa9e0"),
    "ทีล": ("#0f766e", "#4fd1c5"),
    "ม่วงมะเดื่อ": ("#7e3a6e", "#d79ac8"),
    "กราไฟต์": ("#3b4450", "#c9d1d9"),
}
DEFAULT_PALETTE = "ส้มอิฐ"


def accent(name, dark=False):
    light, dark_variant = PALETTES.get(name, PALETTES[DEFAULT_PALETTE])
    return dark_variant if dark else light


def css_vars(name, dark=False):
    """CSS custom properties that assets/style.css builds on."""
    ink = "#111514" if dark else "#ffffff"
    return f":root{{--accent:{accent(name, dark)};--accent-ink:{ink};}}"
