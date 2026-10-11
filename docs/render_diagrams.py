"""
Render docs/diagrams/*.mmd to SVG and PNG with Mermaid in a headless Chrome.

    pip install playwright        # and have Chrome installed (the script uses channel="chrome")
    python docs/render_diagrams.py

Needs internet access once per run (Mermaid and the Prompt font are loaded from a CDN).
The .mmd files are the source of truth; ARCHITECTURE.md embeds the same diagrams as ```mermaid blocks.
"""
import html
import json
import pathlib
import sys

from playwright.sync_api import sync_playwright

MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"
DIAGRAMS = pathlib.Path(__file__).parent / "diagrams"

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Prompt:wght@400;500;600&display=swap" rel="stylesheet">
<style>body{{margin:0;padding:24px;background:#fff;font-family:'Prompt','Noto Sans Thai',sans-serif}}</style>
</head><body><pre class="mermaid" id="d">{source}</pre>
<script src="{mermaid}"></script>
<script>
  mermaid.initialize({{startOnLoad:false, securityLevel:'loose', theme:'base',
    fontFamily:"Prompt,'Noto Sans Thai',sans-serif",
    themeVariables:{{fontSize:'15px', lineColor:'#6b7280', primaryTextColor:'#1f2433'}},
    flowchart:{{htmlLabels:true, curve:'basis', nodeSpacing:36, rankSpacing:46, padding:12}},
    sequence:{{mirrorActors:false, messageMargin:34, actorMargin:60, useMaxWidth:false}}}});
  // Mermaid measures labels when it lays out nodes, so the Thai font must be loaded first
  // (a web font only loads once something uses it, hence the explicit load calls)
  Promise.all(['400','500','600'].map(w => document.fonts.load(w + ' 15px Prompt', 'กขค abc')))
    .then(() => document.fonts.ready)
    .then(() => mermaid.run({{querySelector:'#d'}}))
    .then(() => {{ window.__done = true; }})
    .catch(e => {{ window.__error = String(e); }});
</script></body></html>"""


def main():
    files = sorted(DIAGRAMS.glob("*.mmd"))
    if not files:
        sys.exit("no .mmd files found")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        for path in files:
            page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=2)
            page.set_content(PAGE.format(source=html.escape(path.read_text(encoding="utf-8")), mermaid=MERMAID))
            page.wait_for_function("window.__done || window.__error", timeout=60000)
            error = page.evaluate("window.__error")
            if error:
                sys.exit(f"{path.name}: {error}")
            svg = page.locator("#d svg")
            svg_markup = page.evaluate("document.querySelector('#d svg').outerHTML")
            path.with_suffix(".svg").write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n' + svg_markup.replace("<svg ", '<svg xmlns:xlink="http://www.w3.org/1999/xlink" ', 1)
                if "xmlns:xlink" not in svg_markup else svg_markup, encoding="utf-8")
            svg.screenshot(path=str(path.with_suffix(".png")))
            print(f"{path.stem}: ok  ({json.dumps(svg.bounding_box())})")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
