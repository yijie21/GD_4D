"""Generate a self-contained results gallery (data-URI images) for the S0 correspondence check.

Writes viz/gallery.html — publishable as an Artifact and viewable locally (images inlined).
"""
from __future__ import annotations
import base64, mimetypes
from pathlib import Path

VIZ = Path(__file__).resolve().parent / "viz"
OUT = VIZ / "gallery.html"


def uri(rel: str) -> str:
    p = VIZ / rel
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def figure(rel: str, title: str, cap: str) -> str:
    return (f'<figure><div class="frame"><img alt="{title}" src="{uri(rel)}"></div>'
            f'<figcaption><span class="ftitle">{title}</span> {cap}</figcaption></figure>')


CSS = """
:root{
  --paper:#f4f7f7; --panel:#ffffff; --ink:#0f1619; --muted:#586a70; --line:#dde5e6;
  --accent:#0c888f; --accent-soft:#e3f1f1; --pass:#2c9a55; --pass-soft:#e4f3ea;
  --caveat:#b57a19; --caveat-soft:#f6ecd7;
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#0c1214; --panel:#131c1f; --ink:#e7eef0; --muted:#8ba1a7; --line:#243033;
  --accent:#33c2cb; --accent-soft:#123030; --pass:#46c07a; --pass-soft:#12291d;
  --caveat:#e0a94a; --caveat-soft:#2b2313;
}}
:root[data-theme="light"]{
  --paper:#f4f7f7; --panel:#ffffff; --ink:#0f1619; --muted:#586a70; --line:#dde5e6;
  --accent:#0c888f; --accent-soft:#e3f1f1; --pass:#2c9a55; --pass-soft:#e4f3ea;
  --caveat:#b57a19; --caveat-soft:#f6ecd7;
}
:root[data-theme="dark"]{
  --paper:#0c1214; --panel:#131c1f; --ink:#e7eef0; --muted:#8ba1a7; --line:#243033;
  --accent:#33c2cb; --accent-soft:#123030; --pass:#46c07a; --pass-soft:#12291d;
  --caveat:#e0a94a; --caveat-soft:#2b2313;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:56px 22px 96px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--accent);margin:0 0 14px}
h1{font-size:clamp(28px,4.4vw,44px);line-height:1.08;letter-spacing:-.02em;font-weight:800;
  margin:0 0 12px;text-wrap:balance}
.lede{font-size:19px;color:var(--muted);max-width:64ch;margin:0 0 26px}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 8px}
.chip{font-family:var(--mono);font-size:12.5px;font-weight:600;padding:6px 12px;border-radius:999px;
  display:inline-flex;gap:7px;align-items:center;border:1px solid transparent}
.chip.pass{background:var(--pass-soft);color:var(--pass);border-color:color-mix(in srgb,var(--pass) 30%,transparent)}
.chip.caveat{background:var(--caveat-soft);color:var(--caveat);border-color:color-mix(in srgb,var(--caveat) 30%,transparent)}
.chip .dot{width:7px;height:7px;border-radius:50%;background:currentColor}
section{margin-top:52px}
h2{font-size:13px;font-family:var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
  font-weight:600;margin:0 0 18px;padding-bottom:10px;border-bottom:1px solid var(--line)}
h3{font-size:20px;letter-spacing:-.01em;margin:34px 0 6px;font-weight:700}
h3 .sub{font-family:var(--mono);font-size:12px;color:var(--muted);font-weight:500;letter-spacing:.04em}
p{margin:0 0 14px;max-width:66ch}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px 24px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line)}
th{font-family:var(--mono);font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
td{font-size:15px}
td.metric{font-family:var(--mono);font-weight:600}
td .unit{color:var(--muted);font-weight:400;font-size:12.5px}
tbody tr:last-child td{border-bottom:none}
.probe{display:grid;grid-template-columns:auto 1fr;gap:6px 16px;align-items:baseline}
.probe dt{font-family:var(--mono);font-size:13px;color:var(--accent);font-weight:600}
.probe dd{margin:0 0 10px;color:var(--ink)}
.grid{display:grid;gap:26px}
figure{margin:0}
.frame{border:1px solid var(--line);border-radius:12px;overflow:auto;background:var(--panel)}
.frame img{display:block;width:100%;height:auto}
figcaption{margin-top:10px;font-size:14px;color:var(--muted);max-width:none}
figcaption .ftitle{font-family:var(--mono);font-size:12.5px;color:var(--ink);font-weight:600;
  text-transform:uppercase;letter-spacing:.04em;margin-right:6px}
.foot{margin-top:60px;padding-top:22px;border-top:1px solid var(--line);font-size:13.5px;color:var(--muted)}
.foot code{font-family:var(--mono);font-size:12.5px;background:var(--accent-soft);color:var(--accent);
  padding:2px 7px;border-radius:6px}
.legend{font-family:var(--mono);font-size:12px;color:var(--muted);margin:2px 0 0}
.legend b{color:var(--pass)} .legend i{color:#c0433a;font-style:normal}
"""

HTML = """<style>__CSS__</style>
<div class="wrap">

  <p class="eyebrow">GD-4D · S0 backbone sanity</p>
  <h1>The 4D backbone recovers correspondences</h1>
  <p class="lede">Before building anything on top of it, we checked the one assumption the whole
  idea rests on: that the frozen backbone's point-query interface <b>X(u,&nbsp;t<sub>src</sub>&rarr;t<sub>tgt</sub>)</b>
  returns real, geometrically consistent correspondences. It does &mdash; on a natural moving-camera clip
  and, zero-shot, on robot manipulation (our target domain).</p>
  <div class="chips">
    <span class="chip pass"><span class="dot"></span>PASS &mdash; identity ~1&ndash;2px, cycle ~2&ndash;5px</span>
    <span class="chip pass"><span class="dot"></span>Zero-shot on manipulation</span>
    <span class="chip caveat"><span class="dot"></span>Caveat &mdash; raw confidence saturated</span>
  </div>

  <section>
    <h2>Results</h2>
    <div class="panel" style="overflow-x:auto">
    <table>
      <thead><tr><th>Probe (256&times;256 px)</th><th>HOT3D&nbsp;&mdash;&nbsp;natural, moving cam</th><th>LIBERO&nbsp;&mdash;&nbsp;manipulation, static cam</th></tr></thead>
      <tbody>
        <tr><td>Identity&nbsp;<span class="unit">median / p90</span></td><td class="metric">1.85 / 3.46 <span class="unit">px</span></td><td class="metric">0.91 / 2.10 <span class="unit">px</span></td></tr>
        <tr><td>Fwd&ndash;bwd cycle&nbsp;<span class="unit">visible, to 47-frame horizon</span></td><td class="metric">2.9&ndash;4.6 <span class="unit">px</span></td><td class="metric">1.8&ndash;2.2 <span class="unit">px</span></td></tr>
        <tr><td>Visibility&nbsp;<span class="unit">calibration</span></td><td class="metric">tracks occlusion &amp; return</td><td class="metric">occludes under the arm</td></tr>
        <tr><td>Confidence&nbsp;<span class="unit">sigmoid</span></td><td class="metric" style="color:var(--caveat)">saturated ~1.0</td><td class="metric" style="color:var(--caveat)">saturated ~1.0</td></tr>
      </tbody>
    </table>
    </div>
  </section>

  <section>
    <h2>How it was verified</h2>
    <p>No ground-truth motion was needed. A correct correspondence field must be <em>self-consistent</em>,
    and self-consistency is checkable without labels &mdash; three probes, increasing in strength:</p>
    <div class="panel"><dl class="probe">
      <dt>Identity</dt><dd>Query a pixel at its own frame, <b>X(u,&nbsp;a&rarr;a)</b>. Must return the pixel itself. Checks the head + coordinate wiring. &rarr; ~1&ndash;2&nbsp;px.</dd>
      <dt>Fwd&ndash;bwd cycle</dt><dd>Ask where pixel <b>u</b> in frame <b>a</b> lands in frame <b>b</b>, then ask where <em>that</em> point came from. A hallucinated match can't round-trip &mdash; errors compound. This <em>is</em> the disagreement head's <b>e_cyc</b> feature. &rarr; ~2&ndash;5&nbsp;px on visible points, even at a 47-frame horizon.</dd>
      <dt>Calibration</dt><dd>As points get occluded or leave frame, the <b>visibility</b> flag must drop and recover. It does (so <b>v</b> is usable); <b>confidence</b>, by contrast, is flat ~1.0 &mdash; the caveat below.</dd>
    </dl></div>
    <p style="margin-top:16px;color:var(--caveat)"><b>Design note &rarr; S4.</b> Raw <code style="all:unset;font-family:var(--mono)">confidence</code> is non-discriminative on this checkpoint. The disagreement head must derive D<sub>t</sub> from the validated cycle residual <b>e_cyc</b> and visibility <b>v</b> (plus a computed confidence proxy), not raw confidence.</p>
  </section>

  <section>
    <h2>LIBERO &mdash; manipulation (our target domain)</h2>
    <h3>Source &rarr; target matches <span class="sub">&nbsp; frame 0 &rarr; frame 47</span></h3>
    <p>Query points on frame 0, matched into frame 47. Static-scene points map to the same image
    location (horizontal lines = stable); points near the arm are carried to its new pose. The grid's
    2-D colour arrangement is preserved &mdash; the correspondence field is coherent.</p>
    <div class="grid">
      __LIB_CORR__
      __LIB_CYCLE__
      __LIB_TRAILS__
      __LIB_GIF__
    </div>
  </section>

  <section>
    <h2>HOT3D &mdash; natural video, moving camera</h2>
    <p>Egocentric hand-object clip with large camera motion &mdash; a harder stress test than a static rig.</p>
    <div class="grid">
      __HOT_CORR__
      __HOT_CYCLE__
      __HOT_TRAILS__
    </div>
  </section>

  <p class="foot">Reproduce: <code>src/eval/s0_backbone_sanity.py</code> (metrics) and
  <code>src/eval/s0_visualize_correspondence.py</code> (these figures), on the released OpenD4RT
  48CLIP checkpoint in the <code>gd4d5090</code> env. Full protocol, raw numbers and animations:
  <code>experiments/S0_backbone_sanity/</code>.</p>

</div>
"""

LEGEND_VIS = '<p class="legend">green = <b>visible</b> &nbsp;·&nbsp; red = <i>occluded / out&#8209;of&#8209;frame</i></p>'

parts = {
    "__CSS__": CSS,
    "__LIB_CORR__": figure("libero_spatial_demo0/corr_lines_libero_t47.png",
        "Correspondence lines",
        "85/100 points still in view at t=47; each line links a source pixel to its predicted match."),
    "__LIB_CYCLE__": figure("libero_spatial_demo0/cycle_libero_t47.png",
        "Round-trip",
        "0&rarr;47&rarr;0 on visible points: start (o) and returned (x) coincide &mdash; median 2.2&nbsp;px."),
    "__LIB_TRAILS__": figure("libero_spatial_demo0/trails_libero.png",
        "Forward tracks",
        "Points on the arm are carried along its 3-D sweep (colour = time); background points stay put."),
    "__LIB_GIF__": figure("libero_spatial_demo0/track_libero_small.gif",
        "Animated track",
        "The tracked grid across the episode. " + "green = visible, dimmed = occluded."),
    "__HOT_CORR__": figure("hot3d_puzzle_toy/corr_lines_hot3d_t47.png",
        "Correspondence lines",
        "After a large pan, only 16/100 source points remain in view &mdash; correctly flagged; the visible matches land on the right scene structure."),
    "__HOT_CYCLE__": figure("hot3d_puzzle_toy/cycle_hot3d_t47.png",
        "Round-trip",
        "0&rarr;47&rarr;0 on visible points &mdash; the round-trip returns to the start."),
    "__HOT_TRAILS__": figure("hot3d_puzzle_toy/trails_hot3d.png",
        "Forward tracks",
        "Tracks follow scene structure through heavy egocentric motion."),
}

html = HTML
for k, v in parts.items():
    html = html.replace(k, v)
OUT.write_text(html)
print("wrote", OUT, f"({OUT.stat().st_size/1e6:.1f} MB)")
