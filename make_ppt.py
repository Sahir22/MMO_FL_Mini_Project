"""
Generates MMO-FL implementation walkthrough presentation.
Run: python3 make_ppt.py
Output: MMO_FL_Implementation.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Palette ──────────────────────────────────────────────────────────────────
DARK_BG   = RGBColor(0x1E, 0x1E, 0x2E)   # near-black blue
ACCENT    = RGBColor(0x89, 0xB4, 0xFA)   # soft blue
ACCENT2   = RGBColor(0xA6, 0xE3, 0xA1)   # soft green
ACCENT3   = RGBColor(0xF3, 0x8B, 0xA8)   # soft pink
YELLOW    = RGBColor(0xF9, 0xE2, 0xAF)   # warm yellow
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY= RGBColor(0xCC, 0xD0, 0xDA)
MID_GRAY  = RGBColor(0x45, 0x47, 0x5A)
CODE_BG   = RGBColor(0x18, 0x18, 0x28)

W = Inches(13.33)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

blank_layout = prs.slide_layouts[6]  # completely blank


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_slide():
    s = prs.slides.add_slide(blank_layout)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = DARK_BG
    return s


def box(slide, x, y, w, h, fill=None, border=None, border_w=Pt(1)):
    from pptx.util import Emu
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid() if fill else shape.fill.background()
    if fill:
        shape.fill.fore_color.rgb = fill
    shape.line.fill.background() if border is None else None
    if border:
        shape.line.color.rgb = border
        shape.line.width = border_w
    else:
        shape.line.fill.background()
    return shape


def txt(slide, text, x, y, w, h, size=20, bold=False, color=WHITE,
        align=PP_ALIGN.LEFT, italic=False, wrap=True):
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf.word_wrap = wrap
    p = tf.text_frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tf


def heading(slide, text, y=0.18, size=32, color=ACCENT):
    txt(slide, text, 0.4, y, 12.5, 0.7, size=size, bold=True, color=color,
        align=PP_ALIGN.LEFT)


def subheading(slide, text, y=0.85, color=LIGHT_GRAY):
    txt(slide, text, 0.4, y, 12.5, 0.5, size=18, color=color)


def divider(slide, y=0.82, color=ACCENT):
    line = slide.shapes.add_connector(1, Inches(0.4), Inches(y), Inches(12.93), Inches(y))
    line.line.color.rgb = color
    line.line.width = Pt(1.5)


def bullet_block(slide, items, x, y, w, h, size=17, color=WHITE,
                 marker="▸", marker_color=ACCENT2):
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf.word_wrap = True
    frame = tf.text_frame
    frame.word_wrap = True
    for i, item in enumerate(items):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.space_before = Pt(4)
        # marker
        r0 = p.add_run()
        r0.text = marker + "  "
        r0.font.size = Pt(size)
        r0.font.color.rgb = marker_color
        r0.font.bold = True
        # text
        r1 = p.add_run()
        r1.text = item
        r1.font.size = Pt(size)
        r1.font.color.rgb = color


def code_block(slide, code_lines, x, y, w, h, size=13):
    bg = box(slide, x, y, w, h, fill=CODE_BG, border=MID_GRAY, border_w=Pt(1))
    tf = slide.shapes.add_textbox(Inches(x+0.15), Inches(y+0.12),
                                   Inches(w-0.3), Inches(h-0.25))
    tf.word_wrap = False
    frame = tf.text_frame
    for i, line in enumerate(code_lines):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.name = "Courier New"
        # simple syntax colouring
        if line.strip().startswith("#"):
            r.font.color.rgb = RGBColor(0x6C, 0x70, 0x86)
        elif any(kw in line for kw in ["def ", "class ", "return ", "import ", "from "]):
            r.font.color.rgb = ACCENT
        elif any(kw in line for kw in ["if ", "else", "for ", "while "]):
            r.font.color.rgb = ACCENT3
        else:
            r.font.color.rgb = LIGHT_GRAY


def tag(slide, label, x, y, fill=ACCENT, text_color=DARK_BG, size=13):
    w = len(label) * 0.095 + 0.2
    b = box(slide, x, y, w, 0.28, fill=fill)
    b.adjustments[0] = 0.1
    txt(slide, label, x+0.07, y+0.04, w, 0.25, size=size, bold=True,
        color=text_color, align=PP_ALIGN.LEFT)


def arrow(slide, x1, y1, x2, y2, color=ACCENT, w=Pt(2)):
    from pptx.util import Inches
    from pptx.oxml.ns import qn
    conn = slide.shapes.add_connector(2, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = color
    conn.line.width = w


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
box(s, 0, 0, 13.33, 7.5, fill=DARK_BG)
# accent bar left
box(s, 0, 0, 0.08, 7.5, fill=ACCENT)
# big title
txt(s, "MMO-FL + PMM", 0.5, 1.5, 12.3, 1.4, size=52, bold=True, color=ACCENT,
    align=PP_ALIGN.CENTER)
txt(s, "Multimodal Online Federated Learning", 0.5, 2.9, 12.3, 0.7,
    size=26, color=WHITE, align=PP_ALIGN.CENTER)
txt(s, "with Modality Missing in IoT", 0.5, 3.55, 12.3, 0.6,
    size=26, color=WHITE, align=PP_ALIGN.CENTER)
divider(s, y=4.35, color=MID_GRAY)
txt(s, "Wang et al., IEEE TMC 2026  ·  Implementation Walkthrough",
    0.5, 4.5, 12.3, 0.5, size=16, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)
txt(s, "Paper 4 · Code replication", 0.5, 5.1, 12.3, 0.4,
    size=14, color=MID_GRAY, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Agenda
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "What We'll Cover")
divider(s)
items = [
    ("1", "Problem: Multimodal FL with missing sensors",   ACCENT),
    ("2", "System model & MMO-FL workflow",                ACCENT2),
    ("3", "PMM algorithm — OPC + OPS",                    ACCENT3),
    ("4", "Project structure & key files",                 YELLOW),
    ("5", "Model architectures (UCI-HAR & MVSA)",         ACCENT),
    ("6", "Data loaders & streaming buffers",              ACCENT2),
    ("7", "FL training loop (train.py)",                   ACCENT3),
    ("8", "Benchmarks: FM / PM / ZF / PMM",               YELLOW),
    ("9", "How to run — commands",                         ACCENT),
    ("10","Ablation studies",                              ACCENT2),
]
for i, (num, label, col) in enumerate(items):
    row = i % 5
    col_x = 0.5 if i < 5 else 6.8
    y = 1.15 + row * 1.1
    b = box(s, col_x, y, 0.45, 0.45, fill=col)
    txt(s, num, col_x+0.07, y+0.05, 0.35, 0.38, size=16, bold=True,
        color=DARK_BG, align=PP_ALIGN.CENTER)
    txt(s, label, col_x+0.58, y+0.07, 5.8, 0.38, size=16, color=WHITE)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Problem Statement
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "The Problem: Missing Modalities in IoT FL")
divider(s)

# IoT scenario diagram
box(s, 0.4, 1.05, 12.5, 5.85, fill=RGBColor(0x24, 0x27, 0x3A), border=MID_GRAY)

# Server box
box(s, 5.4, 1.25, 2.55, 0.9, fill=ACCENT, border=None)
txt(s, "☁  Server (Cloud)", 5.45, 1.3, 2.5, 0.8, size=15, bold=True,
    color=DARK_BG, align=PP_ALIGN.CENTER)

# 3 client boxes
for i, (cx, has_miss) in enumerate([(0.6, False), (4.1, True), (7.6, False)]):
    col = ACCENT2 if not has_miss else ACCENT3
    box(s, cx, 3.5, 2.55, 2.9, fill=RGBColor(0x18, 0x18, 0x28), border=col, border_w=Pt(2))
    txt(s, f"Client {i+1}", cx+0.1, 3.6, 2.3, 0.4, size=14, bold=True, color=col)
    modalities = ["Image", "Text", "Audio"] if not has_miss else ["Image", "✗ MISSING", "Audio"]
    for j, m in enumerate(modalities):
        mc = ACCENT2 if "✗" not in m else ACCENT3
        box(s, cx+0.15, 4.15+j*0.7, 2.2, 0.55, fill=RGBColor(0x2A, 0x2D, 0x3E), border=mc)
        txt(s, m, cx+0.2, 4.2+j*0.7, 2.1, 0.45, size=13,
            color=mc if "✗" in m else LIGHT_GRAY, align=PP_ALIGN.CENTER)
    arrow(s, cx+1.3, 3.5, 6.65, 2.15, color=col)

txt(s, "Sensor failure → one modality unavailable for some global rounds",
    0.5, 6.6, 12.3, 0.4, size=15, color=YELLOW, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — MMO-FL Workflow
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "MMO-FL Workflow per Global Round t")
divider(s)

steps = [
    ("1", "Server broadcasts Θᵗ to all clients",           ACCENT),
    ("2", "Each client collects new streaming data D_k^t",  ACCENT2),
    ("3", "Full-modality clients: run OGD, compute local prototypes (OPC)", ACCENT2),
    ("4", "Missing-modality clients: inject prototypes → run OGD (OPS)",    ACCENT3),
    ("5", "All clients upload local model Θ̃_k^{t+1}",     ACCENT),
    ("6", "Server: FedAvg → Θ^{t+1}, update persistent prototypes P̄",     YELLOW),
]
for i, (num, label, col) in enumerate(steps):
    y = 1.1 + i * 0.96
    b = box(s, 0.4, y, 0.5, 0.5, fill=col)
    txt(s, num, 0.4, y+0.05, 0.5, 0.42, size=16, bold=True,
        color=DARK_BG, align=PP_ALIGN.CENTER)
    box(s, 1.1, y, 11.6, 0.58, fill=RGBColor(0x24, 0x27, 0x3A), border=col, border_w=Pt(1))
    txt(s, label, 1.25, y+0.08, 11.2, 0.45, size=16, color=WHITE)
    if i < 5:
        arrow(s, 0.65, y+0.5, 0.65, y+0.96, color=MID_GRAY)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — PMM Algorithm
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "PMM: Prototypical Modality Mitigation")
divider(s)

# OPC column
box(s, 0.4, 1.05, 5.9, 5.85, fill=RGBColor(0x1A, 0x2A, 0x1A), border=ACCENT2, border_w=Pt(2))
txt(s, "OPC — Online Prototypes Construction", 0.6, 1.1, 5.5, 0.5,
    size=15, bold=True, color=ACCENT2)
bullet_block(s, [
    "Runs every round (full-modality clients)",
    "For each class c & modality m:",
    "  p_k^{t,m,c} = mean of θᵐ(x) over class-c samples",
    "Server averages across clients → temporal proto p_c^{t,m}",
    "Persistent update:  p̄_c^{t,m} = ((t-1)·p̄ + p_c^t) / t",
    "Prototypes broadcast to all clients",
], 0.55, 1.7, 5.65, 3.8, size=14, marker="→", marker_color=ACCENT2)
txt(s, "Eq. 15 – 17", 0.6, 5.6, 5.5, 0.4, size=12, color=MID_GRAY, italic=True)

# OPS column
box(s, 7.0, 1.05, 5.9, 5.85, fill=RGBColor(0x2A, 0x1A, 0x1A), border=ACCENT3, border_w=Pt(2))
txt(s, "OPS — Online Prototypes Substitution", 7.2, 1.1, 5.5, 0.5,
    size=15, bold=True, color=ACCENT3)
bullet_block(s, [
    "Runs when a modality m is missing",
    "Client downloads P̄ᵗ from server",
    "For each sample n: look up p̄_{y_n}^{t,m}",
    "Inject as FEATURE (skip encoder entirely)",
    "Head encoder runs on substituted features",
    "Missing encoder gets zero gradient",
], 7.15, 1.7, 5.65, 3.8, size=14, marker="→", marker_color=ACCENT3)
txt(s, "Eq. 18 – 19", 7.2, 5.6, 5.5, 0.4, size=12, color=MID_GRAY, italic=True)

# middle arrow
box(s, 6.15, 3.2, 0.7, 0.5, fill=ACCENT)
txt(s, "uses", 6.17, 3.25, 0.7, 0.42, size=13, bold=True,
    color=DARK_BG, align=PP_ALIGN.CENTER)
arrow(s, 5.9, 3.45, 6.15, 3.45, color=ACCENT)
arrow(s, 6.85, 3.45, 7.0, 3.45, color=ACCENT)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — The Key Fix: OPS at Feature Level
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "Critical Implementation Detail: OPS at Feature Level")
divider(s)

txt(s, "The bug in naive implementation:", 0.4, 1.0, 12.5, 0.5,
    size=18, bold=True, color=ACCENT3)
code_block(s, [
    "# ✗ WRONG — prototype (128,) written into raw input slot (B, 128, 3)",
    "modality_data[missing_m] = proto_feats   # shape mismatch or silent corruption",
    "logits = model(modality_data)            # encoder called on prototype tensor!",
], 0.4, 1.55, 12.5, 1.05, size=13)

txt(s, "Correct implementation — feature-level injection:", 0.4, 2.75, 12.5, 0.5,
    size=18, bold=True, color=ACCENT2)
code_block(s, [
    "# fl/pmm.py — get_proto_features()",
    "def get_proto_features(labels, missing_m, global_prototypes, feat_dim=128):",
    "    rows = [global_prototypes.get((missing_m, y.item()), zero) for y in labels]",
    "    return torch.stack(rows)   # (N, 128) — already a feature tensor",
    "",
    "# models/ucihar_model.py — MMOFLModel.forward_ops()",
    "def forward_ops(self, modality_data, missing_m, proto_features):",
    "    features = []",
    "    for m, (enc, x) in enumerate(zip(self.encoders, modality_data)):",
    "        if m == missing_m:",
    "            features.append(proto_features.detach())  # skip encoder entirely",
    "        else:",
    "            features.append(enc(x))                   # normal encode",
    "    return self.head(torch.cat(features, dim=-1))",
], 0.4, 3.3, 12.5, 3.6, size=12)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Project Structure
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "Project Structure")
divider(s)

tree = [
    ("mmo_fl/",                          WHITE,    0),
    ("├── data/",                         ACCENT,   1),
    ("│   ├── ucihar.py",                ACCENT2,  2),
    ("│   └── mvsa.py",                  ACCENT2,  2),
    ("├── models/",                       ACCENT,   1),
    ("│   ├── ucihar_model.py",          ACCENT2,  2),
    ("│   └── mvsa_model.py",            ACCENT2,  2),
    ("├── fl/",                           ACCENT,   1),
    ("│   ├── client.py",                ACCENT2,  2),
    ("│   ├── server.py",                ACCENT2,  2),
    ("│   └── pmm.py",                   ACCENT2,  2),
    ("├── baselines/benchmarks.py",       ACCENT,   1),
    ("├── ablation/run_ablations.py",     ACCENT,   1),
    ("└── train.py",                      YELLOW,   1),
]
descs = {
    "ucihar.py":        "Loader, Dirichlet split, StreamingBuffer feeds",
    "mvsa.py":          "Image+text loader, tokeniser, Dirichlet split",
    "ucihar_model.py":  "AccelEncoder, GyroEncoder, HeadEncoder, MMOFLModel + forward_ops",
    "mvsa_model.py":    "ImageEncoder, TextEncoder, build_mvsa_model()",
    "client.py":        "StreamingBuffer, DataPool, do_local_update (OGD)",
    "server.py":        "fedavg(), update_global_prototypes() (OPC server)",
    "pmm.py":           "compute_local_prototypes() (OPC), get_proto_features() (OPS)",
    "benchmarks.py":    "run_fm / run_pm / run_zf / run_pmm",
    "run_ablations.py": "λ, α, quantisation, delayed-update sweeps",
    "train.py":         "mmo_fl_train(), evaluate(), main() entry point",
}
for i, (name, col, indent) in enumerate(tree):
    x = 0.55 + indent * 0.25
    y = 1.05 + i * 0.4
    txt(s, name, x, y, 4.5-indent*0.25, 0.38, size=14,
        bold=(indent == 0), color=col)
    stem = name.strip("│├└ /").replace("── ", "").strip()
    if stem in descs:
        txt(s, "—  " + descs[stem], 5.2, y, 7.7, 0.38, size=13, color=LIGHT_GRAY)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Model Architectures
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "Model Architectures")
divider(s)

# UCI-HAR
box(s, 0.4, 1.05, 5.9, 5.85, fill=RGBColor(0x1A, 0x1A, 0x2A), border=ACCENT, border_w=Pt(2))
txt(s, "UCI-HAR", 0.6, 1.1, 5.5, 0.45, size=16, bold=True, color=ACCENT)
for label, detail in [
    ("AccelEncoder", "Conv1D ×5 + AdaptiveAvgPool + FC → (B, 128)"),
    ("GyroEncoder",  "LSTM(3→128) × 1 + FC → (B, 128)"),
    ("HeadEncoder",  "FC(256→128) + ReLU + FC(128→6)"),
    ("Input shapes", "accel: (B, 128, 3)   gyro: (B, 128, 3)"),
    ("Concat dim",   "256 = 128 + 128"),
    ("Classes",      "6 activity classes"),
]:
    idx = list(d[0] for d in [
        ("AccelEncoder",""), ("GyroEncoder",""), ("HeadEncoder",""),
        ("Input shapes",""), ("Concat dim",""), ("Classes",""),
    ]).index(label)
    y = 1.65 + idx * 0.75
    txt(s, label, 0.6, y, 2.0, 0.42, size=13, bold=True, color=ACCENT2)
    txt(s, detail, 2.65, y, 3.5, 0.42, size=13, color=LIGHT_GRAY)

# MVSA
box(s, 7.0, 1.05, 5.9, 5.85, fill=RGBColor(0x1A, 0x1A, 0x2A), border=ACCENT3, border_w=Pt(2))
txt(s, "MVSA-Single", 7.2, 1.1, 5.5, 0.45, size=16, bold=True, color=ACCENT3)
for label, detail in [
    ("ImageEncoder", "Conv2D ×4 + AdaptiveAvgPool2D + FC → (B, 128)"),
    ("TextEncoder",  "Embed(10k, 64) + LSTM(2-layer) + FC → (B, 128)"),
    ("HeadEncoder",  "FC(256→128) + ReLU + FC(128→3)"),
    ("Input shapes", "img: (B, 3, 64, 64)   text: (B, 32) token ids"),
    ("Concat dim",   "256 = 128 + 128"),
    ("Classes",      "3 sentiment classes  (neg / neu / pos)"),
]:
    idx = list(d[0] for d in [
        ("ImageEncoder",""), ("TextEncoder",""), ("HeadEncoder",""),
        ("Input shapes",""), ("Concat dim",""), ("Classes",""),
    ]).index(label)
    y = 1.65 + idx * 0.75
    txt(s, label, 7.2, y, 2.2, 0.42, size=13, bold=True, color=ACCENT2)
    txt(s, detail, 9.45, y, 3.2, 0.42, size=13, color=LIGHT_GRAY)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Data Pipeline
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "Data Pipeline: Streaming Buffers")
divider(s)

# Flow diagram
stages = [
    ("Raw Dataset\n(static)",         ACCENT,  0.5),
    ("Dirichlet Split\n(non-IID α)",  ACCENT2, 3.3),
    ("DataPool\n(long-term)",         ACCENT3, 6.1),
    ("StreamingBuffer\n(local, 500)", YELLOW,  8.9),
    ("get_all()\nbatch",              ACCENT,  11.7),
]
for i, (label, col, x) in enumerate(stages):
    box(s, x, 2.5, 1.7, 1.2, fill=RGBColor(0x24, 0x27, 0x3A), border=col, border_w=Pt(2))
    txt(s, label, x+0.05, 2.58, 1.6, 1.05, size=13, color=col,
        align=PP_ALIGN.CENTER)
    if i < len(stages)-1:
        arrow(s, x+1.7, 3.1, x+2.8, 3.1, color=MID_GRAY)

txt(s, "update(n_new=20)  →  add 20 new, drop 20 oldest  (FIFO maxlen deque)",
    0.4, 4.0, 12.5, 0.4, size=14, color=ACCENT2, align=PP_ALIGN.CENTER)

bullet_block(s, [
    "UCI-HAR  : initial 2000 samples / client  ·  buffer size 500  ·  20 in/out per round",
    "MVSA     : initial 1500 samples / client  ·  buffer size 800  ·  20 in/out per round",
    "Samples stored as (modality_list, label)  —  numpy arrays for UCI-HAR, tensors for MVSA",
    "get_all() detects tensor vs numpy automatically and collates into (modality_data, labels)",
], 0.4, 4.6, 12.5, 2.4, size=15)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Training Loop Code Walkthrough
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "Training Loop — mmo_fl_train() in train.py")
divider(s)

code_block(s, [
    "for t in range(T):",
    "    global_state = copy.deepcopy(server_model.state_dict())",
    "    is_missing_round = (torch.rand(1).item() < lambda_missing)  # λ prob",
    "    missing_modality = 0 if is_missing_round else None",
    "",
    "    for k, client in enumerate(clients):",
    "        client.model.load_state_dict(global_state)",
    "        client.buffer.update(n_new=20)          # streaming update",
    "        batch = client.buffer.get_all()",
    "",
    "        if client_missing is None:",
    "            local_protos = compute_local_prototypes(model, batch)  # OPC",
    "            do_local_update(model, batch, E, lr_t)                 # OGD",
    "        else:",
    "            proto_feat = get_proto_features(labels, missing_m, prototypes)",
    "            do_local_update(model, batch, E, lr_t,                 # OPS",
    "                            missing=missing_m, proto_features=proto_feat)",
    "",
    "    server_model = fedavg(server_model, local_states)             # FedAvg",
    "    prototypes = update_global_prototypes(prototypes, new_protos, t)",
], 0.4, 1.05, 12.5, 6.25, size=12)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Benchmarks
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "Benchmarks: FM / PM / ZF / PMM")
divider(s)

rows = [
    ("FM",  "Full Modality",    "λ = 0",    "All modalities always available. Upper-bound baseline.",            ACCENT2),
    ("PM",  "Partial Modality", "skip enc", "Missing encoder NOT called. Zero-pad feature slot. No compensation.",ACCENT),
    ("ZF",  "Zero Filling",     "zero inp", "Missing encoder's RAW INPUT zeroed. Encoder still runs, gets grad.", YELLOW),
    ("PMM", "Proposed Method",  "proto sub","Prototype feature injected at feature level. Missing enc skipped.",  ACCENT3),
]
for i, (abbr, name, tag_txt, desc, col) in enumerate(rows):
    y = 1.1 + i * 1.4
    box(s, 0.4, y, 12.5, 1.22, fill=RGBColor(0x24, 0x27, 0x3A), border=col, border_w=Pt(2))
    b = box(s, 0.55, y+0.18, 0.75, 0.75, fill=col)
    txt(s, abbr, 0.55, y+0.22, 0.75, 0.65, size=18, bold=True,
        color=DARK_BG, align=PP_ALIGN.CENTER)
    txt(s, name, 1.45, y+0.1, 2.5, 0.42, size=15, bold=True, color=col)
    txt(s, f"[{tag_txt}]", 1.45, y+0.58, 2.5, 0.38, size=13,
        color=LIGHT_GRAY, italic=True)
    txt(s, desc, 4.1, y+0.3, 8.6, 0.65, size=14, color=WHITE)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — How to Run
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "How to Run")
divider(s)

txt(s, "① Install", 0.4, 1.05, 3.0, 0.4, size=15, bold=True, color=ACCENT)
code_block(s, [
    "pip3 install torch torchvision numpy scipy Pillow",
], 0.4, 1.5, 12.5, 0.5, size=14)

txt(s, "② Smoke test (no dataset needed)", 0.4, 2.15, 6.0, 0.4, size=15, bold=True, color=ACCENT2)
code_block(s, [
    "python3 train.py --dataset synthetic --rounds 10 --clients 3",
], 0.4, 2.6, 12.5, 0.5, size=14)

txt(s, "③ UCI-HAR full run (paper settings)", 0.4, 3.25, 7.0, 0.4, size=15, bold=True, color=ACCENT3)
code_block(s, [
    "python3 train.py --dataset ucihar --data-path './UCI HAR Dataset' \\",
    "    --rounds 100 --clients 5 --alpha 1 --lambda-missing 0.5 --seeds 10",
], 0.4, 3.7, 12.5, 0.75, size=14)

txt(s, "④ MVSA-Single full run", 0.4, 4.6, 5.0, 0.4, size=15, bold=True, color=YELLOW)
code_block(s, [
    "python3 train.py --dataset mvsa --data-path './MVSA_Single' \\",
    "    --rounds 120 --lr 0.01 --decay 0.99 --buffer-size 800 --initial-per-client 1500",
], 0.4, 5.05, 12.5, 0.75, size=14)

txt(s, "⑤ Ablation studies", 0.4, 5.95, 5.0, 0.4, size=15, bold=True, color=ACCENT)
code_block(s, [
    "python3 ablation/run_ablations.py --study missing_rate --dataset ucihar --rounds 100",
], 0.4, 6.4, 12.5, 0.5, size=14)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — Ablation Studies
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "Ablation Studies (Section VII-E)")
divider(s)

ablations = [
    ("Fig. 5", "Modality Missing Rate",
     "λ ∈ {0.3, 0.5, 0.7}",
     "Lower λ → better accuracy. PMM degrades gracefully; PM collapses at λ=0.7",
     ACCENT),
    ("Fig. 6", "Non-IID Level",
     "α ∈ {1, 5, 10}",
     "Higher α = more homogeneous → better accuracy. PMM less sensitive than baselines",
     ACCENT2),
    ("Fig. 7", "Quantized Upload",
     "b ∈ {2, 4} bits on prototypes",
     "Minor accuracy drop vs. communication savings. b=4 nearly matches b=32 (full precision)",
     ACCENT3),
    ("Fig. 8", "Delayed OPC Update",
     "DL ∈ {2, 4} rounds between OPC runs",
     "Moderate perf. drop; prototypes stay stale. Good compute/comm trade-off for IoT",
     YELLOW),
]
for i, (fig, name, param, finding, col) in enumerate(ablations):
    y = 1.05 + i * 1.5
    box(s, 0.4, y, 12.5, 1.32, fill=RGBColor(0x24, 0x27, 0x3A), border=col, border_w=Pt(1))
    txt(s, fig, 0.55, y+0.08, 0.9, 0.5, size=13, bold=True, color=col,
        align=PP_ALIGN.CENTER)
    txt(s, name, 1.55, y+0.08, 3.5, 0.4, size=15, bold=True, color=col)
    txt(s, param, 1.55, y+0.58, 3.5, 0.38, size=13, color=LIGHT_GRAY, italic=True)
    txt(s, finding, 5.3, y+0.3, 7.4, 0.75, size=13, color=WHITE)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — Key Equations Summary
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "Key Equations")
divider(s)

eqs = [
    ("Loss (eq. 1)",
     "F_t(Θ, D^t) = (1/K) Σ_k  f_t( θ⁰(Z_k^{t,1}, …, Z_k^{t,M}), Y_k )",
     ACCENT),
    ("OGD update (eq. 3)",
     "Θ_k^{t,τ+1} = Θ_k^{t,τ} − η · ∇F_t(Θ_k^{t,τ}; D_k^t)",
     ACCENT2),
    ("OPC local proto (eq. 15)",
     "p_k^{t,m,c} = (1/|X_k^{t,m,c}|) Σ_{n: y=c}  θᵐ(x_k,n^{t,m})",
     ACCENT3),
    ("OPC persistent update (eq. 17)",
     "p̄_c^{t,m} = ( (t−1)·p̄_c^{t-1,m} + p_c^{t,m} ) / t",
     YELLOW),
    ("OPS substitution (eq. 19)",
     "Z̃_k^{t,m} = [ p̄_c(k,1)^{t,m}, …, p̄_c(k,N)^{t,m} ]",
     ACCENT),
    ("FedAvg (eq. 7)",
     "Θ^{t+1} = (1/K)( Σ_{k∈S_t} Θ_k^{t+1} + Σ_{k∉S_t} Θ̃_k^{t+1} )",
     ACCENT2),
]
for i, (label, eq, col) in enumerate(eqs):
    y = 1.05 + i * 1.02
    box(s, 0.4, y, 2.2, 0.75, fill=col)
    txt(s, label, 0.45, y+0.12, 2.1, 0.55, size=13, bold=True,
        color=DARK_BG, align=PP_ALIGN.CENTER)
    box(s, 2.75, y, 10.2, 0.75, fill=CODE_BG, border=col, border_w=Pt(1))
    txt(s, eq, 2.9, y+0.15, 9.9, 0.5, size=14, color=WHITE)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — Results Summary
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
heading(s, "Paper Results Summary")
divider(s)

txt(s, "Configuration: λ=0.5, α=1 (UCI-HAR) / α=1 (MVSA),  T=100/120 rounds",
    0.4, 1.05, 12.5, 0.45, size=14, color=LIGHT_GRAY, italic=True)

# table header
for j, (hdr, x, w) in enumerate([("Method", 0.4, 2.0), ("UCI-HAR acc", 2.55, 3.0),
                                    ("MVSA acc", 5.7, 3.0), ("vs FM gap", 8.85, 4.0)]):
    box(s, x, 1.6, w, 0.5, fill=MID_GRAY)
    txt(s, hdr, x+0.05, 1.65, w-0.1, 0.4, size=14, bold=True,
        color=WHITE, align=PP_ALIGN.CENTER)

rows = [
    ("FM  (upper)",  "~0.65",  "~0.88", "baseline",     ACCENT2),
    ("PM",           "~0.52",  "~0.71", "−0.13 / −0.17",ACCENT),
    ("ZF",           "~0.55",  "~0.75", "−0.10 / −0.13",YELLOW),
    ("PMM (ours)",   "~0.67",  "~0.90", "0 / +0.02 ✓",  ACCENT3),
]
for i, (method, uci, mvsa, gap, col) in enumerate(rows):
    y = 2.2 + i * 0.9
    bg = RGBColor(0x1E, 0x1E, 0x2E) if i % 2 == 0 else RGBColor(0x24, 0x27, 0x3A)
    for j, (val, x, w) in enumerate([(method, 0.4, 2.0), (uci, 2.55, 3.0),
                                       (mvsa, 5.7, 3.0), (gap, 8.85, 4.0)]):
        box(s, x, y, w, 0.78, fill=bg, border=col if j == 0 else None, border_w=Pt(2))
        txt(s, val, x+0.08, y+0.2, w-0.15, 0.42, size=14,
            color=col if j == 0 else WHITE, align=PP_ALIGN.CENTER,
            bold=(j == 0))

txt(s, "PMM surpasses FM — prototypes act as additional regularisation signal",
    0.4, 6.35, 12.5, 0.5, size=15, bold=True, color=ACCENT3, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 16 — Closing
# ════════════════════════════════════════════════════════════════════════════
s = add_slide()
box(s, 0, 0, 13.33, 7.5, fill=DARK_BG)
box(s, 0, 0, 0.08, 7.5, fill=ACCENT3)

txt(s, "Implementation Complete", 0.5, 1.8, 12.3, 1.0,
    size=40, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

bullet_block(s, [
    "7 files fixed / added — OPS bug, mvsa.py, model factories, CLI",
    "Smoke test passes:  python3 train.py --dataset synthetic --rounds 5",
    "Ready for UCI-HAR and MVSA-Single real-data runs",
    "All 4 ablation studies wired up in ablation/run_ablations.py",
], 1.5, 3.0, 10.3, 2.8, size=18, marker="✓", marker_color=ACCENT2)

divider(s, y=5.8, color=MID_GRAY)
txt(s, "Paper:  Wang et al., 'Multimodal Online Federated Learning With Modality Missing in IoT'  ·  IEEE TMC 2026",
    0.5, 6.0, 12.3, 0.5, size=13, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════
out = "/Users/i569940/Library/CloudStorage/OneDrive-SAPSE/Desktop/Claude Setup/UseCases/paper/mmo_fl/MMO_FL_Implementation.pptx"
prs.save(out)
print(f"Saved: {out}")
print(f"Slides: {len(prs.slides)}")
