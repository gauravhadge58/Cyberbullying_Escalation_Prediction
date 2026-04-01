"""
Generate the architecture diagram for the Cyberbullying Escalation Prediction application.
Outputs: docs/architecture_diagram.png  (relative to the repo root)

Run from the repository root:
    python docs/generate_architecture_diagram.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe


# ─── helpers ──────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, sublabel=None,
        facecolor="#2d3748", edgecolor="#4a5568",
        textcolor="white", fontsize=9, radius=0.015):
    """Draw a rounded box with an optional sub-label."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.5, edgecolor=edgecolor,
        facecolor=facecolor, zorder=3
    )
    ax.add_patch(patch)
    ty = y + (0.012 if sublabel else 0)
    ax.text(x, ty, label, ha="center", va="center",
            fontsize=fontsize, color=textcolor,
            fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.022, sublabel, ha="center", va="center",
                fontsize=7, color="#a0aec0", zorder=4)


def section(ax, x, y, w, h, title, facecolor, edgecolor, titlecolor,
            title_fontsize=9):
    """Draw a large section container with a title banner."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=2, edgecolor=edgecolor,
        facecolor=facecolor, zorder=1
    )
    ax.add_patch(patch)
    # title banner
    banner = FancyBboxPatch(
        (x - w / 2, y + h / 2 - 0.055), w, 0.055,
        boxstyle="round,pad=0,rounding_size=0.015",
        linewidth=0, facecolor=edgecolor, zorder=2
    )
    ax.add_patch(banner)
    ax.text(x, y + h / 2 - 0.027, title, ha="center", va="center",
            fontsize=title_fontsize, color=titlecolor,
            fontweight="bold", zorder=3)


def arrow(ax, x1, y1, x2, y2, label="", color="#718096",
          style="->", lw=1.5, labelcolor="#cbd5e0", fontsize=7):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color,
                        lw=lw, connectionstyle="arc3,rad=0.0"),
        zorder=5
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.005, my, label, ha="left", va="center",
                fontsize=fontsize, color=labelcolor, zorder=6)


def bidir_arrow(ax, x1, y1, x2, y2, label="", color="#718096", lw=1.5,
                labelcolor="#cbd5e0", fontsize=7):
    arrow(ax, x1, y1, x2, y2, label, color, "<->", lw, labelcolor, fontsize)


# ─── canvas ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(18, 11))
fig.patch.set_facecolor("#1a202c")
ax.set_facecolor("#1a202c")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ─── title ────────────────────────────────────────────────────────────────────

ax.text(0.5, 0.965, "Cyberbullying Detection & Escalation Prediction — System Architecture",
        ha="center", va="center", fontsize=14, color="white",
        fontweight="bold", zorder=10)
ax.text(0.5, 0.942, "Full-Stack: React  •  Node.js/Express  •  Python/FastAPI  •  MongoDB  •  Docker",
        ha="center", va="center", fontsize=9, color="#a0aec0", zorder=10)

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — USER / CLIENT
# ══════════════════════════════════════════════════════════════════════════════

section(ax, 0.5, 0.875, 0.92, 0.065,
        "CLIENT LAYER", "#2a2f3a", "#4299e1", "white", 9)

box(ax, 0.22, 0.875, 0.17, 0.038, "Web Browser", "User Interface",
    "#2b6cb0", "#4299e1", "white", 8)
box(ax, 0.5,  0.875, 0.17, 0.038, "React Dashboard", "Vite + Tailwind CSS",
    "#2b6cb0", "#4299e1", "white", 8)
box(ax, 0.78, 0.875, 0.17, 0.038, "Chart.js Analytics", "React Router DOM",
    "#2b6cb0", "#4299e1", "white", 8)

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — FRONTEND PAGES
# ══════════════════════════════════════════════════════════════════════════════

section(ax, 0.5, 0.775, 0.92, 0.075,
        "FRONTEND SERVICE  (React 18 + Vite)   Port 5173   [Docker: frontend]",
        "#1e2a38", "#4299e1", "#90cdf4", 8.5)

pages = [
    (0.12, "Dashboard",   "Stats overview"),
    (0.27, "Monitor",     "Live WS updates"),
    (0.42, "Analytics",   "Historical charts"),
    (0.57, "Training",    "CSV upload & UI"),
    (0.72, "Simulator",   "Test predictions"),
    (0.87, "Nginx Proxy", "Reverse proxy"),
]
for px, name, sub in pages:
    box(ax, px, 0.775, 0.135, 0.042, name, sub,
        "#2c5282", "#4299e1", "white", 7.5)

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — BACKEND
# ══════════════════════════════════════════════════════════════════════════════

section(ax, 0.5, 0.635, 0.92, 0.100,
        "BACKEND SERVICE  (Node.js + Express)   Port 5000   [Docker: backend]",
        "#1e2a2a", "#38a169", "#9ae6b4", 8.5)

backend_boxes = [
    (0.13, "Express Server",  "HTTP + REST API"),
    (0.28, "WebSocket Server","ws — real-time"),
    (0.43, "API Routes",      "/predict /train\n/conversations /stats"),
    (0.58, "Mute Manager",    "Auto-mute logic\nviolation tracking"),
    (0.73, "Mongoose ODM",    "MongoDB models\nMessage, Conversation"),
    (0.87, "Axios Proxy",     "Forward to\nML Service"),
]
for bx, name, sub in backend_boxes:
    box(ax, bx, 0.640, 0.135, 0.055, name, sub,
        "#22543d", "#38a169", "white", 7.5)

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — ML SERVICE
# ══════════════════════════════════════════════════════════════════════════════

section(ax, 0.5, 0.465, 0.92, 0.120,
        "ML SERVICE  (Python 3.10 + FastAPI)   Port 8000   [Docker: ml-service]",
        "#2d1f1a", "#ed8936", "#fbd38d", 8.5)

# routers
ml_router_boxes = [
    (0.13, "/predict",       "Toxicity + Escalation"),
    (0.27, "/train",         "CSV pipeline"),
    (0.41, "/train/progress","Training status"),
    (0.55, "/models",        "Registry CRUD"),
    (0.69, "/conversations", "In-memory cache"),
    (0.83, "/stats + /health","Analytics + health"),
]
for rx, name, sub in ml_router_boxes:
    box(ax, rx, 0.500, 0.125, 0.040, name, sub,
        "#7b341e", "#ed8936", "white", 7, 0.012)

# ML modules row
ml_module_boxes = [
    (0.13, "Preprocessing",   "Text cleaning\nSlang normalise"),
    (0.27, "BERT Detection",  "unitary/toxic-bert\n6 toxicity labels"),
    (0.41, "Escalation Model","Hybrid: Rule+RF\n+LSTM voting"),
    (0.55, "Random Forest",   "100 trees\n9 features"),
    (0.69, "LSTM Model",      "PyTorch\nTemporal seq"),
    (0.83, "Model Registry",  "registry.json\nversioning"),
]
for mx, name, sub in ml_module_boxes:
    box(ax, mx, 0.435, 0.125, 0.050, name, sub,
        "#652b19", "#c05621", "white", 7, 0.012)

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 5 — DATA / INFRA
# ══════════════════════════════════════════════════════════════════════════════

section(ax, 0.5, 0.300, 0.92, 0.095,
        "DATA & INFRASTRUCTURE   [Docker: mongodb + volumes]",
        "#1a1f2e", "#805ad5", "#d6bcfa", 8.5)

data_boxes = [
    (0.13, "MongoDB 6",         "Port 27017"),
    (0.27, "Messages Collection","messageId, userId\ntoxicityScore, isBullying"),
    (0.41, "Conversations Coll.","escalationLevel\nescalationScore"),
    (0.55, "Datasets",          "hateXplain.csv\nformatted_train.csv"),
    (0.69, "Saved Models",      ".joblib + .pth\nregistry.json"),
    (0.83, "Docker Compose",    "4-service orch.\nhealth checks"),
]
for dx, name, sub in data_boxes:
    box(ax, dx, 0.310, 0.125, 0.055, name, sub,
        "#44337a", "#805ad5", "white", 7, 0.012)

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE (side panel)
# ══════════════════════════════════════════════════════════════════════════════

section(ax, 0.5, 0.148, 0.92, 0.095,
        "TRAINING PIPELINE  (CSV Upload → 3-Phase Model Training)",
        "#1a2228", "#667eea", "#c3dafe", 8.5)

train_steps = [
    (0.10, "CSV Upload",     "Frontend → Backend\n→ ML Service"),
    (0.25, "Phase 1: BERT",  "Toxicity detection\nper message"),
    (0.40, "Phase 2: RF",    "Feature extract\n→ Random Forest"),
    (0.55, "Phase 3: LSTM",  "Seq modelling\n20 epochs"),
    (0.70, "Model Registry", "Save + version\nSet active"),
    (0.85, "Deploy Ready",   "Instant prediction\nusing new model"),
]
for tx, name, sub in train_steps:
    box(ax, tx, 0.155, 0.130, 0.055, name, sub,
        "#3c366b", "#667eea", "white", 7, 0.012)

# arrows inside training pipeline (left to right)
for i in range(len(train_steps) - 1):
    x1 = train_steps[i][0] + 0.065
    x2 = train_steps[i + 1][0] - 0.065
    arrow(ax, x1, 0.155, x2, 0.155, color="#667eea", lw=1.5)

# ══════════════════════════════════════════════════════════════════════════════
#  INTER-LAYER ARROWS
# ══════════════════════════════════════════════════════════════════════════════

arrow_color  = "#718096"
ws_color     = "#f6ad55"
http_color   = "#68d391"
db_color     = "#b794f4"

# Browser ↔ Frontend
bidir_arrow(ax, 0.5, 0.842, 0.5, 0.813, "HTTP/HTTPS", arrow_color)

# Frontend ↔ Backend (HTTP REST)
arrow(ax, 0.38, 0.738, 0.38, 0.690, "REST API", http_color, lw=1.5, fontsize=7)
arrow(ax, 0.42, 0.690, 0.42, 0.738, "JSON resp", http_color, lw=1.5, fontsize=7)

# WebSocket frontend ↔ backend
arrow(ax, 0.61, 0.738, 0.61, 0.690, "WS connect", ws_color, lw=1.5, fontsize=7)
arrow(ax, 0.65, 0.690, 0.65, 0.738, "PREDICTION\nUPDATE", ws_color, lw=1.5, fontsize=7)

# Backend → ML Service (HTTP)
arrow(ax, 0.38, 0.590, 0.38, 0.525, "HTTP POST\n/predict /train", http_color, lw=1.5, fontsize=7)
arrow(ax, 0.42, 0.525, 0.42, 0.590, "JSON results", http_color, lw=1.5, fontsize=7)

# Backend → MongoDB
arrow(ax, 0.63, 0.590, 0.63, 0.358, "Mongoose\nread/write", db_color, lw=1.5, fontsize=7)
arrow(ax, 0.67, 0.358, 0.67, 0.590, "Documents", db_color, lw=1.5, fontsize=7)

# ML modules → saved models
arrow(ax, 0.55, 0.410, 0.55, 0.358, "Save model\nfiles", "#ed8936", lw=1.5, fontsize=7)

# ML routers ↔ ML modules
arrow(ax, 0.5, 0.479, 0.5, 0.461, color="#ed8936", lw=1.2)

# ══════════════════════════════════════════════════════════════════════════════
#  LEGEND
# ══════════════════════════════════════════════════════════════════════════════

legend_items = [
    (mpatches.Patch(color="#4299e1"), "React Frontend"),
    (mpatches.Patch(color="#38a169"), "Node.js Backend"),
    (mpatches.Patch(color="#ed8936"), "Python ML Service"),
    (mpatches.Patch(color="#805ad5"), "Data / Infrastructure"),
    (mpatches.Patch(color="#667eea"), "Training Pipeline"),
    (plt.Line2D([0], [0], color="#68d391", lw=2), "REST HTTP"),
    (plt.Line2D([0], [0], color="#f6ad55", lw=2), "WebSocket"),
    (plt.Line2D([0], [0], color="#b794f4", lw=2), "Database I/O"),
]
handles = [h for h, _ in legend_items]
labels  = [l for _, l in legend_items]
leg = ax.legend(handles, labels, loc="lower center",
                ncol=8, bbox_to_anchor=(0.5, 0.008),
                framealpha=0.15, facecolor="#2d3748",
                edgecolor="#4a5568", fontsize=7.5,
                labelcolor="white", handlelength=1.5,
                columnspacing=1.2, handletextpad=0.6)

# ─── save ─────────────────────────────────────────────────────────────────────

# Output path is always alongside this script (docs/architecture_diagram.png)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_diagram.png")
plt.tight_layout(pad=0.3)
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved → {out}")
