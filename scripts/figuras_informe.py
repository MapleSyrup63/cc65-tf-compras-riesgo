"""Genera las figuras del informe a partir de los JSON producidos por los programas Go
(reports/*.json). Solo dibuja: todos los cálculos se hacen en Go."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 160})
OUT = "docs/img/"
eda = json.load(open("reports/eda.json"))
tr = json.load(open("reports/train_report.json"))
bl = json.load(open("reports/bench_loader.json"))
bt = json.load(open("reports/bench_train.json"))
fmt = lambda v: f"{v:,.0f}"

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT + name, bbox_inches="tight"); plt.close(fig)

# 1. órdenes por mes
m = sorted(eda["por_mes_emision"].items())
fig, ax = plt.subplots(figsize=(7.5, 3.2))
xs = [k[3:] for k, _ in m]; ys = [v["ordenes"] for _, v in m]
ax.bar(xs, ys, color=BLUE, width=0.6)
for x, y in zip(xs, ys): ax.text(x, y, f"{y/1000:.0f}k", ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.set_title("Órdenes de compra y servicio por mes de emisión (2026)", loc="left")
ax.set_ylabel("Órdenes"); ax.yaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k"); ax.grid(axis="x", visible=False)
save(fig, "fig_ordenes_mes.png")

# 2. tasa de riesgo por departamento
d = sorted(eda["por_departamento"].items(), key=lambda kv: kv[1]["riesgo"] / kv[1]["ordenes"])
fig, ax = plt.subplots(figsize=(7.5, 6.2))
names = [k.title() for k, _ in d]; rates = [100 * v["riesgo"] / v["ordenes"] for _, v in d]
ax.barh(names, rates, color=BLUE, height=0.65)
for i, r in enumerate(rates): ax.text(r + 0.2, i, f"{r:.1f}%", va="center", fontsize=8, color=INK2)
tot = 100 * eda["total"]["riesgo"] / eda["total"]["ordenes"]
ax.axvline(tot, color=GRAY, ls="--", lw=1); ax.text(tot + 0.2, len(d) - 0.6, f"Promedio nacional {tot:.1f}%", fontsize=8, color=INK2)
ax.set_title("Tasa de órdenes en riesgo (anulación o sobrecosto) por departamento", loc="left")
ax.set_xlabel("% de órdenes en riesgo"); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8.5)
save(fig, "fig_riesgo_departamento.png")

# 3. histograma de montos (escala log)
order = ["10^0", "10^1", "10^2", "10^3", "10^4", "10^5", "10^6", ">=10^7"]
labels = ["1–10", "10–100", "100–1k", "1k–10k", "10k–100k", "100k–1M", "1M–10M", "≥10M"]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
vals = [eda["hist_log10_monto"].get(k, 0) for k in order]
ax.bar(labels, vals, color=BLUE, width=0.6)
for i, v in enumerate(vals): ax.text(i, v, fmt(v), ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_yscale("log"); ax.set_title("Distribución del monto por orden (S/, escala logarítmica)", loc="left")
ax.set_ylabel("Órdenes (log)"); ax.set_xlabel("Rango de monto en soles"); ax.grid(axis="x", visible=False)
save(fig, "fig_hist_monto.png")

# 4. tasa de riesgo por nivel de gobierno y por tipo de contratista
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.1), gridspec_kw={"width_ratios": [2, 1.2]})
lab = {"GOB_NACIONAL": "Gob. nacional", "SALUD": "Salud", "GOB_REGIONAL": "Gob. regional", "UNIVERSIDAD": "Universidad",
       "MUNI_DISTRITAL": "Muni. distrital", "MUNI_PROVINCIAL": "Muni. provincial", "EMPRESA": "Empresa (RUC 20)",
       "PN": "Persona natural (RUC 10)", "OTRO": "Otro"}
for ax, key, title in [(axs[0], "por_nivel_gobierno", "Por nivel de gobierno"), (axs[1], "por_tipo_contratista", "Por tipo de contratista")]:
    it = sorted(eda[key].items(), key=lambda kv: kv[1]["riesgo"] / kv[1]["ordenes"])
    n = [lab[k] for k, _ in it]; r = [100 * v["riesgo"] / v["ordenes"] for _, v in it]
    ax.barh(n, r, color=BLUE, height=0.6)
    for i, v in enumerate(r): ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_title(title, loc="left", fontsize=10); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlim(0, max(r) * 1.25)
fig.suptitle("Tasa de órdenes en riesgo", x=0.02, ha="left", fontweight="bold", color=INK)
save(fig, "fig_riesgo_nivel.png")

# 5. importancia de variables
imp = tr["random_forest"]["importancia"][:12][::-1]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.barh([i["feature"] for i in imp], [100 * i["importancia"] for i in imp], color=BLUE, height=0.6)
for k, i in enumerate(imp): ax.text(100 * i["importancia"] + 0.3, k, f"{100*i['importancia']:.1f}%", va="center", fontsize=8, color=INK2)
ax.set_title("Random Forest: importancia de variables (reducción de Gini, top 12)", loc="left")
ax.set_xlabel("% de la ganancia total"); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8.5)
save(fig, "fig_importancia.png")

# 6. speedup
fig, ax = plt.subplots(figsize=(7.5, 3.6))
series = [("Cargador (Parte A)", BLUE, "o", [(r["workers"], r["speedup"]) for r in bl["resultados"]]),
          ("Random Forest", ORANGE, "s", [(r["workers"], r["speedup"]) for r in bt["resultados"] if r["modelo"] == "random_forest"]),
          ("Regresión logística", AQUA, "^", [(r["workers"], r["speedup"]) for r in bt["resultados"] if r["modelo"] == "logistic_regression"])]
ax.plot([1, 2, 4, 8], [1, 2, 2, 2], color=GRAY, ls="--", lw=1.2, label="Ideal con 2 núcleos")
for name, c, mk, pts in series:
    x, y = zip(*pts); ax.plot(x, y, color=c, lw=2, marker=mk, ms=7, label=name, markeredgecolor="white", markeredgewidth=1.2)
    dy = {"Regresión logística": 0.04, "Cargador (Parte A)": -0.04}.get(name, 0)
    ax.text(x[-1] * 1.08, y[-1] + dy, f"{name.split(' (')[0]} {y[-1]:.2f}x", fontsize=8, va="center", color=INK)
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"]); ax.set_xlim(0.85, 16)
ax.set_ylim(0.8, 2.3); ax.set_xlabel("Workers (goroutines)"); ax.set_ylabel("Speedup vs. 1 worker")
ax.set_title("Escalabilidad medida (máquina de 2 núcleos lógicos)", loc="left"); ax.legend(frameon=False, fontsize=8, loc="lower right")
save(fig, "fig_speedup.png")

# 7. comparación de modelos
rf, lr = tr["random_forest"]["test"], tr["logistic_regression"]["test"]
mets = [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc_average_precision"), ("Precisión", "precision"), ("Recall", "recall"), ("F1", "f1")]
fig, ax = plt.subplots(figsize=(7.5, 3.2)); import numpy as np
x = np.arange(len(mets)); w = 0.36
b1 = ax.bar(x - w/2 - 0.01, [rf[k] for _, k in mets], w, color=BLUE, label="Random Forest")
b2 = ax.bar(x + w/2 + 0.01, [lr[k] for _, k in mets], w, color=ORANGE, label="Regresión logística")
for bars in (b1, b2):
    for b in bars: ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([m for m, _ in mets]); ax.set_ylim(0, 1.08); ax.grid(axis="x", visible=False)
ax.set_title("Desempeño en el conjunto de prueba (463,800 órdenes)", loc="left"); ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")
save(fig, "fig_modelos.png")
print("ok")
