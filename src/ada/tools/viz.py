"""EDA charts — direct ports of Day 1 cells 26-30, retargeted at the canonical
column names (id, text, ts, author, engagement, platform).

All functions accept a DataFrame + an output directory and return the saved
path. They never call plt.show().
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.0)
plt.rcParams["figure.dpi"] = 110
# zh-TW glyph fallback — system font that ships with Windows 10/11
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def temporal_chart(df: pd.DataFrame, out_dir: Path, landfall: pd.Timestamp | None = None) -> Path | None:
    if "ts" not in df.columns:
        return None
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    series = (
        pd.DataFrame({"ts": ts, "id": df["id"]})
        .dropna(subset=["ts"])
        .set_index("ts")
        .resample("6h")["id"]
        .count()
    )
    if series.empty:
        return None
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(series.index, series.values, alpha=0.3, color="steelblue")
    ax.plot(series.index, series.values, color="steelblue", linewidth=2)
    if landfall is not None:
        ax.axvline(landfall, color="red", linewidth=2, linestyle="--", label="landfall")
        ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.set_xlabel("Date / Time (UTC)")
    ax.set_ylabel("Posts (6h bins)")
    ax.set_title("Temporal posting volume", fontweight="bold")
    return _save(fig, out_dir, "temporal")


def platform_author_chart(df: pd.DataFrame, out_dir: Path) -> Path | None:
    has_plat = "platform" in df.columns
    has_auth = "author" in df.columns
    if not has_plat and not has_auth:
        return None
    n_panels = int(has_plat) + int(has_auth)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    i = 0
    if has_plat:
        ax = axes[i]; i += 1
        counts = df["platform"].value_counts()
        ax.pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=sns.color_palette("Set2", len(counts)),
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
            startangle=140,
        )
        ax.set_title("Platform distribution", fontweight="bold")
    if has_auth:
        ax = axes[i]
        counts = df["author"].value_counts(dropna=False)
        bot_terms = {"疑似機器人", "suspected_bot", "bot"}
        colors = ["#e74c3c" if str(a) in bot_terms else "#3498db" for a in counts.index]
        ax.barh(counts.index[::-1], counts.values[::-1],
                color=colors[::-1], edgecolor="white")
        ax.set_xlabel("Count")
        ax.set_title("Author type (red = bots)", fontweight="bold")
    return _save(fig, out_dir, "platform_author")


def engagement_chart(df: pd.DataFrame, out_dir: Path) -> Path | None:
    if "engagement" not in df.columns:
        return None
    eng = pd.to_numeric(df["engagement"], errors="coerce").dropna()
    if eng.empty:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.hist(eng, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
    ax1.set_yscale("log")
    ax1.set_xlabel("Engagement"); ax1.set_ylabel("Posts (log)")
    ax1.set_title("Engagement distribution (log)", fontweight="bold")
    ax1.axvline(eng.median(), color="red", linestyle="--",
                linewidth=1.5, label=f"median: {eng.median():.0f}")
    ax1.axvline(eng.mean(), color="orange", linestyle="--",
                linewidth=1.5, label=f"mean: {eng.mean():.0f}")
    ax1.legend(fontsize=8)

    if "platform" in df.columns:
        cap = eng.quantile(0.95)
        sub = df[pd.to_numeric(df["engagement"], errors="coerce") <= cap]
        order = sub.groupby("platform")["engagement"].median().sort_values(ascending=False).index
        sub.boxplot(column="engagement", by="platform", ax=ax2,
                    grid=False, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.5))
        ax2.set_title("By platform (<= p95)", fontweight="bold")
        ax2.set_xlabel("Platform"); ax2.set_ylabel("Engagement")
        plt.suptitle("")
    return _save(fig, out_dir, "engagement")


def text_length_chart(df: pd.DataFrame, out_dir: Path) -> Path | None:
    if "text" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(11, 4))
    df = df.copy()
    df["_len"] = df["text"].fillna("").astype(str).str.len()
    if "platform" in df.columns:
        order = df.groupby("platform")["_len"].median().sort_values(ascending=False).index
        for p in order:
            ax.hist(df[df["platform"] == p]["_len"], bins=40, alpha=0.55, label=p, density=True)
        ax.legend(title="Platform", fontsize=8)
    else:
        ax.hist(df["_len"], bins=40, color="steelblue", alpha=0.7)
    ax.set_xlim(0, 300)
    ax.set_xlabel("Text length (chars)")
    ax.set_ylabel("Density")
    ax.set_title("Text length distribution", fontweight="bold")
    return _save(fig, out_dir, "text_length")


def top_engagement_chart(df: pd.DataFrame, out_dir: Path, n: int = 15) -> Path | None:
    if "engagement" not in df.columns or "text" not in df.columns:
        return None
    eng = pd.to_numeric(df["engagement"], errors="coerce")
    top = df.assign(_eng=eng).dropna(subset=["_eng"]).nlargest(n, "_eng")
    if top.empty:
        return None
    fig, ax = plt.subplots(figsize=(13, 6))
    type_colors = {
        "一般使用者": "#3498db", "意見領袖": "#9b59b6", "主流媒體": "#e67e22",
        "政府機關": "#27ae60", "公眾人物": "#e74c3c", "疑似機器人": "#34495e",
    }
    if "author" in top.columns:
        bar_colors = [type_colors.get(str(a), "#95a5a6") for a in top["author"]]
    else:
        bar_colors = ["#3498db"] * len(top)
    labels = [
        (f"{(str(row['platform'])[:3] + ' | ') if 'platform' in top.columns else ''}"
         f"{str(row['text'])[:40]}...")
        for _, row in top.iterrows()
    ]
    ax.barh(range(len(top)), top["_eng"].values,
            color=bar_colors, edgecolor="white", linewidth=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Engagement")
    ax.set_title(f"Top {n} posts by engagement", fontweight="bold")
    if "author" in top.columns:
        legend = [Patch(facecolor=c, label=t) for t, c in type_colors.items()
                  if t in top["author"].values]
        if legend:
            ax.legend(handles=legend, loc="lower right", fontsize=8)
    return _save(fig, out_dir, "top_engagement")
