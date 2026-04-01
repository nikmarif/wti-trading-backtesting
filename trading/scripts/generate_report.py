#!/usr/bin/env python3
"""
generate_report.py — PDF report from pipeline artefacts.

Usage
-----
    python scripts/generate_report.py
    python scripts/generate_report.py --artifacts data/artifacts --out data/artifacts/report.pdf
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

W, H = A4

# ── Colour palette ──────────────────────────────────────────────────────────
DARK   = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#0f3460")
LIGHT  = colors.HexColor("#e94560")
GREY   = colors.HexColor("#f5f5f5")
MID    = colors.HexColor("#cccccc")


# ── Styles ───────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "title", fontSize=22, leading=28, textColor=DARK,
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle", fontSize=11, leading=15, textColor=ACCENT,
        fontName="Helvetica", alignment=TA_CENTER, spaceAfter=2,
    )
    styles["caption"] = ParagraphStyle(
        "caption", fontSize=8, leading=11, textColor=colors.grey,
        fontName="Helvetica", alignment=TA_CENTER,
    )
    styles["h2"] = ParagraphStyle(
        "h2", fontSize=13, leading=17, textColor=ACCENT,
        fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4,
    )
    styles["body"] = ParagraphStyle(
        "body", fontSize=9, leading=13, textColor=DARK,
        fontName="Helvetica", spaceAfter=4,
    )
    styles["disclaimer"] = ParagraphStyle(
        "disclaimer", fontSize=7.5, leading=11, textColor=colors.grey,
        fontName="Helvetica-Oblique", alignment=TA_CENTER,
    )
    return styles


# ── Table helpers ─────────────────────────────────────────────────────────────
_BASE_TABLE_STYLE = TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0), ACCENT),
    ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, 0), 9),
    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ("TOPPADDING",    (0, 0), (-1, 0), 6),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GREY, colors.white]),
    ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE",   (0, 1), (-1, -1), 9),
    ("TOPPADDING",    (0, 1), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
    ("GRID",  (0, 0), (-1, -1), 0.4, MID),
    ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ("ALIGN", (0, 0), (0, -1), "LEFT"),
])

def make_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(_BASE_TABLE_STYLE)
    return t


def img(path: Path, width_cm: float = 14) -> Image:
    w = width_cm * cm
    im = Image(str(path), width=w, height=w * 0.55)
    im.hAlign = "CENTER"
    return im


# ── Report builder ────────────────────────────────────────────────────────────
def build_report(artifacts_dir: Path, out_path: Path) -> None:
    styles = make_styles()
    story = []

    # ── Load data ──────────────────────────────────────────────────────────
    with (artifacts_dir / "metrics.json").open() as f:
        metrics = json.load(f)

    sweep = pd.read_csv(artifacts_dir / "threshold_sweep.csv")
    overall = metrics["overall"]
    per_fold = metrics.get("per_fold", [])

    # ── Cover ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 2.5 * cm))
    story.append(Paragraph("WTI 1-Minute Return Predictor", styles["title"]))
    story.append(Paragraph("Out-of-Sample Evaluation Report", styles["subtitle"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="80%", thickness=1.5, color=LIGHT, hAlign="CENTER"))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}  |  "
        f"Data: WTI/USD 1-min  |  Period: 2018–2023  |  Model: XGBoost (GPU)",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 1.5 * cm))

    # Quick-stat boxes as a 4-col table
    qs_data = [
        ["OOS Bars", "Features", "CV Folds", "Sign Accuracy"],
        [
            f"{overall['n_obs']:,}",
            "59",
            str(len(per_fold)),
            f"{overall['sign_accuracy']:.2%}",
        ],
    ]
    qs_style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("BACKGROUND",    (0, 1), (-1, 1), ACCENT),
        ("TEXTCOLOR",     (0, 1), (-1, 1), colors.white),
        ("FONTNAME",      (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 1), (-1, 1), 14),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.white),
        ("ROUNDEDCORNERS", [4]),
    ])
    qs = Table(qs_data, colWidths=[3.8 * cm] * 4, hAlign="CENTER")
    qs.setStyle(qs_style)
    story.append(qs)

    story.append(Spacer(1, 1 * cm))

    # Executive summary
    story.append(Paragraph("Executive Summary", styles["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "This report evaluates an XGBoost machine learning model trained to predict the "
        "next 1-minute log return of WTI crude oil futures (2018–2023). "
        "The model was assessed using a rigorous walk-forward validation framework — "
        "meaning it was always trained on past data and tested on future data it had never seen, "
        "closely mimicking real-world deployment.",
        styles["body"],
    ))
    story.append(Paragraph(
        "The key finding is that 1-minute price returns are extremely difficult to predict from "
        "historical price data alone. The model achieves a directional accuracy of ~52% "
        "(vs 50% for random guessing) and a near-zero correlation with actual returns. "
        "After realistic transaction costs, no threshold tested produced a reliably profitable "
        "trading strategy over the 6-year evaluation period.",
        styles["body"],
    ))
    story.append(Paragraph(
        "This result is consistent with the academic literature on high-frequency return prediction "
        "and does not indicate a flaw in the model — rather, it reflects the efficient nature of "
        "liquid commodity futures markets at the 1-minute horizon. Potential paths to improvement "
        "include incorporating order flow data, macro event signals, or cross-asset features.",
        styles["body"],
    ))

    story.append(PageBreak())

    # ── 1. Regression Metrics ──────────────────────────────────────────────
    story.append(Paragraph("1. Regression Metrics (Out-of-Sample)", styles["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID))
    story.append(Spacer(1, 0.2 * cm))

    reg_data = [
        ["Metric", "Value", "Notes"],
        ["RMSE",          f"{overall['rmse']:.6f}",  "Root mean squared error on OOS predictions"],
        ["MAE",           f"{overall['mae']:.6f}",   "Mean absolute error on OOS predictions"],
        ["Pearson Corr.", f"{overall['corr']:.4f}",  "Linear correlation between y_pred and y_true"],
        ["Sign Accuracy", f"{overall['sign_accuracy']:.4f}",
         "Fraction of bars where sign(pred) == sign(actual)"],
        ["N Observations", f"{overall['n_obs']:,}", "Total out-of-sample bars evaluated"],
    ]
    story.append(make_table(reg_data, col_widths=[4 * cm, 3 * cm, 9.5 * cm]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<b>What does this mean?</b> A Pearson correlation of 0.006 is very close to zero — "
        "the model's predictions and the actual returns are almost unrelated. "
        "This is expected: 1-minute commodity returns are extremely noisy and nearly unpredictable "
        "from price history alone (the efficient market hypothesis in action). "
        "Sign accuracy of ~52% means the model correctly calls the direction of the next bar "
        "slightly more often than a coin flip, but only barely.",
        styles["body"],
    ))

    story.append(Spacer(1, 0.6 * cm))

    # ── 2. Backtest Results ────────────────────────────────────────────────
    story.append(Paragraph("2. Backtest Results — Threshold Sweep", styles["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID))
    story.append(Spacer(1, 0.2 * cm))

    sweep_data = [["Threshold", "Sharpe (ann.)", "Total Log Ret.", "Max Drawdown", "Hit Rate", "Trades"]]
    for _, row in sweep.iterrows():
        sweep_data.append([
            f"{row['threshold']:.5f}",
            f"{row['annualised_sharpe']:.3f}",
            f"{row['total_log_return']:.4f}",
            f"{row['max_drawdown']:.4f}",
            f"{row['hit_rate']:.3f}",
            f"{int(row['n_trades']):,}",
        ])
    story.append(make_table(sweep_data, col_widths=[2.5*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.3*cm, 2.3*cm]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<b>How to read this table:</b> The threshold is the minimum predicted return required to "
        "place a trade. A higher threshold means the model only trades when it is most confident — "
        "resulting in fewer trades but potentially cleaner signals. "
        "The Sharpe ratio measures risk-adjusted return (higher is better; above 1.0 is generally "
        "considered good). A negative Sharpe means the strategy lost money on a risk-adjusted basis. "
        "At threshold=0.001 the Sharpe turns marginally positive (0.032), but with only 8 trades "
        "over 6 years this is statistically meaningless — far too few to draw any conclusion.",
        styles["body"],
    ))

    story.append(PageBreak())

    # ── 3. Plots ───────────────────────────────────────────────────────────
    plot_specs = [
        ("actual_vs_predicted.png",  "3. Actual vs Predicted Log Returns",
         "Left — scatter plot of predicted vs actual returns. A perfect model would show a tight "
         "diagonal line; the near-circular cloud here confirms very low predictive power. "
         "Right — actual (blue) vs predicted (orange) returns over time. Notice how the predicted "
         "line is almost flat near zero: the model has learned that the safest guess is always "
         "\"close to zero\", because that minimises average error when the signal is weak."),
        ("prediction_histogram.png", "4. Distribution of OOS Predictions",
         "How spread out are the model's predictions? Almost everything falls in a tight spike "
         "around zero — between the two dashed threshold lines. This is why so few trades are "
         "triggered: the model almost never predicts a return large enough to clear the ±0.0002 "
         "bar. This behaviour (called regression shrinkage) is mathematically correct given the "
         "low signal, but it means the strategy sits flat most of the time."),
        ("cumulative_returns.png",   "5. Cumulative Strategy Returns",
         "The running total of strategy profit/loss (after 1 bp cost + 0.5 bp slippage per side) "
         "at the default threshold of 0.0002. A flat or rising line would indicate edge; "
         "the downward drift here confirms the model does not generate exploitable alpha "
         "at this frequency and threshold after realistic costs."),
        ("fold_metrics.png",         "6. Per-Fold Performance Over Time",
         "Each point represents one walk-forward validation window (~3.5 days). "
         "RMSE (top) spikes around 2020 — this corresponds to the COVID-19 oil price collapse, "
         "where 1-minute returns became unusually large and unpredictable. "
         "Sign accuracy (bottom) hovers around 50–52%, confirming the marginal directional edge "
         "is consistent across time but very small."),
        ("threshold_sweep.png",      "7. Threshold Sensitivity Analysis",
         "How sensitive are the results to the trading threshold? "
         "As the threshold rises, fewer trades are taken (right axis, falling bars) but the "
         "signal quality per trade may improve. The Sharpe curve (left axis) shows diminishing "
         "losses at higher thresholds, but also vanishing trade count — a classic signal/noise "
         "trade-off with no clearly profitable region."),
        ("feature_importance.png",   "8. Feature Importance (Last Fold)",
         "Which input variables did XGBoost rely on most? Importance here is measured by "
         "\"gain\" — how much each feature reduced prediction error across all tree splits. "
         "Recent lag returns (ret_lag_1, ret_lag_2) and rolling volatility measures dominate, "
         "suggesting short-term momentum and variance are the most informative signals available "
         "in raw price data. Cyclical time features (hour/minute encodings) provide only marginal lift."),
    ]

    for fname, heading, caption in plot_specs:
        p = artifacts_dir / fname
        if not p.exists():
            continue
        story.append(Paragraph(heading, styles["h2"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=MID))
        story.append(Spacer(1, 0.2 * cm))
        story.append(img(p, width_cm=15))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(caption, styles["caption"]))
        story.append(Spacer(1, 0.6 * cm))

    story.append(PageBreak())

    # ── 9. Methodology ────────────────────────────────────────────────────
    story.append(Paragraph("9. Methodology", styles["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID))
    story.append(Spacer(1, 0.2 * cm))

    method_rows = [
        ["Component",      "Detail"],
        ["Data",           "HistData WTI/USD 1-minute OHLCV, 2018–2023 (1,966,702 bars after cleaning)"],
        ["Target",         "1-bar-ahead log return: log(close[t+1] / close[t])"],
        ["Features",       "59 features: lag returns (1–20), rolling mean/std/skew/kurt (5,10,20,60 bars), "
                           "momentum (3,5,10,20 bars), cyclical time (sin/cos of hour, minute, day-of-week)"],
        ["Model",          "XGBoost Regressor (GPU, tree_method=hist) — n_estimators=500, "
                           "max_depth=4, lr=0.04, early_stopping=50 rounds"],
        ["Validation",     "Expanding-window walk-forward CV: 694 folds, 15k-bar min train, "
                           "5k-bar val windows, 2.5k-bar step"],
        ["Backtest",       "Long if pred > threshold, short if pred < -threshold. "
                           "Cost: 1 bp/side + 0.5 bp slippage"],
        ["Annualisation",  "98,000 min/year (252 days × 390 equity-hours equivalent)"],
    ]
    story.append(make_table(method_rows, col_widths=[3.5 * cm, 13 * cm]))

    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "DISCLAIMER: This report is produced from research code for educational and analytical purposes only. "
        "It does not constitute financial advice, investment recommendations, or a solicitation to trade. "
        "Past simulated performance is not indicative of future results. Transaction costs in live trading "
        "may differ significantly from those modelled here.",
        styles["disclaimer"],
    ))

    # ── Build PDF ──────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="WTI 1-Min Return Predictor — Report",
        author="WTI Return Predictor Pipeline",
    )
    doc.build(story)
    print(f"Report saved -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default=None)
    p.add_argument("--out",       default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    artifacts_dir = Path(args.artifacts) if args.artifacts else root / "data" / "artifacts"
    out_path      = Path(args.out)       if args.out       else artifacts_dir / "report.pdf"

    build_report(artifacts_dir, out_path)
