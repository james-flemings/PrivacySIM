"""Privacy-persona clustering for RQ2.

5-quadrant taxonomy borrowing the 2D layout from Dupree et al. (2016) but
labeled by the (stance, exposure) quadrant rather than the original Dupree
archetype names. The cluster predictors are two per-domain z-scored axes
that are INDEPENDENT of the LLM evaluation outcome (questionnaire
`responses`), to avoid circularity:

    stated_z   — privacy_attitudes Likert items (polarity-corrected mean).
                 Used as the *stance* axis (self-reported privacy concern).
    pe_z       — previous_experiences AI-exposure count proxy.
                 Used as the *exposure* axis (AI-tool familiarity).

Note: `pe_z` is AI exposure, not Dupree's original privacy-knowledge axis,
so we label clusters by (stance, exposure) quadrant rather than inheriting
Dupree's archetype names. The rough Dupree correspondence is preserved as
a comment for readers familiar with the prior work.

We deliberately do NOT include `revealed_z` (mean of questionnaire responses)
in the cluster predictors, because `responses` are the same outcome variable
the LLM is evaluated against.  Mixing them would produce circularity in the
per-cluster alignment analysis.

Cluster labels (stance / exposure) and centers in (pe_z, stated_z) space.
Centers are placed at +0.8 / 0.0 / -0.8 z-scores for HIGH / MID / LOW tiers
(≈ the midpoint of each tertile for a standard normal):

    "high / high"  — (pe_z=+0.8, stated_z=+0.8)   [≈ Dupree Fundamentalist]
    "low / high"   — (pe_z=+0.8, stated_z=-0.8)   [≈ Dupree Lazy Expert]
    "high / mid"   — (pe_z= 0.0, stated_z=+0.8)   [≈ Dupree Technician]
    "mid / mid"    — (pe_z= 0.0, stated_z= 0.0)   [≈ Dupree Amateur]
    "low / low"    — (pe_z=-0.8, stated_z=-0.8)   [≈ Dupree Marginally Concerned]

Each user is assigned to the cluster whose center is closest in
(pe_z, stated_z) space (Euclidean nearest-centroid).

All 5 domains, including CGC, are included in the per-cluster analysis.  CGC's
previous_experiences items use ordinal frequency vocabulary (Daily, Always,
Sometimes, ...), and its privacy_attitudes items use familiarity Likert
vocabulary (Moderately/Extremely/Not knowledgeable).  Both extractors below
recognise those token sets so that CGC users get meaningful (stated_z, pe_z)
scores under the same scheme as the other 4 domains.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import evaluation as ev


CSV_FIELD_LIMIT = sys.maxsize
csv.field_size_limit(CSV_FIELD_LIMIT)


# ---------------------------------------------------------------------------
# Feature extraction


GENERIC_LIKERT_5 = {
    "strongly disagree": 0.0,
    "somewhat disagree": 1.0,
    "disagree": 1.0,
    "moderately disagree": 1.0,
    "neither agree nor disagree": 2.0,
    "neutral": 2.0,
    "somewhat agree": 3.0,
    "moderately agree": 3.0,
    "agree": 3.0,
    "strongly agree": 4.0,
    "completely unacceptable": 0.0,
    "somewhat unacceptable": 1.0,
    "somewhat acceptable": 3.0,
    "completely acceptable": 4.0,
    "very unfamiliar": 0.0,
    "unfamiliar": 1.0,
    "somewhat unfamiliar": 1.0,
    "somewhat familiar": 3.0,
    "very familiar": 4.0,
    "knowledgeable": 3.0,
    "very knowledgeable": 4.0,
    "extremely knowledgeable": 4.0,
    "moderately knowledgeable": 2.0,
    "not at all knowledgeable": 0.0,
    "not knowledgeable at all": 0.0,
    "not knowledgeable": 0.0,
    "slightly knowledgeable": 1.0,
}


# Ordinal frequency vocabulary used by CGC's previous_experiences items
# (e.g., "How often do you interact with chatbots in group chats").  Mapped
# to a [0, 1] exposure score so they can be averaged alongside the binary
# yes/no items used by other domains' PE columns.
FREQUENCY_LIKERT = {
    "never": 0.0,
    "rarely": 0.2,
    "less than weekly": 0.25,
    "less than once a week": 0.25,
    "once a week": 0.4,
    "1 time per week": 0.4,
    "sometimes": 0.5,
    "2-3 times per week": 0.65,
    "often": 0.75,
    "usually": 0.75,
    "4-6 times per week": 0.85,
    "always": 1.0,
    "daily": 1.0,
}


CONCERN_KEYWORDS = (
    "concerned",
    "concern",
    "worry",
    "worried",
    "compromise",
    "breach",
    "leak",
    "uncomfortable",
    "sensitive",
    "risk",
    "fear",
    "wary",
    "private",
    "privacy is important",
    "important is privacy",
    "too much",
)

PERMISSIVE_KEYWORDS = (
    "trust",
    "safe",
    "comfortable",
    "willing",
    "confident",
    "rely",
    "will use",
    "plan to",
    "use ... in the future",
    "in the future",
    "remain private",  # phrased as belief privacy will be respected -> low concern
    "useful",
)


def _parse_dict(raw: Any) -> Dict[str, Any]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {}
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return {}
    text = re.sub(r"(?<![\w\"'])\bnan\b(?![\w\"'])", "None", text, flags=re.IGNORECASE)
    try:
        v = ast.literal_eval(text)
        if isinstance(v, dict):
            return {str(k).strip(): val for k, val in v.items()}
    except (ValueError, SyntaxError):
        pass
    try:
        v = json.loads(text)
        if isinstance(v, dict):
            return {str(k).strip(): val for k, val in v.items()}
    except json.JSONDecodeError:
        pass
    return {}


def _to_unit(value: Any) -> Optional[float]:
    """Map a Likert/numeric response to [0, 1].  Returns None if not parseable."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value):
            return None
        v = float(value)
        if 0 <= v <= 4:
            return v / 4.0
        if 0 <= v <= 100:
            return v / 100.0
        return None
    s = str(value).strip().lower()
    if not s or s == "nan":
        return None
    if s in GENERIC_LIKERT_5:
        return GENERIC_LIKERT_5[s] / 4.0
    try:
        v = float(s)
        if 0 <= v <= 4:
            return v / 4.0
        if 0 <= v <= 5:
            return (v - 1) / 4.0  # 1..5 scale
        if 0 <= v <= 100:
            return v / 100.0
        return None
    except ValueError:
        return None


def _polarity(question_text: str) -> int:
    """Return +1 if higher score == more concern, -1 if lower score == more concern."""
    q = question_text.lower()
    if any(w in q for w in CONCERN_KEYWORDS):
        return +1
    if any(w in q for w in PERMISSIVE_KEYWORDS):
        return -1
    return +1


# Ordered ascending in *believed scope of chatbot access*.  Higher = the user
# perceives broader data exposure, which we treat as a higher stance signal
# (the user is more aware of and attuned to the privacy surface).
CGC_MESSAGE_ACCESS_ORDER = [
    "only messages needed for the normal operation of the chatbot",
    "messages where the chatbot is directly mentioned or addressed",
    "i am not sure about the types of messages a chatbot receives",
    "all messages sent in the chat, excluding those that are later deleted",
    "all messages sent in the chat, including those that are later deleted",
]


def _categorical_to_unit(question_text: str, value: Any) -> Optional[float]:
    """Score categorical / multi-select privacy_attitudes items as a stance proxy.

    Returns a [0, 1] score in the *concern direction* (higher = more concerned /
    more privacy-aware), or None if the item type is not recognised.

    Handles two CGC item families:

    1. Single-select 'What type of messages does a chatbot have access to?' —
       options ordered from least → most believed access.  Treat 'I am not sure'
       as mid (0.5).
    2. Multi-select 'What [info] does a chatbot have access to?  Select all that
       apply.' — score by the *fraction* of selectable items the user picked,
       excluding 'I am not sure'.  More options selected = broader perceived
       exposure = higher concern signal.
    """
    q = str(question_text).strip().lower()
    s = str(value).strip().lower()
    if not s or s in {"nan", "(skipped)", "skipped", "prefer not to say"}:
        return None

    # Family 1: ordered single-select on chatbot message access.
    if "type of messages" in q and "chatbot" in q:
        s_norm = s.rstrip(".")
        for i, opt in enumerate(CGC_MESSAGE_ACCESS_ORDER):
            if s_norm == opt.rstrip("."):
                return i / (len(CGC_MESSAGE_ACCESS_ORDER) - 1)
        if "not sure" in s_norm:
            return 0.5
        return None

    # Family 2: multi-select 'select all that apply' on chatbot data access.
    if "select all that apply" in q and "chatbot" in q:
        # Comma-separated chips; commas inside option text (e.g., "All
        # messages sent in the chat, excluding those that are later deleted")
        # are rare for these items but we conservatively split on a chip
        # delimiter that matches '. ,' or '. , ' or sentence-end + ',' patterns.
        chips = [c.strip() for c in re.split(r"\.\s*,", s) if c.strip()]
        if len(chips) <= 1:
            chips = [c.strip() for c in s.split(",") if c.strip()]
        chips = [c for c in chips if "not sure" not in c]
        if not chips:
            # User answered only 'I am not sure'.  Treat as low signal (mid).
            return 0.5
        # Each chip contributes one selected option.  Score is fraction of a
        # nominal max of 5 options (per the CGC instrument design); cap at 1.
        return min(len(chips) / 5.0, 1.0)

    return None


def stated_concern_score(privacy_attitudes_raw: Any) -> Optional[float]:
    """Compute a 0..1 stated-concern score from a privacy_attitudes dict.

    For each item, try Likert/numeric scoring first (`_to_unit`); fall back to
    categorical/multi-select scoring (`_categorical_to_unit`) for items whose
    values are non-Likert but still encode a stance signal.  Polarity is
    applied per item via heuristic keyword match.  Returns None if no items
    contribute.
    """
    d = _parse_dict(privacy_attitudes_raw)
    if not d:
        return None
    vals: List[float] = []
    for q, v in d.items():
        u = _to_unit(v)
        if u is None:
            u = _categorical_to_unit(q, v)
        if u is None:
            continue
        pol = _polarity(q)
        vals.append(u if pol > 0 else 1.0 - u)
    if not vals:
        return None
    return float(np.mean(vals))


def revealed_concern_score(domain: str, responses_raw: Any) -> Optional[float]:
    """Compute a 0..1 revealed-concern score from questionnaire responses.

    For permissive-direction domains (healthcare/AAP/LCA/LC) higher response =
    more permissive, so concern = 1 - mean(val/range).
    For CGC, items are concern-statements where higher Likert agreement =
    more concern, so concern = mean(val/range).
    """
    if responses_raw is None:
        return None
    parsed = ev.parse_user_responses(responses_raw)
    if not parsed:
        return None
    norm_dom = ev.normalize_text(domain)
    rng = ev.DOMAIN_SCORE_RANGE.get(norm_dom)
    if rng is None or rng <= 0:
        return None
    vals = []
    for k, v in parsed.items():
        if not str(k).strip().startswith("Q"):
            continue
        nv = ev.map_value(domain, v)
        if nv is None:
            continue
        scaled = nv / rng
        if norm_dom == "chatbot group chat":
            vals.append(scaled)
        else:
            vals.append(1.0 - scaled)
    if not vals:
        return None
    return float(np.mean(vals))


def pe_exposure_score(previous_experiences_raw: Any) -> Optional[float]:
    """Mean exposure score across previous_experiences items, normalized 0..1.

    Recognises three vocabularies:
      - binary yes/has-experience tokens (Yes, have used, ...)
      - ordinal frequency tokens (Daily, Always, Sometimes, ...) via FREQUENCY_LIKERT
      - numeric familiarity (1..5)

    Items whose value matches none of these are still counted toward the
    denominator with score 0 (e.g., free-text descriptions, multi-select
    chips with no recognised exposure token).
    """
    d = _parse_dict(previous_experiences_raw)
    if not d:
        return None
    score_sum = 0.0
    total = 0
    for _, v in d.items():
        if v is None:
            continue
        s = str(v).strip().lower()
        if not s or s in {"(skipped)", "skipped", "nan", "prefer not to say"}:
            continue
        total += 1
        if s in FREQUENCY_LIKERT:
            score_sum += FREQUENCY_LIKERT[s]
            continue
        if (
            s.startswith("yes")
            or "have used" in s
            or "have any" in s
            or s.startswith("often")
        ):
            score_sum += 1.0
            continue
        # numeric familiarity (1..5)
        try:
            f = float(s)
            if f >= 3:
                score_sum += 1.0
            continue
        except ValueError:
            pass
    if total == 0:
        return None
    return float(score_sum) / float(total)


# ---------------------------------------------------------------------------
# Cluster assignment


CLUSTERS = (
    "high / high",   # ≈ Dupree Fundamentalist
    "low / high",    # ≈ Dupree Lazy Expert
    "high / mid",    # ≈ Dupree Technician
    "mid / mid",     # ≈ Dupree Amateur
    "low / low",     # ≈ Dupree Marginally Concerned
)


def _is_high(z: Optional[float], thr: float) -> bool:
    if z is None:
        return False
    if isinstance(z, float) and math.isnan(z):
        return False
    return float(z) > thr


# Cluster centers in (pe_z, stated_z) space, labeled by (stance / exposure)
# quadrant. The +0.8 / 0.0 / -0.8 representative values correspond to the
# midpoints of the HIGH / MID / LOW tertiles for a standard normal.
CLUSTER_CENTERS: Dict[str, Tuple[float, float]] = {
    "high / high": (0.8, 0.8),    # ≈ Dupree Fundamentalist
    "low / high":  (0.8, -0.8),   # ≈ Dupree Lazy Expert
    "high / mid":  (0.0, 0.8),    # ≈ Dupree Technician
    "mid / mid":   (0.0, 0.0),    # ≈ Dupree Amateur
    "low / low":   (-0.8, -0.8),  # ≈ Dupree Marginally Concerned
}


def assign_cluster_5quadrant(
    stated_z: float,
    revealed_z: float,  # accepted for signature compat but NOT used
    pe_z: Optional[float],
    *,
    stance_thr: float = 0.43,  # kept for signature compat; unused in centroid mode
    pe_thr: float = 0.43,
) -> str:
    """Nearest-centroid 5-quadrant cluster assignment in (stance, exposure) space.

    Each user is mapped to the closest of the 5 cluster centers in
    (pe_z, stated_z) space using squared Euclidean distance. Cluster
    predictors: stated_z (self-reported privacy stance) and pe_z
    (AI exposure). revealed_z is NOT used --- it is the LLM evaluation
    outcome and including it would produce circularity. Missing pe_z
    falls back to mid (0.0). Returns one of the (stance / exposure)
    labels in CLUSTERS.
    """
    del revealed_z, stance_thr, pe_thr  # explicitly unused in centroid mode
    if pd.isna(stated_z):
        return "Unassigned"
    if pe_z is None or (isinstance(pe_z, float) and math.isnan(pe_z)):
        pe_val = 0.0
    else:
        pe_val = float(pe_z)
    best_label = "Unassigned"
    best_dist = float("inf")
    for label, (pe_c, stated_c) in CLUSTER_CENTERS.items():
        d = (float(stated_z) - stated_c) ** 2 + (pe_val - pe_c) ** 2
        if d < best_dist:
            best_dist = d
            best_label = label
    return best_label


# ---------------------------------------------------------------------------
# Pipeline


def build_features(dataset_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    df["stated_concern"] = df["privacy_attitudes"].apply(stated_concern_score)
    df["revealed_concern"] = df.apply(
        lambda r: revealed_concern_score(r["domain"], r["responses"]), axis=1
    )
    df["pe_exposure"] = df["previous_experiences"].apply(pe_exposure_score)
    return df


def empirical_polarity_check(df: pd.DataFrame) -> pd.DataFrame:
    """For each domain, if stated-revealed correlation is negative, flip stated.

    Operates on the raw 0..1 stated_concern.  Returns a new df with possibly
    flipped stated_concern values and a 'stated_polarity_flipped' column.
    """
    df = df.copy()
    df["stated_polarity_flipped"] = False
    for dom in df["domain"].unique():
        mask = df["domain"] == dom
        sub = df.loc[mask, ["stated_concern", "revealed_concern"]].dropna()
        if len(sub) < 5:
            continue
        if sub["stated_concern"].std() == 0 or sub["revealed_concern"].std() == 0:
            continue
        r = sub["stated_concern"].corr(sub["revealed_concern"])
        if r is not None and r < -0.05:  # meaningfully negative
            df.loc[mask, "stated_concern"] = 1.0 - df.loc[mask, "stated_concern"]
            df.loc[mask, "stated_polarity_flipped"] = True
    return df


def zscore_within_domain(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("domain")[col].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0.0
    )


def assign_clusters(
    df: pd.DataFrame,
    *,
    stance_thr: float = 0.43,
    pe_thr: float = 0.43,
) -> pd.DataFrame:
    """Assign cluster labels per domain.

    Cluster predictors: stated_z and pe_z (NOT revealed_z, to avoid circularity
    with the LLM evaluation outcome).  Users with no parseable stated_z fall
    through to 'Unassigned'.
    """
    df = df.copy()
    df["stated_z"] = zscore_within_domain(df, "stated_concern")
    df["revealed_z"] = zscore_within_domain(df, "revealed_concern")
    if df["pe_exposure"].notna().any():
        df["pe_z"] = zscore_within_domain(df, "pe_exposure")
    else:
        df["pe_z"] = np.nan

    cluster_labels: List[str] = []
    for _, r in df.iterrows():
        s_z = r["stated_z"]
        pe_z_val = r.get("pe_z", np.nan)
        if pd.isna(s_z):
            cluster_labels.append("Unassigned")
            continue
        label = assign_cluster_5quadrant(
            s_z, np.nan, pe_z_val, stance_thr=stance_thr, pe_thr=pe_thr
        )
        cluster_labels.append(label)
    df["cluster"] = cluster_labels
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "dataset.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "persona_clustering",
    )
    parser.add_argument(
        "--stance-thr",
        type=float,
        default=0.43,
        help="z-score threshold for HIGH/LOW stated stance (default tertile = 0.43).",
    )
    parser.add_argument(
        "--pe-thr",
        type=float,
        default=0.43,
        help="z-score threshold for HIGH PE exposure (default tertile = 0.43).",
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help=(
            "Run robustness sweep at thresholds {0.30, 0.43, 0.55}. "
            "Emits threshold_sensitivity.csv with cluster sizes per setting."
        ),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = build_features(args.dataset)
    print(
        "Feature extraction summary (n parseable per domain):\n",
        df.groupby("domain")[["stated_concern", "revealed_concern", "pe_exposure"]]
        .count()
        .to_string(),
    )

    df = empirical_polarity_check(df)
    flipped = (
        df[df["stated_polarity_flipped"]]["domain"].drop_duplicates().tolist()
    )
    if flipped:
        print(f"\nEmpirical polarity flip applied for domains: {flipped}")
    else:
        print("\nNo empirical polarity flips needed.")

    df = assign_clusters(df, stance_thr=args.stance_thr, pe_thr=args.pe_thr)

    if args.threshold_sweep:
        sweep_rows = []
        for thr in (0.30, 0.43, 0.55):
            sweep_df = assign_clusters(
                df.drop(columns=["cluster"], errors="ignore"),
                stance_thr=thr,
                pe_thr=thr,
            )
            sizes = sweep_df["cluster"].value_counts().to_dict()
            for cluster_name in list(CLUSTERS) + ["Unassigned"]:
                sweep_rows.append(
                    {
                        "threshold": thr,
                        "cluster": cluster_name,
                        "n": int(sizes.get(cluster_name, 0)),
                    }
                )
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_path = args.out_dir / "threshold_sensitivity.csv"
        sweep_df.to_csv(sweep_path, index=False)
        print(f"\nWrote threshold sweep: {sweep_path}")
        print(sweep_df.pivot(index="cluster", columns="threshold", values="n").to_string())

    print("\nCluster sizes per domain:")
    print(
        df.groupby(["domain", "cluster"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(CLUSTERS), fill_value=0)
        .to_string()
    )

    print("\nCluster centroids (mean stated_concern, revealed_concern, pe_exposure):")
    print(
        df.groupby(["domain", "cluster"])[["stated_concern", "revealed_concern", "pe_exposure"]]
        .mean()
        .round(3)
        .to_string()
    )

    out_cols = [
        "uuid",
        "user_id",
        "domain",
        "stated_concern",
        "revealed_concern",
        "pe_exposure",
        "stated_z",
        "revealed_z",
        "pe_z",
        "stated_polarity_flipped",
        "cluster",
    ]
    out_path = args.out_dir / "cluster_labels.csv"
    df[out_cols].to_csv(out_path, index=False)
    print(f"\nWrote cluster labels: {out_path}")

    centroid_path = args.out_dir / "cluster_centroids.csv"
    cent = (
        df.groupby(["domain", "cluster"])
        .agg(
            n=("uuid", "count"),
            mean_stated=("stated_concern", "mean"),
            mean_revealed=("revealed_concern", "mean"),
            mean_pe_exposure=("pe_exposure", "mean"),
            mean_stated_z=("stated_z", "mean"),
            mean_revealed_z=("revealed_z", "mean"),
        )
        .round(4)
        .reset_index()
    )
    cent.to_csv(centroid_path, index=False)
    print(f"Wrote cluster centroids: {centroid_path}")


if __name__ == "__main__":
    main()
