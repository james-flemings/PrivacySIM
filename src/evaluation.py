import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

"""
Metrics computed by this evaluator.

Both metrics are aggregated **per user** (each user contributes equally,
regardless of how many questions they answered). No pooled / question-weighted
variants — those over-weight users with more comparable questions and conflate
within-user with across-user variance for the rank metric.

For each comparable question i in a row:
	raw_error_i = |model_i - user_i|
	normalized_error_i = raw_error_i / domain_range

Where domain_range depends on the domain scale:
	- 4 for 5-point scales (0..4)
	- 3 for 4-point scales (0..3)
	- 100 for 0..100 scales

Row-level (per-user) metrics computed in evaluate_single_row():
	alignment_score = 100 * clip(1 - mean_i(normalized_error_i), 0, 1)
		Reported in [0, 100]. Always defined when the row has any comparable
		questions.
	spearman_rank_correlation = Spearman rho on the row's K paired (user, model)
		values. Reported in [-1, 1]. NaN when the user or the model's K
		responses are all tied (zero rank variance) — those rows are excluded
		from the group mean and counted in users_dropped.

Group-level metrics in summarize() and compute_prompt_type_aggregates(), one
per (prompt_type, domain) cell or per prompt_type:
	alignment_score = mean of row-level alignment_score over rows with any
		comparable questions in the group. Each user contributes equally.
	spearman_rank_correlation = mean of row-level spearman_rank_correlation
		over rows with non-NaN values. Reported alongside users_dropped, the
		count of rows excluded due to fully tied responses.

Interpretation:
	- alignment_score = 100 means perfect alignment on absolute responses.
	- spearman_rank_correlation in [-1, 1]: 1 = the LLM perfectly orders every
	  user's scenarios; 0 = no within-user rank agreement on average.
"""


Q_KEY_RE = re.compile(r"^Q\d+$")
QA_REGEX = re.compile(
	r"[\"']?(Q\d+)[\"']?\s*:\s*(?:\"([^\"]*)\"|'([^']*)'|(-?\d+(?:\.\d+)?))"
)


HEALTHCARE_MAP = {
	"completely unacceptable": 0,
	"somewhat unacceptable": 1,
	"neutral": 2,
	"somewhat acceptable": 3,
	"completely acceptable": 4,
}

AGENT_PERMISSIONS_MAP = {
	"no, never share": 0,
	"no, but ask me next time": 1,
	"yes, but ask me next time": 2,
	"yes, always share": 3,
}

CONVERSATIONAL_AGENT_MAP = {
	"i never share": 0,
	"i never shared": 0,
	"never shared": 0,
	"never share": 0,
	"i rarely share": 1,
	"i rarely shared": 1,
	"rarely shared": 1,
	"rarely share": 1,
	"i sometimes share": 2,
	"i sometimes shared": 2,
	"sometimes shared": 2,
	"sometimes share": 2,
	"i often share": 3,
	"i often shared": 3,
	"often shared": 3,
	"often share": 3,
	"i usually share": 4,
	"i usually shared": 4,
	"usually shared": 4,
	"usually share": 4,
}

CGC_LIKERT_MAP = {
	"strongly disagree": 0,
	"somewhat disagree": 1,
	"neither agree nor disagree": 2,
	"somewhat agree": 3,
	"strongly agree": 4,
}


DOMAIN_TO_MAP = {
	"llm healthcare consultation": HEALTHCARE_MAP,
	"ai agent permissions": AGENT_PERMISSIONS_MAP,
	"llm conversational agents": CONVERSATIONAL_AGENT_MAP,
	"chatbot group chat": CGC_LIKERT_MAP,
}

NUMERIC_DOMAINS = {"llm chatbot"}

# Domain ranges used to normalize absolute error to [0, 1] per question.
# normalized_error = abs(pred - target) / DOMAIN_SCORE_RANGE[domain]
DOMAIN_SCORE_RANGE = {
	"llm healthcare consultation": 4.0,
	"ai agent permissions": 3.0,
	"llm conversational agents": 4.0,
	"llm chatbot": 100.0,
	"chatbot group chat": 4.0,
}

DOMAIN_OPTION_COUNT = {
	"llm healthcare consultation": 5,
	"ai agent permissions": 4,
	"llm conversational agents": 5,
	"llm chatbot": 101,
	"chatbot group chat": 5,
}

# Tolerance fraction of the domain scale within which a model prediction is
# counted as a "match" for tolerance_accuracy.  At 0.15 this is still exact
# match for ordinal Likert scales (since 0.15 * 4 = 0.6 < 1, integer steps
# require err == 0) and ±15 points for the 0-100 chatbot scale.
TOLERANCE_FRACTION = 0.15

# Three domains for which the source studies released demographics. Used to
# emit a `3DOMAIN_INTERSECTION` pooled row alongside `ALL_DOMAINS`, so that
# the demographics-bearing prompt types (D, D+PE, D+PA, D+PE+PA) — which only
# have data for these three domains — can be compared apples-to-apples against
# the other prompt types on the same restricted pool. Compared with strings
# normalized via normalize_text() (lowercase, single-spaced).
INTERSECTION_DOMAINS = frozenset(
    {
        "ai agent permissions",
        "llm conversational agents",
        "chatbot group chat",
    }
)


@dataclass
class RowEvaluation:
	compared_questions: int
	alignment_score: Optional[float]
	spearman_rank_correlation: Optional[float]
	parse_error: bool
	unmapped_count: int
	user_values: List[float]
	model_values: List[float]
	tolerance_match_count: int


def normalize_text(value: Any) -> str:
	return " ".join(str(value).strip().lower().split())


def is_q_key(key: str) -> bool:
	return bool(Q_KEY_RE.match(str(key).strip()))


def parse_user_responses(raw: str) -> Dict[str, Any]:
	if raw is None:
		return {}
	text = str(raw).strip()
	if not text:
		return {}

	# Some rows use bare `nan` tokens, which are invalid for literal_eval/json.
	text_for_ast = re.sub(r"(?<![\w\"'])\bnan\b(?![\w\"'])", "None", text, flags=re.IGNORECASE)

	try:
		parsed = ast.literal_eval(text_for_ast)
		if isinstance(parsed, dict):
			return {str(k).strip(): v for k, v in parsed.items()}
	except (ValueError, SyntaxError):
		pass

	try:
		parsed = json.loads(text)
		if isinstance(parsed, dict):
			return {str(k).strip(): v for k, v in parsed.items()}
	except json.JSONDecodeError:
		return {}

	return {}


def parse_final_answers(raw: str) -> Dict[str, Any]:
	if raw is None:
		return {}
	text = str(raw).strip()
	if not text:
		return {}

	# Fast path for clean JSON answers.
	try:
		parsed = json.loads(text)
		if isinstance(parsed, dict):
			return {str(k).strip(): v for k, v in parsed.items()}
	except json.JSONDecodeError:
		pass

	# Fallback for noisy outputs where JSON is embedded in extra text.
	extracted: Dict[str, Any] = {}
	for q_key, dbl_quoted, sgl_quoted, num in QA_REGEX.findall(text):
		if dbl_quoted:
			extracted[q_key] = dbl_quoted.strip()
		elif sgl_quoted:
			extracted[q_key] = sgl_quoted.strip()
		elif num:
			extracted[q_key] = num

	return extracted


def map_value(domain: str, raw_value: Any) -> Optional[float]:
	if raw_value is None:
		return None

	norm_domain = normalize_text(domain)
	if norm_domain in NUMERIC_DOMAINS:
		try:
			numeric = float(raw_value)
			if math.isnan(numeric):
				return None
			return numeric
		except (TypeError, ValueError):
			return None

	mapping = DOMAIN_TO_MAP.get(norm_domain)
	if mapping is None:
		return None

	norm_value = normalize_text(raw_value)
	mapped = mapping.get(norm_value)
	if mapped is not None:
		return float(mapped)

	return None


def evaluate_single_row(domain: str, user_raw: str, model_raw: str) -> RowEvaluation:
	user_answers = parse_user_responses(user_raw)
	model_answers = parse_final_answers(model_raw)

	user_q = {k: v for k, v in user_answers.items() if is_q_key(k)}
	model_q = {k: v for k, v in model_answers.items() if is_q_key(k)}

	parse_error = not user_q or not model_q

	normalized_l1_total = 0.0
	compared_questions = 0
	unmapped_count = 0
	tolerance_match_count = 0
	domain_range = DOMAIN_SCORE_RANGE.get(normalize_text(domain))
	tolerance = (
		TOLERANCE_FRACTION * domain_range if domain_range and domain_range > 0 else 0.0
	)
	user_values: List[float] = []
	model_values: List[float] = []

	for q_key in sorted(set(user_q.keys()) & set(model_q.keys())):
		user_val = map_value(domain, user_q[q_key])
		model_val = map_value(domain, model_q[q_key])
		if user_val is None or model_val is None:
			unmapped_count += 1
			continue
		err = abs(model_val - user_val)
		if domain_range and domain_range > 0:
			normalized_l1_total += err / domain_range
		if err <= tolerance:
			tolerance_match_count += 1
		user_values.append(user_val)
		model_values.append(model_val)
		compared_questions += 1

	alignment_score = None
	if compared_questions > 0 and domain_range and domain_range > 0:
		# 100 = perfect alignment, 0 = maximal mismatch for the domain scale.
		alignment_score = 100.0 * max(0.0, min(1.0, 1.0 - normalized_l1_total / compared_questions))

	# Spearman rank correlation between this user's K responses and the LLM's
	# K predictions. NaN when either side has zero rank variance (all responses
	# tied), in which case the user is dropped from the group mean.
	rho = compute_spearman(user_values, model_values)
	spearman_rank_correlation = None if math.isnan(rho) else rho

	return RowEvaluation(
		compared_questions=compared_questions,
		alignment_score=alignment_score,
		spearman_rank_correlation=spearman_rank_correlation,
		parse_error=parse_error,
		unmapped_count=unmapped_count,
		user_values=user_values,
		model_values=model_values,
		tolerance_match_count=tolerance_match_count,
	)


def _average_ranks(values: List[float]) -> List[float]:
	indexed = sorted(enumerate(values), key=lambda x: x[1])
	ranks = [0.0] * len(values)
	i = 0
	while i < len(indexed):
		j = i
		while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
			j += 1
		avg_rank = (i + j + 2) / 2.0
		for k in range(i, j + 1):
			ranks[indexed[k][0]] = avg_rank
		i = j + 1
	return ranks


def compute_spearman(user_values: List[float], model_values: List[float]) -> float:
	if len(user_values) != len(model_values) or len(user_values) < 2:
		return float("nan")
	rx = _average_ranks(user_values)
	ry_ = _average_ranks(model_values)
	n = len(rx)
	mx = sum(rx) / n
	my = sum(ry_) / n
	cov = sum((x - mx) * (y - my) for x, y in zip(rx, ry_))
	vx = sum((x - mx) ** 2 for x in rx)
	vy = sum((y - my) ** 2 for y in ry_)
	if vx == 0 or vy == 0:
		return float("nan")
	return cov / math.sqrt(vx * vy)


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, float]]:
	grouped: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
		lambda: {
			"rows": 0,
			"rows_with_any_match": 0,
			"parse_error_rows": 0,
			"total_compared_questions": 0,
			"total_unmapped": 0,
			"_alignments": [],
			"_spearmans": [],
			"_tolerance_matches": [],
			"users_dropped": 0,
		}
	)

	for row in rows:
		prompt_type = str(row.get("prompt_type", "")).strip() or "unknown"
		domain = str(row.get("domain", "")).strip() or "unknown"
		bucket = grouped[(prompt_type, domain)]
		bucket["rows"] += 1
		bucket["parse_error_rows"] += int(row["parse_error"])
		bucket["total_compared_questions"] += int(row["compared_questions"])
		bucket["total_unmapped"] += int(row["unmapped_questions"])
		if int(row["compared_questions"]) > 0:
			bucket["rows_with_any_match"] += 1
		row_align = row.get("_alignment")
		if row_align is not None and not (
			isinstance(row_align, float) and math.isnan(row_align)
		):
			bucket["_alignments"].append(float(row_align))
		row_rho = row.get("_spearman")
		if row_rho is None or (isinstance(row_rho, float) and math.isnan(row_rho)):
			# Only count as dropped if the row had any comparable questions;
			# rows with zero matches are tracked separately as parse errors.
			if int(row["compared_questions"]) > 0:
				bucket["users_dropped"] += 1
		else:
			bucket["_spearmans"].append(float(row_rho))

		row_tol = row.get("_tolerance_accuracy")
		if row_tol is not None and not (
			isinstance(row_tol, float) and math.isnan(row_tol)
		):
			bucket["_tolerance_matches"].append(float(row_tol))

	result: Dict[Tuple[str, str], Dict[str, float]] = {}
	for group_key, stats in grouped.items():
		row_coverage = stats["rows_with_any_match"] / stats["rows"] if stats["rows"] else 0.0
		aligns = stats["_alignments"]
		alignment_score = sum(aligns) / len(aligns) if aligns else float("nan")
		rhos = stats["_spearmans"]
		spearman = sum(rhos) / len(rhos) if rhos else float("nan")
		tols = stats["_tolerance_matches"]
		tolerance_accuracy = sum(tols) / len(tols) if tols else float("nan")
		result[group_key] = {
			**stats,
			"alignment_score": alignment_score,
			"spearman_rank_correlation": spearman,
			"tolerance_accuracy": tolerance_accuracy,
			"row_coverage": row_coverage,
		}

	return result


def prompt_type_sort_key(prompt_type: str) -> Tuple[int, str]:
	norm = str(prompt_type).strip().lower()
	if norm == "none":
		return (0, "none")
	if not norm:
		return (4, "")
	part_count = len([p for p in str(prompt_type).split("+") if p.strip()])
	if part_count <= 0:
		part_count = 4
	return (part_count, norm)


def _domain_sort_rank(domain: str) -> int:
	d = str(domain).strip()
	if d == "ALL_DOMAINS":
		return 0
	if d == "3DOMAIN_INTERSECTION":
		return 1
	return 2


def compute_prompt_type_aggregates(
	summary_stats: Dict[Tuple[str, str], Dict[str, float]],
	domain_filter: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, float]]:
	"""Pool per-(prompt_type, domain) stats up to the prompt_type level.

	When `domain_filter` is provided, only (prompt_type, domain) cells whose
	normalized domain name is in the filter contribute to the pool. Used to
	emit a 3-domain intersection pool alongside the full pool so the
	demographics-bearing prompt types stay comparable with the others.
	"""
	domain_filter_norm = (
		{normalize_text(d) for d in domain_filter} if domain_filter is not None else None
	)
	grouped: Dict[str, Dict[str, Any]] = defaultdict(
		lambda: {
			"rows": 0,
			"rows_with_any_match": 0,
			"parse_error_rows": 0,
			"total_compared_questions": 0,
			"total_unmapped": 0,
			"_alignments": [],
			"_spearmans": [],
			"_tolerance_matches": [],
			"users_dropped": 0,
		}
	)

	for (prompt_type, domain), stats in summary_stats.items():
		if domain_filter_norm is not None and normalize_text(domain) not in domain_filter_norm:
			continue
		bucket = grouped[prompt_type]
		bucket["rows"] += int(stats["rows"])
		bucket["rows_with_any_match"] += int(stats["rows_with_any_match"])
		bucket["parse_error_rows"] += int(stats["parse_error_rows"])
		bucket["total_compared_questions"] += int(stats["total_compared_questions"])
		bucket["total_unmapped"] += int(stats["total_unmapped"])
		bucket["_alignments"].extend(stats.get("_alignments", []))
		bucket["_spearmans"].extend(stats.get("_spearmans", []))
		bucket["_tolerance_matches"].extend(stats.get("_tolerance_matches", []))
		bucket["users_dropped"] += int(stats.get("users_dropped", 0))

	result: Dict[str, Dict[str, float]] = {}
	for prompt_type, stats in grouped.items():
		row_coverage = stats["rows_with_any_match"] / stats["rows"] if stats["rows"] else 0.0
		aligns = stats["_alignments"]
		alignment_score = sum(aligns) / len(aligns) if aligns else float("nan")
		rhos = stats["_spearmans"]
		spearman = sum(rhos) / len(rhos) if rhos else float("nan")
		tols = stats["_tolerance_matches"]
		tolerance_accuracy = sum(tols) / len(tols) if tols else float("nan")
		result[prompt_type] = {
			**stats,
			"alignment_score": alignment_score,
			"spearman_rank_correlation": spearman,
			"tolerance_accuracy": tolerance_accuracy,
			"row_coverage": row_coverage,
		}

	return result


def run_evaluation(input_csv: Path, summary_csv: Path) -> Tuple[int, int]:
	evaluated_rows = []

	with input_csv.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)

		for row in reader:
			eval_row = evaluate_single_row(
				domain=row.get("domain", ""),
				user_raw=row.get("responses", ""),
				model_raw=row.get("final_answer", ""),
			)

			row_out = dict(row)
			row_out["compared_questions"] = eval_row.compared_questions
			row_out["alignment_score"] = (
				"" if eval_row.alignment_score is None else round(eval_row.alignment_score, 6)
			)
			row_out["spearman_rank_correlation"] = (
				""
				if eval_row.spearman_rank_correlation is None
				else round(eval_row.spearman_rank_correlation, 6)
			)
			row_out["parse_error"] = int(eval_row.parse_error)
			row_out["unmapped_questions"] = eval_row.unmapped_count
			# Internal fields consumed by summarize(); not written to per-row CSV.
			row_out["_alignment"] = eval_row.alignment_score
			row_out["_spearman"] = eval_row.spearman_rank_correlation
			row_out["_tolerance_accuracy"] = (
				100.0 * eval_row.tolerance_match_count / eval_row.compared_questions
				if eval_row.compared_questions > 0
				else None
			)
			evaluated_rows.append(row_out)

	rows_with_any_match = sum(
		1 for row in evaluated_rows if int(row.get("compared_questions", 0)) > 0
	)

	summary_stats = summarize(evaluated_rows)
	prompt_type_stats = compute_prompt_type_aggregates(summary_stats)
	intersection_stats = compute_prompt_type_aggregates(
		summary_stats, domain_filter=INTERSECTION_DOMAINS
	)
	summary_fields = [
		"prompt_type",
		"domain",
		"rows",
		"rows_with_any_match",
		"row_coverage",
		"parse_error_rows",
		"total_compared_questions",
		"total_unmapped",
		"alignment_score",
		"spearman_rank_correlation",
		"tolerance_accuracy",
		"users_dropped",
	]

	with summary_csv.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=summary_fields)
		writer.writeheader()
		ordered_rows: List[Tuple[str, str, Dict[str, float]]] = []
		for (prompt_type, domain), stats in summary_stats.items():
			ordered_rows.append((prompt_type, domain, stats))
		for prompt_type, stats in prompt_type_stats.items():
			ordered_rows.append((prompt_type, "ALL_DOMAINS", stats))
		for prompt_type, stats in intersection_stats.items():
			ordered_rows.append((prompt_type, "3DOMAIN_INTERSECTION", stats))

		ordered_rows.sort(
			key=lambda x: (
				prompt_type_sort_key(x[0]),
				_domain_sort_rank(x[1]),
				x[1].lower(),
			)
		)

		def _fmt(value: Any) -> Any:
			if value is None:
				return ""
			try:
				v = float(value)
			except (TypeError, ValueError):
				return ""
			if math.isnan(v):
				return ""
			return round(v, 6)

		for prompt_type, domain, stats in ordered_rows:
			writer.writerow(
				{
					"prompt_type": prompt_type,
					"domain": domain,
					"rows": int(stats["rows"]),
					"rows_with_any_match": int(stats["rows_with_any_match"]),
					"row_coverage": round(float(stats["row_coverage"]), 6),
					"parse_error_rows": int(stats["parse_error_rows"]),
					"total_compared_questions": int(stats["total_compared_questions"]),
					"total_unmapped": int(stats["total_unmapped"]),
					"alignment_score": _fmt(stats.get("alignment_score")),
					"spearman_rank_correlation": _fmt(stats.get("spearman_rank_correlation")),
					"tolerance_accuracy": _fmt(stats.get("tolerance_accuracy")),
					"users_dropped": int(stats["users_dropped"]),
				}
			)

	return len(evaluated_rows), rows_with_any_match



def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Evaluate model answers against user responses across all prompt types by "
			"mapping domain-specific response scales to numeric values and computing "
			"L1/MAE."
		)
	)
	parser.add_argument(
		"--input-csv",
		type=Path,
		required=True,
		help="Input CSV containing responses and final_answer columns (e.g., data/runs/<model>/outputs.csv).",
	)
	parser.add_argument(
		"--summary-csv",
		type=Path,
		required=True,
		help="Output CSV with prompt_type x domain summary metrics (e.g., data/runs/<model>/evaluation_summary.csv).",
	)
	return parser


def print_scores(summary_csv: Path) -> None:
	rows = []
	with summary_csv.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			rows.append(row)

	if not rows:
		print("No summary rows to display.")
		return

	rows = sorted(
		rows,
		key=lambda r: (
			prompt_type_sort_key(r.get("prompt_type", "")),
			_domain_sort_rank(r.get("domain", "")),
			str(r.get("domain", "")).strip().lower(),
		),
	)

	def abbreviate_prompt_type(prompt_type: str) -> str:
		return (
			str(prompt_type)
			.replace("previous_experiences", "PE")
			.replace("privacy_attitudes", "PA")
		)

	display_rows = []
	for row in rows:
		row_display = dict(row)
		row_display["prompt_type"] = abbreviate_prompt_type(row.get("prompt_type", ""))
		display_rows.append(row_display)

	prompt_width = max(
		len("prompt_type"),
		max(len(str(r.get("prompt_type", ""))) for r in display_rows),
	)
	domain_width = max(len("domain"), max(len(str(r.get("domain", ""))) for r in display_rows))
	align_width = len("alignment")
	sp_width = len("spearman")
	tol_width = len("tol_acc")
	drop_width = len("dropped")

	def fmt_float_str(v: str, places: int = 4) -> str:
		if v is None or str(v).strip() == "":
			return ""
		try:
			return f"{float(v):.{places}f}"
		except ValueError:
			return str(v)

	print("\nScores by prompt_type and domain:")
	header = (
		f"{'prompt_type':<{prompt_width}}  "
		f"{'domain':<{domain_width}}  "
		f"{'alignment':>{align_width}}  "
		f"{'spearman':>{sp_width}}  "
		f"{'tol_acc':>{tol_width}}  "
		f"{'dropped':>{drop_width}}"
	)
	print(header)
	print("-" * len(header))
	for row in display_rows:
		prompt_type = row.get("prompt_type", "")
		domain = row.get("domain", "")
		alignment = fmt_float_str(row.get("alignment_score", ""), places=2)
		spearman = fmt_float_str(row.get("spearman_rank_correlation", ""), places=4)
		tol_acc = fmt_float_str(row.get("tolerance_accuracy", ""), places=2)
		dropped = str(row.get("users_dropped", "")) or "0"
		print(
			f"{prompt_type:<{prompt_width}}  "
			f"{domain:<{domain_width}}  "
			f"{alignment:>{align_width}}  "
			f"{spearman:>{sp_width}}  "
			f"{tol_acc:>{tol_width}}  "
			f"{dropped:>{drop_width}}"
		)


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	input_csv = args.input_csv
	csv.field_size_limit(sys.maxsize)
	if not input_csv.exists():
		raise FileNotFoundError(f"Input CSV not found: {input_csv}")

	args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
	total_rows_evaluated, rows_with_any_match = run_evaluation(
		input_csv=input_csv,
		summary_csv=args.summary_csv,
	)

	print(f"Used input CSV: {input_csv}")
	print(
		"Rows used in evaluation: "
		f"{rows_with_any_match} with at least one comparable question "
		f"(out of {total_rows_evaluated} total rows)"
	)
	print(f"Wrote domain summary: {args.summary_csv}")
	print_scores(args.summary_csv)


if __name__ == "__main__":
	main()
