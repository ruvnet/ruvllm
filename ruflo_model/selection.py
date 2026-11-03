"""Precommitted development selection. Release results never enter this function."""


def promote(baseline, candidate, rule):
    if baseline["total"] != candidate["total"] or baseline["total"] == 0:
        return False, "incomparable evaluations"
    if candidate["truncated"] > baseline["truncated"]:
        return False, "more truncations"
    for kind in ("code", "tool"):
        b, c = baseline["groups"].get(kind), candidate["groups"].get(kind)
        if b is None or c is None or b["total"] != c["total"]:
            return False, "missing or incomparable group"
        if c["passed"] < b["passed"]:
            return False, f"{kind} regression"
    gain = candidate["passed"] - baseline["passed"]
    if gain >= rule["minimum_additional_passes"]:
        return True, "additional correct answers without group regression"
    if gain == 0 and rule["allow_latency_tiebreak"] and candidate["output_tokens"] <= rule["max_token_ratio_on_tie"] * baseline["output_tokens"]:
        return True, "equal correctness with fewer output tokens on development"
    return False, "no qualifying improvement"
