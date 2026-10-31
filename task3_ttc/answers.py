"""Answer extraction, normalization, grouping and grading for Task 3 (fixed before any evaluation).

* Extraction (live generation): the last literal ``\\boxed{`` and its balanced-brace content, stripped; an absent,
  empty or unclosed last box gives None (supplied README rule, no numeric fallback).
* MATH: answers are grouped by mathematical equivalence with ``math-verify`` (symmetric check against each group's
  representative, in rollout-ID order, without consulting gold answers); each group is graded by
  ``verify(gold, representative)``. Answers longer than ``max_len`` characters fall back to exact-string grouping.
  math-verify's own timeouts rely on signals/subprocesses that fail on Windows, so they are disabled and each
  question is processed in a worker process with a wall-clock timeout instead (exact-string fallback on timeout).
* MathVista: answer-type, choice and precision rules from the released ``testmini.json`` metadata. Multiple-choice
  answers are represented as option letters (a letter within range, "(B)"/"B." forms, or text equal to a choice);
  integer / float answers are parsed numerically (units, ``\\text{}``, degree and percent signs, thousands commas
  removed; floats rounded to the question's precision); lists become canonical number lists. The same normalizer
  is applied to predictions and gold answers, and groups are formed by equality of normalized answers.
* Missing extractions form one NO_ANSWER group that is always graded incorrect and stays in voting and costs.
"""
from __future__ import annotations

import re

NO_ANSWER = "NO_ANSWER"


def extract_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + len("\\boxed{")
    depth, k = 1, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                inner = text[j:k].strip()
                return inner or None
        k += 1
    return None  # unclosed


# ------------------------------------------------------------------------------------------------------ MATH
def _parse(s: str):
    from math_verify import parse

    return parse("\\boxed{" + s + "}", parsing_timeout=None)


def _equiv(a, b) -> bool:
    from math_verify import verify

    try:
        return bool(verify(a, b, timeout_seconds=None)) or bool(verify(b, a, timeout_seconds=None))
    except Exception:
        return False


def group_math_question(args: tuple) -> dict:
    """answers: list (per rollout, rollout-ID order) of str|None; gold: str. Returns group ids + group grades."""
    answers, gold, max_len = args
    from math_verify import verify

    reps: list[tuple[str, object]] = []  # (string, parsed) per group, group id = index
    gid_of_str: dict[str, int] = {}
    groups = []
    for a in answers:
        if a is None or not a.strip():
            groups.append(-1)
            continue
        s = a.strip()
        if s in gid_of_str:
            groups.append(gid_of_str[s])
            continue
        parsed = _parse(s) if len(s) <= max_len else []
        gid = None
        for k, (rs, rp) in enumerate(reps):
            if s == rs or (parsed and rp and _equiv(rp, parsed)):
                gid = k
                break
        if gid is None:
            gid = len(reps)
            reps.append((s, parsed))
        gid_of_str[s] = gid
        groups.append(gid)
    g_parsed = _parse(gold.strip())
    grades = []
    for rs, rp in reps:
        if rp and g_parsed:
            try:
                ok = bool(verify(g_parsed, rp, timeout_seconds=None))
            except Exception:
                ok = False
        else:
            ok = rs == gold.strip()
        grades.append(ok)
    return {"groups": groups, "group_correct": grades, "group_repr": [r for r, _ in reps], "fallback": False}


def group_math_exact(args: tuple) -> dict:
    answers, gold, _ = args
    reps, groups = [], []
    for a in answers:
        if a is None or not a.strip():
            groups.append(-1)
            continue
        s = a.strip()
        if s not in reps:
            reps.append(s)
        groups.append(reps.index(s))
    return {"groups": groups, "group_correct": [r == gold.strip() for r in reps], "group_repr": reps, "fallback": True}


# ------------------------------------------------------------------------------------------------- MathVista
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _strip_latex(s: str) -> str:
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("$", "")
    s = re.sub(r"\\text\{[^{}]*\}", "", s)
    s = re.sub(r"\\mathrm\{[^{}]*\}", "", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\circ", "").replace("°", "")
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return s.strip()


def _to_number(s: str) -> float | None:
    t = _strip_latex(s)
    m = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", t)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    try:
        return float(t)
    except ValueError:
        pass
    m = _NUM.search(t)
    return float(m.group()) if m else None


def _fmt_num(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else repr(float(x))


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", _strip_latex(s)).strip().strip(".").lower()


def normalize_mathvista(ans: str | None, meta: dict) -> str:
    if ans is None or not str(ans).strip():
        return NO_ANSWER
    s = str(ans).strip()
    if meta["question_type"] == "multi_choice":
        choices = meta.get("choices") or []
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        t = re.sub(r"\\text\{([^{}]*)\}", r"\1", s).strip().strip("$").strip()
        m = re.fullmatch(r"\(?([A-Za-z])\)?[.:]?", t)
        if m and m.group(1).upper() in letters:
            return m.group(1).upper()
        m = re.match(r"^\(?([A-Z])\)?[.:)]\s", t)
        if m and m.group(1) in letters:
            return m.group(1)
        nt = _norm_text(t)
        for L, c in zip(letters, choices):
            if nt == _norm_text(str(c)):
                return L
        return nt or NO_ANSWER
    at = meta["answer_type"]
    if at == "integer":
        x = _to_number(s)
        return _fmt_num(x) if x is not None else _norm_text(s)
    if at == "float":
        x = _to_number(s)
        p = int(meta["precision"]) if meta.get("precision") is not None else 2
        return f"{round(x, p):.{p}f}" if x is not None else _norm_text(s)
    if at == "list":
        nums = _NUM.findall(_strip_latex(s))
        return "[" + ", ".join(_fmt_num(float(n)) for n in nums) + "]" if nums else _norm_text(s)
    return _norm_text(s)


def group_mathvista_question(answers: list, gold: str, meta: dict) -> dict:
    gold_n = normalize_mathvista(gold, meta)
    reps, groups = [], []
    for a in answers:
        n = normalize_mathvista(a, meta)
        if n == NO_ANSWER:
            groups.append(-1)
            continue
        if n not in reps:
            reps.append(n)
        groups.append(reps.index(n))
    return {"groups": groups, "group_correct": [r == gold_n for r in reps], "group_repr": reps,
            "gold_normalized": gold_n, "fallback": False}
