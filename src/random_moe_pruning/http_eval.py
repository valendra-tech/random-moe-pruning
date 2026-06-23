"""Evaluate remote OpenAI-compatible endpoints on a shared JSONL test set."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_CHAT_PATH = "/v1/chat/completions"
VALID_MATCH_MODES = frozenset({"contains", "exact", "regex"})
_WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class EvalCase:
    """One text-generation check loaded from a JSONL line."""

    name: str
    messages: tuple[dict[str, str], ...]
    expected: str
    match: str = "exact"


@dataclass(frozen=True)
class HttpEndpointConfig:
    """Connection details for one remote model endpoint."""

    base_url: str
    model: str
    chat_path: str = DEFAULT_CHAT_PATH
    api_key: str | None = None
    extra_headers: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")


@dataclass(frozen=True)
class RequestConfig:
    """Generation settings shared by both endpoints."""

    temperature: float = 0.0
    max_tokens: int = 256
    timeout: float = 60.0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass(frozen=True)
class CaseScore:
    """Evaluation result for one model on one case."""

    response: str
    passed: bool
    exact_match: bool
    token_f1: float


@dataclass(frozen=True)
class ModelSummary:
    """Aggregate metrics for one evaluated model."""

    model: str
    base_url: str
    passed: int
    total: int
    accuracy: float
    avg_token_f1: float


@dataclass(frozen=True)
class PairedCaseResult:
    """Side-by-side result for the baseline and candidate models."""

    name: str
    match: str
    expected: str
    baseline: CaseScore
    candidate: CaseScore


@dataclass(frozen=True)
class ComparisonReport:
    """Comparison of baseline and candidate models on the same case set."""

    baseline: ModelSummary
    candidate: ModelSummary
    cases: tuple[PairedCaseResult, ...]

    @property
    def accuracy_delta(self) -> float:
        return self.candidate.accuracy - self.baseline.accuracy

    @property
    def token_f1_delta(self) -> float:
        return self.candidate.avg_token_f1 - self.baseline.avg_token_f1

    @property
    def regressions(self) -> tuple[str, ...]:
        return tuple(
            case.name
            for case in self.cases
            if case.baseline.passed and not case.candidate.passed
        )

    @property
    def improvements(self) -> tuple[str, ...]:
        return tuple(
            case.name
            for case in self.cases
            if not case.baseline.passed and case.candidate.passed
        )


def load_eval_cases(path: str | Path, system_prompt: str | None = None) -> list[EvalCase]:
    """Load evaluation cases from a JSONL file."""

    cases: list[EvalCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_no}: expected a JSON object")
            cases.append(_parse_case(payload, line_no, system_prompt))
    if not cases:
        raise ValueError("cases file is empty")
    return cases


def compare_models(
    cases: list[EvalCase],
    baseline: HttpEndpointConfig,
    candidate: HttpEndpointConfig,
    request_config: RequestConfig,
) -> ComparisonReport:
    """Evaluate both models on the same cases and return side-by-side metrics."""

    baseline_scores = _evaluate_endpoint(cases, baseline, request_config)
    candidate_scores = _evaluate_endpoint(cases, candidate, request_config)

    paired_cases = tuple(
        PairedCaseResult(
            name=case.name,
            match=case.match,
            expected=case.expected,
            baseline=baseline_score,
            candidate=candidate_score,
        )
        for case, baseline_score, candidate_score in zip(
            cases, baseline_scores, candidate_scores, strict=True
        )
    )

    return ComparisonReport(
        baseline=_summarize_model(baseline, baseline_scores),
        candidate=_summarize_model(candidate, candidate_scores),
        cases=paired_cases,
    )


def format_report(report: ComparisonReport) -> str:
    """Render a concise human-readable report."""

    lines = [
        _format_summary("Baseline", report.baseline),
        _format_summary("Candidate", report.candidate),
        "Delta:",
        f"  accuracy: {report.accuracy_delta:+.3f}",
        f"  avg token F1: {report.token_f1_delta:+.3f}",
    ]
    if report.regressions:
        lines.append("Regressions:")
        for case in report.cases:
            if case.name not in report.regressions:
                continue
            lines.append(
                f"  - {case.name}: expected={case.expected!r} "
                f"baseline={case.baseline.response!r} "
                f"candidate={case.candidate.response!r}"
            )
    else:
        lines.append("Regressions: none")

    if report.improvements:
        lines.append("Improvements:")
        for name in report.improvements:
            lines.append(f"  - {name}")

    return "\n".join(lines)


def report_to_dict(report: ComparisonReport) -> dict[str, Any]:
    """Convert the report to a JSON-serializable dictionary."""

    return {
        "baseline": {
            "model": report.baseline.model,
            "base_url": report.baseline.base_url,
            "passed": report.baseline.passed,
            "total": report.baseline.total,
            "accuracy": report.baseline.accuracy,
            "avg_token_f1": report.baseline.avg_token_f1,
        },
        "candidate": {
            "model": report.candidate.model,
            "base_url": report.candidate.base_url,
            "passed": report.candidate.passed,
            "total": report.candidate.total,
            "accuracy": report.candidate.accuracy,
            "avg_token_f1": report.candidate.avg_token_f1,
        },
        "delta": {
            "accuracy": report.accuracy_delta,
            "avg_token_f1": report.token_f1_delta,
            "regressions": list(report.regressions),
            "improvements": list(report.improvements),
        },
        "cases": [
            {
                "name": case.name,
                "match": case.match,
                "expected": case.expected,
                "baseline": {
                    "response": case.baseline.response,
                    "passed": case.baseline.passed,
                    "exact_match": case.baseline.exact_match,
                    "token_f1": case.baseline.token_f1,
                },
                "candidate": {
                    "response": case.candidate.response,
                    "passed": case.candidate.passed,
                    "exact_match": case.candidate.exact_match,
                    "token_f1": case.candidate.token_f1,
                },
            }
            for case in report.cases
        ],
    }


def write_report_json(report: ComparisonReport, path: str | Path) -> None:
    """Persist a JSON report to disk."""

    Path(path).write_text(
        json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_case(
    payload: dict[str, Any],
    line_no: int,
    system_prompt: str | None,
) -> EvalCase:
    expected = payload.get("expected")
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError(f"line {line_no}: expected must be a non-empty string")

    match = payload.get("match", "exact")
    if match not in VALID_MATCH_MODES:
        raise ValueError(
            f"line {line_no}: match must be one of {sorted(VALID_MATCH_MODES)}"
        )

    messages = _build_messages(payload, line_no, system_prompt)
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        name = f"case-{line_no}"

    return EvalCase(
        name=name,
        messages=messages,
        expected=expected,
        match=match,
    )


def _build_messages(
    payload: dict[str, Any],
    line_no: int,
    system_prompt: str | None,
) -> tuple[dict[str, str], ...]:
    prompt = payload.get("prompt")
    messages = payload.get("messages")
    if (prompt is None) == (messages is None):
        raise ValueError(
            f"line {line_no}: provide exactly one of prompt or messages"
        )

    built_messages: list[dict[str, str]] = []
    if system_prompt is not None:
        built_messages.append({"role": "system", "content": system_prompt})

    if isinstance(prompt, str):
        built_messages.append({"role": "user", "content": prompt})
        return tuple(built_messages)
    if prompt is not None:
        raise ValueError(f"line {line_no}: prompt must be a string")

    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_no}: messages must be a non-empty list")
    for message_idx, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(
                f"line {line_no}: message {message_idx} must be an object"
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(
                f"line {line_no}: message {message_idx} role must be a string"
            )
        if not isinstance(content, str):
            raise ValueError(
                f"line {line_no}: message {message_idx} content must be a string"
            )
        built_messages.append({"role": role, "content": content})
    return tuple(built_messages)


def _evaluate_endpoint(
    cases: list[EvalCase],
    endpoint: HttpEndpointConfig,
    request_config: RequestConfig,
) -> tuple[CaseScore, ...]:
    return tuple(
        _score_case(
            case,
            _generate_response(endpoint, case.messages, request_config),
        )
        for case in cases
    )


def _generate_response(
    endpoint: HttpEndpointConfig,
    messages: tuple[dict[str, str], ...],
    request_config: RequestConfig,
) -> str:
    payload = {
        "model": endpoint.model,
        "messages": list(messages),
        "temperature": request_config.temperature,
        "max_tokens": request_config.max_tokens,
        "stream": False,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    headers.update(dict(endpoint.extra_headers))
    if endpoint.api_key and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"

    url = _join_url(endpoint.base_url, endpoint.chat_path)
    http_request = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=request_config.timeout) as response:
            raw_response = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc

    payload = json.loads(raw_response)
    return _extract_text(payload)


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response does not contain choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("response choice must be an object")

    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        return _flatten_content(content)

    text = choice.get("text")
    if isinstance(text, str):
        return text
    raise ValueError("response choice does not contain message.content or text")


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    raise ValueError("unsupported message content format")


def _score_case(case: EvalCase, response: str) -> CaseScore:
    exact_match = _normalize_text(response) == _normalize_text(case.expected)
    if case.match == "exact":
        passed = exact_match
    elif case.match == "contains":
        passed = _normalize_text(case.expected) in _normalize_text(response)
    else:
        passed = re.search(case.expected, response, flags=re.IGNORECASE) is not None

    return CaseScore(
        response=response,
        passed=passed,
        exact_match=exact_match,
        token_f1=_token_f1(response, case.expected),
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _WORD_RE.findall(prediction.lower())
    ref_tokens = _WORD_RE.findall(reference.lower())
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    same = sum(overlap.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _summarize_model(
    endpoint: HttpEndpointConfig,
    scores: tuple[CaseScore, ...],
) -> ModelSummary:
    total = len(scores)
    passed = sum(score.passed for score in scores)
    return ModelSummary(
        model=endpoint.model,
        base_url=endpoint.base_url,
        passed=passed,
        total=total,
        accuracy=passed / total,
        avg_token_f1=sum(score.token_f1 for score in scores) / total,
    )


def _format_summary(label: str, summary: ModelSummary) -> str:
    return "\n".join(
        [
            f"{label}: {summary.model} @ {summary.base_url}",
            f"  accuracy: {summary.passed}/{summary.total} = {summary.accuracy:.3f}",
            f"  avg token F1: {summary.avg_token_f1:.3f}",
        ]
    )


def _join_url(base_url: str, path: str) -> str:
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{base_url.rstrip('/')}{normalized_path}"