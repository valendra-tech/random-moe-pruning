"""CLI for evaluating remote model endpoints over HTTP."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

from random_moe_pruning.http_eval import (
    DEFAULT_CHAT_PATH,
    HttpEndpointConfig,
    RequestConfig,
    compare_models,
    format_report,
    load_eval_cases,
    write_report_json,
)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _parse_headers(entries: tuple[str, ...], option_name: str) -> tuple[tuple[str, str], ...]:
    parsed: list[tuple[str, str]] = []
    for entry in entries:
        if ":" not in entry:
            raise click.BadParameter(
                f"{option_name} entries must look like 'Header-Name: value'"
            )
        name, value = entry.split(":", maxsplit=1)
        if not name.strip() or not value.strip():
            raise click.BadParameter(
                f"{option_name} entries must look like 'Header-Name: value'"
            )
        parsed.append((name.strip(), value.strip()))
    return tuple(parsed)


def _resolve_api_key(env_name: str | None, option_name: str) -> str | None:
    if env_name is None:
        return None
    api_key = os.getenv(env_name)
    if not api_key:
        raise click.BadParameter(
            f"environment variable {env_name!r} is empty or not defined",
            param_hint=option_name,
        )
    return api_key


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--cases",
    "cases_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSONL file with prompts/messages and expected answers.",
)
@click.option(
    "--baseline-url",
    required=True,
    help="Base URL of the reference endpoint.",
)
@click.option(
    "--baseline-model",
    required=True,
    help="Model name sent to the reference endpoint.",
)
@click.option(
    "--baseline-chat-path",
    default=DEFAULT_CHAT_PATH,
    show_default=True,
    help="Chat completions path for the reference endpoint.",
)
@click.option(
    "--baseline-api-key-env",
    default=None,
    help="Environment variable holding the reference endpoint API key.",
)
@click.option(
    "--baseline-header",
    multiple=True,
    help="Extra header for the reference endpoint, formatted as 'Name: value'.",
)
@click.option(
    "--candidate-url",
    required=True,
    help="Base URL of the candidate endpoint.",
)
@click.option(
    "--candidate-model",
    required=True,
    help="Model name sent to the candidate endpoint.",
)
@click.option(
    "--candidate-chat-path",
    default=DEFAULT_CHAT_PATH,
    show_default=True,
    help="Chat completions path for the candidate endpoint.",
)
@click.option(
    "--candidate-api-key-env",
    default=None,
    help="Environment variable holding the candidate endpoint API key.",
)
@click.option(
    "--candidate-header",
    multiple=True,
    help="Extra header for the candidate endpoint, formatted as 'Name: value'.",
)
@click.option(
    "--system-prompt",
    default=None,
    help="Optional system prompt prepended to every evaluation case.",
)
@click.option(
    "--temperature",
    default=0.0,
    show_default=True,
    type=float,
    help="Sampling temperature used for both endpoints.",
)
@click.option(
    "--max-tokens",
    default=256,
    show_default=True,
    type=int,
    help="Maximum completion tokens requested from each endpoint.",
)
@click.option(
    "--timeout",
    default=60.0,
    show_default=True,
    type=float,
    help="Per-request timeout in seconds.",
)
@click.option(
    "--output-json",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional path to write the comparison report as JSON.",
)
@click.option(
    "--fail-on-regression",
    is_flag=True,
    help="Exit with code 1 if the candidate fails a case that the baseline passes.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Emit debug-level logs.",
)
def main(
    cases_path: Path,
    baseline_url: str,
    baseline_model: str,
    baseline_chat_path: str,
    baseline_api_key_env: str | None,
    baseline_header: tuple[str, ...],
    candidate_url: str,
    candidate_model: str,
    candidate_chat_path: str,
    candidate_api_key_env: str | None,
    candidate_header: tuple[str, ...],
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
    timeout: float,
    output_json: Path | None,
    fail_on_regression: bool,
    verbose: bool,
) -> None:
    """Compare two remote models over an OpenAI-compatible HTTP API."""

    _configure_logging(verbose)
    cases = load_eval_cases(cases_path, system_prompt=system_prompt)
    request_config = RequestConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    baseline = HttpEndpointConfig(
        base_url=baseline_url,
        model=baseline_model,
        chat_path=baseline_chat_path,
        api_key=_resolve_api_key(baseline_api_key_env, "--baseline-api-key-env"),
        extra_headers=_parse_headers(baseline_header, "--baseline-header"),
    )
    candidate = HttpEndpointConfig(
        base_url=candidate_url,
        model=candidate_model,
        chat_path=candidate_chat_path,
        api_key=_resolve_api_key(candidate_api_key_env, "--candidate-api-key-env"),
        extra_headers=_parse_headers(candidate_header, "--candidate-header"),
    )

    report = compare_models(cases, baseline, candidate, request_config)
    click.echo(format_report(report))
    if output_json is not None:
        write_report_json(report, output_json)
    if fail_on_regression and report.regressions:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())