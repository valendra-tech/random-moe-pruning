"""Tests for remote HTTP evaluation helpers."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from click.testing import CliRunner

from random_moe_pruning.http_eval import (
    HttpEndpointConfig,
    RequestConfig,
    compare_models,
    load_eval_cases,
)
from random_moe_pruning.http_eval_cli import main


def test_load_eval_cases_supports_prompt_messages_and_system_prompt(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "name": "capital-france",
                        "prompt": "Capital of France?",
                        "expected": "Paris",
                    }
                ),
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Reply with yes or no: is 14 even?"}
                        ],
                        "expected": "yes",
                        "match": "contains",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_eval_cases(cases_path, system_prompt="Be terse.")

    assert [case.name for case in cases] == ["capital-france", "case-2"]
    assert cases[0].messages[0] == {"role": "system", "content": "Be terse."}
    assert cases[0].messages[1] == {
        "role": "user",
        "content": "Capital of France?",
    }
    assert cases[1].match == "contains"


def test_compare_models_reports_accuracy_delta_and_regressions(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "name": "capital-france",
                        "prompt": "Capital of France?",
                        "expected": "Paris",
                    }
                ),
                json.dumps(
                    {
                        "name": "parity",
                        "prompt": "Reply with yes or no: is 14 even?",
                        "expected": "yes",
                        "match": "contains",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    responses = {
        ("baseline", "Capital of France?"): "Paris",
        ("baseline", "Reply with yes or no: is 14 even?"): "yes",
        ("candidate", "Capital of France?"): "Lyon",
        ("candidate", "Reply with yes or no: is 14 even?"): "Yes, it is even.",
    }

    with _chat_server(responses) as base_url:
        report = compare_models(
            load_eval_cases(cases_path),
            HttpEndpointConfig(base_url=base_url, model="baseline"),
            HttpEndpointConfig(base_url=base_url, model="candidate"),
            RequestConfig(max_tokens=32, timeout=5.0),
        )

    assert report.baseline.accuracy == 1.0
    assert report.candidate.accuracy == 0.5
    assert report.accuracy_delta == -0.5
    assert report.regressions == ("capital-france",)
    assert report.candidate.avg_token_f1 < report.baseline.avg_token_f1


def test_http_eval_cli_writes_json_and_can_fail_on_regression(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "name": "capital-france",
                "prompt": "Capital of France?",
                "expected": "Paris",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"
    responses = {
        ("baseline", "Capital of France?"): "Paris",
        ("candidate", "Capital of France?"): "Lyon",
    }
    runner = CliRunner()

    with _chat_server(responses) as base_url:
        result = runner.invoke(
            main,
            [
                "--cases",
                str(cases_path),
                "--baseline-url",
                base_url,
                "--baseline-model",
                "baseline",
                "--candidate-url",
                base_url,
                "--candidate-model",
                "candidate",
                "--output-json",
                str(output_path),
            ],
        )
        failing_result = runner.invoke(
            main,
            [
                "--cases",
                str(cases_path),
                "--baseline-url",
                base_url,
                "--baseline-model",
                "baseline",
                "--candidate-url",
                base_url,
                "--candidate-model",
                "candidate",
                "--fail-on-regression",
            ],
        )

    assert result.exit_code == 0
    assert "capital-france" in result.output
    report_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert report_payload["candidate"]["accuracy"] == 0.0
    assert report_payload["delta"]["regressions"] == ["capital-france"]
    assert failing_result.exit_code == 1


class _ChatHandler(BaseHTTPRequestHandler):
    responses: dict[tuple[str, str], str] = {}

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        model = payload["model"]
        messages = payload["messages"]
        prompt = messages[-1]["content"]
        response_text = self.responses[(model, prompt)]
        body = json.dumps(
            {"choices": [{"message": {"content": response_text}}]}
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return None


class _chat_server:
    def __init__(self, responses: dict[tuple[str, str], str]) -> None:
        self._responses = responses

    def __enter__(self) -> str:
        _ChatHandler.responses = self._responses
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __exit__(self, exc_type, exc, tb) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()