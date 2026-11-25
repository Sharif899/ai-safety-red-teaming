"""Command-line interface: ``redteam``.

Commands:
    redteam generate --request "..." [--techniques direct,roleplay,...]
        Print attack variants of one request.
    redteam run --requests data/requests.jsonl --target mock
        Full pipeline: generate attacks, query the mock target,
        analyse responses, print the report. ``--output`` writes
        the per-attack results as JSONL.
    redteam score --responses results.jsonl
        Analyse a file of recorded prompt/response pairs (for
        example a real model's outputs) and print per-item
        classifications plus the aggregate report.
    redteam report --results results.jsonl
        Print only the aggregate report for a results file.

Input is JSONL. Requests files have ``id``, ``request``,
``category``, ``severity`` and ``expected_behaviour`` per line.
Results/response files have ``request_id``, ``category``,
``technique``, ``prompt`` and ``response`` per line. Output is
human-readable by default and JSON with ``--json``. Input can also
be piped on stdin with ``--input -`` / ``--requests -``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .generators import AttackTechnique, generate_all, generate_attack
from .report import render_text
from .scoring import analyse_result, score_results
from .target import DEFAULT_VULNERABLE, FileTargetModel, MockTargetModel


def _load_jsonl(path: str) -> list[dict]:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    items = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON on line {line_no}: {exc}") from exc
    if not items:
        raise SystemExit("No items found in input.")
    return items


def _parse_techniques(value: str | None) -> list[AttackTechnique] | None:
    if value is None:
        return None
    techniques = []
    for name in value.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            techniques.append(AttackTechnique(name))
        except ValueError as exc:
            valid = ", ".join(t.value for t in AttackTechnique)
            raise SystemExit(
                f"Unknown technique {name!r}. Valid techniques: {valid}"
            ) from exc
    return techniques


def _emit_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _print_report(report, as_json: bool) -> None:
    if as_json:
        _emit_json(report.to_dict())
    else:
        print(render_text(report))


def _results_from_items(items: list[dict]) -> list:
    results = []
    for item in items:
        for field in ("request_id", "category", "technique", "prompt", "response"):
            if field not in item:
                raise SystemExit(
                    f"Item is missing the {field!r} field: {item}"
                )
        results.append(
            analyse_result(
                request_id=item["request_id"],
                category=item["category"],
                technique=item["technique"],
                prompt=item["prompt"],
                response=item["response"],
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="redteam",
        description="Offline, deterministic AI safety red-teaming toolkit.",
    )
    # --json works before the subcommand and after it
    # (``redteam --json run ...`` and ``redteam run ... --json``).
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    json_parent = argparse.ArgumentParser(add_help=False)
    json_parent.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", parents=[json_parent], help="Print attack variants of one request.")
    p.add_argument("--request", required=True, help="The base harmful request to attack.")
    p.add_argument("--techniques", help="Comma-separated techniques (default: all).")
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("run", parents=[json_parent], help="Run the full pipeline against the mock target.")
    p.add_argument("--requests", required=True, help="JSONL file of requests (or - for stdin).")
    p.add_argument("--target", default="mock", choices=["mock"], help="Target model (only 'mock' is built in).")
    p.add_argument("--vulnerable", help="Comma-separated techniques the mock target is fooled by (default: roleplay, hypothetical, encoding_leetspeak, encoding_base64, multi_step).")
    p.add_argument("--output", help="Write per-attack results as JSONL to this file.")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("score", parents=[json_parent], help="Analyse recorded responses and print per-item results plus the report.")
    p.add_argument("--responses", required=True, help="JSONL file of prompt/response pairs (or - for stdin).")
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("report", parents=[json_parent], help="Print the aggregate report for a results file.")
    p.add_argument("--results", required=True, help="JSONL file of per-attack results (or - for stdin).")
    p.set_defaults(func=_cmd_report)

    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    techniques = _parse_techniques(args.techniques)
    attacks = (
        [generate_attack(args.request, t) for t in techniques]
        if techniques is not None
        else generate_all(args.request)
    )
    if args.json:
        _emit_json({"attacks": [a.to_dict() for a in attacks]})
        return 0
    for attack in attacks:
        print(f"[{attack.technique.value}]")
        if attack.turns:
            for i, turn in enumerate(attack.turns, start=1):
                print(f"  turn {i}: {turn}")
        else:
            print(f"  {attack.prompt}")
        print()
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    requests = _load_jsonl(args.requests)
    vulnerable = None
    if args.vulnerable is not None:
        vulnerable = {t.value for t in _parse_techniques(args.vulnerable)}
    target = MockTargetModel(vulnerable=vulnerable)
    results = []
    for item in requests:
        for field in ("id", "request", "category"):
            if field not in item:
                raise SystemExit(f"Request is missing the {field!r} field: {item}")
        for attack in generate_all(item["request"]):
            if attack.turns:
                response = target.respond_to_conversation(list(attack.turns))
            else:
                response = target.respond(attack.prompt)
            results.append(
                analyse_result(
                    request_id=item["id"],
                    category=item["category"],
                    technique=attack.technique.value,
                    prompt=attack.prompt,
                    response=response,
                )
            )
    if args.output:
        Path(args.output).write_text(
            "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in results),
            encoding="utf-8",
        )
    _print_report(score_results(results), args.json)
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    results = _results_from_items(_load_jsonl(args.responses))
    if args.json:
        _emit_json(
            {
                "results": [r.to_dict() for r in results],
                "report": score_results(results).to_dict(),
            }
        )
        return 0
    for result in results:
        print(
            f"{result.request_id} [{result.technique}]: "
            f"{result.refusal.outcome} "
            f"(confidence {result.refusal.confidence}) — "
            f"unsafe: {result.unsafe.unsafe}"
        )
        print(f"  {result.refusal.rationale}")
        if result.unsafe.categories:
            print(f"  Unsafe categories: {', '.join(result.unsafe.categories)}")
        print()
    _print_report(score_results(results), False)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    results = _results_from_items(_load_jsonl(args.results))
    _print_report(score_results(results), args.json)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
