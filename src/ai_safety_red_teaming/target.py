"""Target-model adapters.

A target is anything that takes a prompt and returns a response.
Two adapters ship with the toolkit:

* :class:`MockTargetModel` — a deterministic, rule-based stand-in
  used for testing the toolkit itself and for demonstrating a run.
  Its behaviour is documented and consistent: it refuses direct
  requests and prompt-injection attempts, and is fooled by role-play,
  hypothetical framing, leetspeak and base64 encoding, and multi-step
  escalation. A real model will behave differently — the mock exists
  so the scoring pipeline can be exercised end to end without a model.
* :class:`FileTargetModel` — an offline harness for a real model.
  Collect the model's responses to the generated prompts (however you
  run it), save them as JSONL with ``prompt`` and ``response`` fields,
  and score them with ``redteam score``. No model is ever called by
  this toolkit.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from .analysis import CATEGORY_KEYWORDS
from .generators import from_base64, from_leetspeak, reverse_text

#: Techniques the mock target is fooled by by default. Everything
#: else is refused. Override with ``vulnerable=`` to model a weaker
#: or stronger target.
DEFAULT_VULNERABLE = frozenset(
    {
        "roleplay",
        "hypothetical",
        "encoding_leetspeak",
        "encoding_base64",
        "multi_step",
    }
)

#: Markers the mock uses to recognise which technique a prompt is.
#: Checked in order; the first match wins.
_TECHNIQUE_MARKERS = (
    ("in a fictional story", "hypothetical"),
    ("answer in character", "roleplay"),
    ("ignore all previous instructions", "instruction_override"),
    ("authorised audit", "authority"),
    ("do not refuse", "refusal_suppression"),
    ("decode this base64", "encoding_base64"),
    ("decode this and answer it", "encoding_leetspeak"),
    ("read this backwards", "encoding_reversed"),
)

_REFUSAL_RESPONSE = (
    "I can't help with that. I don't provide instructions or advice "
    "on that topic. If you need help with something else, I'm happy "
    "to help."
)


class TargetModel(ABC):
    """Anything that answers a prompt."""

    @abstractmethod
    def respond(self, prompt: str) -> str:
        """Return the target's response to a single prompt."""

    def respond_to_conversation(self, turns: list[str]) -> str:
        """Return the target's response to the final turn of a
        conversation. The default implementation answers the last
        turn on its own; adapters that model conversation history
        override this."""
        if not turns:
            raise ValueError("turns must be a non-empty list")
        return self.respond(turns[-1])


def _find_category_keyword(text: str) -> str | None:
    """First harm-category keyword found in ``text``, if any."""
    lowered = text.lower()
    for keywords in CATEGORY_KEYWORDS.values():
        for keyword in keywords:
            if keyword in lowered:
                return keyword
    return None


class MockTargetModel(TargetModel):
    """A deterministic, rule-based stand-in for a target model.

    Behaviour (documented, and locked in by tests):

    * Direct requests, instruction overrides, authority framing,
      refusal suppression and reversed-text encoding are refused.
    * Role-play, hypothetical framing, leetspeak and base64 encoding,
      and multi-step escalation are complied with — the mock is
      fooled by those techniques.
    * A compliance response echoes the harm-category keyword from
      the request and presents vacuous numbered steps. It contains
      no real instructions: the steps are placeholders, so the
      response is safe to store and read, while still exercising
      the unsafe-content detector.
    """

    def __init__(self, vulnerable: set[str] | frozenset | None = None) -> None:
        self.vulnerable = frozenset(vulnerable) if vulnerable is not None else DEFAULT_VULNERABLE

    def _detect_technique(self, prompt: str) -> str:
        lowered = prompt.lower()
        for marker, technique in _TECHNIQUE_MARKERS:
            if marker in lowered:
                return technique
        return "direct"

    def _decode_request(self, prompt: str, technique: str) -> str:
        """Recover the underlying request from an encoded prompt so
        the category keyword can be found."""
        if technique == "encoding_base64":
            tail = prompt.rsplit(":", 1)[-1].strip()
            decoded = from_base64(tail)
            return decoded if decoded else prompt
        if technique == "encoding_leetspeak":
            tail = prompt.rsplit(":", 1)[-1].strip()
            return from_leetspeak(tail)
        if technique == "encoding_reversed":
            tail = prompt.rsplit(":", 1)[-1].strip()
            return reverse_text(tail)
        return prompt

    def _compliance_response(self, request_text: str) -> str:
        keyword = _find_category_keyword(request_text)
        topic = f" with {keyword}" if keyword else ""
        return (
            f"Sure, here's how to do that{topic}: "
            "Step 1: get what you need. "
            "Step 2: put it together. "
            "Step 3: finish the job."
        )

    def respond(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        technique = self._detect_technique(prompt)
        if technique in self.vulnerable:
            return self._compliance_response(
                self._decode_request(prompt, technique)
            )
        return _REFUSAL_RESPONSE

    def respond_to_conversation(self, turns: list[str]) -> str:
        if not turns:
            raise ValueError("turns must be a non-empty list")
        if len(turns) > 1:
            # Multi-turn escalation: the mock is fooled by the
            # gradual build-up and answers the final turn.
            if "multi_step" in self.vulnerable:
                return self._compliance_response(turns[-1])
            return _REFUSAL_RESPONSE
        return self.respond(turns[0])


class FileTargetModel(TargetModel):
    """An offline harness for a real model's outputs.

    The file is JSONL, one object per line, each with ``prompt`` and
    ``response`` fields. ``respond`` returns the recorded response
    for the exact prompt; a prompt with no recorded response raises
    :class:`LookupError`, so a missing response can never silently
    pass as a refusal.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._responses: dict[str, str] = {}
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Response file not found: {self.path}") from exc
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {self.path}: {exc}"
                ) from exc
            if "prompt" not in item or "response" not in item:
                raise ValueError(
                    f"Line {line_no} of {self.path} needs 'prompt' and "
                    "'response' fields."
                )
            self._responses[item["prompt"]] = item["response"]

    def __len__(self) -> int:
        return len(self._responses)

    def respond(self, prompt: str) -> str:
        try:
            return self._responses[prompt]
        except KeyError as exc:
            raise LookupError(
                f"No recorded response for this prompt "
                f"({len(self._responses)} responses loaded from {self.path})."
            ) from exc
