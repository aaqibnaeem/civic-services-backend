"""DeepSeek tier — the highest-quality analyzer.

VERIFIED API FACTS THIS MODULE DEPENDS ON (checked 2026-08-08)
--------------------------------------------------------------
Training data for every LLM is wrong about most of these. Do not "fix" them from
memory.

* ``deepseek-chat`` and ``deepseek-reasoner`` were **retired 2026-07-24** and now
  return errors. The only valid model ids are ``deepseek-v4-flash`` and
  ``deepseek-v4-pro``.
* **Thinking mode is ON BY DEFAULT.** When it is on, the JSON frequently lands in
  ``reasoning_content`` or gets prefixed with prose, and structured output breaks.
  ``extra_body={"thinking": {"type": "disabled"}}`` is mandatory on every call.
* ``response_format={"type": "json_object"}`` is supported.
  ``{"type": "json_schema"}`` is **not** on the main endpoint.
* The literal word ``json`` must appear in the prompt, and ``max_tokens`` must be
  set or the JSON truncates mid-string.
* The API can return **empty content**. Documented, real, and retryable — never a
  crash.
* Never retry 400 / 401 / 402 / 422. Retry 429 / 5xx / timeouts / connection
  errors / empty content / JSON and validation failures.
* No vision endpoint and no embeddings endpoint exist.

INPUT      raw complaint text plus optional location context.
PROCESSING one chat completion against a long, byte-stable system prompt (which
           earns the 50x cheaper prefix cache hit), temperature 0, JSON mode,
           thinking disabled; the parsed dict is validated by Pydantic, and one
           corrective retry is issued with the validation error fed back.
OUTPUT     a fully populated :class:`AnalysisResult` with ``source="llm"`` and
           real token telemetry.

LIMITATIONS
    * Non-deterministic in practice. temperature=0 reduces but does not remove
      variation, so the same complaint can be classified differently on two runs.
      This is why the golden-set evaluation reports an agreement rate.
    * Network-dependent: 2-6 s typical, and a hard dependency on a third-party
      service with ~99.79% uptime and multi-hour outages.
    * Self-reported confidence. The model is not calibrated; 0.9 from an LLM does
      not mean 90% of such answers are right, unlike the ML tier's Platt-scaled
      probability.
    * Prompt-injectable in principle — a complaint containing instructions is
      untrusted input. Mitigated by JSON mode plus strict Pydantic validation, so
      the blast radius is a wrong category, never code execution or data access.
    * Costs money and cannot run offline or without an API key.
    * Text only: cannot see the attached photo.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.ai.base import (
    DEPARTMENT_BY_CATEGORY,
    AIAnalyzer,
    AnalysisResult,
    AnalyzerUnavailable,
)
from app.ai.circuit_breaker import llm_breaker
from app.ai.prompts import TRIAGE_RETRY_SUFFIX, TRIAGE_SYSTEM_PROMPT

logger = logging.getLogger("app.ai.llm")

#: Status codes that are worth retrying. 400/401/402/422 are permanent config or
#: billing faults — retrying them only burns the demo window.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

#: Max characters of complaint text sent upstream. Guards against a pathological
#: 5000-char paste blowing out latency; the tail of a civic complaint is almost
#: never where the classification signal lives.
MAX_INPUT_CHARS = 4000

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class LLMPermanentError(RuntimeError):
    """A non-retryable upstream error (bad key, no balance, malformed request)."""


class LLMTransientError(RuntimeError):
    """A retryable upstream error: rate limit, 5xx, timeout, empty content."""


def _strip_fences(raw: str) -> str:
    """Remove markdown fences the model sometimes adds despite JSON mode."""
    text = _FENCE_RE.sub("", raw.strip())
    # If there is prose around the object, keep the outermost {...}.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


class DeepSeekAnalyzer(AIAnalyzer):
    """Concrete :class:`AIAnalyzer` backed by DeepSeek ``deepseek-v4-flash``.

    Inherits the result schema, value normalisation, latency stamping and the
    ``safe_analyze`` template method; overrides only the three things that are
    actually LLM-specific — availability (needs a key and a closed breaker),
    identity, and the analysis call itself.
    """

    source = "llm"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout: float | None = None,
                 max_attempts: int | None = None) -> None:
        from app.core.config import settings  # lazy: module must import mid-build

        self.api_key = api_key if api_key is not None else getattr(settings, "DEEPSEEK_API_KEY", "")
        self.base_url = base_url or getattr(settings, "DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or getattr(settings, "DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.timeout = float(timeout if timeout is not None
                             else getattr(settings, "AI_TIMEOUT_SECONDS", 25))
        self.max_attempts = int(max_attempts if max_attempts is not None
                                else getattr(settings, "AI_MAX_RETRIES", 3)) + 1
        self.max_attempts = max(1, min(self.max_attempts, 4))  # research cap: 4 attempts
        self.name = self.model
        self._client: Any | None = None
        self._last_error: str | None = None

    # -- availability --------------------------------------------------------

    def is_available(self) -> bool:
        """Key present, AI globally enabled, and the circuit breaker not open."""
        from app.core.config import settings

        if not getattr(settings, "AI_ENABLED", True):
            return False
        if not self.api_key or not str(self.api_key).strip():
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return llm_breaker.allow()

    def configured(self) -> bool:
        """Key + SDK present, ignoring breaker state. Used by ``/ai/health``."""
        from app.core.config import settings

        if not getattr(settings, "AI_ENABLED", True) or not str(self.api_key or "").strip():
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    @property
    def last_error(self) -> str | None:
        return self._last_error or llm_breaker.last_error

    # -- client --------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise AnalyzerUnavailable("openai SDK not installed") from exc
        if not self.api_key:
            raise AnalyzerUnavailable("DEEPSEEK_API_KEY is not set")
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            # We own the retry policy via tenacity; stacking the SDK's own retries
            # on top would silently multiply the worst-case latency by 3.
            max_retries=0,
        )
        return self._client

    # -- one raw call --------------------------------------------------------

    def _call(self, messages: list[dict[str, str]], max_tokens: int = 500) -> tuple[str, Any]:
        """One chat completion. Classifies failures into permanent vs transient."""
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,      # required, or JSON truncates mid-string
                temperature=0,
                # REQUIRED. Thinking is on by default and corrupts JSON output.
                extra_body={"thinking": {"type": "disabled"}},
            )
        except APITimeoutError as exc:
            raise LLMTransientError(f"timeout after {self.timeout}s") from exc
        except APIConnectionError as exc:
            raise LLMTransientError(f"connection error: {exc}") from exc
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status in RETRYABLE_STATUS:
                raise LLMTransientError(f"HTTP {status}") from exc
            if status == 402:
                raise LLMPermanentError(
                    "HTTP 402 insufficient balance — DeepSeek account has no credit"
                ) from exc
            raise LLMPermanentError(f"HTTP {status}: {str(exc)[:200]}") from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMTransientError("response contained no choices")
        content = choices[0].message.content
        # Documented DeepSeek edge case: occasional empty content. Retryable.
        if not content or not content.strip():
            raise LLMTransientError("empty content returned by DeepSeek")
        return content, getattr(response, "usage", None)

    def _call_with_retry(self, messages: list[dict[str, str]], max_tokens: int = 500
                         ) -> tuple[str, Any]:
        """Exponential backoff with jitter: base 1 s, cap 20 s, <=4 attempts, <=45 s."""
        try:
            from tenacity import (
                retry,
                retry_if_exception_type,
                stop_after_attempt,
                stop_after_delay,
                wait_exponential_jitter,
            )
        except ImportError:  # tenacity absent -> single attempt, still correct
            return self._call(messages, max_tokens)

        @retry(
            retry=retry_if_exception_type(LLMTransientError),
            wait=wait_exponential_jitter(initial=1, max=20),
            # Whichever fires first. The delay stop protects the request budget on
            # a 0.1-CPU instance; Render/Vercel/the browser give up around 60 s.
            stop=(stop_after_attempt(self.max_attempts) | stop_after_delay(45)),
            reraise=True,
        )
        def _attempt() -> tuple[str, Any]:
            return self._call(messages, max_tokens)

        return _attempt()

    # -- telemetry -----------------------------------------------------------

    @staticmethod
    def _usage(usage: Any) -> dict[str, int]:
        """Extract token counters, including DeepSeek's cache-hit/miss fields."""
        if usage is None:
            return {}

        def _get(*names: str) -> int:
            for n in names:
                v = getattr(usage, n, None)
                if v is None and isinstance(usage, dict):
                    v = usage.get(n)
                if isinstance(v, int | float):
                    return int(v)
            return 0

        return {
            "prompt_tokens": _get("prompt_tokens"),
            "completion_tokens": _get("completion_tokens"),
            "prompt_cache_hit_tokens": _get("prompt_cache_hit_tokens"),
            "prompt_cache_miss_tokens": _get("prompt_cache_miss_tokens"),
        }

    # -- main entry point ----------------------------------------------------

    def analyze(self, text: str, context: dict[str, Any] | None = None) -> AnalysisResult:
        if not self.is_available():
            state = llm_breaker.state
            if state != "closed":
                raise AnalyzerUnavailable(
                    f"circuit breaker {state}, retry in {llm_breaker.retry_after_seconds()}s"
                )
            raise AnalyzerUnavailable("DeepSeek not configured")

        context = context or {}
        started = time.perf_counter()

        # Per-request content lives ONLY in the user message. Nothing is ever
        # interpolated into the system prompt, or the prefix cache is lost.
        parts = [f"COMPLAINT:\n{(text or '').strip()[:MAX_INPUT_CHARS]}"]
        if context.get("location_text"):
            parts.append(f"LOCATION: {str(context['location_text'])[:200]}")
        if context.get("area"):
            parts.append(f"AREA: {str(context['area'])[:100]}")
        if context.get("category"):
            parts.append(
                f"CITIZEN'S OWN CATEGORY GUESS (may be wrong, verify it): {context['category']}"
            )
        user_message = "\n".join(parts) + "\n\nReturn the json object now."

        messages = [
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            result = self._attempt_parse(messages, started)
        except (LLMPermanentError, LLMTransientError, AnalyzerUnavailable) as exc:
            self._last_error = str(exc)
            llm_breaker.record_failure(str(exc))
            logger.warning("deepseek analyze failed: %s", exc)
            raise AnalyzerUnavailable(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            llm_breaker.record_failure(self._last_error)
            logger.exception("deepseek analyze crashed")
            raise AnalyzerUnavailable(self._last_error) from exc

        llm_breaker.record_success()
        self._last_error = None
        return result

    def _attempt_parse(self, messages: list[dict[str, str]], started: float) -> AnalysisResult:
        """Call, parse, validate. One corrective retry with the error fed back."""
        from pydantic import ValidationError

        raw, usage = self._call_with_retry(messages)
        tokens = self._usage(usage)
        logger.info(
            "deepseek call model=%s cache_hit=%s cache_miss=%s completion=%s",
            self.model,
            tokens.get("prompt_cache_hit_tokens", 0),
            tokens.get("prompt_cache_miss_tokens", 0),
            tokens.get("completion_tokens", 0),
        )

        try:
            return self._build(raw, tokens, started)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as first_error:
            logger.warning("deepseek output rejected, retrying once: %s", first_error)
            # Feed the validator error back. The correction goes in a fresh user
            # turn so the cached system prefix stays byte-identical.
            corrective = [
                messages[0],
                messages[1],
                {"role": "assistant", "content": raw[:1500]},
                {"role": "user",
                 "content": TRIAGE_RETRY_SUFFIX + str(first_error)[:600]},
            ]
            raw2, usage2 = self._call_with_retry(corrective)
            tokens2 = self._usage(usage2)
            # Bill both attempts honestly.
            for key in tokens:
                tokens2[key] = tokens2.get(key, 0) + tokens.get(key, 0)
            try:
                return self._build(raw2, tokens2, started)
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as second:
                raise LLMTransientError(
                    f"invalid json after corrective retry: {second}"
                ) from second

    def _build(self, raw: str, tokens: dict[str, int], started: float) -> AnalysisResult:
        """Parse + validate one raw response into an AnalysisResult."""
        payload = json.loads(_strip_fences(raw))
        if not isinstance(payload, dict):
            raise TypeError(f"expected a json object, got {type(payload).__name__}")

        result = AnalysisResult(
            category=payload.get("category"),
            priority=payload.get("priority"),
            summary=payload.get("summary") or "",
            department_suggestion=payload.get("department_suggestion") or "",
            confidence=payload.get("confidence", 0.7),
            source="llm",
            model_name=self.model,
            reasoning=(payload.get("reasoning") or None),
            keywords=payload.get("keywords") or [],
            sentiment=payload.get("sentiment"),
            is_emergency=bool(payload.get("is_emergency", False)),
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            **tokens,
        )

        # Post-conditions the prompt asks for but the model may still violate.
        # The contract is enforced here, not hoped for.
        if result.priority == "critical":
            result.is_emergency = True
        elif result.is_emergency and result.priority != "critical":
            # Model flagged an emergency at a lower level: trust the flag and
            # escalate, because under-triage is the expensive error.
            result.priority = "critical"
        if not result.department_suggestion:
            result.department_suggestion = DEPARTMENT_BY_CATEGORY[result.category]
        if len(result.summary) > 380:
            result.summary = result.summary[:377].rstrip() + "..."
        return result
