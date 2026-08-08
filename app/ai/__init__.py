"""The AI layer: three interchangeable analyzers behind one abstract interface.

    app.ai.base            AIAnalyzer ABC + AnalysisResult schema + value normalisation
    app.ai.llm_analyzer    DeepSeekAnalyzer   (deepseek-v4-flash, JSON mode)
    app.ai.ml_analyzer     MLAnalyzer         (TF-IDF + calibrated LinearSVC, local)
    app.ai.rule_analyzer   RuleBasedAnalyzer  (weighted keywords, zero dependencies)
    app.ai.pipeline        the fallback chain, result cache and persistence
    app.ai.duplicates      TF-IDF cosine duplicate detection with geo/time gating
    app.ai.assistant       plan -> SQL -> prose civic assistant
    app.ai.circuit_breaker keeps a DeepSeek outage from stalling the whole API

Nothing is imported eagerly here. ``app.api.v1.router`` loads this package
defensively while other agents are still editing the modules it depends on, and a
heavyweight import at package level (sklearn, openai) would slow every boot and
turn one broken file into a dead API. Import the submodule you need directly.
"""

from __future__ import annotations

__all__ = [
    "AIAnalyzer",
    "AnalysisResult",
    "AnalyzerUnavailable",
    "DeepSeekAnalyzer",
    "MLAnalyzer",
    "RuleBasedAnalyzer",
    "analyze_and_store",
    "analyze_text",
    "find_duplicates",
]


def __getattr__(name: str):
    """PEP 562 lazy re-exports, so ``from app.ai import MLAnalyzer`` works cheaply."""
    if name in {"AIAnalyzer", "AnalysisResult", "AnalyzerUnavailable"}:
        from app.ai import base

        return getattr(base, name)
    if name == "DeepSeekAnalyzer":
        from app.ai.llm_analyzer import DeepSeekAnalyzer

        return DeepSeekAnalyzer
    if name == "MLAnalyzer":
        from app.ai.ml_analyzer import MLAnalyzer

        return MLAnalyzer
    if name == "RuleBasedAnalyzer":
        from app.ai.rule_analyzer import RuleBasedAnalyzer

        return RuleBasedAnalyzer
    if name in {"analyze_text", "analyze_and_store"}:
        from app.ai import pipeline

        return getattr(pipeline, name)
    if name == "find_duplicates":
        from app.ai.duplicates import find_duplicates

        return find_duplicates
    raise AttributeError(f"module 'app.ai' has no attribute {name!r}")
