from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from deep_researcher.contracts import (
    AgentRole,
    AgentSpec,
    BudgetUsage,
    EvidenceRelation,
)
from deep_researcher.kernel import ModelAdapter, ModelRequest, ModelResponse
from deep_researcher.knowledge.normalization import normalize_text

from .models import SemanticJudgment, SemanticLabel, SemanticVerificationAdapter


_TOKEN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)*)(?:%|‰|[a-zA-Z]+)?")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN.findall(normalize_text(value))
        if len(token) > 1 and token.casefold() not in _STOP_WORDS
    }


class DeterministicSemanticVerificationAdapter(SemanticVerificationAdapter):
    """Frozen/offline verifier with explicit lexical and numeric-entailment rules."""

    async def judge(
        self,
        *,
        run_id: str,
        task_id: str | None,
        subject_id: str,
        statement: str,
        evidence_id: str,
        relation: EvidenceRelation,
        passages: tuple[str, ...],
    ) -> SemanticJudgment:
        del run_id, task_id, subject_id, evidence_id
        normalized_statement = normalize_text(statement)
        corpus = normalize_text("\n".join(passages))
        statement_tokens = _tokens(normalized_statement)
        corpus_tokens = _tokens(corpus)
        lexical_coverage = (
            len(statement_tokens & corpus_tokens) / len(statement_tokens)
            if statement_tokens
            else 0.0
        )
        if (
            normalized_statement
            and normalized_statement.casefold() in corpus.casefold()
        ):
            lexical_coverage = 1.0
        statement_numbers = tuple(dict.fromkeys(_NUMBER.findall(normalized_statement)))
        corpus_numbers = set(_NUMBER.findall(corpus))
        absent_numbers = tuple(
            item for item in statement_numbers if item not in corpus_numbers
        )
        numeric_factor = (
            1.0
            if not statement_numbers
            else (
                (len(statement_numbers) - len(absent_numbers)) / len(statement_numbers)
            )
        )
        score = max(0.0, min(1.0, lexical_coverage * (0.75 + 0.25 * numeric_factor)))
        if relation == EvidenceRelation.SUPPORTS and score >= 0.2:
            label = SemanticLabel.SUPPORTS
        elif relation == EvidenceRelation.REFUTES and score >= 0.2:
            label = SemanticLabel.REFUTES
        else:
            label = SemanticLabel.NEUTRAL
        contradiction_fragments = (
            absent_numbers if label == SemanticLabel.REFUTES else ()
        )
        return SemanticJudgment(
            label=label,
            score=score,
            overreach_fragments=(
                absent_numbers if relation == EvidenceRelation.SUPPORTS else ()
            ),
            contradiction_fragments=contradiction_fragments,
            decision_summary=(
                f"Deterministic verification found {lexical_coverage:.3f} statement-token coverage "
                f"and {numeric_factor:.3f} numeric grounding; relation evaluated as {label.value}."
            ),
        )


class AgentSpecSemanticVerificationAdapter(SemanticVerificationAdapter):
    """Uses the independent verifier AgentSpec through the shared ModelAdapter."""

    def __init__(
        self,
        *,
        agent_spec: AgentSpec,
        model: ModelAdapter,
        max_output_tokens: int | None = None,
    ) -> None:
        if agent_spec.role != AgentRole.EVIDENCE_VERIFIER:
            raise ValueError(
                "semantic verification requires an evidence_verifier AgentSpec"
            )
        self.agent_spec = agent_spec
        self.model = model
        self.max_output_tokens = max_output_tokens or agent_spec.reserved_output_tokens

    async def judge(
        self,
        *,
        run_id: str,
        task_id: str | None,
        subject_id: str,
        statement: str,
        evidence_id: str,
        relation: EvidenceRelation,
        passages: tuple[str, ...],
    ) -> SemanticJudgment:
        request = ModelRequest(
            run_id=run_id,
            task_id=task_id or f"task_verify_{subject_id}",
            actor_id=self.agent_spec.agent_spec_id,
            system=(
                "Independently classify whether the supplied passages support or refute the statement. "
                "Return only the requested structured judgment. Identify unsupported quantitative or "
                "scope fragments. Provide a short decision summary, never hidden chain-of-thought."
            ),
            messages=(
                {
                    "role": "user",
                    "content": {
                        "subject_id": subject_id,
                        "statement": statement,
                        "evidence_id": evidence_id,
                        "declared_relation": relation.value,
                        "passages": list(passages),
                    },
                },
            ),
            command_schema=SemanticJudgment.model_json_schema(),
            model_version=self.agent_spec.model.version,
            prompt_version=self.agent_spec.prompt.version,
            max_output_tokens=self.max_output_tokens,
            metadata={
                "agent_spec_id": self.agent_spec.agent_spec_id,
                "verification_policy_version": (
                    self.agent_spec.verification_policy.version
                    if self.agent_spec.verification_policy is not None
                    else None
                ),
            },
        )
        response = await self.model.complete(request)
        usage = response.usage
        try:
            judgment = self._parse(response)
        except (ValidationError, TypeError, ValueError) as exc:
            repaired = await self.model.repair(request, response, (str(exc),))
            judgment = self._parse(repaired)
            usage = self._combine_usage(usage, repaired.usage)
        return judgment.model_copy(
            update={"usage": self._combine_usage(judgment.usage, usage)}
        )

    @staticmethod
    def _parse(response: ModelResponse) -> SemanticJudgment:
        payload: Any = response.structured
        if isinstance(payload, dict) and "judgment" in payload:
            payload = payload["judgment"]
        if payload is None:
            raise ValueError("verifier model did not return structured output")
        return SemanticJudgment.model_validate(payload, strict=False)

    @staticmethod
    def _combine_usage(left: BudgetUsage, right: BudgetUsage) -> BudgetUsage:
        return left.plus(
            input_tokens=right.input_tokens,
            output_tokens=right.output_tokens,
            cost_usd=right.cost_usd,
            wall_time_seconds=right.wall_time_seconds,
            model_calls=right.model_calls,
            tool_calls=right.tool_calls,
            search_calls=right.search_calls,
            retries=right.retries,
            errors=right.errors,
        )
