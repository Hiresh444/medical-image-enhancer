"""Agentic chat assistant for run-specific Q&A.

Users can ask questions about a pipeline run (metrics, issues, plan, validation)
and the assistant answers using only the stored run context — never hallucinating
unseen details or outputting PHI.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from pipeline.storage import get_run, save_chat_message, get_chat_history

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

CHAT_SYSTEM_PROMPT = """\
You are a medical imaging QA assistant.  You answer questions about a specific
image enhancement run.  You MUST follow these rules:

1. **Only answer using the provided run context.**  If the context does not
   contain the information, say "I don't have that information for this run."
2. **Never output PHI** (patient names, IDs, dates of birth, etc.).
3. **Never echo full DICOM tags.**  Only reference safe metadata (Modality,
   BodyPartExamined, StudyDescription).
4. **Never hallucinate** metric values, parameters, or results not in the context.
5. Use bullet points and short explanations.
6. If asked about a metric you can explain (SSIM, PSNR, NIQE, SNR, CNR, entropy,
   edge density, Laplacian energy, histogram spread), provide a brief definition.
7. If asked how to improve results, suggest concrete parameter adjustments based
   on the run's plan and validation results.

## RUN CONTEXT
{run_context}
"""


def _build_run_context(run_data: dict[str, Any]) -> str:
    """Build a concise, non-PHI context string from stored run data."""
    parts: list[str] = []

    parts.append(f"Run ID: {run_data.get('run_id', 'unknown')}")
    parts.append(f"Timestamp: {run_data.get('timestamp', 'unknown')}")
    parts.append(f"Input file: {run_data.get('input_filename', 'unknown')}")
    parts.append(f"Status: {run_data.get('status', 'unknown')}")

    meta = run_data.get("metadata_summary", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}
    if meta:
        parts.append(f"Metadata: {json.dumps(meta)}")

    issues = run_data.get("issues", [])
    parts.append(f"Detected issues: {json.dumps(issues)}")

    mb = run_data.get("metrics_before", {})
    ma = run_data.get("metrics_after", {})
    parts.append(f"Metrics BEFORE: {json.dumps(mb, indent=2)}")
    parts.append(f"Metrics AFTER: {json.dumps(ma, indent=2)}")

    val = run_data.get("validation", {})
    parts.append(f"Validation: {json.dumps(val, indent=2)}")

    ops = run_data.get("applied_ops", [])
    parts.append(f"Applied operations: {json.dumps(ops)}")

    plan = run_data.get("plan_json", "")
    if plan:
        parts.append(f"Enhancement plan JSON: {plan}")

    expl = run_data.get("explainability", {})
    if expl:
        parts.append(f"Explainability: {json.dumps(expl, default=str)}")

    model = run_data.get("genai_model", "")
    if model:
        parts.append(f"Model used: {model}")
        parts.append(f"LLM calls: {run_data.get('genai_llm_calls', 0)}")

    return "\n".join(parts)


def handle_chat(
    run_id: str,
    user_message: str,
    model: str | None = None,
) -> str:
    """Process a chat message for a specific run.

    Parameters
    ----------
    run_id : str
        The pipeline run to reference.
    user_message : str
        The user's question.
    model : str | None
        OpenAI model override.

    Returns
    -------
    str
        The assistant's response.
    """
    from agents import Agent, ModelSettings, Runner

    if model is None:
        model = _DEFAULT_MODEL

    # Load run data
    run_data = get_run(run_id)
    if run_data is None:
        return f"Run '{run_id}' not found. Please check the run ID."

    # Build context
    run_context = _build_run_context(run_data)
    system_prompt = CHAT_SYSTEM_PROMPT.format(run_context=run_context)

    # Load chat history
    history = get_chat_history(run_id)

    # Build conversation
    conversation_parts: list[str] = []
    for msg in history[-10:]:  # Last 10 messages for context window
        role = msg["role"]
        content = msg["content"]
        conversation_parts.append(f"[{role}]: {content}")
    conversation_parts.append(f"[user]: {user_message}")

    input_text = "\n".join(conversation_parts) if conversation_parts else user_message

    # Save user message
    save_chat_message(run_id, "user", user_message)

    # Run chat agent
    try:
        # Some models (gpt-5-mini, o-series) reject the temperature param
        _no_temp = ("o1", "o3", "o4", "gpt-5")
        model_lower = model.lower()
        if any(model_lower.startswith(p) for p in _no_temp):
            settings = ModelSettings(max_tokens=1024)
        else:
            settings = ModelSettings(temperature=0.3, max_tokens=1024)

        agent = Agent(
            name="ChatAssistant",
            model=model,
            instructions=system_prompt,
            model_settings=settings,
        )

        result = Runner.run_sync(agent, input=input_text, max_turns=3)
        response = result.final_output
        if not isinstance(response, str):
            response = str(response)

    except Exception as exc:
        logger.error("Chat agent failed: %s", exc)
        response = (
            "I encountered an error processing your question. "
            "Please try rephrasing or check the system logs."
        )

    # Save assistant response
    save_chat_message(run_id, "assistant", response)

    return response
