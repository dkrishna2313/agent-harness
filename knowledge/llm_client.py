"""Provider-agnostic LLM client using native SDKs.

The model string determines the provider automatically:
  Gemini:    "gemini-2.5-flash"      needs GEMINI_API_KEY
  Anthropic: "claude-sonnet-5"       needs ANTHROPIC_API_KEY
  OpenAI:    "gpt-4o-mini"           needs OPENAI_API_KEY
"""

from __future__ import annotations

import json
import logging

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"


def _provider(model: str) -> str:
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    raise ValueError(f"Cannot determine provider from model string: {model!r}. "
                     "Expected model starting with 'gemini', 'claude', 'gpt-', 'o1', 'o3', or 'o4'.")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def chat_completion(
    model: str,
    messages: list[dict],
    *,
    system: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """Return the model's text response."""
    p = _provider(model)
    if p == "gemini":
        return _gemini_chat(model, messages, system=system, max_tokens=max_tokens)
    if p == "anthropic":
        return _anthropic_chat(model, messages, system=system, max_tokens=max_tokens)
    return _openai_chat(model, messages, system=system, max_tokens=max_tokens)


def tool_use(
    model: str,
    messages: list[dict],
    *,
    tool_name: str,
    tool_description: str,
    tool_parameters: dict,
    system: str | None = None,
    max_tokens: int = 2048,
) -> dict | None:
    """Force a single tool call and return the parsed input dict, or None on failure."""
    p = _provider(model)
    if p == "gemini":
        return _gemini_tool_use(model, messages, tool_name=tool_name,
                                tool_description=tool_description,
                                tool_parameters=tool_parameters,
                                system=system, max_tokens=max_tokens)
    if p == "anthropic":
        return _anthropic_tool_use(model, messages, tool_name=tool_name,
                                   tool_description=tool_description,
                                   tool_parameters=tool_parameters,
                                   system=system, max_tokens=max_tokens)
    return _openai_tool_use(model, messages, tool_name=tool_name,
                            tool_description=tool_description,
                            tool_parameters=tool_parameters,
                            system=system, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Gemini (google-genai)
# ---------------------------------------------------------------------------


def _gemini_chat(model: str, messages: list[dict], *, system: str | None, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        system_instruction=system or "",
    )
    response = client.models.generate_content(
        model=model,
        config=config,
        contents=_to_gemini_contents(messages),
    )
    return response.text or ""


def _gemini_tool_use(
    model: str,
    messages: list[dict],
    *,
    tool_name: str,
    tool_description: str,
    tool_parameters: dict,
    system: str | None,
    max_tokens: int,
) -> dict | None:
    from google import genai
    from google.genai import types

    client = genai.Client()
    config_kwargs: dict = dict(
        max_output_tokens=max_tokens,
        system_instruction=system or "",
        tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=tool_name,
                description=tool_description,
                parameters=tool_parameters,
            )
        ])],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
    )
    # Disable thinking for tool-use calls — thinking output lands in a separate
    # candidate part that causes content=None on the function-call candidate.
    try:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass  # older SDK versions without ThinkingConfig

    try:
        response = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(**config_kwargs),
            contents=_to_gemini_contents(messages),
        )
    except Exception as exc:
        LOGGER.warning("llm_client: Gemini tool_use call failed — %s", exc)
        return None

    # SDK shortcut (google-genai >= 0.8)
    fn_calls = getattr(response, "function_calls", None)
    if fn_calls:
        for fc in fn_calls:
            if fc.name == tool_name:
                return dict(fc.args)

    # Manual scan across all candidates
    for candidate in response.candidates or []:
        content = candidate.content
        if content is None:
            finish_reason = getattr(candidate, "finish_reason", "unknown")
            LOGGER.debug("llm_client: candidate has no content — finish_reason=%s", finish_reason)
            continue
        for part in content.parts or []:
            fc = getattr(part, "function_call", None)
            if fc and fc.name == tool_name:
                return dict(fc.args)

    LOGGER.warning(
        "llm_client: Gemini did not call %r tool — finish_reasons=%s",
        tool_name,
        [getattr(c, "finish_reason", "?") for c in (response.candidates or [])],
    )
    return None


def _to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-style messages to Gemini contents format."""
    role_map = {"user": "user", "assistant": "model"}
    return [
        {"role": role_map.get(m["role"], "user"), "parts": [{"text": m["content"]}]}
        for m in messages
        if m["role"] in role_map
    ]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _anthropic_chat(model: str, messages: list[dict], *, system: str | None, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic()
    kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=messages)
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text if response.content else ""


def _anthropic_tool_use(
    model: str,
    messages: list[dict],
    *,
    tool_name: str,
    tool_description: str,
    tool_parameters: dict,
    system: str | None,
    max_tokens: int,
) -> dict | None:
    import anthropic

    client = anthropic.Anthropic()
    tool = {"name": tool_name, "description": tool_description, "input_schema": tool_parameters}
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        tools=[tool],
        tool_choice={"type": "any"},
        messages=messages,
    )
    if system:
        kwargs["system"] = system

    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:
        LOGGER.warning("llm_client: Anthropic tool_use call failed — %s", exc)
        return None

    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input

    LOGGER.warning("llm_client: Anthropic did not call %r tool", tool_name)
    return None


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def _openai_chat(model: str, messages: list[dict], *, system: str | None, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI()
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens, messages=full_messages
    )
    return response.choices[0].message.content or ""


def _openai_tool_use(
    model: str,
    messages: list[dict],
    *,
    tool_name: str,
    tool_description: str,
    tool_parameters: dict,
    system: str | None,
    max_tokens: int,
) -> dict | None:
    from openai import OpenAI

    client = OpenAI()
    tool = {"type": "function", "function": {
        "name": tool_name,
        "description": tool_description,
        "parameters": tool_parameters,
    }}
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            messages=full_messages,
        )
    except Exception as exc:
        LOGGER.warning("llm_client: OpenAI tool_use call failed — %s", exc)
        return None

    for choice in response.choices:
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                if tc.function.name == tool_name:
                    try:
                        return json.loads(tc.function.arguments)
                    except json.JSONDecodeError as exc:
                        LOGGER.warning("llm_client: failed to parse OpenAI tool arguments — %s", exc)
                        return None

    LOGGER.warning("llm_client: OpenAI did not call %r tool", tool_name)
    return None
