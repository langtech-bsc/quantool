"""
Utilities for converting conversational dataset examples to plain text and
applying chat templates when available.

This module contains a compact, dependency-light reimplementation inspired by
the helpers found in TRL (https://github.com/huggingface/trl). It intentionally
recreates only the small subset of functionality needed here (chat-template
application and conversational example normalization) to avoid adding TRL as a
dependency.
"""

from itertools import takewhile
from typing import Any, Callable, Dict, Optional, Union


def has_chat_template(tokenizer_or_processor: Any, verify: bool = False) -> bool:
    """
    Heuristically detect if a tokenizer/processor supports chat templates.

    Returns True if:
      - object exposes `.apply_chat_template`, AND
      - `.chat_template` attribute is a non-empty string
        (or, if verify=True, calling apply_chat_template on a minimal
         conversation doesn't raise).

    This works for both tokenizers and processors.
    """
    obj = tokenizer_or_processor
    if not hasattr(obj, "apply_chat_template"):
        return False

    # Fast path: explicit template string present
    tmpl = getattr(obj, "chat_template", None)
    if isinstance(tmpl, str) and tmpl.strip():
        return True

    # Some classes ship a default template even if the string attr is empty.
    if verify:
        try:
            _ = obj.apply_chat_template(
                [{"role": "user", "content": "ping"}],
                tokenize=False,
                add_generation_prompt=False,
            )
            return True
        except Exception:
            return False

    return False


def is_conversational(sample: Union[str, Any]):
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
    example_keys = {key for key in sample.keys() if key in supported_keys}

    if example_keys:
        key = example_keys.pop()
        maybe_messages = sample[key]
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            if (
                isinstance(maybe_message, dict)
                and "role" in maybe_message
                and "content" in maybe_message
            ):
                return True

    return False


def _validate_example_keys(example: dict) -> set:
    """Validate that example has supported keys in valid combinations."""
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    valid_sets = [
        {"messages"},
        {"prompt"},
        {"prompt", "completion"},
        {"prompt", "chosen", "rejected"},
        {"chosen", "rejected"},
        {"prompt", "completion", "label"},
    ]
    if example_keys not in valid_sets:
        raise KeyError(f"Invalid keys in the example: {example_keys}")
    return example_keys


def _extract_common_prefix(str1: str, str2: str) -> str:
    """Extract the common prefix between two strings."""
    return "".join(x for x, _ in takewhile(lambda x: x[0] == x[1], zip(str1, str2)))


def _process_messages(
    example: dict, tokenizer: Any, tools: Optional[list], kwargs: dict
) -> Dict[str, str]:
    """Process examples with 'messages' key."""
    messages = tokenizer.apply_chat_template(
        example["messages"],
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    )
    return {"text": messages}


def _process_prompt_only(
    example: dict, tokenizer: Any, tools: Optional[list], kwargs: dict
) -> str:
    """Process the prompt field and return rendered prompt string."""
    last_role = example["prompt"][-1].get("role")
    if last_role == "user":
        add_generation_prompt = True
        continue_final_message = False
    elif last_role == "assistant":
        add_generation_prompt = False
        continue_final_message = True
    else:
        raise ValueError(f"Invalid role in the last message: {last_role}")

    return tokenizer.apply_chat_template(
        example["prompt"],
        tools=tools,
        continue_final_message=continue_final_message,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def _process_prompt_with_response(
    example: dict,
    prompt: str,
    response_key: str,
    tokenizer: Any,
    tools: Optional[list],
    kwargs: dict,
) -> tuple[str, str]:
    """
    Process prompt with a response field (chosen/rejected/completion).
    Returns updated prompt and the response text.
    """
    prompt_response = tokenizer.apply_chat_template(
        example["prompt"] + example[response_key],
        tools=tools,
        tokenize=False,
        **kwargs,
    )
    common = _extract_common_prefix(prompt, prompt_response)
    response = prompt_response[len(common) :]
    return common, response


def _process_implicit_responses(
    example: dict, tokenizer: Any, tools: Optional[list], kwargs: dict
) -> Dict[str, str]:
    """Process examples with chosen/rejected but no explicit prompt."""
    output = {}
    if "chosen" in example:
        chosen = tokenizer.apply_chat_template(
            example["chosen"],
            tools=tools,
            tokenize=False,
            **kwargs,
        )
        output["chosen"] = chosen
    if "rejected" in example:
        rejected = tokenizer.apply_chat_template(
            example["rejected"],
            tools=tools,
            tokenize=False,
            **kwargs,
        )
        output["rejected"] = rejected
    return output


def convert_row(
    example: dict[str, list[dict[str, str]]],
    tokenizer: Any,
    tools: Optional[list[Union[dict, Callable]]] = None,
    **template_kwargs,
) -> Dict[str, str]:
    """
    Apply a chat template to a conversational example if the tokenizer supports it.

    This is a compact, dependency-light implementation inspired by TRL's
    `apply_chat_template`/`maybe_apply_chat_template` helpers. If the example
    is not conversational or the tokenizer doesn't provide `apply_chat_template`,
    the function returns the example unchanged.

    Supported input shapes (same as TRL):
      - {"messages": [...]}
      - {"prompt": [...]} (prompt-only)
      - {"prompt": [...], "completion": [...]} (prompt+completion)
      - {"prompt", "chosen", "rejected"} (preference)
      - {"chosen","rejected"} (implicit prompt)
      - {"prompt","completion","label"} (unpaired preference)

    Returns a dict with string fields produced by the tokenizer's
    `apply_chat_template` calls (e.g. 'text', 'prompt', 'chosen', 'rejected', 'completion').
    """
    # Quick bail-out: not conversational or tokenizer lacks template capability
    if not is_conversational(example) or not has_chat_template(tokenizer):
        return example  # leave unchanged for non-conversational examples
    elif is_conversational(example) and not has_chat_template(tokenizer):
        raise ValueError(
            "Example appears conversational but tokenizer lacks chat template support."
        )

    _validate_example_keys(example)

    # Merge kwargs supplied per-example
    ext_kwargs = example.get("chat_template_kwargs", {}) or {}
    kwargs = {**ext_kwargs, **template_kwargs}

    output: Dict[str, str] = {}

    try:
        # Handle messages (language modeling / chat-style)
        if "messages" in example:
            return _process_messages(example, tokenizer, tools, kwargs)

        # Handle prompt-based examples
        if "prompt" in example:
            prompt = _process_prompt_only(example, tokenizer, tools, kwargs)

            # Process prompt with responses
            if "chosen" in example:
                prompt, chosen = _process_prompt_with_response(
                    example, prompt, "chosen", tokenizer, tools, kwargs
                )
                output["chosen"] = chosen

            if "rejected" in example:
                prompt, rejected = _process_prompt_with_response(
                    example, prompt, "rejected", tokenizer, tools, kwargs
                )
                output["rejected"] = rejected

            if "completion" in example:
                prompt, completion = _process_prompt_with_response(
                    example, prompt, "completion", tokenizer, tools, kwargs
                )
                output["completion"] = completion

            output["prompt"] = prompt
        else:
            # Implicit prompt case: chosen/rejected alone
            output = _process_implicit_responses(example, tokenizer, tools, kwargs)

        if "label" in example:
            output["label"] = example["label"]

    except Exception:
        # If template application fails for any reason, return the example as-is
        # so downstream code can handle non-chat formats.
        return example

    return output
