import pytest

from quantool.utils import dataset_textifier as dt


class FakeTokenizer:
    """A minimal fake tokenizer that implements apply_chat_template.

    It will render messages by joining role:content lines and, when asked to
    render combined prompt+completion, returns a predictable concatenation so
    slicing logic can be tested.
    """

    def __init__(self, prefix="[BOS]", suffix="[EOS]"):
        self.prefix = prefix
        self.suffix = suffix
        self.chat_template = "<chat_template>"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        # messages is a list of {"role":..., "content":...}
        parts = []
        for m in messages:
            parts.append(f"{m.get('role','user')}:{m.get('content','')}")
        out = self.prefix + "|".join(parts) + self.suffix
        if add_generation_prompt:
            out += "|GEN"
        return out


class NoTemplateTok:
    """Tokenizer without chat template support."""

    pass


# Fixtures
@pytest.fixture
def fake_tokenizer():
    """Provide a tokenizer with chat template support."""
    return FakeTokenizer()


@pytest.fixture
def no_template_tokenizer():
    """Provide a tokenizer without chat template support."""
    return NoTemplateTok()


# Tests for has_chat_template
@pytest.mark.parametrize("verify", [False, True])
def test_has_chat_template_with_template(fake_tokenizer, verify):
    """Test has_chat_template returns True when tokenizer has chat template."""
    assert dt.has_chat_template(fake_tokenizer, verify=verify) is True


def test_has_chat_template_without_template(no_template_tokenizer):
    """Test has_chat_template returns False when tokenizer lacks chat template."""
    assert dt.has_chat_template(no_template_tokenizer) is False


def test_has_chat_template_verify_mode(fake_tokenizer):
    """Test has_chat_template with verify=True actually calls apply_chat_template."""
    assert dt.has_chat_template(fake_tokenizer, verify=True) is True


# Tests for is_conversational
@pytest.mark.parametrize(
    "example,expected",
    [
        ({"messages": [{"role": "user", "content": "hi"}]}, True),
        ({"prompt": [{"role": "user", "content": "hi"}]}, True),
        ({"chosen": [{"role": "assistant", "content": "yes"}]}, True),
        ({"rejected": [{"role": "assistant", "content": "no"}]}, True),
        ({"completion": [{"role": "assistant", "content": "done"}]}, True),
        ({"text": "plain text"}, False),
        ({"input": "some input", "output": "some output"}, False),
    ],
)
def test_is_conversational(example, expected):
    """Test is_conversational correctly identifies conversational examples."""
    assert dt.is_conversational(example) is expected


# Tests for convert_row
@pytest.mark.parametrize(
    "example,expected_keys,assertions",
    [
        # Messages format
        (
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            },
            ["text"],
            lambda out: "user:hi" in out["text"] and "assistant:hello" in out["text"],
        ),
        # Prompt + completion
        (
            {
                "prompt": [{"role": "user", "content": "q?"}],
                "completion": [{"role": "assistant", "content": "a."}],
            },
            ["prompt", "completion"],
            lambda out: isinstance(out["prompt"], str)
            and isinstance(out["completion"], str),
        ),
        # Prompt + chosen + rejected
        (
            {
                "prompt": [{"role": "user", "content": "Which?"}],
                "chosen": [{"role": "assistant", "content": "ChoiceA"}],
                "rejected": [{"role": "assistant", "content": "ChoiceB"}],
            },
            ["prompt", "chosen", "rejected"],
            lambda out: out["chosen"] != out["rejected"],
        ),
        # Implicit prompt (chosen + rejected only)
        (
            {
                "chosen": [{"role": "assistant", "content": "Yes"}],
                "rejected": [{"role": "assistant", "content": "No"}],
            },
            ["chosen", "rejected"],
            lambda out: out["chosen"] != out["rejected"],
        ),
        # With label field
        (
            {
                "prompt": [{"role": "user", "content": "q?"}],
                "completion": [{"role": "assistant", "content": "a."}],
                "label": "positive",
            },
            ["prompt", "completion", "label"],
            lambda out: out.get("label") == "positive",
        ),
    ],
)
def test_convert_row_valid_formats(fake_tokenizer, example, expected_keys, assertions):
    """Test convert_row with various valid conversational formats."""
    out = dt.convert_row(example, tokenizer=fake_tokenizer)
    # Check expected keys are present
    for key in expected_keys:
        assert key in out, f"Expected key '{key}' not found in output"
    # Run custom assertions
    assert assertions(out), "Custom assertions failed"


def test_convert_row_no_template_support(no_template_tokenizer):
    """Test convert_row returns example unchanged when tokenizer lacks chat template."""
    example = {"messages": [{"role": "user", "content": "ping"}]}
    out = dt.convert_row(example, tokenizer=no_template_tokenizer)
    assert out is example


def test_convert_row_non_conversational(fake_tokenizer):
    """Test convert_row returns example unchanged for non-conversational data."""
    example = {"text": "plain text", "label": 1}
    out = dt.convert_row(example, tokenizer=fake_tokenizer)
    assert out is example


@pytest.mark.parametrize(
    "invalid_example",
    [
        {
            "prompt": [{"role": "user", "content": "q?"}],
            "messages": [{"role": "user", "content": "hi"}],
        },
        {"chosen": [{"role": "assistant", "content": "yes"}]},  # missing rejected
        {"completion": [{"role": "assistant", "content": "done"}]},  # missing prompt
    ],
)
def test_convert_row_invalid_key_combinations(fake_tokenizer, invalid_example):
    """Test convert_row raises KeyError for invalid key combinations."""
    with pytest.raises(KeyError):
        dt.convert_row(invalid_example, tokenizer=fake_tokenizer)
