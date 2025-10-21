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

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        # messages is a list of {"role":..., "content":...}
        parts = []
        for m in messages:
            parts.append(f"{m.get('role','user')}:{m.get('content','')}")
        out = self.prefix + "|".join(parts) + self.suffix
        if add_generation_prompt:
            out += "|GEN"
        return out
    


def test_has_chat_template_true():
    tok = FakeTokenizer()
    assert dt.has_chat_template(tok) is True


def test_convert_messages():
    tok = FakeTokenizer()
    example = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    out = dt.convert_row(example, tokenizer=tok)
    assert "text" in out
    assert "user:hi" in out["text"]


def test_convert_prompt_completion():
    tok = FakeTokenizer()
    example = {
        "prompt": [{"role": "user", "content": "q?"}],
        "completion": [{"role": "assistant", "content": "a."}],
    }
    out = dt.convert_row(example, tokenizer=tok)
    # prompt and completion should be produced and be strings
    assert isinstance(out.get("prompt"), str)
    assert isinstance(out.get("completion"), str)
    assert "user:q?" in out["prompt"] or out["prompt"].startswith("[BOS]")


def test_convert_chosen_rejected_with_prompt():
    tok = FakeTokenizer()
    example = {
        "prompt": [{"role": "user", "content": "Which?"}],
        "chosen": [{"role": "assistant", "content": "ChoiceA"}],
        "rejected": [{"role": "assistant", "content": "ChoiceB"}],
    }
    out = dt.convert_row(example, tokenizer=tok)
    assert "chosen" in out and "rejected" in out and "prompt" in out
    assert out["chosen"] != out["rejected"]


def test_convert_chosen_rejected_implicit():
    tok = FakeTokenizer()
    example = {
        "chosen": [{"role": "assistant", "content": "Yes"}],
        "rejected": [{"role": "assistant", "content": "No"}],
    }
    out = dt.convert_row(example, tokenizer=tok)
    assert "chosen" in out and "rejected" in out


def test_fallback_when_no_apply():
    class NoTemplateTok:
        pass

    tok = NoTemplateTok()
    example = {"messages": [{"role": "user", "content": "ping"}]}
    out = dt.convert_row(example, tokenizer=tok)
    # Since tokenizer lacks apply_chat_template, convert_row should return the example unchanged
    assert out is example
