from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quantool.methods.llm_compressor.llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

modifier_map = {
    "gptq": GPTQModifier,
    "awq": AWQModifier,
    "smoothquant": SmoothQuantModifier,
}