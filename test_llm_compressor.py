from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.utils import dispatch_for_generation


# Select calibration dataset.
DATASET_ID = "mit-han-lab/pile-val-backup"
DATASET_SPLIT = "validation"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 4
MAX_SEQUENCE_LENGTH = 256

from llmcompressor.utils import dispatch_for_generation
model_name = "amd/AMD-Llama-135m"
# Load model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
)

recipe = [
    AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
]

[
    AWQMapping(
        "re:.*input_layernorm",
        ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
    ),
    AWQMapping("re:.*v_proj", ["re:.*o_proj"]),
    AWQMapping(
        "re:.*post_attention_layernorm",
        ["re:.*gate_proj", "re:.*up_proj"],
    ),
    AWQMapping(
        "re:.*up_proj",
        ["re:.*down_proj"],
    ),
]

# Load dataset.
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
# Preprocess the data into the format the model is trained with.
def preprocess(example):
    return {"text": 
            tokenizer.apply_chat_template(example["messages"], tokenize=False,)
            }
ds = ds.map(preprocess)

# Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)
# ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
# ds = ds.shuffle(seed=42)


# def preprocess(example):
#     return {
#         "text": tokenizer.apply_chat_template(
#             [{"role": "user", "content": example["text"]}],
#             tokenize=False,
#         )
#     }


# ds = ds.map(preprocess)
# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
print(tokenizer.decode(model.generate(input_ids, max_new_tokens=32)[0]))