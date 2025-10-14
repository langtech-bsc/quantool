---
# Quantization Model Card Template
# This template is used by quantool to generate model cards for quantized models
{{ card_data }}
---

# {{ model_name | default("Quantized Model", true) }}

{{ model_description | default("", true) }}

{% if base_model %}
## Base Model

This model is a quantized version of [{{ base_model }}](https://huggingface.co/{{ base_model }}).

**Original Model:** `{{ base_model }}`
{% endif %}

## Quantization Details

{% if quantization_method %}
**Method:** {{ quantization_method }}
{% endif %}

{% if quantization_config %}
### Configuration

| Parameter | Value |
|-----------|-------|
{% for key, value in quantization_config.items() -%}
| `{{ key }}` | `{{ value }}` |
{% endfor %}
{% endif %}

{% if model_size_info %}
### Model Size

{{ model_size_info }}
{% endif %}

## Usage

{% if usage_example %}
```python
{{ usage_example }}
```
{% elif library_name == "transformers" %}
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{{ model_id | default('path/to/model', true) }}")
tokenizer = AutoTokenizer.from_pretrained("{{ model_id | default('path/to/model', true) }}")

# Generate text
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
{% elif library_name == "llama.cpp" or (quantization_config and quantization_config.format == "gguf") %}
### With llama.cpp

```bash
./main -m {{ model_file | default("model.gguf", true) }} -p "Your prompt here" -n 128
```

### With llama-cpp-python

```python
from llama_cpp import Llama

# Load model
llm = Llama(model_path="{{ model_file | default('model.gguf', true) }}")

# Generate text
output = llm("Your prompt here", max_tokens=128)
print(output['choices'][0]['text'])
```
{% else %}
```python
# Usage example
# Please refer to the model documentation for specific usage instructions
```
{% endif %}

{% if performance_metrics %}
## Performance

{{ performance_metrics }}
{% endif %}

{% if intended_use %}
## Intended Use

{{ intended_use }}
{% endif %}

{% if out_of_scope_use %}
## Out-of-Scope Use

{{ out_of_scope_use }}
{% endif %}

{% if limitations %}
## Limitations and Biases

{{ limitations }}
{% endif %}

{% if ethical_considerations %}
## Ethical Considerations

{{ ethical_considerations }}
{% endif %}

{% if citation_bibtex or citations %}
## Citation

{% if citation_bibtex %}
**BibTeX:**

```bibtex
{{ citation_bibtex }}
```
{% endif %}

{% if citations %}
**References:**

{% for citation in citations -%}
{% if citation.startswith('http') -%}
- [{{ citation }}]({{ citation }})
{% else -%}
- {{ citation }}
{% endif %}
{% endfor %}
{% endif %}
{% endif %}

{% if acknowledgments %}
## Acknowledgments

{{ acknowledgments }}
{% endif %}

---

<sub>This model card was automatically generated using [quantool](https://github.com/langtech-bsc/quantool) - A unified quantization toolkit for LLMs.</sub>
