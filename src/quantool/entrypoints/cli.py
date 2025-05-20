from transformers import HfArgumentParser
from quantool.args.quantization_args import ModelArguments, QuantizationArguments, CalibrationArguments, ExportArguments
from quantool.core.registry import QuantizerRegistry
from quantool.core.helpers import PipelineBase, LoggerFactory
import sys

# configure CLI logger
logger = LoggerFactory.get_logger(__name__)

def main():
    """
    Main entry point for the CLI.
    """
    # Initialize argument parser
    parser = HfArgumentParser((ModelArguments, QuantizationArguments, CalibrationArguments, ExportArguments))
    
    # If the user passes a single YAML file, load from it:
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        model_args, quant_args, calib_args, export_args = parser.parse_yaml_file(
            sys.argv[1], allow_extra_keys=False
        )
    else:
        # Fallback to normal CLI parsing:
        model_args, quant_args, calib_args, export_args = parser.parse_args_into_dataclasses()
        logger.info("Parsed arguments from CLI.")
    
    # Initialize the pipeline
    # Add steps to the pipeline
    pipeline = (
        PipelineBase()
        .add_step(load_model_step,   name="load_model")
        .add_step(quantize_step,     name="quantize")
        .add_step(export_step,       name="export")
        .add_step(readme_step,       name="generate_readme")
    )
    state = {
        "model_args": model_args,
        "quant_args": quant_args,
        "calib_args": calib_args,
        "export_args": export_args,
    }
    pipeline.run(state)

# define pipeline steps
def load_model_step(state):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    margs = state["model_args"]
    state["tokenizer"] = AutoTokenizer.from_pretrained(
        margs.tokenizer_name or margs.model_name_or_path,
        cache_dir=margs.cache_dir,
        use_auth_token=margs.use_auth_token,
        revision=margs.revision,
    )
    state["model"] = AutoModelForCausalLM.from_pretrained(
        margs.model_name_or_path,
        config=margs.config_name,
        cache_dir=margs.cache_dir,
        use_auth_token=margs.use_auth_token,
        revision=margs.revision,
    )
    logger.info("Model and tokenizer loaded.")
    return state

def quantize_step(state):
    qargs = state["quant_args"]
    quantizer = QuantizerRegistry.get(qargs.method)
    level = qargs.quant_level or str(qargs.bit_width)
    quantizer.quantize(state["model"], level, **vars(state["calib_args"]))
    state["quantizer"] = quantizer
    state["quantized_model"] = state["model"]
    logger.info(f"Model quantized using {qargs.method} at level {level}.")
    return state

def export_step(state):
    eargs = state["export_args"]
    state["quantizer"].export(state["quantized_model"], eargs.output_path)
    logger.info(f"Quantized model exported to {eargs.output_path}.")
    return state

def readme_step(state):
    from quantool.core.base import BaseTemplateRenderer
    BaseTemplateRenderer().generate_readme(state)
    logger.info("Generated README for quantized model.")
    return state

if __name__ == "__main__":
    main()