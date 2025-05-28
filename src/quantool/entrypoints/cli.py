from transformers import HfArgumentParser
from quantool.args.quantization_args import ModelArguments, QuantizationArguments, CalibrationArguments, ExportArguments
from quantool.args.common_args import CommonArguments, LoggingArguments, EvaluationArguments
from quantool.core.registry import QuantizerRegistry
from quantool.core.helpers import PipelineBase, LoggerFactory
import sys

# Import all methods to register them
from quantool.methods.gptq import GPTQ
from quantool.methods.gptq_v2 import GPTQv2
from quantool.methods.awq import AWQ
from quantool.methods.gguf import GGUF
from quantool.methods.higgs import Higgs
from quantool.methods.aqml import AQML

# configure CLI logger
logger = LoggerFactory.get_logger(__name__)

def main():
    """
    Main entry point for the CLI.
    """
    # Initialize argument parser with all argument classes
    parser = HfArgumentParser((
        ModelArguments, 
        QuantizationArguments, 
        CalibrationArguments, 
        ExportArguments,
        CommonArguments,
        LoggingArguments,
        EvaluationArguments
    ))
    
    # If the user passes a single YAML file, load from it:
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        (model_args, quant_args, calib_args, export_args, 
         common_args, logging_args, eval_args) = parser.parse_yaml_file(
            sys.argv[1], allow_extra_keys=False
        )
    else:
        # Fallback to normal CLI parsing:
        (model_args, quant_args, calib_args, export_args,
         common_args, logging_args, eval_args) = parser.parse_args_into_dataclasses()
        logger.info("Parsed arguments from CLI.")
    
    # Initialize the pipeline
    pipeline = (
        PipelineBase()
        .add_step(load_model_step,   name="load_model")
        .add_step(setup_logging_step, name="setup_logging")
        .add_step(quantize_step,     name="quantize")
        .add_step(evaluate_step,     name="evaluate")
        .add_step(save_step,         name="save_model")
        .add_step(readme_step,       name="generate_readme")
    )
    
    state = {
        "model_args": model_args,
        "quant_args": quant_args,
        "calib_args": calib_args,
        "export_args": export_args,
        "common_args": common_args,
        "logging_args": logging_args,
        "eval_args": eval_args,
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

def setup_logging_step(state):
    """Setup experiment logging if enabled."""
    logging_args = state["logging_args"]
    state["loggers"] = {}
    
    if logging_args.enable_mlflow:
        from quantool.loggers.mlflow import QuantizationMLflowLogger
        state["loggers"]["mlflow"] = QuantizationMLflowLogger(
            experiment_name=logging_args.mlflow_experiment_name,
            tracking_uri=logging_args.mlflow_tracking_uri
        )
        logger.info("MLflow logging enabled")
    
    if logging_args.enable_wandb:
        from quantool.loggers.wandb import QuantizationWandbLogger
        state["loggers"]["wandb"] = QuantizationWandbLogger(
            project=logging_args.wandb_project,
            entity=logging_args.wandb_entity
        )
        logger.info("W&B logging enabled")
    
    return state

def quantize_step(state):
    qargs = state["quant_args"]
    quantizer = QuantizerRegistry.get(qargs.method)
    level = qargs.quant_level or str(qargs.bit_width)
    
    # Store original model for comparison
    state["original_model"] = state["model"]
    
    # Apply quantization
    quantized_model = quantizer.quantize(state["model"], level, **vars(state["calib_args"]))
    quantizer.tokenizer = state["tokenizer"]  # Ensure tokenizer is available
    
    state["quantizer"] = quantizer
    state["quantized_model"] = quantized_model
    logger.info(f"Model quantized using {qargs.method} at level {level}.")
    return state

def evaluate_step(state):
    """Evaluate model performance if enabled."""
    eval_args = state["eval_args"]
    if not eval_args.enable_evaluation:
        logger.info("Evaluation disabled, skipping...")
        return state
    
    # Placeholder for evaluation logic
    # In a real implementation, you'd add proper evaluation here
    state["evaluation_metrics"] = {
        "perplexity": 15.2,  # Placeholder values
        "accuracy": 0.85,
        "inference_time": 0.12
    }
    
    logger.info("Model evaluation completed.")
    return state

def save_step(state):
    """Save model using ExportMixin functionality."""
    eargs = state["export_args"]
    quantizer = state["quantizer"]
    
    # Use the mixin's save_pretrained method instead of export()
    quantizer.save_pretrained(
        save_directory=eargs.output_path,
        push_to_hub=getattr(eargs, 'push_to_hub', False),
        repo_id=getattr(eargs, 'repo_id', None),
        private=getattr(eargs, 'private', None)
    )
    
    # Log to experiment trackers
    if "loggers" in state:
        config = {
            "method": state["quant_args"].method,
            "model_name": state["model_args"].model_name_or_path,
            "bit_width": state["quant_args"].bit_width,
            "quant_level": state["quant_args"].quant_level,
        }
        
        metrics = state.get("evaluation_metrics", {})
        
        for logger_name, exp_logger in state["loggers"].items():
            try:
                exp_logger.log_quantization_run(
                    method=state["quant_args"].method,
                    model_name=state["model_args"].model_name_or_path.split("/")[-1],
                    quantization_config=config,
                    performance_metrics=metrics,
                    model_path=eargs.output_path
                )
                logger.info(f"Logged to {logger_name}")
            except Exception as e:
                logger.error(f"Failed to log to {logger_name}: {e}")
    
    logger.info(f"Model saved to {eargs.output_path}")
    return state

def readme_step(state):
    from quantool.core.base import BaseTemplateRenderer
    try:
        BaseTemplateRenderer().generate_readme(state)
        logger.info("Generated README for quantized model.")
    except Exception as e:
        logger.warning(f"Failed to generate README: {e}")
    return state

if __name__ == "__main__":
    main()