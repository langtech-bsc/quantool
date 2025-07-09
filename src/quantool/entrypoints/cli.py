from accelerate.commands.config.config_args import cache_dir
from transformers import HfArgumentParser
from quantool.args import (
    CommonArguments,
    LoggingArguments,
    ModelArguments,
    QuantizationArguments,
    # CalibrationArguments,
    EvaluationArguments,
    ExportArguments
)
from quantool.core.registry import QuantizerRegistry
from quantool.core.helpers import PipelineBase, LoggerFactory
from quantool.utils import load_model
import sys
import os

# Must import quantization methods to register them
from quantool import methods

# configure CLI logger
logger = LoggerFactory.get_logger(__name__)

def main():
    """
    Main entry point for the CLI.
    """
    try:
        # Initialize argument parser with all argument classes
        parser = HfArgumentParser((
            ModelArguments, 
            QuantizationArguments, 
            # CalibrationArguments, 
            EvaluationArguments,
            ExportArguments,
            CommonArguments,
            LoggingArguments,
        ))
        
        # If the user passes a single YAML file, load from it:
        if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
            (model_args, 
            quant_args, 
            #  calib_args,
            evaluation_args,
            export_args, 
            common_args, logging_args) = parser.parse_yaml_file(
                sys.argv[1], allow_extra_keys=False
            )
            logger.info(f"Parsed arguments from YAML file: {sys.argv[1]}")
        else:
            # Fallback to normal CLI parsing:
            (model_args, 
            quant_args, 
            #  calib_args,
            evaluation_args, 
            export_args,
            common_args, 
            logging_args) = parser.parse_args_into_dataclasses()
            logger.info("Parsed arguments from CLI.")
        
        # Initialize the pipeline
        pipeline = (
            PipelineBase()
            .add_step(load_model_step,     name="load_model")
            .add_step(setup_logging_step,  name="setup_logging")
            .add_step(validate_args_step,  name="validate_args")
            .add_step(quantize_step,       name="quantize")
            .add_step(model_card_step,         name="generate_readme")
            .add_step(save_step,           name="save_model")
        )
        
        state = {
            "model_args": model_args,
            "quant_args": quant_args,
            # "calib_args": calib_args,
            "export_args": export_args,
            "common_args": common_args,
            "logging_args": logging_args,
        }
        
        logger.info("Starting quantization pipeline...")
        result = pipeline.run(state)
        logger.info("Quantization pipeline completed successfully!")
        return result
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

# define pipeline steps
def load_model_step(state):
    """Load model and tokenizer from Hugging Face."""

    margs = state["model_args"]
    
    try:
        # Load model
        model_path = load_model(margs.model_id, revision = margs.revision, cache_dir = margs.cache_dir)
        state["model_path"] = model_path
        logger.info(f"Model downloaded successfully from {margs.model_id} to {model_path}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e

    return state

def setup_logging_step(state):
    """Setup experiment logging if enabled."""
    logging_args = state["logging_args"]
    state["loggers"] = {}
    
    # if logging_args.enable_mlflow:
    #     from quantool.loggers.mlflow import QuantizationMLflowLogger
    #     state["loggers"]["mlflow"] = QuantizationMLflowLogger(
    #         experiment_name=logging_args.mlflow_experiment_name,
    #         tracking_uri=logging_args.mlflow_tracking_uri
    #     )
    #     logger.info("MLflow logging enabled")
    
    # if logging_args.enable_wandb:
    #     from quantool.loggers.wandb import QuantizationWandbLogger
    #     state["loggers"]["wandb"] = QuantizationWandbLogger(
    #         project=logging_args.wandb_project,
    #         entity=logging_args.wandb_entity
    #     )
    #     logger.info("W&B logging enabled")
    
    return state

def validate_args_step(state):
    """Validate arguments and quantizer availability."""
    qargs = state["quant_args"]
    
    # Check if the requested quantization method is available
    available_methods = QuantizerRegistry.list()
    if qargs.method not in available_methods:
        raise ValueError(f"Quantization method '{qargs.method}' not available. "
                        f"Available methods: {available_methods}")
    
    # Validate quantization level if provided
    if hasattr(qargs, 'quant_level') and qargs.quant_level:
        # We'll validate the level during quantization since it's method-specific
        logger.info(f"Using quantization level: {qargs.quant_level}")
    # elif hasattr(qargs, 'bit_width') and qargs.bit_width:
    #     logger.info(f"Using bit width: {qargs.bit_width}")
    else:
        logger.warning("No quantization level or bit width specified, using method defaults")
    
    logger.info(f"Validated arguments for {qargs.method} quantization")
    return state

def quantize_step(state):
    """Apply quantization using the specified method."""
    qargs = state["quant_args"]
    margs = state["model_args"]
    # Get source model path - use the actual path/repo ID from model arguments
    source_model_path = state.get("model_path", margs.model_id)
    # cargs = state["calib_args"]
    
    try:
        # Create quantizer instance
        quantizer = QuantizerRegistry.create(qargs.method, model_id = margs.model_id, 
                                             **qargs.quantization_config)
        logger.info(f"Created {qargs.method} quantizer: {quantizer.__class__.__name__}")
        
        # Determine quantization level
        level = qargs.quant_level #or str(getattr(qargs, 'bit_width', 'Q4_K_M'))
        
        # Store original model for comparison (if loaded)
        if state.get("model") is not None:
            state["original_model"] = state["model"]
        

        # Prepare calibration arguments, filtering out None values
        # calib_kwargs = {k: v for k, v in vars(cargs).items() if v is not None}
        
        logger.info(f"Starting quantization: method={qargs.method}, level={level}, source={source_model_path}")
        
        # Apply quantization - pass the source model path instead of loaded model
        # Most quantizers work better with the original model path/repo ID
        print("Quantization args",qargs.quantization_config)
        quantized_output = quantizer.quantize(
            model=source_model_path, 
            level=level, 
            **qargs.quantization_config
        )
        
        # Store quantizer and results
        state["quantizer"] = quantizer
        state["quantized_output"] = quantized_output
        
        logger.info(f"Quantization completed successfully")
        logger.info(f"Quantized output: {quantized_output}")
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise RuntimeError(f"Failed to quantize model using {qargs.method}: {e}") from e
        
    return state


def save_step(state):
    """Save model using ExportMixin functionality."""
    eargs = state["export_args"]
    quantizer = state["quantizer"]
    
    try:
        # Check if push_to_hub is requested
        should_push_to_hub = getattr(eargs, 'push_to_hub', False)
        repo_id = getattr(eargs, 'repo_id', None)
        private = getattr(eargs, 'private', None)
        
        if should_push_to_hub and repo_id:
            # Push directly to hub using the mixin's push_to_hub method
            commit_message = f"Upload quantized model using {state['quant_args'].method}"
            logger.info(f"Pushing model to Hugging Face Hub: {repo_id}")
            
            result = quantizer.push_to_hub(
                repo_id=repo_id,
                private=private,
                commit_message=commit_message
            )
            logger.info(f"Model successfully pushed to hub: {repo_id}")
            logger.info(f"Upload result: {result}")
            
        else:
            # Save locally using the mixin's save_pretrained method
            output_path = eargs.output_path
            if not output_path:
                # Create default output path
                output_path = f"./output/{state['quant_args'].method}_{state['model_args'].model_id.replace('/', '_')}" # TODO: Use a more robust path handling
            
            logger.info(f"Saving model locally to: {output_path}")
            quantizer.save_pretrained(save_directory=output_path)
            logger.info(f"Model successfully saved to {output_path}")
        
        # # Log to experiment trackers
        # if "loggers" in state:
        #     config = {
        #         "method": state["quant_args"].method,
        #         "model_name": state["model_args"].model_id,
        #         # "bit_width": getattr(state["quant_args"], "bit_width", None),
        #         "quant_level": getattr(state["quant_args"], "quant_level", None),
        #     }
            
        #     metrics = state.get("evaluation_metrics", {})
            
        #     for logger_name, exp_logger in state["loggers"].items():
        #         try:
        #             exp_logger.log_quantization_run(
        #                 method=state["quant_args"].method,
        #                 model_name=state["model_args"].model_name_or_path.split("/")[-1],
        #                 quantization_config=config,
        #                 performance_metrics=metrics,
        #                 model_path=output_path if not should_push_to_hub else repo_id
        #             )
        #             logger.info(f"Successfully logged to {logger_name}")
        #         except Exception as e:
        #             logger.error(f"Failed to log to {logger_name}: {e}")
                    
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise RuntimeError(f"Model saving failed: {e}") from e
    
    return state

def model_card_step(state):
    """Generate README using the quantizer's template card."""
    try:
        quantizer = state["quantizer"]
        # The ExportMixin's _save_model_card method handles README generation
        quantizer.save_model_card(
            save_directory=state["export_args"].output_path
        )
        logger.info("README generated successfully")
    except Exception as e:
        logger.warning(f"Failed to check README generation capability: {e}")
    return state

if __name__ == "__main__":
    main()