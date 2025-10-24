from accelerate.commands.config.config_args import cache_dir
from transformers import HfArgumentParser
from quantool.args import (
    CommonArguments,
    LoggingArguments,
    ModelArguments,
    QuantizationArguments,
    CalibrationArguments,
    EvaluationArguments,
    ExportArguments,
)
from quantool.core.registry import QuantizerRegistry
from quantool.core.helpers import PipelineBase, LoggerFactory
from quantool.utils import load_model
from quantool.core.helpers.calibration import CalibrationArtifact
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
            CalibrationArguments,
            EvaluationArguments,
            ExportArguments,
            CommonArguments,
            LoggingArguments,
        ))
        
        # If the user passes a single YAML file, load from it:
        if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
            (model_args,
            quant_args,
            calibration_args,
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
            calibration_args,
            evaluation_args,
            export_args,
            common_args,
            logging_args) = parser.parse_args_into_dataclasses()
            logger.info("Parsed arguments from CLI.")
        
        # Initialize the pipeline
        pipeline = (
            PipelineBase()
            .add_step(setup_logging_step, name="setup_logging")
            .add_step(validate_args_step, name="validate_args")
            .add_step(load_model_step, name="load_model")
            .add_step(prepare_calibration_step, name="prepare_calibration")
            .add_step(quantize_step, name="quantize")
            .add_step(model_card_step, name="generate_readme")
            .add_step(save_step, name="save_model")
        )
        
        state = {
            "model_args": model_args,
            "quant_args": quant_args,
            "calibration_args": calibration_args,
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


def prepare_calibration_step(state):
    """Prepare calibration data or descriptor and store it in state['calibration_data'].

    Behavior:
      - If no calibration args provided -> state['calibration_data'] = None
      - If `load_in_pipeline` is False -> store a lightweight descriptor (dataset_id or dataset_path)
      - If `load_in_pipeline` is True -> attempt to load and optionally preprocess dataset
    """
    cargs = state.get("calibration_args")
    # Nothing requested
    if not cargs or (not getattr(cargs, "dataset_id", None) and not getattr(cargs, "dataset_path", None)):
        state["calibration_data"] = None
        logger.info("No calibration dataset requested; skipping calibration preparation")
        return state

    # If the user prefers the pipeline not to load the dataset, just keep the descriptor
    if not getattr(cargs, "load_in_pipeline", False):
        if getattr(cargs, "dataset_id", None):
            artifact = CalibrationArtifact(type="dataset_id", payload=cargs.dataset_id, meta={"sample_size": cargs.sample_size, "split": cargs.split})
        else:
            artifact = CalibrationArtifact(type="dataset_path", payload=cargs.dataset_path, meta={"sample_size": cargs.sample_size, "split": cargs.split})
        state["calibration_data"] = artifact
        logger.info(f"Prepared calibration descriptor: {artifact.type}={artifact.payload}")
        return state

    # Otherwise, try loading the dataset now
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("Calibration requested but the `datasets` package is not installed. Install with `pip install datasets`.") from e

    # Load dataset by id or from local path
    if getattr(cargs, "dataset_id", None):
        ds = load_dataset(cargs.dataset_id, cargs.dataset_config or None, split=cargs.split, cache_dir=cargs.cache_dir)
    else:
        # assume a local file (json/ndjson/csv); let datasets infer via data_files
        ds = load_dataset("json", data_files=cargs.dataset_path, split=cargs.split)

    # Shuffle and sample
    if getattr(cargs, "shuffle", True):
        try:
            ds = ds.shuffle(seed=cargs.seed)
        except Exception:
            pass
    if getattr(cargs, "sample_size", None):
        try:
            n = min(len(ds), cargs.sample_size)
            ds = ds.select(range(n))
        except Exception:
            # some datasets/streaming may not support length/select; ignore
            pass

    # Optional preprocess function (module.func string)
    if getattr(cargs, "preprocess_fn", None):
        try:
            module_name, fn_name = cargs.preprocess_fn.rsplit(".", 1)
            module = __import__(module_name, fromlist=[fn_name])
            fn = getattr(module, fn_name)
            ds = ds.map(lambda ex: fn(ex), batched=False)
        except Exception as e:
            logger.warning(f"Failed to run preprocess_fn '{cargs.preprocess_fn}': {e}")

    artifact = CalibrationArtifact(type="hf_dataset", payload=ds, meta={"sample_size": getattr(cargs, "sample_size", None), "split": cargs.split})
    state["calibration_data"] = artifact
    logger.info(f"Loaded calibration dataset (samples={artifact.meta.get('sample_size')})")
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

        # Build method-specific calibration kwargs from pipeline artifact
        cal_artifact = state.get("calibration_data")
        extra_kwargs = {}
        if cal_artifact is not None:
            try:
                art_type = getattr(cal_artifact, "type", None) or cal_artifact.get("type")
            except Exception:
                art_type = None

            if art_type == "dataset_id":
                extra_kwargs["dataset"] = cal_artifact.payload
                if cal_artifact.meta.get("sample_size"):
                    extra_kwargs["num_calibration_samples"] = cal_artifact.meta.get("sample_size")
            elif art_type == "dataset_path":
                extra_kwargs["dataset_path"] = cal_artifact.payload
            elif art_type in ("hf_dataset", "dataloader"):
                # pass as calibration_dataloader which llm-compressor or other methods may accept
                extra_kwargs["calibration_dataloader"] = cal_artifact.payload

        logger.info(f"Quantization args: {qargs.quantization_config}")
        merged_kwargs = dict(qargs.quantization_config or {})
        merged_kwargs.update(extra_kwargs)

        quantized_output = quantizer.quantize(
            model=source_model_path,
            level=level,
            **merged_kwargs,
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