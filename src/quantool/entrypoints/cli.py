import sys
from transformers import HfArgumentParser
from quantool.args.quantization_args import ModelArguments, QuantizationArguments


def main():
    """
    Main entry point for the CLI.
    """
    # Initialize argument parser
    parser = HfArgumentParser((ModelArguments, QuantizationArguments))
    
    # If the user passes a single YAML file, load from it:
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        model_args, quant_args, calib_args, export_args = parser.parse_yaml_file(
            sys.argv[1], allow_extra_keys=False
        )
    else:
        # Fallback to normal CLI parsing:
        model_args, quant_args, calib_args, export_args = parser.parse_args_into_dataclasses()
