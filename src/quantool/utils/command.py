import os
import subprocess
import sys
import logging
from typing import List


def run_command(logger: logging.Logger, command: List[str], cwd: str = "."):
    """
    Run a command using subprocess.run and log the output.
    
    Args:
        logger: Logger instance to log output.
        command: Command to run as a list of strings.
        cwd: Optional working directory to run the command in.
        
    Returns:
        The command output as a string if successful.
    """
    logger.debug(
        f"Running: '{' '.join(command)}' in {os.path.join(os.getcwd(), cwd)}"
    )
    
    try:
        result = subprocess.run(
            command,
            cwd=os.path.join(os.getcwd(), cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True  # Raise CalledProcessError if the command fails
        )
        
        # Log the output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(line)
                sys.stdout.write(line + '\n')
                sys.stdout.flush()
        
        # Log any stderr even if the command succeeded
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning(line)
                sys.stderr.write(line + '\n')
                sys.stderr.flush()
                
        logger.info(f"Command '{' '.join(command)}' completed successfully")
        return result.stdout
    
    except subprocess.CalledProcessError as e:
        # Log the error and any output that was captured
        if e.stdout:
            for line in e.stdout.splitlines():
                logger.info(line)
                sys.stdout.write(line + '\n')
                sys.stdout.flush()
                
        if e.stderr:
            for line in e.stderr.splitlines():
                logger.warning(line)
                sys.stderr.write(line + '\n')
                sys.stderr.flush()
        
        logger.exception(
            f"Running: '{' '.join(command)}' in {os.path.join(os.getcwd(), cwd)}: Error code: {e.returncode}",
            exc_info=True,
            stack_info=True,
        )
        sys.exit(127) # Exit with error code 127 to indicate command failure