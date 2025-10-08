import pkgutil
import importlib
from quantool.core.helpers import LoggerFactory
logger = LoggerFactory.get_logger(__name__)

# Loop over every .py in this package and import it:
for _finder, name, _ispkg in pkgutil.iter_modules(__path__):
    try:
        importlib.import_module(f"{__name__}.{name}")
        logger.info(f"Imported module: {name}")
    except Exception as e:
        logger.error(f"Failed to import module {name}: {e}")