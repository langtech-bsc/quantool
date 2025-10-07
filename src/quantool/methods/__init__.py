import pkgutil
import importlib

# Loop over every .py in this package and import it:
for _finder, name, _ispkg in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{name}")