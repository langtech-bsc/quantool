from quantool.core.base import BaseQuantizer


class Registry:
    def __init__(self):
        self._plugins: dict[str, type[BaseQuantizer]] = {}

    def register(self, plugin_cls: type[BaseQuantizer]):
        if not hasattr(plugin_cls, "name"):
            raise ValueError(f"{plugin_cls.__name__} must have a 'name' attribute")
        name = plugin_cls.name
        if name in self._plugins:
            raise KeyError(f"Plugin {name!r} already registered")
        self._plugins[name] = plugin_cls
        return plugin_cls

    def create(self, name: str, **kwargs):
        plugin_cls = self._plugins[name]
        return plugin_cls(**kwargs)

    def list(self):
        return list(self._plugins.keys())


QuantizerRegistry = Registry()
