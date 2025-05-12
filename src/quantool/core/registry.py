class Registry:
    def __init__(self):
        self._plugins = {}

    def register(self, plugin_cls):
        self._plugins[plugin_cls.name] = plugin_cls()
        return plugin_cls

    def get(self, name):
        return self._plugins[name]

    def list(self):
        return list(self._plugins.keys())

QuantizerRegistry = Registry()
