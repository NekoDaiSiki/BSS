class Registry:
    """A registry to map strings to classes.
    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    def get(self, key):
        """Get the registry record.
        Args:
            key (str): The class name in string format.
        Returns:
            class: The corresponding class.
        """
        return self._module_dict.get(key, None)

    def register_module(self):

        def register(cls):
            if cls.__name__ in self._module_dict:
                raise KeyError('{} is already registered in {}'.format(cls.__name__, self.name))
            self._module_dict[cls.__name__] = cls
            return cls
        return register

def build_from_cfg(cfg, registry):
    args = dict(cfg)
    name = args.pop('type')
    cls = registry.get(name)
    if cls is None:
        raise KeyError('{} is not registered in {}'.format(name, registry.name))
    return cls(**args)
