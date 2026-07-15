import pickle

import torch


class LegacyHabitatConfig(dict):
    """Plain placeholder for Habitat 0.1.x Config objects in old checkpoints."""


class LegacyCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "habitat.config.default" and name == "Config":
            return LegacyHabitatConfig
        return super().find_class(module, name)


class LegacyCheckpointPickleModule:
    __name__ = "pickle"
    Unpickler = LegacyCheckpointUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)


def load_torch_checkpoint_compat(checkpoint_path, *args, **kwargs):
    """Load current checkpoints normally and recover legacy Habitat configs."""
    try:
        return torch.load(checkpoint_path, *args, **kwargs)
    except (KeyError, RecursionError):
        if "pickle_module" in kwargs:
            raise
        return torch.load(
            checkpoint_path,
            *args,
            pickle_module=LegacyCheckpointPickleModule,
            **kwargs,
        )
