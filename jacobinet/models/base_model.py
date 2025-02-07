import keras
from jacobinet.layers.layer import (
    BackwardLayer,
)

from .utils import is_linear_layer
from typing import Any, Union, List


class BackwardModel(keras.Model):
    def __init__(
        self,
        *args,
        **kwargs: Any,
    ):
        if "n_input" in kwargs:
            self.n_input = kwargs.pop("n_input")
        else:
            self.n_input = 1
        super().__init__(*args, **kwargs)
        self.is_linear = min(
            [
                is_linear_layer(layer)
                for layer in self.layers
                if isinstance(layer, BackwardLayer)
                or isinstance(layer, BackwardModel)
                or isinstance(layer, BackwardSequential)
            ]
        )

    def get_config(self):
        config = super().get_config()
        config["n_input"] = self.n_input

        return config

    """
    @classmethod
    def from_config(cls, config):
        backward_model = super().from_config(config)
        n_input = config.pop('n_input')
        model = keras.Model.from_config(config)
        instance = BackwardModel(model.inputs, model.outputs)
        return instance
    """


class BackwardSequential(keras.Sequential):

    def __init__(self, layers=None, trainable=True, name=None):
        self.is_linear = True
        self.n_input = 1
        if len(self.layers):
            self.is_linear = min(
                [is_linear_layer(layer) for layer in self.layers]
            )
        super().__init__(layers=layers, trainable=trainable, name=name)

    def add(self, layer, rebuild=True):
        """Adds a layer instance on top of the layer stack.

        Args:
            layer: layer instance.
        """
        super().add(layer=layer, rebuild=rebuild)
        # recompute is_linear
        self.is_linear = min([self.is_linear, is_linear_layer(layer)])

    def pop(self, rebuild=True):
        """Removes the last layer in the model."""
        super().pop(rebuild=rebuild)
        self.is_linear = min([is_linear_layer(layer) for layer in self.layers])

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        backward_model = super().from_config(config)
        return backward_model
