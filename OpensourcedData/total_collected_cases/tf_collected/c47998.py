import attr

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential

@attr.s(eq=False)
class TestLayer(layers.Layer):
    _n_out: int = attr.ib(validator=lambda i, a, x: x>=8)
    
    def __attrs_post_init__(self):
        super().__init__()
        self.fn = layers.Conv2D(self._n_out, 3, activation='relu')
    
    def call(self, x, **kwargs):
        return self.fn(x, **kwargs)
    

m = Sequential([TestLayer(8), layers.Dense(1)])

m.build([None, 128, 128, 3])


Possible cause of issue from within `Sequential` implementation:

from tensorflow.python.util import nest

nest.flatten(TestLayer(8)) # returns [8], should be [ TestLayer at <0x???????>]

