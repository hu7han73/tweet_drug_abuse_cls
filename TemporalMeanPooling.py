from keras.engine.topology import Layer, InputSpec
from keras import backend as T

class TemporalMeanPooling(Layer):
    """
This is a custom Keras layer. This pooling layer accepts the temporal
sequence output by a recurrent layer and performs temporal pooling,
looking at only the non-masked portion of the sequence. The pooling
layer converts the entire variable-length hidden vector sequence
into a single hidden vector, and then feeds its output to the Dense
layer.

input shape: (nb_samples, nb_timesteps, nb_features)
output shape: (nb_samples, nb_features)
"""
    def __init__(self, **kwargs):
        super(TemporalMeanPooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None): #mask: (nb_samples, nb_timesteps)
        if mask is None:
            mask = T.mean(T.ones_like(x), axis=-1)
        ssum = T.sum(x,axis=-2) #(nb_samples, np_features)
        mask = T.cast(mask,T.floatx())
        rcnt = T.sum(mask,axis=-1,keepdims=True) #(nb_samples)
        return ssum/rcnt
        #return rcnt

    def compute_mask(self, input, mask):
        return None
