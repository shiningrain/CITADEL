import tensorflow as tf
nbins = -16
value_range = [0.0, 5.0]
new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
indices = tf.histogram_fixed_width_bins(new_values, value_range, nbins=nbins)
indices.numpy()

Outputs:

array([0, 0, 0, 0, 0, 0], dtype=int32)


**Describe the current behavior**
`tf.histogram_fixed_width_bins` has an argument `nbins` which should be a **positive** integer. However, it does not perform any validity checking and can accept a **negative** value like `-16`.  `tf.histogram_fixed_width` (another API with similar functionality) can detect this error and raise an `InvalidArgumentError`:

import tensorflow as tf
nbins = -16
value_range = [0.0, 5.0]
new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
indices = tf.histogram_fixed_width(new_values, value_range, nbins=nbins)
indices.numpy()
# InvalidArgumentError: nbins should be a positive number, but got '-16' [Op:HistogramFixedWidth]

