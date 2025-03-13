import tensorflow as tf

labels = tf.ragged.constant([["a", "b"], ["a"]])
vals = tf.constant([0.1, 0.2])
ds = tf.data.Dataset.from_tensors(dict(labels=labels, vals=vals, other=vals))

parts = ["labels", "vals"]

def _flatten(ex):
  flat_ds = tf.data.Dataset.from_tensor_slices({k: ex[k] for k in parts})

  def _merge(_flat_ex):
    _flat_ex["other"] = tf.constant([0.1, 0.2])
    return _flat_ex

  return flat_ds.map(_merge)
ds = ds.flat_map(_flatten)

for ex in ds.as_numpy_iterator():
  print(ex)

