import tensorflow as tf
from tensorflow.python.training.tracking import autotrackable

_FILENAME = "file.txt"
_SAVEDIR = "model"

# Write contents to asset file
with open(_FILENAME, 'w') as f:
    f.write("foo")

# First Save
trackable = autotrackable.AutoTrackable()
asset = tf.saved_model.Asset(_FILENAME)
trackable.asset = asset
tf.saved_model.save(trackable, _SAVEDIR)

# Change asset contents
with open(_FILENAME, 'w') as f:
    f.write("bar")

# Second Save
tf.saved_model.save(trackable, _SAVEDIR)

# Load and compare with in-memory trackable
loaded_trackable = tf.saved_model.load(_SAVEDIR)
print("Expected Result: {}".format(tf.io.read_file(trackable.asset)))      # bar
print("Loaded Result: {}".format(tf.io.read_file(loaded_trackable.asset))) # foo

