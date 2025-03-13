import tensorflow as tf
import pandas as pd
import logging
from pathlib import Path

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level = logging.INFO)

path = "https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff"
if not Path("freMTPL2freq.csv").exists():
  logging.info("Loading data file")
  df = pd.read_csv(path)
  df.to_csv("freMTPL2freq.csv")
else:
  logging.info("Data file already exists")
  df = pd.read_csv(path)

logging.info("Opening data")
data = tf.data.experimental.make_csv_dataset("freMTPL2freq.csv",
                                             num_rows_for_inference=100,
                                             batch_size=256,
                                             shuffle=False,
                                             sloppy=False,
                                             #num_parallel_reads=4,
                                             num_epochs=1,)
# this is even slower?
#data = tf.data.Dataset.from_tensor_slices(df.to_dict('list'))

logging.info("Reading data")
# This part is slower than expected
tmp = list(data.as_numpy_iterator())
logging.info("Done reading data")
lookup = tf.keras.layers.experimental.preprocessing.StringLookup()
feature_ds = data.map(lambda x: x['Area'])
logging.info("Starting adapt")
# This part is also slower than expected
lookup.adapt(feature_ds)
logging.info("Finished adapt")

# Commented out IPython magic to ensure Python compatibility.
# %timeit tmp = list(data.as_numpy_iterator())

# Commented out IPython magic to ensure Python compatibility.
# %timeit lookup.adapt(feature_ds)

# Commented out IPython magic to ensure Python compatibility.
# %timeit df["Area"].unique()

# Commented out IPython magic to ensure Python compatibility.
# %timeit pd.read_csv(path)

