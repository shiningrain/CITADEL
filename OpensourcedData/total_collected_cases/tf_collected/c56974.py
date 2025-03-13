When I create a tf.lookup.StaticHashTable with tf.lookup.TextFileInitializer, then change the text file and create a new tf.lookup.StaticHashTable, it uses the contents from the original text file. If this caching behavior is expected, it's not clear to me how to clear this cache.



### Standalone code to reproduce the issue

shell
https://colab.research.google.com/drive/1mJ2rTixQXxaePjkdoEfbRMfNbJjlAFSe?usp=sharing


import tensorflow as tf
import tempfile

f = tempfile.NamedTemporaryFile(delete=False)
csv_data1 = f"""0,0
1,1
2,2
"""
with open(f.name, "w") as csv_file:
  csv_file.write(csv_data1)

!cat {f.name}

table1 = tf.lookup.StaticHashTable(
    initializer=tf.lookup.TextFileInitializer(
        f.name,
        key_dtype=tf.int64, key_index=0,
        value_dtype=tf.int64, value_index=1,
        delimiter=",",
    ),
    default_value=-1,
)

csv_data2 = f"""0,1
1,2
2,3
"""
with open(f.name, "w") as csv_file:
  csv_file.write(csv_data2)

!cat {f.name}

table2 = tf.lookup.StaticHashTable(
    initializer=tf.lookup.TextFileInitializer(
        f.name,
        key_dtype=tf.int64, key_index=0,
        value_dtype=tf.int64, value_index=1,
        delimiter=",",
    ),
    default_value=-1,
)

table1.export(), table2.export()

