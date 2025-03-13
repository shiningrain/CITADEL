import tensorflow as tf
import tensorflow_datasets as tfds

ds_train, ds_test = tfds.load("imdb_reviews", split=["train", "test"], as_supervised=True)


encoder1 = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=10000)
encoder2 = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=10000, output_sequence_length=200)

encoder1.adapt(ds_train.map(lambda x, y: x))
encoder2.adapt(ds_train.map(lambda x, y: x))

for x, y in ds_train:
  print(x)
  print(encoder1(x)) # This one works
  print(encoder2(x)) # This one fails



tf.Tensor(b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it.", shape=(), dtype=string)
tf.Tensor(
[  11   14   34  412  384   18   90   28    1    8   33 1322 3560   42
  487    1  191   24   85  152   19   11  217  316   28   65  240  214
    8  489   54   65   85  112   96   22 5596   11   93  642  743   11
   18    7   34  394 9522  170 2464  408    2   88 1216  137   66  144
   51    2    1 7558   66  245   65 2870   16    1 2860    1    1 1426
 5050    3   40    1 1579   17 3560   14  158   19    4 1216  891 8040
    8    4   18   12   14 4059    5   99  146 1241   10  237  704   12
   48   24   93   39   11 7339  152   39 1322    1   50  398   10   96
 1155  851  141    9], shape=(116,), dtype=int64)

---------------------------------------------------------------------------

InvalidArgumentError                      Traceback (most recent call last)

<ipython-input-3-fe1d367865bf> in <module>()
      2   print(x)
      3   print(encoder1(x))
----> 4   print(encoder2(x))

8 frames

/usr/local/lib/python3.7/dist-packages/six.py in raise_from(value, from_value)

InvalidArgumentError: slice index 1 of dimension 0 out of bounds. [Op:StridedSlice] name: text_vectorization_1/strided_slice/


