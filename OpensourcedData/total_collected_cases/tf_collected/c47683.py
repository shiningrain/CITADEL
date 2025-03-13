!pip install tensorflow-text
!pip install sentencepiece

!python -c "import tensorflow as tf; print(tf.version.GIT_VERSION, tf.version.VERSION)"
!python --version

import tensorflow as tf
import tensorflow_text as tf_text
import sentencepiece

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

tokenizer_prefix = "/tmp/tokenizer"
corpus_path = tokenizer_prefix + ".txt"
model_path = tokenizer_prefix + ".model"

with open(corpus_path, "w") as f: 
  content = f.write("a b c d e")

sentencepiece.SentencePieceTrainer.Train(input=corpus_path, model_prefix=tokenizer_prefix, vocab_size=9, character_coverage=1.0)

with tf.device("/TPU:0"):
  tokenizer = tf_text.SentencepieceTokenizer(model=tf.io.gfile.GFile(model_path, "rb").read(), add_eos=True)
  @tf.function
  def tf_detokenize(input):
    return tokenizer.detokenize(input)
  print(tf_detokenize(tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)))

