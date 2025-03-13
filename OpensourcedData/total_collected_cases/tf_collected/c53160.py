import sys
import tensorflow as tf
import numpy as np

def build_model_():

	input_a_size = 20
	input_b_size = 4
	num_classes = 2
	len_embedding = 256

	input_a = tf.keras.layers.Input(shape=(input_a_size,), name='input_a', dtype=np.uint8)
	input_b = tf.keras.layers.Input(shape=(input_b_size,), name='input_b', dtype=np.float32)

	x = tf.keras.layers.Embedding(len_embedding, 100)(input_a)
	x = tf.keras.layers.Conv1D(128, 4, activation='relu')(x)
	x = tf.keras.layers.MaxPooling1D(4)(x)
	x = tf.keras.layers.Flatten()(x)
	branch_a = tf.keras.layers.Dense(64, activation='relu')(x)

	x = tf.keras.layers.Dense(32, activation='relu')(input_b)
	branch_b = tf.keras.layers.Dense(32, activation='relu')(x)

	concat = tf.keras.layers.Concatenate()([
				                            branch_a,
				                            branch_b,
				                           ])

	x = tf.keras.layers.Dense(512, activation = 'relu')(concat)
	output = tf.keras.layers.Dense(num_classes, name='output', activation='softmax')(x)

	model = tf.keras.models.Model(inputs=[
				                          input_a,
				                          input_b,
				                         ],
				                  outputs=[output])

	return model

strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])
with strategy.scope():
    model = build_model_()
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

y_train = True
y_train = tf.keras.utils.to_categorical(y_train, 2)

dataset = tf.data.Dataset.from_tensors(
    (
        {"input_a": [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]], 
         "input_b": [[1.], [1.], [1.], [1.]],}, 
        {"output": y_train},
    )
).repeat(1000000).batch(256)

history = model.fit(
    x = dataset,
    epochs=10,
    verbose = 1,
)


