import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import keras_tuner as kt

class ConvHyperModel(kt.HyperModel):
    def build(self, hp):
        input_shape = (60, 300) #change
        prev_phrase_in = tf.keras.Input(shape=input_shape, name='prev_phrase')
        phrase_in      = tf.keras.Input(shape=input_shape, name='phrase')
        party_in       = tf.keras.Input(shape=(1, 78), name='party')

        # parameters tuning
        hp_number_conv_layers = hp.Int("num_conv_layers", 1, 3)
        hp_dropout_rate = hp.Choice('dropout_rate', values=[0.25, 0.5, 0.8])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        hp_filters = hp.Choice('filters', values=[100, 150])

        conv_pool_layers = []
        # Convolutions with previous phrase
        for i in range(hp_number_conv_layers):
            conv = layers.Conv1D(filters=hp_filters, kernel_size=i+2)(prev_phrase_in)
            conv = layers.BatchNormalization()(conv)
            conv = layers.Activation("relu")(conv)
            # max pooling
            conv_pool = layers.MaxPooling1D(pool_size=conv.shape[1], strides=1)(conv)
            conv_pool_layers.append(conv_pool)

        # Convolutions with current phrase
        for i in range(hp_number_conv_layers):
            conv = layers.Conv1D(filters=hp_filters, kernel_size=i+2)(phrase_in)
            conv = layers.BatchNormalization()(conv)
            conv = layers.Activation("relu")(conv)
            # max pooling
            conv_pool = layers.MaxPooling1D(pool_size=conv.shape[1], strides=1)(conv)
            conv_pool_layers.append(conv_pool)

        # Append the correspond political party
        conv_pool_layers.append(party_in)

        # Concatenation of (feature embeddings + party encodings)
        x = layers.Concatenate()(conv_pool_layers)
        x = layers.Flatten()(x)

        for i in range(hp.Int("num_hidden_layers", 0, 1)):
            x = layers.Dropout(rate=hp_dropout_rate)(x)
            x = layers.Dense(units=hp.Int(f"units_{i}", min_value=64, max_value=128, step=32), 
                             activation=None)(x)
            if hp.Boolean("batchnorm_after_hidden_layer"):
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation='sigmoid')(x)

        # Final Fully Connected
        x = layers.Dropout(rate=hp_dropout_rate)(x)
        x = layers.Dense(7, activation=None)(x)
        if hp.Boolean("relu_before_softmax"):
            x = layers.Activation("relu")(x)

        model = tf.keras.Model(inputs={'prev_phrase' : prev_phrase_in, 'phrase' : phrase_in, 'party' : party_in}, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['categorical_accuracy', 
                                tfa.metrics.F1Score(num_classes=7, average='macro', name='f1_score_macro')])

        return model

    def fit(self, hp, model, *args, **kwargs):
        #hp_batch_size = hp.Int('batch_size', min_value=8, max_value=128, step=8)

        return model.fit(
            *args,
            #batch_size=hp_batch_size,
            **kwargs,
        )