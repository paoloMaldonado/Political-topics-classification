import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import keras_tuner as kt

class ConvHyperModel(kt.HyperModel):
    def build(self, hp):
        input_shape = (80, 100) #change
        inputs = tf.keras.Input(shape=input_shape)

        # parameters tuning
        hp_dropout_rate = hp.Choice('dropout', values=[0.0, 0.15, 0.25, 0.5, 0.75, 0.8])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        hp_filters = hp.Choice('filters', values=[10, 50, 100, 150])
        hp_activation = hp.Choice('activation_linear_layer', values=["relu", "sigmoid"])

        conv_layers = []
        for i in range(hp.Int("num_conv_layers", 1, 3)):
            conv = layers.Conv1D(filters=hp_filters, kernel_size=i+2)(inputs)
            if hp.Boolean("batchnorm"):
                conv = layers.BatchNormalization()(conv)
            conv = layers.Activation("relu")(conv)
            # max pooling
            conv_pool = layers.MaxPooling1D(pool_size=conv.shape[1], strides=1)(conv)
            conv_layers.append(conv_pool)

        x = layers.Concatenate()(conv_layers)
        x = layers.Flatten()(x)

        for i in range(hp.Int("num_layers", 0, 3)):
            x = layers.Dropout(rate=hp_dropout_rate)(x)
            x = layers.Dense(units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32), 
                             activation=None)(x)
            if hp.Boolean("batchnorm"):
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation=hp_activation)(x)

        # Final Fully Connected
        x = layers.Dropout(rate=hp_dropout_rate)(x)
        x = layers.Dense(5, activation=None)(x)
        if hp.Boolean("batchnorm"):
            x = layers.BatchNormalization()(x)
        if hp.Boolean("relu_before_softmax"):
            x = layers.Activation("relu")(x)
        x = layers.Activation("softmax")(x)

        model=tf.keras.Model(inputs, x)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=[tfa.metrics.F1Score(num_classes=5, average='micro', name='f1_score_micro'), 
                               tfa.metrics.F1Score(num_classes=5, average='macro', name='f1_score_macro')])

        return model

    def fit(self, hp, model, *args, **kwargs):
        hp_batch_size = hp.Int('batch_size', min_value=8, max_value=128, step=8)

        return model.fit(
            *args,
            batch_size=hp_batch_size,
            **kwargs,
        )

class SimpleConvHyperModel(kt.HyperModel):
    def build(self, hp):
        input_shape = (80, 100) #change
        inputs = tf.keras.Input(shape=input_shape)

        # parameters tuning
        hp_dropout_rate = hp.Choice('dropout', values=[0.0, 0.15, 0.25, 0.5, 0.75, 0.8])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        hp_filters = hp.Choice('filters', values=[10, 50, 100, 150])
        hp_activation = hp.Choice('activation_linear_layer', values=["relu", "sigmoid"])
        hp_kernel_size = hp.Choice('kernel_size', values=[2, 3, 4])

        conv = layers.Conv1D(filters=hp_filters, kernel_size=hp_kernel_size)(inputs)
        if hp.Boolean("batchnorm"):
            conv = layers.BatchNormalization()(conv)
        conv = layers.Activation("relu")(conv)
        x = layers.MaxPooling1D(pool_size=conv.shape[1], strides=1)(conv)

        x = layers.Flatten()(x)

        for i in range(hp.Int("num_layers", 0, 3)):
            x = layers.Dropout(rate=hp_dropout_rate)(x)
            x = layers.Dense(units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32), 
                             activation=None)(x)
            if hp.Boolean("batchnorm"):
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation=hp_activation)(x)

        # Final Fully Connected
        x = layers.Dropout(rate=hp_dropout_rate)(x)
        x = layers.Dense(5, activation=None)(x)
        if hp.Boolean("batchnorm"):
            x = layers.BatchNormalization()(x)
        if hp.Boolean("relu_before_softmax"):
            x = layers.Activation("relu")(x)
        x = layers.Activation("softmax")(x)

        model=tf.keras.Model(inputs, x)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=[tfa.metrics.F1Score(num_classes=5, average='micro', name='f1_score_micro'), 
                               tfa.metrics.F1Score(num_classes=5, average='macro', name='f1_score_macro')])

        return model

    def fit(self, hp, model, *args, **kwargs):
        hp_batch_size = hp.Int('batch_size', min_value=8, max_value=128, step=8)

        return model.fit(
            *args,
            batch_size=hp_batch_size,
            **kwargs,
        )