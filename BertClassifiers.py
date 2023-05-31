import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFBertTokenizer, TFBertModel

class BertClassifier:
    def build(self, bert_tokenizer, bert_model):
        text_in = tf.keras.Input(shape=(), dtype=tf.string, name='text')

        encoder_inputs = bert_tokenizer(text_in, padding="longest", truncation=True)
        outputs = bert_model(encoder_inputs)
        x = outputs['pooler_output']
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(7, activation=None, name='classifier')(x)

        model = tf.keras.Model(inputs={'text' : text_in}, outputs=x)
        return model


class BertClassifierForTwoPhrases:
    def build(self, bert_tokenizer, bert_model):
        prev_text_in  = tf.keras.Input(shape=(), dtype=tf.string, name='prev_text')
        text_in       = tf.keras.Input(shape=(), dtype=tf.string, name='text')

        encoder_inputs = bert_tokenizer(prev_text_in, text_in, padding="longest", truncation=True)
        outputs = bert_model(encoder_inputs)
        x = outputs['pooler_output']
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(7, activation=None, name='classifier')(x)

        model = tf.keras.Model(inputs={'prev_text' : prev_text_in, 'text' : text_in}, outputs=x)
        return model
    
# class BertClassifierForTwoPhrasesParty:
#     def build(self, bert_tokenizer, bert_model):
#         prev_text_in  = tf.keras.Input(shape=(), dtype=tf.string, name='prev_text')
#         text_in       = tf.keras.Input(shape=(), dtype=tf.string, name='text')
#         party_in      = tf.keras.Input(shape=(78), name='partyname')

#         encoder_inputs = bert_tokenizer(prev_text_in, text_in, padding="longest", truncation=True)
#         outputs = bert_model(encoder_inputs)
#         x = outputs['pooler_output']
#         # concatenate bert output CLS token with encoded party (both are one-dimensional vectors)
#         x = layers.Concatenate()([x, party_in])
#         x = layers.Dropout(0.5)(x)
#         x = layers.Dense(7, activation=None, name='classifier')(x)

#         model = tf.keras.Model(inputs={'prev_text' : prev_text_in, 'text' : text_in, 'partyname' : party_in}, outputs=x)
#         return model

