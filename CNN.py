import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import keras_tuner as kt
from CNN_hypermodel import ConvHyperModel
from LSTM_hypermodel import LSTMHyperModel
from GRU_hypermodel import BiGRUHyperModel
from GRU_hypermodel import GRUHyperModel

def instanciateHypermodel(hypermodel):
    if hypermodel == "cnn":
        return ConvHyperModel()
    elif hypermodel == "lstm":
        return LSTMHyperModel()
    elif hypermodel == "gru":
        return GRUHyperModel()
    elif hypermodel == "bi_gru":
        return BiGRUHyperModel()
    return


class AutoCNN:
    def __init__(self, hypermodel):
        self.tuner = None
        self.model = None
        self.hypermodel = hypermodel

    def fit_and_tune(self, X, y=None, epochs=50, validation_split=0.2, validation_data=None, objective=["val_categorical_accuracy"], factor=3, refit=True, build_only=False, overwrite=True, checkpoint_path=None, **kwargs):        
        objective_list = []

        if "val_categorical_accuracy" in objective:
            objective_list.append(kt.Objective("val_categorical_accuracy", direction="max"))
        if "val_f1_score_macro" in objective:
            objective_list.append(kt.Objective("val_f1_score_macro", direction="max"))
        if "val_loss" in objective:
            objective_list.append("val_loss")
        else:
            print("Objective function is unknown")

        # tuner = kt.BayesianOptimization(instanciateHypermodel(self.hypermodel),
        #                                 objective=objective_list,
        #                                 max_trials=n_trials,
        #                                 overwrite=True,
        #                                 seed=2023,
        #                                 directory="../CNN_finetune",
        #                                 project_name="politics")

        tuner = kt.Hyperband(instanciateHypermodel(self.hypermodel),
                            objective=objective_list,
                            max_epochs=50,
                            factor=factor,
                            hyperband_iterations=1,
                            directory=kwargs['directory'],
                            project_name=kwargs['project_name'],
                            overwrite=overwrite,
                            seed=2023)


        self.tuner = tuner
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
        tuner.search(X, y, validation_split=validation_split, validation_data=validation_data, callbacks=[stop_early])
        
        best_hps = self.get_best_hyperparameters()

        print("Refit with best hyperparameters")
        if not refit:
            return 

        tf_callbacks = []
        if checkpoint_path != None:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
            tf_callbacks.append(cp_callback)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=kwargs['log_dir'])
        tf_callbacks.append(tensorboard_callback)
        tf_callbacks.append(stop_early)

        hypermodel = instanciateHypermodel(self.hypermodel)
        model = hypermodel.build(best_hps)
        if build_only:
            print("Model built, skipping training, now you can use predict() with saved weights")
            self.model = model
            return
        # hypermodel call model.fit() internally, after training, save the model in self.model attribute
        history = hypermodel.fit(best_hps, model, x=X, y=y, epochs=epochs, validation_split=validation_split, validation_data=validation_data, callbacks=tf_callbacks)   
        self.model = model

    def get_best_hyperparameters(self, verbose=1):
        if self.tuner == None:
            print("No tuner was found, please run fit_and_tune() method first")
            return
        
        best_hps=self.tuner.get_best_hyperparameters()[0]
        
        if verbose == 1:
            print("Hyperparameters")
            print("*"*10)
            if self.hypermodel == "cnn":
                print("Number of convolutional layers:", best_hps.get('num_conv_layers'))
                print("Number of filters:", best_hps.get('filters'))
            if self.hypermodel == "bi_gru" or self.hypermodel == "gru" or self.hypermodel == "lstm":  # COMMIT 
                print("RNN units:", best_hps.get('units'))
                
            print("Dropout rate:", best_hps.get('dropout_rate'))
            print("Number of hidden linear layers:", best_hps.get('num_hidden_layers'))

            for i in range(best_hps.get('num_hidden_layers')):
                print(f"Dense_{i+1}:", best_hps.get(f"units_{i}"))

            print("Batch normalization after hidden layer:", best_hps.get('batchnorm_after_hidden_layer'))
            print("Relu before softmax:", best_hps.get('relu_before_softmax'))
            print("Learning rate", best_hps.get('learning_rate'))
            print("*"*10)

        return best_hps

    def fit(self, X, y=None, epochs=20, validation_split=0.0, validation_data=None, checkpoint_path=None):
        best_hps = self.get_best_hyperparameters(0)

        tf_callbacks = []
        if checkpoint_path != None:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
            tf_callbacks.append(cp_callback)

        hypermodel = instanciateHypermodel(self.hypermodel)
        model = hypermodel.build(best_hps)
        model.fit(X, y, epochs=epochs, validation_split=validation_split, validation_data=validation_data, callbacks=tf_callbacks)
        self.model = model
        return self.model
    
    def predict(self, X, y=None):
        print("Predicting with test data")
        eval_result = self.model.evaluate(X, y)
        print("[test loss, test f1 score,[macro]]:", eval_result)
        return 





