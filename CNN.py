import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import keras_tuner as kt
from CNN_hypermodel import ConvHyperModel, SimpleConvHyperModel

def instanciateHypermodel(hypermodel):
    if hypermodel == "cnn":
        return ConvHyperModel()
    elif hypermodel == "simple_cnn":
        return SimpleConvHyperModel()
    return


class AutoCNN:
    def __init__(self, hypermodel):
        self.tuner = None
        self.model = None
        self.hypermodel = hypermodel

    def fit_and_tune(self, X, y, epochs=50, validation_split=0.2, objective=["val_f1_score_micro"], n_trials=100, refit=True, **kwargs):        
        objective_list = []

        if "val_f1_score_micro" in objective:
            objective_list.append(kt.Objective("val_f1_score_micro", direction="max"))
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
                            hyperband_iterations=3,
                            directory=kwargs['directory'],
                            project_name=kwargs['project_name'],
                            overwrite=True,
                            seed=2023)


        self.tuner = tuner
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        tuner.search(X, y, epochs=epochs, validation_split=validation_split, callbacks=[stop_early])
        
        best_hps = self.get_best_hyperparamenters()

        print("Refit with best hyperparameters")
        if not refit:
            return 

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="../logs")

        hypermodel = instanciateHypermodel(self.hypermodel)
        model = hypermodel.build(best_hps)
        history = hypermodel.fit(best_hps, model, x=X, y=y, epochs=epochs, validation_split=validation_split, callbacks=[tensorboard_callback, stop_early])   

        # val_per_epoch = history.history[objective]
        # if objective == "val_loss":
        #     best_epoch = val_per_epoch.index(min(val_per_epoch)) + 1
        # best_epoch = val_per_epoch.index(max(val_per_epoch)) + 1
        # print("*"*10)
        # print('Best epoch: %d' % (best_epoch,))

    def get_best_hyperparamenters(self, verbose=1):
        if self.tuner == None:
            print("No tuner was found, please run fit_and_tune() method first")
            return
        
        best_hps=self.tuner.get_best_hyperparameters()[0]
        
        if verbose == 1:
            print("Hyperparameters")
            print("*"*10)
            if self.hypermodel == "cnn":
                print("Number of convolutional layers:", best_hps.get('num_conv_layers'))
            if self.hypermodel == "simple_cnn":
                print("Kernel size:", best_hps.get('kernel_size'))
            print("Number of filters:", best_hps.get('filters'))
            print("Dropout:", best_hps.get('dropout'))
            print("Number of linear layers:", best_hps.get('num_layers'))
            print("Dropout:", best_hps.get('dropout_active"'))

            for i in range(best_hps.get('num_layers')):
                print(f"Dense_{i+1}:", best_hps.get(f"units_{i}"))

            print("Activation linear layer:", best_hps.get('activation_linear_layer'))
            print("Batch normalization:", best_hps.get('batchnorm'))
            print("Relu before softmax:", best_hps.get('relu_before_softmax'))
            print("Batch size:", best_hps.get('batch_size'))
            print("Learning rate", best_hps.get('learning_rate'))
            print("*"*10)

        return best_hps

    def fit(self, X, y, epochs, validation_split=0.0):
        best_hps = self.get_best_hyperparamenters(0)

        hypermodel = instanciateHypermodel(self.hypermodel)
        model = hypermodel.build(best_hps)
        model.fit(X, y, batch_size=best_hps.get('batch_size'), epochs=epochs, validation_split=validation_split)
        #model = hypermodel.fit(best_hps, model, x=X, y=y, epochs=epochs, validation_split=validation_split)
        self.model = model
        return model
    
    def predict(self, X_test, y_test):
        best_hps = self.get_best_hyperparamenters(0)    

        print("Predicting with test data")
        eval_result = self.model.evaluate(X_test, y_test, batch_size=best_hps.get('batch_size'))
        print("[test loss, test f1 score,[macro]]:", eval_result)
        return 





