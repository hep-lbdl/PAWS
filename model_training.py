import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
from wandb.keras import WandbCallback
import wandb
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os
from utils import send_slack_message, send_slack_plot
import pickle

#training data
decay = "qqq"
extra = True
            
extra_str = "_extra" if extra else ""

x_data_qq = np.load(f"/pscratch/sd/g/gupsingh/x_parametrized_data_{decay}{extra_str}.npy")
y_data_qq = np.load(f"/pscratch/sd/g/gupsingh/y_parametrized_data_{decay}{extra_str}.npy")
# x_data_qq = np.load("/pscratch/sd/g/gupsingh/x_parametrized_data_qq_extra_23score.npy")
# y_data_qq = np.load("/pscratch/sd/g/gupsingh/y_parametrized_data_qq_extra_23score.npy")

noise_dims = 0
if noise_dims == 0:
    noise = False
else:
    noise = True
    
# noise_dims_remove = [i for i in range(np.shape(x_data_qq)[1] - (3 + noise_dims), 5, -1)]
# x_data_qq = np.delete(x_data_qq, noise_dims_remove, axis = 1)

X_train_qq, X_val_qq, Y_train_qq, Y_val_qq = train_test_split(x_data_qq, y_data_qq, test_size=0.5, random_state = 24)

pscratch_dir = "/pscratch/sd/g/gupsingh/"
os.environ["WANDB_DIR"] = pscratch_dir

config = {
    "layer_1_neurons": 256,
    "layer_2_neurons": 128,
    "layer_3_neurons": 64,
    "output_neurons": 1,
    "activation": "relu",
    "output_activation": "sigmoid",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "loss": "binary_crossentropy",
    "epochs": 5000,
    "batch_size": 1024
}

wandb.init(project="SemiWeakly",
           group="Parametrized",
           entity='gup-singh',
           mode = 'online',
           config=config)

config = wandb.config
run_name = wandb.run.name
key = f"{run_name}{decay}{noise_dims}{extra_str}"
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

def train_parametrized(X_train, Y_train, X_val, Y_val, config, return_history=False):
    print(np.shape(X_train))
    model_parametrized = Sequential()
    model_parametrized.add(Dense(config["layer_1_neurons"], input_dim=np.shape(X_train_qq)[1], activation=config["activation"]))
    model_parametrized.add(Dense(config["layer_2_neurons"], activation=config["activation"]))
    model_parametrized.add(Dense(config["layer_3_neurons"], activation=config["activation"]))
    model_parametrized.add(Dense(config.output_neurons, activation=config["output_activation"]))
    model_parametrized.compile(loss=config["loss"], optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), metrics=['accuracy'])

    with tf.device('/GPU:0'):
        history_parametrized = model_parametrized.fit(X_train, Y_train, epochs=config.epochs, validation_data=(X_val, Y_val), batch_size=config.batch_size, callbacks=[es, WandbCallback()])

    if return_history:
        with open("history_parametrized.pkl", "wb") as f:
            pickle.dump(history_parametrized, f)
        return model_parametrized, history_parametrized
    else:
        return model_parametrized

# send_slack_message(f"Training: " + key)
model_parametrized, history_parametrized = train_parametrized(X_train_qq, Y_train_qq, X_val_qq, Y_val_qq, config, return_history = True)
model_parametrized.save(pscratch_dir + key)

wandb.finish()
send_slack_message(f"Done Training: " + key)

#Diagonistic Plot
plt.figure()
epochs = [x for x in range(len(history_parametrized.history["loss"]))]
plt.plot(epochs, history_parametrized.history["loss"])
plt.plot(epochs, history_parametrized.history["val_loss"])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
img_path = f"plots/parametrized_trainingloss" + key + ".png"
plt.savefig(img_path)
plt.legend()
plt.show()
send_slack_plot(img_path)