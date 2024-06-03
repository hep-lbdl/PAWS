from common import *
import sys
from tensorflow.keras.constraints import Constraint
from data import load_data

qq = "qq"
qqq = "qqq"

class WeightConstraint(Constraint):
    def __init__(self, min_value=0.5, max_value=6):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, weights):
        return tf.clip_by_value(weights, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}
    
class MinMaxRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self):
        self.min_val = 0.5
        self.max_val = 6
        self.l = 1

    def __call__(self, x):
        x_under = tf.cast(tf.less(x, self.min_val), dtype=tf.float32)
        x_over  = tf.cast(tf.greater(x, self.max_val), dtype=tf.float32)
        under_penalty = (tf.exp(self.min_val - x) - 1)
        over_penalty = (tf.exp(x - self.max_val) - 1)
        penalty = self.l * tf.math.reduce_sum(((x_under * under_penalty) + (x_over * over_penalty)))
        return penalty

    def get_config(self):
        return {'l': float(self.l), 'min_val': float(self.min_val), 'max_val': float(self.max_val)}

def simpleModel(weight):
    input_layer = tf.keras.Input(shape=(1,))
    simple_model = Dense(1,use_bias = False,activation="relu",
                         kernel_initializer=tf.keras.initializers.Constant(weight), kernel_constraint=WeightConstraint())(input_layer)
    model = Model(inputs=input_layer, outputs=simple_model)
    return model


#CWOLA comparison
def compileCWOLA(feature_dims, m1, m2):
    model_cwola = Sequential()
    model_cwola.add(Dense(256, input_dim=feature_dims, activation='relu'))
    model_cwola.add(Dense(128, activation='relu'))
    model_cwola.add(Dense(64, activation='relu'))
    model_cwola.add(Dense(1, activation='sigmoid'))
    model_cwola.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001))
    return model_cwola

epsilon = 1e-4
#SemiWeak Model
def compileSemiWeakly(sigfrac, model, feature_dims, params, m1, m2, w1, w2):
    
    for l in model.layers:
        l.trainable=False
        
    inputs_hold = tf.keras.Input(shape=(1,))
    simple_model = Dense(1,use_bias = False,activation='relu',kernel_initializer=tf.keras.initializers.Constant(w1), kernel_regularizer=MinMaxRegularizer())(inputs_hold)
    model3 = Model(inputs = inputs_hold, outputs = simple_model)

    inputs_hold2 = tf.keras.Input(shape=(1,))
    simple_model2 = Dense(1,use_bias = False,activation='relu',kernel_initializer=tf.keras.initializers.Constant(w2), kernel_regularizer=MinMaxRegularizer())(inputs_hold2)
    model32 = Model(inputs = inputs_hold2, outputs = simple_model2)

    inputs_hold3 = tf.keras.Input(shape=(1,))
    simple_model3 = tf.exp(Dense(1,use_bias = False,activation='linear',kernel_initializer=tf.keras.initializers.Constant(-4))(inputs_hold3))
    model33 = Model(inputs = inputs_hold3, outputs = simple_model3)

    inputs = tf.keras.Input(shape=(feature_dims,))
    inputs2 = tf.keras.layers.concatenate([inputs,model3(tf.ones_like(inputs)[:,0]),model32(tf.ones_like(inputs)[:,0])])
    epsilon = 1e-4
    #physics prior
    hidden_layer_1 = model(inputs2)
    LLR = hidden_layer_1 / (1.-hidden_layer_1 + epsilon)

    if params == 2:
        LLR_xs = 1.+sigfrac*LLR - sigfrac
    elif params == 3:
        LLR_xs = 1. + model33(tf.ones_like(inputs)[:,0])*LLR - model33(tf.ones_like(inputs)[:,0])
    else:
        print("Choose 2 or 3 parameters")
    ws = LLR_xs / (1.+LLR_xs)

    SemiWeakModel = Model(inputs = inputs, outputs = ws)
    SemiWeakModel.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.05, clipvalue=0.0001, clipnorm=0.0001))
    return SemiWeakModel

def compileSemiWeakly3Prong(sigfrac, model_qq, model_qqq, feature_dims, parameters, m1, m2, w1, w2):
    #freeze both two pronged and three pronged prior models
    for l in model_qq.layers:
        l.trainable=False

    for l in model_qqq.layers:
        l.trainable=False

    inputs_hold = tf.keras.Input(shape=(1,))
    simple_model = Dense(1,use_bias = False,activation='relu',kernel_initializer=tf.keras.initializers.Constant(w1), kernel_regularizer=MinMaxRegularizer())(inputs_hold)
    model3 = Model(inputs = inputs_hold, outputs = simple_model)

    inputs_hold2 = tf.keras.Input(shape=(1,))
    simple_model2 = Dense(1,use_bias = False,activation='relu',kernel_initializer=tf.keras.initializers.Constant(w2), kernel_regularizer=MinMaxRegularizer())(inputs_hold2)
    model32 = Model(inputs = inputs_hold2, outputs = simple_model2)

    inputs_hold3 = tf.keras.Input(shape=(1,))
    simple_model3 = tf.exp(Dense(1,use_bias = False,activation='linear',kernel_initializer=tf.keras.initializers.Constant(-4))(inputs_hold3))
    model33 = Model(inputs = inputs_hold3, outputs = simple_model3)

    inputs_hold4 = tf.keras.Input(shape=(1,))
    simple_model4 = Dense(1,use_bias = False,activation='sigmoid',kernel_initializer=tf.keras.initializers.Constant(.37))(inputs_hold4)
    model34 = Model(inputs = inputs_hold4, outputs = simple_model4)

    inputs = tf.keras.Input(shape=(feature_dims,))
    inputs2 = tf.keras.layers.concatenate([inputs,model3(tf.ones_like(inputs)[:,0]),model32(tf.ones_like(inputs)[:,0])])
    hidden_layer_1 = model_qq(inputs2)
    hidden_layer_13 = model_qqq(inputs2)
    LLR2 = hidden_layer_1 / (1.-hidden_layer_1+0.0001)
    LLR3 = hidden_layer_13 / (1.-hidden_layer_13+0.0001)
    
    if parameters == 2:
        LLR_xs = 1.+ sigfrac*LLR3 - sigfrac
    if parameters == 4:
        LLR_xs_fixed = 1 + model33(tf.ones_like(inputs)[:,0])*model34(tf.ones_like(inputs)[:,0]) * LLR3 + (1-model34(tf.ones_like(inputs)[:,0]))*LLR2*model33(tf.ones_like(inputs)[:,0]) - model33(tf.ones_like(inputs)[:,0])
    ws = LLR_xs_fixed / (1.+LLR_xs_fixed+0.0001)
    SemiWeak3Prong = Model(inputs = inputs, outputs = ws)
    SemiWeak3Prong.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.05, clipvalue=0.0001, clipnorm=0.0001))
    return SemiWeak3Prong