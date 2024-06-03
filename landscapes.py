from common import *
import os
#from models import compileSemiWeakly #recursive issue?
from data import load_data
import sys
import time
import argparse

#custom moduiles
from utils import send_slack_message, send_slack_plot
from plotting import create_3D_loss_manifold, loss_landscape_nofit
from models import compileSemiWeakly, compileSemiWeakly3Prong

#load all necessary data/files
noise_dims = 10
x = load_data("/pscratch/sd/g/gupsingh/x_array_fixed_EXTRAQCD.pkl", noise_dims = noise_dims)
#model_name = "decent-sun-87qq"
model_name = "robust-river-109qq10"
p_dir = "/pscratch/sd/g/gupsingh/"
model_path = p_dir + model_name
model_qq = tf.keras.models.load_model(model_path)

noise = True
start_time = time.time()
decay = "qq"
def eval_loss_landscape(feature_dims, parameters, m1, m2, step, decay):
    qq = decay
    noise = True
    alpha = 0.5
    model_name = "robust-river-109qq10"
    
    # for l in model_qqq.layers:
    #     l._name = f"{l.name}_model_qqq"
    start_time = time.time()
    #check if loss dictionary exists, if it does load it, if not create empty one
    dir_path = os.getcwd()
    
    #if using extra QCD background
    if len(x[0,0,qq, noise]) > 121352:
        extra = True
        
    extra_str = "_extra" if extra else ""
    file_name = f"data/landscapes/z_{feature_dims}_{parameters}_{m1}{m2}_{step}_{decay}{extra_str}_{model_name}_{noise_dims}.npy"
    file_path = os.path.join(dir_path, file_name)
    
    if os.path.exists(file_path):
        z = np.load(file_name, allow_pickle = True).item()
    else:
        print("Dictionary doesn't exist, creating one...")
        z = {}
    
    losses_list = []
    epsilon = 1e-4
    sigspace = np.logspace(-3.5, -1.3, 10)
    
    start = 0.5
    end = 6
    step = step

    weight_list = np.arange(start, end + step, step)
    
    for sigfrac in sigspace:
        print("Signal Fraction: ", sigfrac)
        count = 0
        for w1 in weight_list:
            for w2 in weight_list:
                if count % 100 == 0:
                    print(f"reached {w1} {w2}")
                count+=1
                #print(w1, w2)
                
                m1 = m1
                m2 = m2
                
                #if computed this mass pair, break
                key = (sigfrac, m1, m2, decay)
                if key in z:
                    break
                
                test_background = int(1/2 *len(x[0,0, qq, noise])+1)
                train_background = int(1/4 * len(x[0,0,qq, noise]))
                train_data = int(1/4 * len(x[0,0,qq, noise]))
                train_reference = int(1/4 * len(x[0,0,qq, noise]))
                #signal
                test_signal_length = int(1/2*len(x[m1,m2,qq, noise]))

                if decay == "qq":
                    model_qq = tf.keras.models.load_model(f"/pscratch/sd/g/gupsingh/{model_name}")
                    model_semiweak = compileSemiWeakly(sigfrac, model_qq, feature_dims, parameters, m1, m2, w1, w2)
                    
                    #randomize signal events
                    random_test_signal_length = random.randint(0, test_signal_length - 1)
                    N = int(1/4 * (len(x[0,0,qq, noise])))
                    signal = x[m1, m2,qq, noise][random_test_signal_length:random_test_signal_length + int(sigfrac*N)]
                    
                    x_data_ = np.concatenate([x[0,0,qq, noise][test_background:],signal])
                    y_data_ = np.concatenate([np.zeros(train_reference),np.ones(train_data),np.ones(len(signal))])
                    
                    X_train_, X_val_, Y_train_, Y_val_ = train_test_split(x_data_, y_data_, test_size=0.5, random_state = 42)


                if decay == "qqq":
                    model_semiweak = compileSemiWeakly3Prong(model_qq, model_qqq, feature_dims, parameters, m1, m2, w1, w2)
                    
                    print(f"Number of 3 Pronged Evengs: {sigfrac * N *alpha}")
                    print(f"Number of 2 Pronged Evengs: {sigfrac * N *(1-alpha)}")

                    #mix both samples
                    signal_mixed = np.concatenate([x[m1, m2, decay, noise][int(random_test_signal_length):int(random_test_signal_length) + int(sigfrac * N * alpha)], x[m1, m2, qq, noise][int(random_test_signal_length):int(random_test_signal_length) + int(sigfrac * N * (1-alpha))]])

                    x_data_mixed = np.concatenate([x[0,0, qq, noise][test_background:],signal_mixed])
                    y_data_mixed = np.concatenate([np.zeros(train_reference),np.ones(train_data),np.ones(len(signal_mixed))])
                    
                    X_train_, X_val_, Y_train_, Y_val_ = train_test_split(x_data_mixed, y_data_mixed, test_size=0.5, random_state = 42)

                with tf.device('/GPU:0'):
                    loss = model_semiweak.evaluate(X_val_, Y_val_, verbose = 0)
                losses_list.append(loss)
                
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)
        print(f"Time taken: {elapsed_time} seconds")
        if key in z:
            print(f"Loss Landscape for m1 = {m1} and m2 = {m2} already exists for {sigfrac} signal fraction and decay {decay}")
        else:
            z[sigfrac, m1, m2, decay] = losses_list
            losses_list = []
            np.save(file_name, z)
    end_time_total = time.time()

    elapsed_time_total = round(end_time_total - start_time, 3)
    print(f"Total elapsed time: {elapsed_time_total} seconds")

def eval_AUC_landscape(feature_dims, parameters, m1, m2, step, decay):
    qq = decay
    noise = False
    
    start_time = time.time()
    
    #if using extra QCD background
    if len(x[0,0,qq, noise]) > 121352:
        extra = True
        
    extra_str = "_extra" if extra else ""

    #check if AUC dictionary exists, if it does load it, if not create empty one
    dir_path = os.getcwd()
    file_name = f"data/landscapes/a_{feature_dims}_{parameters}_{m1}{m2}_{step}_{decay}{extra_str}.npy"
    file_path = os.path.join(dir_path, file_name)
    
    if os.path.exists(file_path):
        a = np.load(file_name, allow_pickle = True).item()
    else:
        print("Dictionary doesn't exist, creating one...")
        a = {}
    
    AUC_list = []

    epsilon = 1e-4
    sigspace = np.logspace(-3,-1,10)

    start = 0.5
    end = 6
    step = step

    weight_list = np.arange(start, end + step, step)
    
    for sigfrac in sigspace:
        print("Signal Fraction: ", sigfrac)
        count = 0
        for w1 in weight_list:
            for w2 in weight_list:
                if count % 100 == 0:
                    print(f"reached {w1} {w2}")
                count+=1
                #print(w1, w2)

                if decay == "qq":
                    for l in model.layers:
                        l.trainable=False
                    model_semiweak = compileSemiWeakly(model, feature_dims, parameters, m1, m2, w1, w2)
                    
                if decay == "qqq":
                    for l in model_qqq.layers:
                        l.trainable=False
                    model_semiweak = compileSemiWeakly3Prong(model_qq, model_qqq, feature_dims, parameters, m1, m2, w1, w2)

                m1 = m1
                m2 = m2
                
                #if computed this mass pair, break
                
                key = (sigfrac,m1,m2, decay)
                if key in a:
                    break

                test_background = int(1/2 *len(x[0,0, qq, noise])+1)
                train_background = int(1/4 * len(x[0,0,qq, noise]))
                train_data = int(1/4 * len(x[0,0,qq, noise]))
                train_reference = int(1/4 * len(x[0,0,qq, noise]))
                #signal
                test_signal_length = int(1/2*len(x[m1,m2,qq, noise]))

                #randomize signal events
                random_test_signal_length = random.randint(0, test_signal_length - 1)
                N = int(1/4 * (len(x[0,0,qq, noise])))
                signal = x[m1, m2,qq, noise][random_test_signal_length:random_test_signal_length + int(sigfrac*N)]

                x_data_ = np.concatenate([x[0,0,qq, noise][test_background:],signal])
                y_data_ = np.concatenate([np.zeros(train_reference),np.ones(train_data),np.ones(len(signal))])

                X_train_, X_val_, Y_train_, Y_val_ = train_test_split(x_data_, y_data_, test_size=0.5, random_state = 42)
                
                with tf.device('/GPU:0'):
                    y_val_pred = model_semiweak.predict(X_val_)
                    auc = roc_auc_score(Y_val_, y_val_pred)
                AUC_list.append(auc)
                
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)
        print(f"Time taken: {elapsed_time} seconds")
        if key in a:
            print(f"AUC Landscape for m1 = {m1} and m2 = {m2} already exists for {sigfrac} signal fraction and decay {decay}")
        else:
            a[sigfrac, m1, m2, decay] = AUC_list
            AUC_list = []
            np.save(file_name, a)
    end_time_total = time.time()

    elapsed_time_total = round(end_time_total - start_time, 3)
    print(f"Total elapsed time: {elapsed_time_total} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dims", type=int, help="Number of feature dimensions")
    parser.add_argument("--parameters", type=int, help="Number of parameters")
    parser.add_argument("--m1", type=int, help="Value for m1")
    parser.add_argument("--m2", type=int, help="Value for m2")
    parser.add_argument("--step", type=float, help="Resolution of Weight Space")
    parser.add_argument("--case", type=str, help="Which to evaluate? Loss or AUC landscape?")
    args = parser.parse_args()
    
    message = (
    "```"
    f"---------- Creating {args.case} Landscape With the Following Parameters ----------\n"
    f"Feature dimensions: {args.feature_dims}\n"
    f"Parameters: {args.parameters}\n"
    f"m1: {args.m1}\n"
    f"m2: {args.m2}\n"
    f"model: {model_path}\n"
    "----------------------------------------------------------------------\n"
    "```"
)
    
    send_slack_message(message)
    print(message)
    
    feature_dims = args.feature_dims
    parameters = args.parameters
    m1 = args.m1
    m2 = args.m2
    step = args.step
    qq = "qq"
    
    if args.case == "Loss":
        eval_loss_landscape(args.feature_dims, args.parameters, args.m1, args.m2, args.step, decay)
        #if using extra QCD background
        if len(x[0,0,qq, noise]) > 121352:
            extra = True

        extra_str = "_extra" if extra else ""
        
        filename = f"data/landscapes/z_{feature_dims}_{parameters}_{m1}{m2}_{step}_{decay}{extra_str}_{model_name}.npy"
        z = np.load(filename, allow_pickle = True).item()
        sigfrac = 0.1
        elv = 60
        azim = 20

        create_3D_loss_manifold(sigfrac, m1, m2, z, step, elv, azim, save = True)
        loss_landscape_nofit(sigfrac, m1, m2, z, step, save = True)
        img_paths = [f"plots/landscapes/l_{float(m1)}{float(m2)}_{decay}{extra_str}.png", f"plots/manifolds/lm_{float(m1)}{float(m2)}_{decay}{extra_str}.png"]
        send_slack_plot(img_paths)
        send_slack_message("Done!")
    
    elif args.case == "AUC":
        eval_AUC_landscape(sigspace, args.feature_dims, args.parameters, args.m1, args.m2, args.step, decay)
        a = np.load(filename, allow_pickle = True).item()
        AUC_landscape_nofit(sigfrac, m1, m2, a, step=0.25, save = True)
        img_path = f"plots/landscapes/AUCL_{float(m1)}{float(m2)}.png"
        send_slack_plot(img_path)
        send_slack_message("Done!")
    elif args.case == "both":
        z = np.load(filename, allow_pickle = True).item()
        a = np.load(filename, allow_pickle = True).item()
        eval_loss_landscape(model, args.feature_dims, args.parameters, args.m1, args.m2, args.step, decay)
        eval_AUC_landscape(sigspace, model, args.feature_dims, args.parameters, args.m1, args.m2, args.step, decay)
        plot_landscapes(0.1, args.m1, args.m2, z, a, args.step, save = True)
        img_path = f"plots/bothlandscape{float(m1)}{float(m2)}.png"
        send_slack_plot(img_paths)
        send_slack_message("Done!")
    else:
        print("Case Error: Possible Cassses are Loss, AUC, or Both")