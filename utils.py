import numpy as np
import pandas as pd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import dotenv_values

def computemjj_pd(event):
    px1 = event[["pxj1"]].to_numpy()
    py1 = event[["pyj1"]].to_numpy()
    pz1 = event[["pzj1"]].to_numpy()
    pE1 = np.sqrt(px1**2+py1**2+pz1**2+event[["mj1"]].to_numpy()**2)
    
    px2 = event[["pxj2"]].to_numpy()
    py2 = event[["pyj2"]].to_numpy()
    pz2 = event[["pzj2"]].to_numpy()
    pE2 = np.sqrt(px2**2+py2**2+pz2**2+event[["mj2"]].to_numpy()**2)
    
    m2 = (pE1+pE2)**2-(px1+px2)**2-(py1+py2)**2-(pz1+pz2)**2
    return np.array(np.sqrt(m2)).flatten()

def computemjj_txt(event):
    pT1 = np.array([float(event[2*i][0]) for i in range(int(len(event)/2))])
    eta1 = np.array([float(event[2*i][1]) for i in range(int(len(event)/2))])
    phi1 = np.array([float(event[2*i][2]) for i in range(int(len(event)/2))])
    m1 = np.array([float(event[2*i][3]) for i in range(int(len(event)/2))])
    px1 = pT1*np.cos(phi1)
    py1 = pT1*np.sin(phi1)
    pz1 = pT1*np.sinh(eta1)
    pE1 = np.sqrt(px1**2+py1**2+pz1**2+m1**2)
    
    pT2 = np.array([float(event[2*i+1][0]) for i in range(int(len(event)/2))])
    eta2 = np.array([float(event[2*i+1][1]) for i in range(int(len(event)/2))])
    phi2 = np.array([float(event[2*i+1][2]) for i in range(int(len(event)/2))])
    m2 = np.array([float(event[2*i+1][3]) for i in range(int(len(event)/2))])
    px2 = pT2*np.cos(phi2)
    py2 = pT2*np.sin(phi2)
    pz2 = pT2*np.sinh(eta2)
    pE2 = np.sqrt(px2**2+py2**2+pz2**2+m2**2)
    
    m2 = (pE1+pE2)**2-(px1+px2)**2-(py1+py2)**2-(pz1+pz2)**2
    return np.array(np.sqrt(m2)).flatten()

#find the weights that didn't work and the corresponding mass pair
def get_stuck_weights(sigspace, injections, m_initializations, m1, m2, weight_list1, weight_list2, decay):
    # weight_list1 = np.load(f"data/weight_list1_runs_notebook{float(m1)}{float(m2)}_{decay}.npy", allow_pickle = True)
    # weight_list2 = np.load(f"data/weight_list2_runs_notebook{float(m1)}{float(m2)}_{decay}.npy", allow_pickle = True)
    
    missed = {}

    found_one = 0
    found_both = 0
    found_none = 0

    for sigfrac in range(len(sigspace)):

        for injection in range(injections):

            m1_diff = abs(np.array(np.ones(len(weight_list1[sigfrac][injection]))*m1) - np.array(weight_list1[sigfrac][injection]))
            m2_diff = abs(np.array(np.ones(len(weight_list2[sigfrac][injection]))*m2) - np.array(weight_list2[sigfrac][injection]))

            count = 0
            for diff1, diff2 in zip(m1_diff, m2_diff):
                case1 = diff1 > 0.2 and diff2 > 0.2
                case2 = diff1 < 0.2 and diff2 < 0.2
                case3 = diff1 > 0.2 and diff2 < 0.2 or diff1 < 0.2 and diff2 > 0.2

                if case1:
                    found_none+=1
                    missed[(sigspace[sigfrac], injection)] = (weight_list1[sigfrac][injection][count], weight_list2[sigfrac][injection][count])
                    print(np.array(weight_list1[sigfrac][injection][count]), np.array(weight_list2[sigfrac][injection][count]))
                elif case2:
                    found_both+=1
                elif case3:
                    found_one+=1
                    missed[(sigspace[sigfrac], injection)] = (weight_list1[sigfrac][injection][count], weight_list2[sigfrac][injection][count])
                    print(np.array(weight_list1[sigfrac][injection][count]), np.array(weight_list2[sigfrac][injection][count]))
                count+=1
                
    print(f"Found Both: {found_both}")
    print(f"Found None: {found_none}")
    print(f"Found One: {found_one}")
            
    return missed

def pred_accuracy(y_test, scores):
    background_count, signal_count = 0, 0

    predictions_list = []
    for pred in scores:
        #arbitrary cutoff of 0.5
        if float(pred) > 0.5:
            predictions_list.append(int(1))
            signal_count+=1
        elif float(pred) < 0.5:
            predictions_list.append(int(0))
            background_count+=1
            
    accuracy = np.mean(predictions_list == y_test)
    return accuracy


def send_slack_message(message=""):
    env_vars = dotenv_values(".env")

    SLACK_API_TOKEN = env_vars["SLACK_API_TOKEN"]
    SLACK_CHANNEL_ID = env_vars["SLACK_CHANNEL_ID"]
    
    client = WebClient(token=SLACK_API_TOKEN)
    channel_id = SLACK_CHANNEL_ID

    message = message

    try:
        response = client.chat_postMessage(channel=channel_id, text=message)
        print("Message sent successfully! :", response["ts"])
    except SlackApiError as e:
        print("Error sending message:", e.response["error"])
        
def send_slack_plot(image_path, message="Here is a plot"):
    env_vars = dotenv_values(".env")

    SLACK_API_TOKEN = env_vars["SLACK_API_TOKEN"]
    SLACK_CHANNEL_ID = env_vars["SLACK_CHANNEL_ID"]
    
    client = WebClient(token=SLACK_API_TOKEN)
    try:
        response = client.files_upload(
            channels=SLACK_CHANNEL_ID,
            file=image_path,
            initial_comment=message
        )
        print("Plot sent successfully:", response["ts"])
    except SlackApiError as e:
        print("Error sending plot:", e.response["error"])