import os 
import argparse
import json

import torch
import torch.nn as nn
from tqdm.auto import tqdm

import numpy as np
import random
seed = 42
random.seed(seed)
np.random.seed(seed)

from model_npc import FarmerLstmModelNPC

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Advice Train')
    parser.add_argument('--landlord', type=str,
            default='baselines\douzero_WP\landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='baselines\douzero_WP\landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='baselines\douzero_WP\landlord_down.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

#     data_list = glob.glob(os.path.join("data_adviser", "*.json"))
#     print(data_list)
    path = "data_adviser\douzero_WP_douzero_WP.json"
    with open(path) as f:
        data = json.load(f)

    data_LU = []
    data_LD = []
    lu = [0, 0]
    ld = [0, 0]
    for episode in data:
        record_list = episode["record"]
        label = 1 if episode["winner"] == "farmer" else 0

        # print(label)
        for record in record_list:
            if record["pos"] == "landlord_up":
                data_LU.append((record["advicer_data"], label))
                lu[label] += 1
            elif record["pos"] == "landlord_down":
                data_LD.append((record["advicer_data"], label))
                ld[label] += 1

    # print(len(data_LU), len(data_LD))
    print(lu, ld)

    random.shuffle(data_LU)
    random.shuffle(data_LD)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model_Advice_LU = FarmerLstmModelNPC()
    model_Advice_LD = FarmerLstmModelNPC()

    # Load pretraining
    model_Advice_LU.load_state_dict(torch.load("baselines\douzero_WP\landlord_up.ckpt", map_location=torch.device(device)))
    model_Advice_LD.load_state_dict(torch.load("baselines\douzero_WP\landlord_down.ckpt", map_location=torch.device(device)))
    

    lr = 0.001
    n_epochs = 10
    criterion = nn.BCELoss()

#     scheduler = 


    # Training
#     z_batch, x_batch = data_LU[0][0]
#     z_batch, x_batch = torch.tensor(z_batch), torch.tensor(x_batch)
# #     print(z_batch, x_batch)
#     pred = model_Advice_LU(z_batch, x_batch)
#     print(pred)
    
    optimizer = torch.optim.Adam(model_Advice_LU.parameters(), lr=0.001)
    model_Advice_LU.train()
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        total_loss = 0
        for ((z_batch, x_batch), label) in tqdm(data_LU):
        #     print(label)
            optimizer.zero_grad() 
            z_batch, x_batch = torch.tensor(z_batch).to(device), torch.tensor(x_batch).to(device)
            pred = model_Advice_LU(z_batch, x_batch)
            true = torch.FloatTensor([[label]]).to(device)
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        print("total loss:", total_loss)
    
    torch.save(model_Advice_LU.state_dict(), "advicer_LU.ckpt")


    optimizer = torch.optim.Adam(model_Advice_LD.parameters(), lr=0.001)
    model_Advice_LD.train()
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        total_loss = 0
        for ((z_batch, x_batch), label) in tqdm(data_LD):
        #     print(label)
            optimizer.zero_grad() 
            z_batch, x_batch = torch.tensor(z_batch).to(device), torch.tensor(x_batch).to(device)
            pred = model_Advice_LD(z_batch, x_batch)
            true = torch.FloatTensor([[label]]).to(device)
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        print("total loss:", total_loss)
    
    torch.save(model_Advice_LD.state_dict(), "advicer_LD.ckpt")