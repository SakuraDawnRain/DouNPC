import os 
import argparse
import json

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from model_npc import FarmerLstmModelNPC

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Advice Train')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_ADP/landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='baselines/sl/landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='baselines/sl/landlord_down.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

#     data_list = glob.glob(os.path.join("data_adviser", "*.json"))
#     print(data_list)
    path = "data_adviser\douzero_ADP_sl.json"
    with open(path) as f:
        data = json.load(f)

    data_LU = []
    data_LD = []
    for episode in data:
        record_list = episode["record"]
        label = 1 if episode["winner"] == "farmer" else 0
        # print(label)
        for record in record_list:
            if record["pos"] == "landlord_up":
                data_LU.append((record["advicer_data"], label))
            elif record["pos"] == "landlord_down":
                data_LD.append((record["advicer_data"], label))

    print(len(data_LU), len(data_LD))

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model_Advice_LU = FarmerLstmModelNPC()

    # Load pretraining
    model_Advice_LU.load_state_dict(torch.load("baselines\sl\landlord_up.ckpt", map_location=torch.device(device)))
    
    # LossFuc, Optimizer, Scheduler
    lr = 0.001
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_Advice_LU.parameters(), lr=0.001)
#     scheduler = 


    # Training
    model_Advice_LU.train()
#     z_batch, x_batch = data_LU[0][0]
#     z_batch, x_batch = torch.tensor(z_batch), torch.tensor(x_batch)
# #     print(z_batch, x_batch)
#     pred = model_Advice_LU(z_batch, x_batch)
#     print(pred)

    n_epochs = 1
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        for ((z_batch, x_batch), label) in tqdm(data_LU):
        #     print(label)
            optimizer.zero_grad() 
            z_batch, x_batch = torch.tensor(z_batch).to(device), torch.tensor(x_batch).to(device)
            pred = model_Advice_LU(z_batch, x_batch)
            true = torch.FloatTensor([[label]]).to(device)
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()