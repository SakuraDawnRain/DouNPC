import torch
import numpy as np

from douzero.env.env_npc import get_obs, get_obs_npc

def _load_model(position, model_path):
    from douzero.dmc.models import model_dict
    model = model_dict[position]()
    model_state_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

class DeepAgent:

    def __init__(self, position, model_path):
        from model_npc import FarmerLstmModelNPC
        self.model = _load_model(position, model_path)
        self.model_advice = FarmerLstmModelNPC()
        self.model_decide = FarmerLstmModelNPC()
        if position == "landlord_up":
            self.model_advice.load_state_dict(torch.load("advicer_LU.ckpt", map_location='cpu'))
            self.model_decide.load_state_dict(torch.load("decider_LU.ckpt", map_location='cpu'))
        elif position == "landlord_down":
            self.model_advice.load_state_dict(torch.load("advicer_LD.ckpt", map_location='cpu'))
            self.model_decide.load_state_dict(torch.load("decider_LD.ckpt", map_location='cpu'))
        self.position = position
        

    def decide(self, infoset, best_action):
        obs = get_obs_npc(infoset, action=best_action)
        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()

        y_pred = self.model_decide.forward(z_batch, x_batch)
        y_pred = y_pred.detach().cpu().numpy()

        self.decide_list.append(y_pred)

        return y_pred>0.5
    

    def advice(self, infoset):
        # advice = 0
        # print(infoset.player_position)
        # print(infoset.player_hand_cards)
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0], ()

        obs = get_obs(infoset) 

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        # print("act:", infoset.player_position)
        # print(y_pred)

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        advice = 1
        
        advicer_input = ()

        if best_action == []:
            advice = 1
        else:
            obs = get_obs_npc(infoset, action=best_action)
            z_batch = torch.from_numpy(obs['z_batch']).float()
            x_batch = torch.from_numpy(obs['x_batch']).float()
            # print(obs)
            # if torch.cuda.is_available():
            #     z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
            y_pred = 0
            if self.position == "landlord_up": 
                y_pred = self.model_advice.forward(z_batch, x_batch)
                y_pred = y_pred.detach().cpu().numpy()

            if self.position == "landlord_down": 
                y_pred = self.model_advice.forward(z_batch, x_batch)
                y_pred = y_pred.detach().cpu().numpy()
                
            if y_pred>0.5:
                advice = 0
            else:
                advice = 1
            advicer_input = (z_batch.cpu().numpy().tolist(), x_batch.cpu().numpy().tolist())
        # print(advicer_input)
        return advice, advicer_input
    
    def get_act_data(self, infoset):
        obs = get_obs(infoset) 

        return [obs['z_batch'], obs['x_batch']]


    def act(self, infoset):
        # print(infoset.player_position)
        # print(infoset.player_hand_cards)
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs(infoset) 

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        # print("act:", infoset.player_position)
        # print(y_pred)

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action
