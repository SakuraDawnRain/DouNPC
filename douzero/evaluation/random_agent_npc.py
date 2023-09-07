import random
import torch

from douzero.env.env_npc import get_obs_npc

class RandomAgent():

    def __init__(self, position):
        self.name = 'Random'
        from model_npc import FarmerLstmModelNPC
        # self.model_npc = _load_model(position, model_path)
        self.model_advice = FarmerLstmModelNPC()
        if position == "landlord_up":
            self.model_advice.load_state_dict(torch.load("advicer_LU.ckpt", map_location='cpu'))
        elif position == "landlord_down":
            self.model_advice.load_state_dict(torch.load("advicer_LD.ckpt", map_location='cpu'))
        self.position = position

    def advice(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0], ()

        best_action = random.choice(infoset.legal_actions)

        advice = 1
        
        advicer_input = ()

        if best_action == []:
            advice = 1
        else:
            obs = get_obs_npc(infoset, action=best_action)
            z_batch = torch.from_numpy(obs['z_batch']).float()
            x_batch = torch.from_numpy(obs['x_batch']).float()
            # print(obs)
            y_pred = 0
            if torch.cuda.is_available():
                z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
            if self.position == "landlord_up": 
                y_pred = self.model_advice.forward(z_batch, x_batch)
                y_pred = y_pred.detach().cpu().numpy()

            if self.position == "landlord_down": 
                y_pred = self.model_advice.forward(z_batch, x_batch)
                y_pred = y_pred.detach().cpu().numpy()

            # print(y_pred)
            if y_pred>0.5:
                advice = 0
            else:
                advice = 1
        return advice, ()

    def act(self, infoset):
        return random.choice(infoset.legal_actions)
