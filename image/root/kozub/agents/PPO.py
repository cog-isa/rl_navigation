import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from torch.distributions import Categorical
import numpy as np
import random

#class impemented ppo agent

class PPO(nn.Module):
    def __init__(self,  device, learning_rate = 1e-3, gamma = 0.99, eps_clip=0.5):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()
        self.device = device

        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Conv2d(1, 4, 4, stride = 4)
        self.fc2 = nn.Conv2d(4, 2, 4, stride = 4)
        self.fc3 = nn.Conv2d(2, 1, 4, stride = 4)

        self.fc_pi = nn.Linear(18, 3)
        self.fc_v = nn.Linear(18, 3)


        self.optimizer = optimizers.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.batch_size = 20 #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    def pi(self, x, y, softmax_dim=0):

        #path image from conv layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        #rehape to vo vector
        x = x.flatten(start_dim=1, end_dim=3)

        #join conv(image) with gps part
        x = torch.cat((x, y), 1)


        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)

        return prob

    def v(self, x, y):

        #path image from conv layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        #rehape to vo vector
        x = x.flatten(start_dim=1, end_dim=3)

        #join conv(image) with gps part
        x = torch.cat((x, y), 1)

        v = self.fc_v(x)
        return v

    def act(self, rgb, comp):
        rgb = torch.tensor(rgb[None], dtype=torch.float, device=self.device)
        #rgb = rgb.permute([0,3,1,2])self.

        comp = torch.tensor(comp[None], dtype=torch.float, device=self.device)

        prob_a = self.pi(rgb, comp, softmax_dim = 1)
        m = Categorical(prob_a)
        a = m.sample().item()

        return prob_a[0][a].item(), a


    def put_data(self, transition):
        self.data.append(transition)

    def data_len(self):
        return len(self.data)

    def clear_data(self):
        self.data = []

    def make_batch(self, size):

        rgb_lst, comp_lst, a_lst, r_lst, rgb_prime_lst, comp_prime_lst, done_lst, prob_a_lst = [], [], [], [], [], [], [], []

        ind = random.sample(range(self.data_len()), 2)

        for i in ind:
            transition  = self.data[i]
            rgb, comp, a, r, rgb_prime, comp_prime,  done , prob_a  = transition

            rgb_lst.append(np.moveaxis(rgb, 2,0))
            comp_lst.append(comp)
            a_lst.append([a])
            r_lst.append([r])
            rgb_prime_lst.append(np.moveaxis(rgb_prime, 2, 0))
            comp_prime_lst.append(comp_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        rgb = torch.tensor(rgb_lst, dtype=torch.float, device=self.device)
        comp = torch.tensor(comp_lst, dtype=torch.float, device=self.device)
        a = torch.tensor(a_lst, device=self.device)
        r = torch.tensor(r_lst, device=self.device)
        rgb_prime = torch.tensor(rgb_prime_lst, dtype=torch.float, device=self.device)
        comp_prime = torch.tensor(comp_prime_lst, dtype=torch.float, device=self.device)
        done_mask = torch.tensor(done_lst, dtype=torch.float, device=self.device)
        prob_a = torch.tensor(prob_a_lst, device=self.device)

        return rgb, comp, a, r, rgb_prime, comp_prime, done_mask, prob_a

    def train_net(self, epochs):

        rgb, comp, a, r, rgb_prime, comp_prime, done_mask, prob_a = self.make_batch(self.batch_size)

        for i in range(epochs):

            td_target = r + self.gamma * self.v(rgb_prime, comp_prime)*done_mask

            # delta = td_target - self.v(rgb, comp)
            # delta = delta.detach().cpu().numpy()
            # advantage_lst = []
            # advantage = 0.0
            # for adv in delta[::-1]:
            #   advantage = self.gamma*advantage + adv[0]
            #   advantage_lst.append(advantage)
            #advantage_lst.append([advantage]*len(delta))


            # advantage_lst.reverse()
            # advantage = torch.tensor(advantage_lst, dtype=torch.float, device=device).reshape((-1,1))
            advantage = td_target

            # getting pi_a
            pi = self.pi(rgb, comp, softmax_dim=1)
            pi_a = pi.gather(1, a)

            # computes first part of surrogate function
            surr1 = torch.exp(torch.log(pi_a) - torch.log(prob_a))*advantage

            # second part
            surr2 = torch.clamp(torch.exp(torch.log(pi_a) - torch.log(prob_a)), 1 + self.eps_clip, 1 - self.eps_clip) * advantage

            #ascent for policy
            loss1 = -torch.min(surr1, surr2)

            #descent for V
            loss2 = F.smooth_l1_loss(self.v(rgb, comp), td_target.detach())
            loss = (loss1 + loss2).mean()

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

        self.clear_data()
        return loss.detach().cpu().numpy()
            ################################

