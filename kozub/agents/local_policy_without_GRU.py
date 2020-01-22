import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from torch.distributions import Categorical
import numpy as np

#class impement ppo agent
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()


class PPO(nn.Module):
    def __init__(self, learning_rate, gamma, eps_clip):
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

    def pi(self, x, y, softmax_dim=0):
        """
        define computation graph for pi
        :param x: input
        :param softmax_dim:
        :return:
        """
        #print(x.size())
        x = F.relu(self.fc1(x))
        #print(x.size())
        x = F.relu(self.fc2(x))
        #print(x.size())
        x = F.relu(self.fc3(x))
        #print(x.size())
        x = x.flatten(start_dim=1, end_dim=3)
        x = torch.cat((x, y), 1)

        #x = x.reshape(1,-1,18)
        x = self.fc_pi(x)

        prob = F.softmax(x, dim=softmax_dim)

        return prob

    def v(self, x, y):
        """
        define computation graph for v
        :param x: вход
        :return:
        """
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = F.relu(self.fc2(x))
        # print(x.size())
        x = F.relu(self.fc3(x))
        # print(x.size())
        x = x.flatten(start_dim=1, end_dim=3)
        x = torch.cat((x, y), 1)

        v = self.fc_v(x)
        return v

    def act(self, rgb, comp):
        rgb = torch.tensor(rgb[None], dtype=torch.float, device=device)
        rgb = rgb.permute([0,3,1,2])
        comp = torch.tensor(comp[None], dtype=torch.float, device=device)

        prob_a = self.pi(rgb, comp, softmax_dim = 1)
        m = Categorical(prob_a)
        a = m.sample().item()

        return prob_a[0][a].item(), a


    def put_data(self, transition):
        self.data.append(transition)

    def data_len(self):
        return len(self.data)

    def make_batch(self):

        rgb_lst, comp_lst, a_lst, r_lst, rgb_prime_lst, comp_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], [], [], []

        for transition in self.data:
            rgb, comp, a, r, rgb_prime, comp_prime, prob_a, done = transition

            rgb_lst.append(np.moveaxis(rgb, 2,0))
            comp_lst.append(comp)
            a_lst.append([a])
            r_lst.append([r])
            rgb_prime_lst.append(np.moveaxis(rgb_prime, 2, 0))
            comp_prime_lst.append(comp_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        rgb = torch.tensor(rgb_lst, dtype=torch.float, device=device)
        comp = torch.tensor(comp_lst, dtype=torch.float, device=device)
        a = torch.tensor(a_lst, device=device)
        r = torch.tensor(r_lst, device=device)
        rgb_prime = torch.tensor(rgb_prime_lst, dtype=torch.float, device=device)
        comp_prime = torch.tensor(comp_prime_lst, dtype=torch.float, device=device)
        done_mask = torch.tensor(done_lst, dtype=torch.float, device=device)
        prob_a = torch.tensor(prob_a_lst, device=device)
        self.data = []
        return rgb, comp, a, r, rgb_prime, comp_prime, done_mask, prob_a

    def train_net(self, epochs):
        """
        training function
        :param epochs: number of epochs
        :return:
        """
        rgb, comp, a, r, rgb_prime, comp_prime, done_mask, prob_a = self.make_batch()

        for i in range(epochs):

            td_target = r + self.gamma * self.v(rgb_prime, comp_prime)*done_mask

            delta = td_target - self.v(rgb, comp)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0


            for adv in delta[::-1]:
              advantage = self.gamma*advantage + adv[0]
              advantage_lst.append(advantage)
            #advantage_lst.append([advantage]*len(delta))


            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float, device=device).reshape((-1,1))

            # getting pi_a
            pi = self.pi(rgb, comp, softmax_dim=1)
            pi_a = pi.gather(1, a)

            # computes first part of surrogate function
            surr1 = torch.exp(torch.log(pi_a) - torch.log(prob_a))*advantage

            # second part
            surr2 = torch.clamp(torch.exp(torch.log(pi_a) - torch.log(prob_a)), 1 + self.eps_clip, 1 - self.eps_clip) * advantage

            loss1 = -torch.min(surr1, surr2)
            loss2 = F.smooth_l1_loss(self.v(rgb, comp), td_target.detach())
            loss = (loss1 + loss2).mean()

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()
            return loss.detach().cpu().numpy()
            ################################

