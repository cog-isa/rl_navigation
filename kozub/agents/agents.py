#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
import numpy as np
from torch.distributions import Categorical

input_size = 2
outpout_size = 3


class PPO(nn.Module):
    def __init__(self, learning_rate, gamma, eps_clip):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(input_size, 256)
        self.fc_pi = nn.Linear(256, outpout_size)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optimizers.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def pi(self, x, softmax_dim=0):
        """
        define computation graph for pi
        :param x: input
        :param softmax_dim:
        :return:
        """
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def act(self, s):
        prob = self.pi(torch.from_numpy(s).float())
        m = Categorical(prob)
        a = m.sample().item()
        return a


    def v(self, x):
        """
        define computation graph for v
        :param x: вход
        :return:
        """
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        """
        memorizing transitions
        :param transition:
        :return:
        """
        self.data.append(transition)

    def make_batch(self):
        """
        we have already seen, almost all the errors that you can get creating batches,
        so don't bother yourself coding this :)
        :return:
        """
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s = torch.tensor(s_lst, dtype=torch.float)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
        done_mask = torch.tensor(done_lst, dtype=torch.float)
        prob_a = torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self, epochs):
        """
        training function
        :param epochs: number of epochs
        :return:
        """
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        #print(self.v(s).size(), done_mask.size(), (self.v(s)*done_mask).size())
        for i in range(epochs):
            # compute td
            # td_target =
            ###### Your code here ##########
            #raise NotImplementedError
            td_target = r + self.gamma * self.v(s_prime)*done_mask
            ################################
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            # compute advantage_lst (pay attention to the order)
            ###### Your code here ##########
            #raise NotImplementedError
            for adv in delta[::-1]:
              advantage = self.gamma*advantage + adv[0]
              advantage_lst.append([advantage])
            ################################
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # getting pi_a
            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)

            # computes first part of surrogate function
            # surr1 =
            ###### Your code here ##########
            #raise NotImplementedError
            surr1 = torch.exp(torch.log(pi_a) - torch.log(prob_a))*advantage
            ####################### #########

            # second part
            # surr2 = torch.clamp(pi_a/prob_a, 1 + eps, 1 - eps) * adv
            # pi_a/prob_a == exp(log(pi_a) - log(prob_a))
            ###### Your code here ##########
            #raise NotImplementedError
            surr2 = torch.clamp(torch.exp(torch.log(pi_a) - torch.log(prob_a)), 1 + self.eps_clip, 1 - self.eps_clip) * advantage
            ################################
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())
            #print(loss.size(), surr1.size(), surr2.size(), F.smooth_l1_loss(self.v(s), td_target.detach()).size())
            # make optimizer step, just copy/paste it from previous seminar :)
            ###### Your code here ##########
            #raise NotImplementedError
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            ################################

