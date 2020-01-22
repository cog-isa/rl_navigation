import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# если видеокарта доступна, то будем ее использовать
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

import numpy as np



#Define model
class LocalPolicy(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        определение сети
        """
        super().__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.fc1 = nn.Conv2d(3, 2, 4, stride = 4)
        self.fc2 = nn.Conv2d(2, 2, 4, stride = 4)
        self.fc3 = nn.Conv2d(2, 1, 4, stride = 4)

        #self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers=1)

        #self.input_size = input_size
        #self.hidden_size = hidden_size
        #self.reset_hidden()

    def forward(self, x,y):
        """
        определение графа вычислений
        :param x: вход
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
        print(x.size(), y.size())
        x = torch.cat((x, y), 1)

        x = x.reshape(1,-1,18)

        #print(x.size())

        x, self.hidden = self.gru(x, self.hidden)
        x = x.reshape(-1, self.hidden_size)

        #print(x.size())
        return F.softmax(x, dim=1)

    def reset_hidden(self):
        self.hidden = torch.zeros(1, 3, self.hidden_size)

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




def main():
    l_policy = LocalPolicy(18, 4)
    test_data = torch.ones(3, 3, 256, 256)
    comp_test_data = torch.zeros(3, 2)
    print(l_policy)
    print(l_policy(test_data, comp_test_data))
    print(l_policy.hidden)

if __name__ == '__main__':
    main()
