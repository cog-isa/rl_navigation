import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np

from utils.distributions import Categorical, DiagGaussian
from utils.model import get_grid, ChannelPool, Flatten, NNBase

from utils.map_builder import MapBuilder


# Global Policy model code
class Global_Policy(NNBase):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1):
        super(Global_Policy, self).__init__(recurrent, hidden_size,
                                            hidden_size)

        out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(10, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.linear1 = nn.Linear(out_size * 32 + 8, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.orientation_emb = nn.Embedding(72, 8)
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):
        x = self.main(inputs)
        orientation_emb = self.orientation_emb(extras).squeeze(1)
        x = torch.cat((x, orientation_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs


# Neural SLAM Module code
class Neural_SLAM_Module(nn.Module):
    """
    """

    def __init__(self, args, full_obs):
        super(Neural_SLAM_Module, self).__init__()
        
        self.width = args.frame_width
        self.height = args.frame_height
        
        self.args = args
        
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['fov'] =  self.args.hfov
        params['resolution'] = self.args.map_resolution
        params['map_size_cm'] = self.args.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.args.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['visualize'] = self.args.visualize
        params['obs_threshold'] = self.args.obs_threshold
        
        self.map_size_cm = args.map_size_cm
        
        
        self.last_sim_location = self.get_sim_location(full_obs)
        
        self.mapper = MapBuilder(params)
        
    
    def _preprocess_depth(self, depth):
        depth = depth[:, :, 0]*1
        mask2 = depth > 0.99
        depth[mask2] = 0.

        for i in range(depth.shape[1]):
            depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

        mask1 = depth == 0
        depth[mask1] = np.NaN
        depth = depth*1000.
        return depth

    def get_sim_location(self, full_obs):
        
        x_arr = []
        y_arr = []
        o_arr = []

        for obs in full_obs:
            x = obs['gps'][2]
            y = obs['gps'][0]
            o = obs['compass'][0]
            
            x_arr.append(x)
            y_arr.append(y)
            o_arr.append(o)
        
        return x, y, o
    
    def get_gt_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do


    def forward(self, full_obs):

        answer_map = []
        answer_exp = []
        answer_sem = []
        answer_pose = []
        
        self.last_sim_location = self.get_sim_location(full_obs)

        
        for index in range(len(full_obs)):
            obs = full_obs[index]
            
            depth = obs['depth']
            semantic = obs['semantic']
            
            
            self.mapper.reset_map(self.map_size_cm)
            
            self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]
            
            self.curr_loc_gt = self.curr_loc
            self.last_loc_gt = self.curr_loc_gt
            
            self.last_loc = self.curr_loc
            

            # Convert pose to cm and degrees for mapper
            mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))
            

            proj_pred, map_pred, fp_exp_pred, exp_pred, fp_sem_pred, semantic_pred = self.mapper.update_map(self._preprocess_depth(depth), obs, mapper_gt_pose)

            pose_pred = obs['gps']

            answer_map.append(map_pred)
            answer_exp.append(exp_pred)
            answer_sem.append(semantic_pred)
            answer_pose.append(pose_pred)
            
        answer_map = np.array(answer_map)
        answer_exp = np.array(answer_exp)
        answer_sem = np.array(answer_sem)
        answer_pose = np.array(answer_pose)
            
        return torch.tensor(answer_map), torch.tensor(answer_exp), torch.tensor(answer_sem), torch.tensor(answer_pose)


# Local Policy model code
class Local_IL_Policy(NNBase):

    def __init__(self, input_shape, num_actions, recurrent=False,
                 hidden_size=512, deterministic=False):

        super(Local_IL_Policy, self).__init__(recurrent, hidden_size,
                                              hidden_size)

        self.deterministic = deterministic
        self.dropout = 0.5

        resnet = models.resnet18(pretrained=True)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])

        # Extra convolution layer
        self.conv = nn.Sequential(*filter(bool, [
            nn.Conv2d(512, 64, (1, 1), stride=(1, 1)),
            nn.ReLU()
        ]))

        # convolution output size
        input_test = torch.randn(1, 3, input_shape[1], input_shape[2])
        conv_output = self.conv(self.resnet_l5(input_test))
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layers
        self.proj1 = nn.Linear(self.conv_output_size, hidden_size - 16)
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
        self.linear = nn.Linear(hidden_size, hidden_size)

        # Short-term goal embedding layers
        self.embedding_angle = nn.Embedding(72, 8)
        self.embedding_dist = nn.Embedding(24, 8)

        # Policy linear layer
        self.policy_linear = nn.Linear(hidden_size, num_actions)

        self.train()

    def forward(self, rgb, rnn_hxs, masks, extras):
        if self.deterministic:
            x = torch.zeros(extras.size(0), 3)
            for i, stg in enumerate(extras):
                if stg[0] < 3 or stg[0] > 68:
                    x[i] = torch.tensor([0.0, 0.0, 1.0])
                elif stg[0] < 36:
                    x[i] = torch.tensor([0.0, 1.0, 0.0])
                else:
                    x[i] = torch.tensor([1.0, 0.0, 0.0])
        else:
            resnet_output = self.resnet_l5(rgb[:, :3, :, :])
            conv_output = self.conv(resnet_output)
            
            proj1 = nn.ReLU()(self.proj1(conv_output.view(
                -1, self.conv_output_size)))
            if self.dropout > 0:
                proj1 = self.dropout1(proj1)

            angle_emb = self.embedding_angle(extras[:, 0]).view(-1, 8)
            dist_emb = self.embedding_dist(extras[:, 1]).view(-1, 8)
            
            x = torch.cat((proj1, angle_emb, dist_emb), 1)
            x = nn.ReLU()(self.linear(x))
            if self.is_recurrent:
                x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

            x = nn.Softmax(dim=1)(self.policy_linear(x))

        action = torch.argmax(x, dim=1)

        return action, x, rnn_hxs


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):

    def __init__(self, obs_shape, action_space, model_type=0,
                 base_kwargs=None):

        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if model_type == 0:
            self.network = Global_Policy(obs_shape, **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            return self.network(inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs