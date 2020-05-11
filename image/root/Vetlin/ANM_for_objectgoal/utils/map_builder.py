import numpy as np

import utils.depth_utils as du


class MapBuilder(object):
    def __init__(self, params):
        self.params = params
        frame_width = params['frame_width']
        frame_height = params['frame_height']
        fov = params['fov']
        self.camera_matrix = du.get_camera_matrix(
            frame_width,
            frame_height,
            fov)
        self.vision_range = params['vision_range']

        self.map_size_cm = params['map_size_cm']
        
        agent_min_z = params['agent_min_z']
        agent_max_z = params['agent_max_z']
        self.z_bins = [agent_min_z, agent_max_z]
        self.du_scale = params['du_scale']
        self.visualize = params['visualize']
        self.obs_threshold = params['obs_threshold']
        
        self.resolution = params['resolution']
        
        # replace in the future
        self.semantic_obs_threshold = params['obs_threshold']

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)
        
        self.semantic_map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

        self.agent_height = params['agent_height']
        self.agent_view_angle = params['agent_view_angle']
        
        # prepare semantic mapper
        self.mapping = [1, 40, 1, 16, 4, 2, 4, 39, 4, 1, 17, 0, 1, 1, 40, 4, 1, 1, 14, 4, 24, 1, 0, 1, -1, 39, 17, 4, 5, 1, 1, 39, 21, 1, 2, 1, 15, 39, 40, 1, 1, 4, 1, 7, 28, 39, 40, 28, 40, 40, 39, 40, 39, 39, 39, 39, 39, 2, 17, 1, 40, 1, 1, 5, 1, 34, 15, 1, 15, 1, 20, 1, 15, 1, 15, 20, 20, 39, 4, 39, 39, 39, 39, 39, 28, 28, 28, 40, 1, 12, 9, 1, 0, 4, 4, 40, 23, 17, 2, 1, 1, 4, 23, 39, 2, 17, 40, 40, 34, 1, 34, 1, 1, 9, 1, 28, 9, 4, 28, 39, 28, 40, 2, 28, 28, 38, 4, 4, 0, 11, 1, 17, 1, 1, 7, 0, 15, 40, 0, 21, 1, 7, 4, 0, 28, 4, 1, 4, 39, 2, 39, 39, 20, 39, 39, 39, 39, 39, 39, 23, 2, 1, 1, 1, 4, 23, 4, 17, 1, 24, 40, 1, 1, 24, 1, 4, 17, 14, 4, 4, 4, 1, 14, 28, 4, 4, 14, 35, 39, 28, 28, 28, 39, 2, 28, 16, 2, 39, 2, 1, 37, 7, 40, 31, 1, 17, 1, 4, 1, 26, 4, 1, 1, 4, 7, 4, 20, 1, 20, 22, 40, 40, 3, 40, 3, 3, 16, 5, 39, 5, 20, 39, 3, 3, 3, 3, 3, 20, 20, 17, 4, 1, 2, 1, 40, 38, 4, 1, 17, 1, 1, 40, 2, 17, 40, 1, 17, 9, 9, 1, 39, 16, 9, 9, 40, 40, 40, 4, 1, 17, 40, 26, 1, 2, 16, 1, 39]
        self.objectgoal_mapping = {'wall': 1, 'misc': 40, 'stairs': 16, 'door': 4, 'floor': 2, 'objects': 39, 'ceiling': 17, 'void': 0, 'plant': 14, 'column': 24, '': -1, 'table': 5, 'mirror': 21, 'sink': 15, 'cabinet': 7, 'lighting': 28, 'seating': 34, 'towel': 20, 'curtain': 12, 'window': 9, 'shower': 23, 'clothes': 38, 'bed': 11, 'board_panel': 35, 'appliances': 37, 'shelving': 31, 'counter': 26, 'tv_monitor': 22, 'chair': 3}
        self.goal_mapper = {0: 'chair', 1: 'table', 2: 'picture', 3: 'cabinet', 4: 'cushion', 5: 'sofa', 6: 'bed', 7: 'chest_of_drawers', 8: 'plant', 9: 'sink', 10: 'toilet', 11: 'stool', 12: 'towel', 13: 'tv_monitor', 14: 'shower', 15: 'bathtub', 16: 'counter', 17: 'fireplace', 18: 'gym_equipment', 19: 'seating', 20: 'clothes'}

        return
    
    def prepare_semantic_observation(self, semantic):
        return np.take(self.mapping, semantic)

    def update_map(self, depth, obs, current_pose):
        
#         current_pose = obs['gps']
        
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN
        point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix, \
                                                scale=self.du_scale)
        
#         import matplotlib.pyplot as plt
#         print(depth.shape)
#         plt.imshow(depth)
        
        prepared_semantic_obs = self.prepare_semantic_observation(obs['semantic'])
        semantic_goal = self.goal_mapper[obs['objectgoal'][0]]
        prepared_semantic_obs = (prepared_semantic_obs == self.objectgoal_mapping[semantic_goal])
        
        semantic_depth = depth * prepared_semantic_obs
        
        semantic_depth[semantic_depth == 0] = np.NaN
        semantic_point_cloud = du.get_point_cloud_from_z(semantic_depth, self.camera_matrix, scale=self.du_scale)
        
#         print(semantic_point_cloud[51])

        agent_view = du.transform_camera_view(point_cloud,
                                              self.agent_height,
                                              self.agent_view_angle)
        
        semantic_agent_view = du.transform_camera_view(semantic_point_cloud,
                                              self.agent_height,
                                              self.agent_view_angle)
        
        shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        agent_view_centered = du.transform_pose(agent_view, shift_loc)
        semantic_agent_view_centered = du.transform_pose(semantic_agent_view, shift_loc)

        
        agent_view_flat = du.bin_points(
            agent_view_centered,
            self.vision_range,
            self.z_bins,
            self.resolution)
        
        semantic_agent_view_flat = du.bin_points(
            semantic_agent_view_centered,
            self.vision_range,
            self.z_bins,
            self.resolution)
        
        semantic_agent_view_cropped = semantic_agent_view_flat[:, :, 1]
        
        semantic_agent_view_cropped = semantic_agent_view_cropped / self.semantic_obs_threshold
        semantic_agent_view_cropped[semantic_agent_view_cropped >= 0.5] = 1.0
        semantic_agent_view_cropped[semantic_agent_view_cropped < 0.5] = 0.0

        agent_view_cropped = agent_view_flat[:, :, 1]

        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0

        geocentric_pc = du.transform_pose(agent_view, current_pose)
        
        semantic_geocentric_pc = du.transform_pose(semantic_agent_view, current_pose)

        geocentric_flat = du.bin_points(
            geocentric_pc,
            self.map.shape[0],
            self.z_bins,
            self.resolution)
        
        semantic_geocentric_flat = du.bin_points(
            semantic_geocentric_pc,
            self.map.shape[0],
            self.z_bins,
            self.resolution)

        self.map = self.map + geocentric_flat
        self.semantic_map = self.semantic_map + semantic_geocentric_flat

        map_gt = self.map[:, :, 1] / self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0
        
        semantic_map_gt = self.semantic_map[:, :, 1] / self.semantic_obs_threshold
        semantic_map_gt[semantic_map_gt >= 0.5] = 1.0
        semantic_map_gt[semantic_map_gt < 0.5] = 0.0

        explored_gt = self.map.sum(2)
        explored_gt[explored_gt > 1] = 1.0
                
        return agent_view_cropped, map_gt, agent_view_explored, explored_gt, semantic_agent_view_cropped, semantic_map_gt

    def get_st_pose(self, current_loc):
        loc = [- (current_loc[0] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               - (current_loc[1] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               90 - np.rad2deg(current_loc[2])]
        return loc

    def reset_map(self, map_size):
        self.map_size_cm = map_size

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

    def get_map(self):
        return self.map
