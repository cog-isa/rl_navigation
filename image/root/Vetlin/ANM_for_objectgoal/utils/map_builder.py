import numpy as np

import utils.depth_utils as du


class MapBuilder(object):
    def __init__(self, params, env):
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
        self.resolution = params['resolution']
        agent_min_z = params['agent_min_z']
        agent_max_z = params['agent_max_z']
        self.z_bins = [agent_min_z, agent_max_z]
        self.du_scale = params['du_scale']
        self.visualize = params['visualize']
        self.obs_threshold = params['obs_threshold']
        
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
        
        scene = env._env._sim.semantic_annotations()
        instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
        
        self.mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])
        self.objectgoal_mapping = {obj.category.name(): obj.category.index() for obj in scene.objects }
        
        return
    
    def prepare_semantic_observation(self, semantic):
        return np.take(self.mapping, semantic)

    def update_map(self, depth, current_pose, obs, env):
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN
        point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix, \
                                                scale=self.du_scale)
        
#         import matplotlib.pyplot as plt
#         print(depth.shape)
#         plt.imshow(depth)
        
        prepared_semantic_obs = self.prepare_semantic_observation(obs['semantic'])
#         semantic_goal = env.current_episode.goals[0].object_category
        semantic_goal = 'chair'
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
