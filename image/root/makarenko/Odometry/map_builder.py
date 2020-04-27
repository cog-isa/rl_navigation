import numpy as np

import depth_utils as du


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
        self.resolution = params['resolution']
        agent_min_z = params['agent_min_z']
        agent_max_z = params['agent_max_z']
        self.z_bins = [agent_min_z, agent_max_z]
        self.du_scale = params['du_scale']
        self.visualize = params['visualize']
        self.obs_threshold = params['obs_threshold']

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

        self.agent_height = params['agent_height']
        self.agent_view_angle = params['agent_view_angle']
        return

    def get_top_down_map(self, depth):
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN
#         print(depth.shape)
        point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix, \
                                                scale=self.du_scale)
#         print(point_cloud.shape)
        agent_view = du.transform_camera_view(point_cloud,
                                              self.agent_height,
                                              self.agent_view_angle)

        shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        agent_view_centered = du.transform_pose(agent_view, shift_loc)
#         print(agent_view_centered.shape)

        agent_view_flat = du.bin_points(
            agent_view_centered,
            self.vision_range,
            self.z_bins,
            self.resolution)
        # Change here mismatch dimensions 0
        agent_view_cropped = agent_view_flat[:, :, :, 1]

        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        return agent_view_cropped