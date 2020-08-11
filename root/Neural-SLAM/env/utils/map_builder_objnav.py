import numpy as np
# from numba import njit
import env.utils.depth_utils as du
import time
import skimage.measure


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
        self.resolution_semantic = params['resolution']# * 4
        
        agent_min_z = params['agent_min_z']
        agent_max_z = params['agent_max_z']
        self.z_bins = [agent_min_z, agent_max_z]
        self.du_scale = params['du_scale']
        self.visualize = params['visualize']
        self.obs_threshold = params['obs_threshold']
        
        self.classes_number = params['classes_number']

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)
        self.semantic_map = np.zeros((self.classes_number,
                                      self.map_size_cm // self.resolution_semantic,
                                      self.map_size_cm // self.resolution_semantic,
                                      ), dtype=np.float32)
        
        self.agent_height = params['agent_height']
        self.agent_view_angle = params['agent_view_angle']
        return

#     @njit
    def update_map(self, depth, current_pose, semantic):
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN
        
        #### Process walls and explored area maps
        
        point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix, \
                                                scale=self.du_scale)
    
        
        shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        agent_view = du.transform_camera_view(point_cloud,
                                              self.agent_height,
                                              self.agent_view_angle)
        agent_view_centered = du.transform_pose(agent_view, shift_loc)
        agent_view_flat = du.bin_points(
            agent_view_centered,
            self.vision_range,
            self.z_bins,
            self.resolution)
        
        
        agent_view_cropped = agent_view_flat[:, :, 1]
        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0

        geocentric_pc = du.transform_pose(agent_view, current_pose)
        geocentric_flat = du.bin_points(
            geocentric_pc,
            self.map.shape[0],
            self.z_bins,
            self.resolution)

        self.geocentric_view = geocentric_flat.sum(2)
        
        # Update map
        self.map = self.map + geocentric_flat
        
        map_gt = self.map[:, :, 1] / self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        explored_gt = self.map.sum(2)
        explored_gt[explored_gt > 1] = 1.0
        
        
        #### Process semantic maps
        
#         tm = time.time()
#         print('startt')
#         depth = skimage.measure.block_reduce(depth, (3,3), np.min)
#         semantic = skimage.measure.block_reduce(semantic, (3,3), np.min)

        # HxWxN binary
        depth_semantic = np.zeros([self.classes_number, *semantic.shape[:2]])
        for i in range(self.classes_number):
            depth_semantic[i] = depth
            nan_map = (semantic == i).astype(np.float)
            nan_map[nan_map == 0] = np.NaN
            depth_semantic[i] = depth_semantic[i] * nan_map
        self.depth_semantic = depth_semantic.copy()
        
#         print('1', time.time() - tm)
#         tm = time.time()
        
        point_cloud_semantic = du.get_point_cloud_from_z(depth_semantic,
                                                         self.camera_matrix,
                                                         scale=self.du_scale)
        
#         print('2', time.time() - tm)
#         tm = time.time()
        
        shift_loc = [self.vision_range * self.resolution_semantic // 2, 0, np.pi / 2.0]
        agent_view_semantic_not_centered = \
            du.transform_camera_view(point_cloud_semantic,
                                     self.agent_height,
                                     self.agent_view_angle)
        
#         print('3', time.time() - tm)
#         tm = time.time()
        
        agent_view_centered_semantic = du.transform_pose(agent_view_semantic_not_centered, 
                                                         shift_loc)
        
#         print('4', time.time() - tm)
#         tm = time.time()
        
        agent_view_semantic = du.project_points(
            agent_view_centered_semantic,
            self.vision_range,
            self.resolution_semantic)
    
#         print('5', time.time() - tm)
#         tm = time.time()
        
        agent_view_semantic[agent_view_semantic >= 0.5] = 1.0
        agent_view_semantic[agent_view_semantic < 0.5] = 0.0
        
#         print('6', time.time() - tm)
#         tm = time.time()
        
        geocentric_pc_semantic = du.transform_pose(agent_view_semantic_not_centered, 
                                                   current_pose)
        
        
#         print('7', time.time() - tm)
#         tm = time.time()
        
        geocentric_flat_semantic = du.project_points(
            geocentric_pc_semantic,
            self.semantic_map.shape[1],
            self.resolution_semantic)
        
#         print('8', time.time() - tm)
#         tm = time.time()
        
        # Update map
        self.semantic_map = self.semantic_map + geocentric_flat_semantic
        
        map_semantic_gt = self.semantic_map.copy()
        map_semantic_gt[map_semantic_gt >= 0.5] = 1.0
        map_semantic_gt[map_semantic_gt < 0.5] = 0.0
        
#         print('9', time.time() - tm)
#         tm = time.time()
        
        
#         agent_view_cropped = np.zeros((64, 64))
#         map_gt = np.zeros((480, 480))
#         agent_view_explored = np.zeros((64, 64))
#         explored_gt = np.zeros((480, 480))
#         agent_view_semantic = np.zeros((40, 64, 64))
#         map_semantic_gt = np.zeros((40, 480, 480))

        return agent_view_cropped, map_gt, agent_view_explored, explored_gt, \
               agent_view_semantic, map_semantic_gt

#     @njit
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
        self.semantic_map = np.zeros((self.classes_number,
                                      self.map_size_cm // self.resolution_semantic,
                                      self.map_size_cm // self.resolution_semantic,
                                      ), dtype=np.float32)
        
    def get_map(self):
        return self.map
