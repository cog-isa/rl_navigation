import sys
import cv2
import numpy as np
import habitat
from environment import environments
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

half_angle_of_view = np.pi/2 - np.arctan(0.78)

# Переводит карту глубины в облако точек
def depth2ptsCloud(depth, cx=127.5, cy=127.5, fx=100, fy=100):
    h, w = depth.shape
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    xv, yv = np.meshgrid(x[::-1], y)
    xv = xv.reshape((-1))
    yv = yv.reshape((-1))
    dfl = depth.T.reshape((-1))
    return np.concatenate(((dfl*(xv - cx)/fx)[:,None], (dfl*(yv-cy)/fy)[:,None], dfl[:,None]), 1)  


# Переводит облако точек в локальную карту
def ptsCloud2localMap(pts, cell_size=0.001, floor_lvl=None, head_lvl=None):
    len_z = max(pts[:,0]) - min(pts[:,0])
    if floor_lvl == None:
        floor_lvl = min(pts[:,0]) + 0.1*len_z
    if head_lvl == None:
        head_lvl = max(pts[:,0]) - 0.3*len_z
    points_tdv = pts[(pts[:,0] > floor_lvl) * (pts[:,0] < head_lvl)][:,1:]
    
    map_size_x = ((max(points_tdv[:,0]) - min(points_tdv[:,0])) // cell_size).astype(np.int) + 2
    map_size_y = (max(points_tdv[:,1]) // cell_size).astype(np.int) + 1
    points_local_map = (points_tdv // cell_size).astype(np.int)
    zero_pos = (-min(points_local_map[:,0]))
    points_local_map[:,0] += zero_pos
    
    threshold = cell_size * 1000
    local_map = np.zeros((map_size_y, map_size_x))
    for point in points_local_map:
        local_map[map_size_y-point[1]-1, point[0]] += 1
    local_map = (local_map > threshold)
    
    return (local_map, zero_pos)


# Округление вверх по модулю
def round_abs_up(a, b=1.):
    sign_a = 2*(1/2-1*(a < 0))
    return int(int(a/b) + sign_a*(a%b > 0))


# Обновляет глобальную карту по облаку точек
def adopt_global_map_with_pts_cloud(old_global_map, x, y, angle, pts, cell_size=0.001, floor_lvl=None, head_lvl=None):
    """
        old_global_map : предыдущая глобальная карта до обновления
        x, y, angle : текущее положение агента после локализации
        pts : облако точек, построенное по текущей карте глубины
        cell_size : отвечает за размер глобальной карты (должен быть константой, относительно всех шагов adopt)
        floor_lvl : "уровень пола", граница, точки ниже которой в облаке точек не рассматриваются 
            [если не указано, то нижние 10%]
        head_lvl : "уровень головы", граница, точки выше которой в облаке точек не рассматриваются
            [если не указано, то верхние 30%]
    """
    len_z = max(pts[:,0]) - min(pts[:,0])
    if floor_lvl == None:
        floor_lvl = min(pts[:,0]) + 0.1*len_z
    if head_lvl == None:
        head_lvl = max(pts[:,0]) - 0.3*len_z
    points_tdv = pts[(pts[:,0] > floor_lvl) * (pts[:,0] < head_lvl)][:,1:]
    points_tdv[:,1] = -points_tdv[:,1]
    
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    points_tdv = points_tdv @ rot_matrix
    
    min_y, max_y = np.min(points_tdv[:,1]), np.max(points_tdv[:,1])
    min_x, max_x = np.min(points_tdv[:,0]), np.max(points_tdv[:,0])
    
    left_adj = max(0, round_abs_up(- x - min_x/cell_size))
    right_adj = max(0, round_abs_up(x + max_x/cell_size - old_global_map.shape[1] + 1))
    top_adj = max(0, round_abs_up(- y - min_y/cell_size))
    down_adj = max(0, round_abs_up(y + max_y/cell_size - old_global_map.shape[0] + 1))
    
    points_tdv = points_tdv / cell_size
    points_tdv += np.array([x+left_adj, y+top_adj])[None,:]
    points_tdv = np.round(points_tdv).astype(np.int)
    
    new_global_map = np.zeros((old_global_map.shape[0] + top_adj + down_adj, old_global_map.shape[1] + right_adj + left_adj))
    
    threshold = cell_size * 1000
    for point in points_tdv:
        new_global_map[point[1], point[0]] += 1
    new_global_map = (new_global_map > threshold)
    
    new_global_map[top_adj : top_adj + old_global_map.shape[0],
                   left_adj : left_adj + old_global_map.shape[1]] = ((new_global_map[top_adj : top_adj + old_global_map.shape[0],
                                                                                   left_adj : left_adj + old_global_map.shape[1]] + old_global_map) > 0)
    
    return new_global_map, x + left_adj, y + top_adj


# Выводит агента на карте с областью зрения
def plot_agent_on_map(x, y, angle, map_sh, map_sw):
    plt.plot(x, y, 'ro', mew=3)
    
    ang1 = angle + half_angle_of_view
    while(ang1 < 0):
        ang1 += 2*np.pi
    while(ang1 >= 2*np.pi):
        ang1 -= 2*np.pi
    
    ang2 = angle - half_angle_of_view
    while(ang2 < 0):
        ang2 += 2*np.pi
    while(ang2 >= 2*np.pi):
        ang2 -= 2*np.pi
    
    a1, a2 = 1./np.tan(ang1), 1./np.tan(ang2)
    b1, b2 = y-a1*x, y-a2*x
    
    xx = np.arange(map_sw)
    yy1, yy2 = np.array([a1*x_+b1 for x_ in xx]), np.array([a2*x_+b2 for x_ in xx])
    xx1, xx2 = xx[(yy1>0)*(yy1<map_sh-1)], xx[(yy2>0)*(yy2<map_sh-1)]
    yy1, yy2 = yy1[(yy1>0)*(yy1<map_sh-1)], yy2[(yy2>0)*(yy2<map_sh-1)]
    
    if (ang1 <= np.pi):
        yy1 = yy1[xx1 <= x]
        xx1 = xx1[xx1 <= x]
    else:
        yy1 = yy1[xx1 >= x]
        xx1 = xx1[xx1 >= x]
    
    if (ang2 <= np.pi):
        yy2 = yy2[xx2 <= x]
        xx2 = xx2[xx2 <= x]
    else:
        yy2 = yy2[xx2 >= x]
        xx2 = xx2[xx2 >= x]
    
    plt.plot(xx1, yy1, 'g')
    plt.plot(xx2, yy2, 'g')


# Базовая функция для получения текущей позиции
def tracking_position(x, y, ang, action, cell_size, rot_size=np.pi/18, step_size=0.025):
    if (action == 'w'):
        real_step_size = step_size/cell_size
        sin, cos = np.sin(ang), np.cos(ang)
        dx, dy = sin * real_step_size, cos * real_step_size
        return x-dx, y-dy, ang
    elif (action == 'a'):
        return x, y, ang+rot_size
    elif (action == 'd'):
        return x, y, ang-rot_size


    
# Класс в котором будет обновляться карта и позиция
class map_and_localize:
    """
        Этот класс задаётся в начале движения агента и хранит
        в себе информацию о его положении и глобальной карте
    """
    def __init__(self, depth, cell_size=0.001, x=0., y=0., ang=0., localizer=None):
        """
            depth : карта глубины на стартовом шаге
            cell_size : отвечает за детальность карты
            x, y, ang : стартовое пположение агента (ang в радианах)
            localizer : функция, отвечающая за уточнение координат агента после совершения
                действия (т.к. данная функция может работать основываясь очень на разных наблюдениях - не задавал явного вида
                входящих переменных, а значит нужно будет незначительно менять сам код класса, чтобы ввести другую функцию)
        """
        self.cell_size = cell_size
        self.x = x
        self.y = y
        self.ang = ang
        self.global_map = np.zeros((10, 10))
        self.last_depth = depth
        pts = depth2ptsCloud(depth)
        self.global_map, self.x, self.y = adopt_global_map_with_pts_cloud(self.global_map,
                                                      self.x, self.y, self.ang, pts, self.cell_size)
        if localizer == None:
            # функция отслеживает позицию агента, если движение не зашумлённое и детерменированное
            self.localizer = tracking_position
        else:
            self.localizer = localizer
       
    
    def update(self, new_depth, action):
        """
        Обновляет глобальную карту
            new_depth : новая карта глубины после совершения действия
            action : совершённое действие ('w' : forward, 'a' : turn left, 'd' : turn right)
        return : (self.global_map : новая глобальная карта,
                  self.x, self.y, self.ang : новое положение агента)
        """
        pts = depth2ptsCloud(new_depth)  # строим облако точек по текущей карте глубины
        
        # !!!!!!!!! ИЗМЕНЯТЬ КОД ЗДЕСЬ ПРИ ИЗМЕНЕНИИ localizer ФУНКЦИИ !!!!!!!!!!
        self.x, self.y, self.ang = self.localizer(self.x, self.y, self.ang,
                                                     action, self.cell_size)  # корректируем положение агента
        
        self.global_map, self.x, self.y = adopt_global_map_with_pts_cloud(self.global_map,
                                              self.x, self.y, self.ang, pts, self.cell_size)  # обновляем глобальную карту, используя облако точек
        
        self.last_depth = new_depth  # записываем последнюю карту глубины
        
        return self.global_map, self.x, self.y, self.ang
    
    
    def draw_local_map(self, create_fig=True, cell_size=None):
        """
        Выводит последнюю локальную карту
            create_fig : если True, то результат будет сразу выведен на экран, а если False, то будет отправлен в текущий fig, ax без вывода
            cell_size : кастомный cell_size, если хочется посмотреть на локальную карту в другой детализации
        """
        if cell_size == None:
            cell_size = self.cell_size
        
        pts = depth2ptsCloud(self.last_depth)
        local_map, zero_pos = ptsCloud2localMap(pts, cell_size)
        
        if create_fig:
            fig, ax = plt.subplots(figsize=(24, 10))
        
        plt.imshow(local_map, cmap='gray')
        plot_agent_on_map(zero_pos, local_map.shape[0]-1, 0, *local_map.shape)
        
        if create_fig:
            plt.show()
            
        
    def draw_pts_cloud_3d(self):
        """
        Выводит 3D изображение последнего облака точек
        """
        pts = depth2ptsCloud(self.last_depth)
        
        fig = plt.figure()
        ax = Axes3D(fig)

        # y, z, x because of 3d matplotlib axes _x /y |z
        ax.scatter(pts[:,1], pts[:,2], pts[:,0], s=0.01)
        plt.show()


    def draw_pts_cloud_full(self, create_fig=True):
        """
        Выводит облако точек (вид сверху) до обрезания уровня пола и уровня головы
            create_fig : если True, то результат будет сразу выведен на экран, а если False, то будет отправлен в текущий fig, ax без вывода
        """
        pts = depth2ptsCloud(self.last_depth)
        
        if create_fig:
            fig, ax = plt.subplots(figsize=(24, 10))
        
        plt.scatter(pts[:,1], pts[:,2],s=0.01)

        s_ = 1000
        x_ = np.linspace(np.min(pts[:,1]), np.max(pts[:,1]), s_)
        plt.plot(x_[x_<0], -0.78*x_[x_<0], 'g', alpha=0.5)
        plt.plot(x_[x_>0], 0.78*x_[x_>0], 'g', alpha=0.5)
        plt.plot(0, 0, 'r^', mew=5)
        
        if create_fig:
            plt.show()
    
    
    def draw_global_map(self, create_fig=True):
        """
        Выводит текущую глобальную карту
            create_fig : если True, то результат будет сразу выведен на экран, а если False, то будет отправлен в текущий fig, ax без вывода    
        """
        if create_fig:
            fig, ax = plt.subplots(figsize=(24, 10))
            
        plt.imshow(self.global_map, cmap='gray')
        plot_agent_on_map(self.x, self.y, self.ang, *self.global_map.shape)
            
        if create_fig:
            plt.show()
    
    
    def get_map(self):
        return self.global_map
    
    
    def get_pos(self):
        return (self.x, self.y, self.ang)