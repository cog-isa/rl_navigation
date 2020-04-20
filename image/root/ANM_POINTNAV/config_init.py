import fileinput
import habitat
from habitat_baselines.config.default import get_config





filename = "/habitat-api/habitat_baselines/config/pointnav/ddppo_pointnav.yaml"
with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace('\"configs/tasks/pointnav_gibson.yaml\"', '\"/habitat-api/configs/tasks/pointnav_gibson.yaml\"'), end='')
with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:        
    for line in file:    
        print(line.replace('GLOO', 'NCCL'), end='')
        
def get_config_init():
    W = 256#640
    H = 256#360
    config_paths="/data/challenge_pointnav2020.local.rgbd.yaml"
    config = habitat.get_config(config_paths=config_paths)
    config.defrost()
    config.SIMULATOR.RGB_SENSOR.HEIGHT = H
    config.SIMULATOR.RGB_SENSOR.WIDTH = W
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = H
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = W
    config.DATASET.DATA_PATH = '/data/v1/{split}/{split}.json.gz'
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS = ["HEADING_SENSOR", "COMPASS_SENSOR", "GPS_SENSOR", "POINTGOAL_SENSOR", "POINTGOAL_WITH_GPS_COMPASS_SENSOR"]
    config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 3
    config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "CARTESIAN"
    config.TASK.POINTGOAL_SENSOR.DIMENSIONALITY = 3
    config.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "CARTESIAN"
    config.TASK.GPS_SENSOR.DIMENSIONALITY = 3
    config.TASK.GPS_SENSOR.GOAL_FORMAT = "CARTESIAN"
    config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
    config.TASK.AGENT_POSITION_SENSOR.TYPE = "position_sensor"
    config.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
    config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
    config.SIMULATOR.TURN_ANGLE = 0.5
    config.SIMULATOR.TILT_ANGLE = 0.5
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.05
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 500*20
    config.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = 500*20
    config.DATASET.SCENES_DIR = '/data'
    config.DATASET.SPLIT = 'train'
    config.SIMULATOR.SCENE = '/data/gibson/Aldrich.glb'
    #config.SIMULATOR_GPU_ID = 0
    config.freeze()
    
    config2 = get_config('/habitat-api/habitat_baselines/config/pointnav/ddppo_pointnav.yaml', [])
    ii = 0
    for i in config2.TASK_CONFIG.keys():
        config2.TASK_CONFIG[i] = config[i]
        ii+=1
    config = config2    
    
    config.defrost()
    config.TASK_CONFIG.DATASET.DATA_PATH = '/data/v1/{split}/{split}.json.gz'
    config.TASK_CONFIG.DATASET.SCENES_DIR = '/data'
    config.TASK_CONFIG.DATASET.SPLIT = 'train'
    config.TASK_CONFIG.SIMULATOR.SCENE = '/data/gibson/Aldrich.glb'
    config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID = 'pos'#'pointgoal_with_gps_compass'
    config.NUM_UPDATES = 50000
    config.ENV_NAME = 'MyRLEnvNew'
    config.NUM_PROCESSES = 4
    config.RL.FRAMESKIP = 20
    config.RL.FRAMESTACK = 1
    config.freeze()
    
    return config