import habitat


def make_env_func(conf:habitat.Config, dataset, rank:int):
    env = habitat.Env(config=conf, dataset=dataset)
    env.seed(rank)
    return env

def create_envs(num_processes, config):
    num_processes = num_processes
    config = config
    configs = []

    dataset = habitat.make_dataset(config.DATASET.TYPE, config=config.DATASET)
    datasets = dataset.get_splits(num_processes)
    configs = [config] * num_processes

    vecenv = habitat.VectorEnv(make_env_fn=make_env_func,
                               env_fn_args=tuple(tuple(zip(configs, datasets, range(num_processes)))),
                               multiprocessing_start_method='fork')
    return vecenv