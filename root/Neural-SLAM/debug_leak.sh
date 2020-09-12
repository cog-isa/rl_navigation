CUDA_VISIBLE_DEVICES=0 python main.py --max_episode_length 200 --split val \
	 -n 2 --num_processes_per_gpu 1 --auto_gpu_config 1 --exp_name test
