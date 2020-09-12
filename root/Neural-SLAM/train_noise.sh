CUDA_VISIBLE_DEVICES=1,2,3 python main.py --split train \
	--load_global pretrained_models/model_best.global \
	--load_local pretrained_models/model_best.local \
	--load_slam pretrained_models/model_best.slam \
	--exp_name "noise-anm-vanila" --auto_gpu_config 1 --num_processes_per_gpu 12 --num_processes_on_first_gpu 10
