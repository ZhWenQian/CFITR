cd two_stage_fast_retrieval

#train
python run.py with task_ft_irtr_be_ce per_gpu_batchsize=8
# nohup python run.py with task_ft_irtr_be_ce per_gpu_batchsize=8 &  


# test
# save data
python run.py with task_ft_irtr_be_ce per_gpu_batchsize=8 load_path="/root/autodl-tmp/CFDS-ITR/result/FT_irtr_BE_CE_seed0_from_vilt_200k_mlm_itm/version_7/checkpoints/last.ckpt" test_only=True save=True
# nohup python run.py with task_ft_irtr_be_ce per_gpu_batchsize=8 load_path="/root/autodl-tmp/CFDS-ITR/result/FT_irtr_BE_CE_seed0_from_vilt_200k_mlm_itm/version_1/checkpoints/last.ckpt" test_only=True &
# reference
python run.py with task_ft_irtr_be_ce per_gpu_batchsize=8 load_path="/root/autodl-tmp/CFDS-ITR/result/FT_irtr_BE_CE_seed0_from_vilt_200k_mlm_itm/version_7/checkpoints/last.ckpt" test_only=True save=False