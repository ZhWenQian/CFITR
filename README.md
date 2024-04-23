# Fast two stage image-text retrieval for remote sensing images

# Dataset Preparation

We follow [ViLT](https://github.com/dandelin/ViLT) and use `pyarrow` to serialize the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.

# Training Stage

python run.py with data_root=<ARROW_ROOT> task_finetune_irtr_rsitmd_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> load_path="/x/vilt_200k_mlm_itm.ckpt"

# Testing Stage

python run.py with data_root=<ARROW_ROOT> task_finetune_irtr_rsitmd_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> test_only=True load_path=<TRAINED_MODEL>

The returned values are IR R@1, R@5, R@10 and TR R@1, R@5, R@10. (Only one gpu) 