# Towards Efficient and Accurate Remote sensing Image-Text Retrieval with a Coarse-to-Fine Approach

# Download Pretrained Weights

We leverage the pretrained weight from ViLT as ICML 2021 (long talk) paper: "[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)" 

ViLT-B/32 Pretrained with MLM+ITM for 200k steps on GCC+SBU+COCO+VG (ViLT-B/32 200k) [this link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt)


# Dataset Preparation

We follow [ViLT](https://github.com/dandelin/ViLT) and use `pyarrow` to serialize the datasets. The resulted files can also be downloaded from here: [[Baidu Pan](https://pan.baidu.com/s/1YrWcz090kdqOZ0lrbqXJJA) (code:nq9y)]. Extract it to `./data/`.

# Training Stage

python run.py with data_root=<ARROW_ROOT> task_ft_irtr_be_ce per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> load_path="/x/vilt_200k_mlm_itm.ckpt"

Examples

```
python run.py with task_ft_irtr_be_ce per_gpu_batchsize=8
```
You can download our pretrained model here: [[Baidu Pan](https://pan.baidu.com/s/1YrWcz090kdqOZ0lrbqXJJA) (code:nq9y)].

# Testing Stage

## Save data

python run.py with data_root=<ARROW_ROOT> task_ft_irtr_be_ce per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> test_only=True load_path=<TRAINED_MODEL> save=True

Examples:

```
python run.py with task_ft_irtr_be_ce per_gpu_batchsize=8 load_path="/root/autodl-tmp/CFDS-ITR/result/FT_irtr_BE_CE_seed0_from_vilt_200k_mlm_itm/version_0/checkpoints/last.ckpt" test_only=True save=True
```

## reference 

python run.py with data_root=<ARROW_ROOT> task_ft_irtr_be_ce per_gpu_batchsize=<BS_FITS_YOUR_GPU> num_gpus=1 num_nodes=1 precision=<PRECISION, as 32 or 16> test_only=True load_path=<TRAINED_MODEL> save=False

Examples:

```
python run.py with task_ft_irtr_be_ce per_gpu_batchsize=8 load_path="/root/autodl-tmp/CFDS-ITR/result/FT_irtr_BE_CE_seed0_from_vilt_200k_mlm_itm/version_0/checkpoints/last.ckpt" test_only=True save=False
```

The returned values are IR R@1, R@5, R@10 and TR R@1, R@5, R@10. (Only one gpu) 
