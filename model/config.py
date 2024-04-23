from sacred import Experiment

ex = Experiment("ViLT")


def _loss_names(d):
    ret = {
        "irtr": 0,
        "irtr_BE": 0,
        "hash":0
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vilt"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32  #32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Hash Setting
    hidden_dim = 1024 * 4
    hash_dim = 128
    beta = 0.001
    gamma = 0.01

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    json = "/root/autodl-tmp/CFDS-ITR/model/data/rsicd/dataset_RSICD.json"
    recall_mode = ""

    # save embeddings and hash codes
    save = True
    save_path = '/root/autodl-tmp/CFDS-ITR/model/precompute_data'
    image_file = r'/root/autodl-tmp/CFDS-ITR/model/precompute_data_v7/image_embed.h5'
    image_hash_file = r'/root/autodl-tmp/CFDS-ITR/model/precompute_data_v7/image_hash_code.h5'
    text_file = r'/root/autodl-tmp/CFDS-ITR/model/precompute_data_v7/text_embed.h5'
    text_hash_file = r'/root/autodl-tmp/CFDS-ITR/model/precompute_data_v7/text_hash_code.h5'

# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def env_dandelin():
    data_root = "/data2/dsets/dataset"
    log_dir = "/data2/vilt/result"
    num_gpus = 1
    num_nodes = 1


# ##conbine two stages
@ex.named_config
def task_ft_irtr_be_ce():
    data_root = "/root/autodl-tmp/CFDS-ITR/model/data/rsicd"
    exp_name = "FT_irtr_BE_CE"
    datasets = ["rsicd"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"irtr_BE": 1, "irtr": 1, 'hash':1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15  # 先都将其设置为0
    learning_rate = 1e-4
    val_check_interval = 1.0  #0.5
    json = "/root/autodl-tmp/CFDS-ITR/model/data/rsicd/dataset_RSICD.json"
    load_path = "/root/autodl-tmp/CFDS-ITR/model/data/vilt_200k_mlm_itm.ckpt"
    recall_mode = "be_ce"