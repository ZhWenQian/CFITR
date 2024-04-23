import torch
import torch.nn as nn
import pytorch_lightning as pl
import model.modules.vision_transformer as vit
from .utils import LFEM
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from model.modules import heads, objectives, vilt_utils
from einops import rearrange
import math

TASK_OBJ_DICT = {
    "irtr_BE": objectives.compute_irtr_be,
    "irtr": objectives.compute_irtr,
}


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        # HASH
        self.image_module = nn.Sequential(
            nn.Linear(config["hidden_size"], config['hidden_dim'], bias=True),
            nn.BatchNorm1d(config['hidden_dim']),
            nn.ReLU(True),
            nn.Linear(config['hidden_dim'], config['hash_dim'], bias=True),
            nn.Tanh()
        )
        self.text_module = nn.Sequential(
            nn.Linear(config["hidden_size"], config['hidden_dim'], bias=True),
            nn.BatchNorm1d(config['hidden_dim']),
            nn.ReLU(True),
            nn.Linear(config['hidden_dim'], config['hash_dim'], bias=True),
            nn.Tanh()
        )
        self.hash_init = True
        self.max = 0
        self.beta = config['beta']
        self.gamma = config['gamma']

        # LFEM
        self.res = LFEM(768,768)

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)


        self.transformer = getattr(vit, self.hparams.config["vit"])(
            pretrained=False, config=self.hparams.config
        )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        self.itm_score = heads.ITMHead(config["hidden_size"])
        self.itm_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
                self.hparams.config["load_path"] != ""
                and not self.hparams.config["test_only"]
        ):
            print("#" * 40)
            print(f"Pretrained model loaded from: {self.hparams.config['load_path']}")
            print("#" * 40)
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False
        if self.hparams.config["loss_names"]["irtr_BE"] > 0:
            self.imagepooler_be = heads.AveragePooler(145)
            self.imagepooler_be.apply(objectives.init_weights)

            self.textpooler_be = heads.AveragePooler(40)
            self.textpooler_be.apply(objectives.init_weights)
        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
    
    def init_hashes(self):
        """
        Initialize hash values (either zeros or random, see below)

        :return: initialized hash values
        """
        dataset_size = len(self.trainer.datamodule.train_dataset)
        B = torch.randn(dataset_size, self.hparams.config['hash_dim']).sign().to(self.device)
        Hi = torch.zeros(dataset_size, self.hparams.config['hash_dim']).sign().to(self.device)
        Ht = torch.zeros(dataset_size, self.hparams.config['hash_dim']).sign().to(self.device)
        return B, Hi, Ht

    def get_embeddings(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"
        do_mlm = "_mlm" if mask_text else ""

        text_ids = batch[f"text_ids{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        ret = {
            "text_embeds": text_embeds,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "image_embeds": image_embeds,
            "image_labels": image_labels,
            "patch_index": patch_index,
            "image_masks": image_masks,
        }
        return ret

    def bi_encoders(self, batch):
        text_embeds, image_embeds = batch["text_embeds"], batch["image_embeds"]
        text_masks, image_masks = batch["text_masks"], batch["image_masks"]
        ret = batch

        # Image embed
        x = image_embeds

        for i in range(6):
            blk = self.transformer.blocks[i]
            x, _ = blk(x, mask=image_masks)
        ###################res block
        # img_cls, image_feats_block = x[:, 0], x[:, 1:]
        # img_cls = img_cls.unsqueeze(dim=1)
        # image_feats_res = self.res(image_feats_block)
        # x = torch.cat([img_cls, image_feats_res], dim=1)
        ####################

        image_feats = x
        ret["be_image_embeds"] = x
        image_feats = self.transformer.norm(image_feats)
        image_feats = self.imagepooler_be(image_feats)

        # Text embed
        x = text_embeds
        for i in range(6):
            blk = self.transformer.blocks[i]
            x, _ = blk(x, mask=text_masks)
        text_feats = x
        ret["be_text_embeds"] = x
        text_feats = self.transformer.norm(text_feats)
        text_feats = self.textpooler_be(text_feats)

        
        ret["be_image_feats"] = image_feats
        ret["be_text_feats"] = text_feats
        return ret

    def cross_encoders(self, batch):
        co_embeds = batch['co_embeds']
        co_masks = batch['co_masks']
        x = co_embeds

        for i in range(len(self.transformer.blocks)):
            if i <= 5:
                continue
            blk = self.transformer.blocks[i]
            x, _attn = blk(x, mask=co_masks)


        x = self.transformer.norm(x) 
        cls_feats = self.pooler(x)
        ret = batch
        ret["cls_feats"] = cls_feats
        return ret

    def infer_be(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
    ):

        batch = self.get_embeddings(
            batch,
            mask_text,
            mask_image,
            image_token_type_idx,
            image_embeds,
            image_masks,
        )

        ret = self.bi_encoders(batch)
        return ret
    
    def infer_hash(self, batch):
        h_img = self.image_module(batch['be_image_feats'])
        h_txt = self.text_module(batch['be_text_feats'])
        batch['h_img'] = h_img
        batch['h_txt'] = h_txt
        return batch

    def infer_ce(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
    ):
        ret = self.cross_encoders(batch)
        return ret

    def forward(self, batch):
        ret = dict()
        ret.update(objectives.compute_irtr_be(self, batch))
        batch = ret
        ret.update(objectives.compute_irtr_hash(self, batch))
        batch = ret
        ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def on_train_epoch_end(self, _outs):
        self.B = ((self.Hi +self.Ht) / 2).sign()
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        if self.hash_init:
            self.B, self.Hi, self.Ht = self.init_hashes()
            self.hash_init = False
            print('*#'*40)
        
        vilt_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_end(self):
        vilt_utils.epoch_wrapup(self)
        
    def test_step(self, batch, batch_idx):
        ret = dict()
        return ret

    def on_test_epoch_end(self):
        if self.hparams.config["save"]:
            vilt_utils.save_data(self)
        else:
            vilt_utils.corase_to_fine_retrieval(self)


    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
