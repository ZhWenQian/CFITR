import functools
import glob
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from einops import rearrange
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler

from model.modules.dist_utils import all_gather
import math
import h5py
import numpy as np
import sys
import time


############# for bi-encoder ########################
def cost_matrix_cosine(x, y, eps=1e-5):
    """
    Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]
    """
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs"""
    im = l2norm(im, dim=-1)
    s = l2norm(s, dim=-1)
    w12 = im.mm(s.t())
    return w12


def calcul_loss(scores, size, margin, max_violation=False):
    diagonal = scores.diag().view(size, 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    mask = torch.eye(scores.size(0)) > 0.5
    I = Variable(mask)
    if torch.cuda.is_available():
        I = I.cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

    return cost_s.mean() + cost_im.mean()


############################################
def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


def compute_irtr_be(pl_module, batch):
    margin = 0.2  
    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)

    infer = pl_module.infer_be(
        {
            "image": batch["image"],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
        }
    )

    be_text_feats = infer['be_text_feats']
    be_text_feats = rearrange(be_text_feats, "(bs fs) l -> bs fs l", bs=_bs, fs=false_len + 1)
    be_text_feats = be_text_feats[:, 0, :].squeeze()

    be_image_embeds = infer['be_image_embeds']
    _, c, l = be_image_embeds.shape
    be_image_embeds = be_image_embeds.unsqueeze(1).expand(_bs, false_len + 1, c, l)
    be_image_embeds = rearrange(be_image_embeds, "bs fs c l -> (bs fs) c l")
    be_image_masks = infer['image_masks']
    be_image_masks = be_image_masks.unsqueeze(1).expand(_bs, false_len + 1, c)
    be_image_masks = rearrange(be_image_masks, "bs fs l -> (bs fs) l")

    text_masks = rearrange(text_masks, "bs fs tl -> (bs fs) tl")

    co_embeds = torch.cat([infer['be_text_embeds'], be_image_embeds], dim=1)
    co_masks = torch.cat([text_masks, be_image_masks], dim=1)

    ret = {
        "be_text_feats": be_text_feats,
        "be_image_feats": infer["be_image_feats"],
        "co_embeds": co_embeds,
        "co_masks": None,
        "co_masks": co_masks,
        'raw_index': batch['raw_index']  # try
    }

    return ret


def calc_score_from_batch(pl_module, txt_batch, ie, im, mode):
    """
    mode: ce (cross encoder) / be (bi-encoder)
    """
    if mode == "be":
        batch = pl_module.infer_be(
            {
                "text_ids": txt_batch["text_ids"],
                "text_masks": txt_batch["text_masks"],
                "text_labels": txt_batch["text_labels"],
            },
            image_embeds=ie,
            image_masks=im,
        )
        score = cosine_sim(
            batch["be_image_feats"][0].unsqueeze(0), batch["be_text_feats"]
        ).squeeze(0)
    elif mode == "ce":
        batch = pl_module.infer_ce(
            {
                "text_ids": txt_batch["text_ids"],
                "text_masks": txt_batch["text_masks"],
                "text_labels": txt_batch["text_labels"],
            },
            image_embeds=ie,
            image_masks=im,
        )
        score = pl_module.rank_output(batch["cls_feats"])[:, 0]
    else:
        raise NotImplementedError
    return score


####################################################
def calculate_losses(self, h_img, h_txt, ind):
    """
    Calculate losses
    :param: h_img1: batch of image hashes #1 (original)
    :param: h_img2: batch of image hashes #2 (augmented)
    :param: h_txt1: batch of text hashes #1 (original)
    :param: h_txt2: batch of text hashes #2 (augmented)
    :param: ind: indexes of samples in current batch

    :returns: total epoch loss and updated dictionary with epoch losses
    """

    loss_quant = calc_quantization_loss(self, h_img, h_txt, ind) * self.beta
    loss_bb = calc_bit_balance_loss(h_img, h_txt) * self.gamma

    err = loss_quant + loss_bb

    return err


def calc_ntxent_loss(h_img, h_txt):
    loss_ntxent = contr_loss(h_img, h_txt)
    return loss_ntxent


def contr_loss(out_1, out_2, temperature=1, eps=1e-6):
    out_1 = F.normalize(out_1)
    out_2 = F.normalize(out_2)

    out = torch.cat([out_1, out_2], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^1 to remove similarity measure for x1.x1
    row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).sum()
    return loss


def calc_quantization_loss(self, h_img, h_txt, ind):
    """
    Calculate Quantization Loss

    :param: h_img1: batch of image hashes #1 (original)
    :param: h_img2: batch of image hashes #2 (augmented)
    :param: h_txt1: batch of text hashes #1 (original)
    :param: h_txt2: batch of text hashes #2 (augmented)

    :returns: Quantization Loss
    """
    loss_quant_img = torch.sum(torch.pow(self.B[ind, :] - h_img, 2))
    loss_quant_txt = torch.sum(torch.pow(self.B[ind, :] - h_txt, 2))
    loss_quant = (loss_quant_img + loss_quant_txt)
    return loss_quant


def calc_bit_balance_loss(h_img, h_txt):
    """
    Calculate Bit Balance Loss

    :param: h_img1: batch of image hashes #1 (original)
    :param: h_img2: batch of image hashes #2 (augmented)
    :param: h_txt1: batch of text hashes #1 (original)
    :param: h_txt2: batch of text hashes #2 (augmented)

    :returns: Bit Balance Loss
    """
    loss_bb_img = torch.sum(torch.pow(torch.sum(h_img, dim=1), 2))
    loss_bb_txt = torch.sum(torch.pow(torch.sum(h_txt, dim=1), 2))
    loss_bb = (loss_bb_img + loss_bb_txt)
    return loss_bb


def compute_irtr_hash(pl_module, batch):
    margin = 0.2 
    _bs, _ = batch["be_image_feats"].shape

    infer = pl_module.infer_hash(batch)
    score = cosine_sim(infer["h_img"], infer["h_txt"])
    _loss = calcul_loss(
        score,
        _bs,
        margin=margin,
        max_violation=True,
    )
    hash_loss = calculate_losses(pl_module, infer['h_img'], infer['h_txt'], batch['raw_index'])
    ret = {
        "hash_loss": hash_loss + _loss,
    }

    phase = "train" if pl_module.training else "val"
    hash_loss = getattr(pl_module, f"{phase}_hash_loss")(ret["hash_loss"])

    pl_module.log(f"irtr/{phase}/hash_loss", hash_loss, prog_bar=True)

    return ret


def compute_irtr(pl_module, batch):
    _bs, _, _ = batch["co_embeds"].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    bs = _bs // false_len
    infer = pl_module.infer_ce(batch)
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]  # 128
    score = rearrange(score, "(bs fs) -> bs fs", bs=bs, fs=false_len + 1)  # ([8, 16])
    answer = torch.zeros(bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss, prog_bar=True)

    return ret


def calc_hamming_dist(B1, B2):
    """
    Hamming distance

    :param B1: binary codes
    :param B2: binary codes
    :return:
    """
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = -0.5 * (q - B1.mm(B2.t()))
    return distH


def get_nn_avg_dist(scores, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    all_distances_1, _ = scores.topk(knn, dim=1, largest=True, sorted=True)
    average_dist1 = torch.mean(all_distances_1, dim=1).unsqueeze_(1)
    all_distances_2, _ = scores.topk(knn, dim=0, largest=True, sorted=True)
    average_dist2 = torch.mean(all_distances_2, dim=0).unsqueeze_(0)
    average_dist1 = average_dist1.expand(scores.shape)
    average_dist2 = average_dist2.expand(scores.shape)
    scores = scores * 2 - average_dist1 - average_dist2
    return scores


@torch.no_grad()
def compute_irtr_recall(pl_module, phase):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    all_repeat_text = text_dset.get_single_text()

    json_path = pl_module.hparams.config["json"]
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = SequentialSampler(image_dset)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    need_text_all = list()

    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
                "text": _b['text']
            }
        )

        for t_ in range(len(_b["text"])):
            need_text_all.append([_b["text"][t_]])

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        (ie, im, _, _) = pl_module.transformer.visual_embed(
            _b["image"][0].to(pl_module.device),
            max_image_len=pl_module.hparams.config["max_image_len"],
            mask_it=False,
        )
        image_preload.append((ie, im, _b["img_index"][0]))

    be_rank_scores = list()
    be_rank_iids = list()
    ce_rank_scores = list()
    ce_rank_iids = list()
    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape

        be_img_batch_score = list()
        ce_img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, l)

            with torch.cuda.amp.autocast():
                batch = pl_module.infer_be(
                    {
                        "text_ids": txt_batch["text_ids"],
                        "text_masks": txt_batch["text_masks"],
                        "text_labels": txt_batch["text_labels"],
                    },
                    image_embeds=ie,
                    image_masks=im,
                )
                infer = pl_module.infer_hash(batch)
                Bi = torch.sign(infer["h_img"][0].unsqueeze(0))
                Bt = torch.sign(infer["h_txt"])
                be_score = calc_hamming_dist(Bi, Bt).squeeze(0)
                co_embeds = torch.cat([batch['be_text_embeds'], batch['be_image_embeds']], dim=1)
                co_masks = torch.cat([batch["text_masks"], batch['image_masks']], dim=1)
                batch = pl_module.infer_ce({
                    "co_embeds": co_embeds,
                    "co_masks": co_masks
                })
                ce_score = pl_module.rank_output(batch["cls_feats"])[:, 0]
            be_img_batch_score.append(be_score)
            ce_img_batch_score.append(ce_score)

        be_img_batch_score = torch.cat(be_img_batch_score)
        ce_img_batch_score = torch.cat(ce_img_batch_score)
        be_rank_scores.append(be_img_batch_score.cpu().tolist())
        ce_rank_scores.append(ce_img_batch_score.cpu().tolist())
        be_rank_iids.append(_iid)
        ce_rank_iids.append(_iid)
    ############################################################################################
    be_ir_tr = ir_tr_topk(be_rank_scores, be_rank_iids, tiids)
    Evaluation_Metrics(pl_module, be_ir_tr, 'be')
    ce_ir_tr = ir_tr_topk(ce_rank_scores, ce_rank_iids, tiids)
    Evaluation_Metrics(pl_module, ce_ir_tr, 'ce')


@torch.no_grad()
def precompute_data(pl_module, phase):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer

    json_path = pl_module.hparams.config["json"]
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = SequentialSampler(image_dset)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    text_ids_all = list()
    text_masks_all = list()
    text_labels_all = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_ids_all.append(_b["text_ids"])
        text_masks_all.append(_b["text_masks"])
        text_labels_all.append(_b["text_labels"])
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
                "text": _b['text']
            }
        )

    text_ids_all = torch.cat(text_ids_all)
    text_masks_all = torch.cat(text_masks_all).to(pl_module.device)
    text_labels_all = torch.cat(text_labels_all)

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    im_all = list()
    ie_all = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        (ie, im, _, _) = pl_module.transformer.visual_embed(
            _b["image"][0].to(pl_module.device),
            max_image_len=pl_module.hparams.config["max_image_len"],
            mask_it=False,
        )
        image_preload.append((ie, im, _b["img_index"][0]))
        im_all.append(im)
        ie_all.append(ie)
    im_all = torch.cat(im_all)
    ie_all = torch.cat(ie_all)

    the_metric = 0
    rank_scores = list()
    rank_iids = list()
    im_embeds_all = list()
    im_hash_code = list()
    for img_batch in tqdm.tqdm(image_preload, desc="first stage rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape

        img_batch_score = list()
        text_embeds_all = list()
        text_hash_code = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            with torch.cuda.amp.autocast():
                batch = pl_module.infer_be(
                    {
                        "text_ids": txt_batch["text_ids"].to(pl_module.device),
                        "text_masks": txt_batch["text_masks"].to(pl_module.device),
                        "text_labels": txt_batch["text_labels"].to(pl_module.device),
                    },
                    image_embeds=_ie,
                    image_masks=_im,
                )
                infer = pl_module.infer_hash(batch)
                Bi = torch.sign(infer["h_img"][0].unsqueeze(0))
                Bt = torch.sign(infer["h_txt"])
                end = time.time()
                score = calc_hamming_dist(Bi, Bt).squeeze(0)
            img_batch_score.append(score)
            text_embeds_all.append(batch["be_text_embeds"])
            text_hash_code.append(Bt)
        text_embeds_all = torch.concat(text_embeds_all)
        text_hash_code = torch.concat(text_hash_code)
        im_embeds_all.append(batch["be_image_embeds"])
        im_hash_code.append(Bi)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    im_embeds_all = torch.cat(im_embeds_all)
    im_hash_code = torch.cat(im_hash_code)
    gather_rank_iids = all_gather(rank_iids)
    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
  
    image_embed = im_embeds_all.cpu().detach().numpy()
    text_embed = text_embeds_all.cpu().detach().numpy()
    image_hash_code = im_hash_code.cpu().detach().numpy()
    text_hash_code = text_hash_code.cpu().detach().numpy()
    image_iid = iids.cpu().detach().numpy()
    text_iid = tiids.cpu().detach().numpy()
    image_mask = im_all.cpu().detach().numpy()
    text_mask = text_masks_all.cpu().detach().numpy()

    # save
    pre_data_path = pl_module.hparams.config["save_path"]
    if not os.path.exists(pre_data_path):
        os.makedirs(pre_data_path)
    # write_hdf5(pre_data_path, 'image_embed', image_embed.astype(np.float32), image_iid.astype(np.int64), image_mask.astype(np.int64))
    # write_hdf5(pre_data_path, 'image_hash_code', image_hash_code.astype(np.float16), image_iid.astype(np.int16))
    # write_hdf5(pre_data_path,'text_embed', text_embed.astype(np.float32), text_iid.astype(np.int64), text_mask.astype(np.int64))
    # write_hdf5(pre_data_path,'text_hash_code', text_hash_code.astype(np.float16), text_iid.astype(np.int16))


def ir_tr_topk(rank_scores, rank_iids, tiids):
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    # scores = get_nn_avg_dist(scores, 10)

    topk10 = scores.topk(10, dim=1, sorted=True)
    topk5 = scores.topk(5, dim=1, sorted=True)
    topk1 = scores.topk(1, dim=1, sorted=True)

    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0, sorted=True)
    topk5 = scores.topk(5, dim=0, sorted=True)
    topk1 = scores.topk(1, dim=0, sorted=True)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1 * 100, ir_r5 * 100, ir_r10 * 100, tr_r1 * 100, tr_r5 * 100, tr_r10 * 100)


def Evaluation_Metrics(pl_module, ir_tr, name):
    the_metric = 0
    phase = "train" if pl_module.training else "val"
    (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = ir_tr
    all = sum([ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10]) / 6
    mean = round(all.item(), 2)
    ir_r1 = round(ir_r1.item(), 2)
    ir_r5 = round(ir_r5.item(), 2)
    ir_r10 = round(ir_r10.item(), 2)
    tr_r1 = round(tr_r1.item(), 2)
    tr_r5 = round(tr_r5.item(), 2)
    tr_r10 = round(tr_r10.item(), 2)

    print("################ {} Evaluation Metrics  ################".format(name))
    print(
        "ir_top1: {}%".format(ir_r1),
        ", ir_top5: {}%".format(ir_r5),
        ", ir_top10: {}%".format(ir_r10),
        ", tr_top1: {}%".format(tr_r1),
        ", tr_top5: {}%".format(tr_r5),
        ", tr_top10: {}%".format(tr_r10),
        " and this is mean: {}%".format(mean),
        pl_module.global_step)
    print("\n")

    pl_module.logger.experiment.add_scalar(
        "recalls/ir_r1", ir_r1, pl_module.global_step
    )
    pl_module.logger.experiment.add_scalar(
        "recalls/ir_r5", ir_r5, pl_module.global_step
    )
    pl_module.logger.experiment.add_scalar(
        "recalls/ir_r10", ir_r10, pl_module.global_step
    )
    pl_module.logger.experiment.add_scalar(
        "recalls/tr_r1", tr_r1, pl_module.global_step
    )
    pl_module.logger.experiment.add_scalar(
        "recalls/tr_r5", tr_r5, pl_module.global_step
    )
    pl_module.logger.experiment.add_scalar(
        "recalls/tr_r10", tr_r10, pl_module.global_step
    )
    the_metric += ir_r1 + tr_r1

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        value = 0
        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "nlvr2":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()

                value = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()
        elif loss_name == "irtr":
            pl_module.log(
                f"{loss_name}/{phase}/irtr_loss_epoch",
                getattr(pl_module, f"{phase}_irtr_loss").compute(),
            )
            getattr(pl_module, f"{phase}_irtr_loss").reset()
        elif loss_name == "mppd" or loss_name == "mpfr":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            pl_module.log(
                f"{loss_name}/{phase}/wpa_loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").reset()
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)


def ir_top_k(text_idx, relate, img_idx, k):
    img_idx = img_idx.permute(1, 0)
    batch, _ = img_idx.shape
    img_idx = img_idx[:, :k]
    top_sum = 0
    for i in range(batch):
        for j in range(k):
            img_index = img_idx[i][j]
            all_relation_index = text_idx[i]
            all_relation_index = "{}".format(all_relation_index.item())
            if img_index in relate[all_relation_index]:
                top_sum += 1
                break
    return top_sum / batch


def tr_top_k(img_idx, relate, idx, k):
    text_idx = idx[:, :k]

    batch, _ = text_idx.shape
    top_sum = 0
    for i in range(batch):
        flag = 0
        for j in range(k):
            if flag == 0:
                rep_text = text_idx[i][j]
                all_relation_index = "{}".format(rep_text.item())
                all_relation = relate[all_relation_index]

                if img_idx[0, i] in all_relation:
                    top_sum += 1
                    flag = 1

    return top_sum / batch


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")


def write_hdf5(out_file,dataset_name, data, iid, mask=None):
    """
    Write to h5 file

    :param out_file: file name
    :param data: data to write
    :return:
    """
    out_file = os.path.join(out_file, dataset_name + '.h5')
    with h5py.File(out_file, 'w') as hf:
        print("Saved as '.h5' file to", out_file)
        hf.create_dataset(dataset_name, data=data)
        hf.create_dataset(dataset_name+'_iid', data=iid)
        if mask is not None:
            hf.create_dataset(dataset_name+'_mask', data=mask)


def read_hdf5(file_name, dataset_name):
    with h5py.File(file_name, 'r') as hf:
        print("Read from:", file_name)
        datasets = list(hf.keys())
        image_embed = torch.from_numpy(hf[datasets[0]][:]).cuda()
        iid = torch.from_numpy(hf[datasets[1]][:]).cuda()
        if len(datasets)==3:
            mask = torch.from_numpy(hf[datasets[2]][:]).cuda()
            return image_embed, iid, mask
        else:
            return image_embed, iid
    

def shard_dis(images, captions, shard_size=128, lengths=None):
    """compute image-caption pairwise distance during validation and test"""

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))

        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            Bi = images[im_start:im_end].float()
            Bt = captions[cap_start:cap_end].float()
            hamming_distance = calc_hamming_dist(Bi, Bt)
            d[im_start:im_end, cap_start:cap_end] = hamming_distance.data.cpu().numpy()
    sys.stdout.write('\n')
    return torch.from_numpy(d)


def two_stage_retrieval(pl_module):
    image_file = pl_module.hparams.config["image_file"]
    image_hash_file = pl_module.hparams.config["image_hash_file"]
    text_file = pl_module.hparams.config["text_file"]
    text_hash_file = pl_module.hparams.config["text_hash_file"]

    image_embeds, iids, image_masks = read_hdf5(image_file, 'image_embed')
    image_hash_codes, _ = read_hdf5(image_hash_file, 'image_hash_code')
    text_embeds, tiids, text_masks = read_hdf5(text_file, 'text_embed')
    text_hash_codes, _ = read_hdf5(text_hash_file, 'text_hash_code')
    first_stage_scores = shard_dis(image_hash_codes, text_hash_codes)
    first_stage_scores = get_nn_avg_dist(first_stage_scores, 10)

    # topK###################################################################
    # i2t
    K = 70
    topK = first_stage_scores.topk(K, dim=1, sorted=True)
    topK_iids = tiids[topK.indices]

    rank_scores = list()
    _, l, c = image_embeds.shape
    for i in tqdm.tqdm(range(len(image_embeds)), desc="second stage i2t rank loop"):
        fblen = K
        image_embed = torch.unsqueeze(image_embeds[i], 0).expand(fblen, l, c)
        image_mask = torch.unsqueeze(image_masks[i], 0).expand(fblen, l)
        text_embed = text_embeds[topK.indices[i]]
        text_mask = text_masks[topK.indices[i]]
        co_embeds = torch.cat([text_embed, image_embed], dim=1)
        co_masks = torch.cat([text_mask, image_mask], dim=1)
        with torch.cuda.amp.autocast():
            batch = pl_module.infer_ce({
                "co_embeds": co_embeds,
                "co_masks": co_masks
            })
            score = pl_module.rank_output(batch["cls_feats"])[:, 0]
        rank_scores.append(score.cpu().tolist())

    gather_rank_scores = all_gather(rank_scores)

    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    # 重排
    topk10 = scores.topk(10, dim=1, sorted=True)
    topk5 = scores.topk(5, dim=1, sorted=True)
    topk1 = scores.topk(1, dim=1, sorted=True)

    topk10_iids = torch.tensor([topK_iids[i][topk10.indices[i]].tolist() for i in range(len(topK_iids))]).cuda()
    topk5_iids = torch.tensor([topK_iids[i][topk5.indices[i]].tolist() for i in range(len(topK_iids))]).cuda()
    topk1_iids = torch.tensor([topK_iids[i][topk1.indices[i]].tolist() for i in range(len(topK_iids))]).cuda()

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()
    
    ##########################################t2i##################################################
    topK = first_stage_scores.topk(K, dim=0, sorted=True)
    topK_iids = iids[topK.indices].T  # (100,K)

    rank_scores = list()
    _, l, c = text_embeds.shape
    for i in tqdm.tqdm(range(len(text_embeds)), desc="second stage t2i rank loop"):
        fblen = K
        text_embed = torch.unsqueeze(text_embeds[i], 0).expand(fblen, l, c)
        text_mask = torch.unsqueeze(text_masks[i], 0).expand(fblen, l)
        image_embed = image_embeds[topK_iids[i]]
        image_mask = image_masks[topK_iids[i]]
        co_embeds = torch.cat([text_embed, image_embed], dim=1)
        co_masks = torch.cat([text_mask, image_mask], dim=1)
        with torch.cuda.amp.autocast():
            batch = pl_module.infer_ce({
                "co_embeds": co_embeds,
                "co_masks": co_masks
            })
            score = pl_module.rank_output(batch["cls_feats"])[:, 0]
        rank_scores.append(score.cpu().tolist())

    gather_rank_scores = all_gather(rank_scores)

    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(tiids), -1)

    # 重排
    topk10 = scores.topk(10, dim=1, sorted=True)
    topk5 = scores.topk(5, dim=1, sorted=True)
    topk1 = scores.topk(1, dim=1, sorted=True)

    topk10_iids = torch.tensor([topK_iids[i][topk10.indices[i]].tolist() for i in range(len(topK_iids))]).cuda()
    topk5_iids = torch.tensor([topK_iids[i][topk5.indices[i]].tolist() for i in range(len(topK_iids))]).cuda()
    topk1_iids = torch.tensor([topK_iids[i][topk1.indices[i]].tolist() for i in range(len(topK_iids))]).cuda()

    # print("this is the shape of topk10_iids:{}".format(topk10_iids.shape))
    ir_r10 = (tiids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    ir_r5 = (tiids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    ir_r1 = (tiids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    ir_r1 = ir_r1 * 100
    ir_r5 = ir_r5 * 100
    ir_r10 = ir_r10 * 100
    tr_r1 = tr_r1 * 100
    tr_r5 = tr_r5 * 100
    tr_r10 = tr_r10 * 100

    all = sum([ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10]) / 6
    mean = round(all.item(), 2)
    ir_r1 = round(ir_r1.item(), 2)
    ir_r5 = round(ir_r5.item(), 2)
    ir_r10 = round(ir_r10.item(), 2)
    tr_r1 = round(tr_r1.item(), 2)
    tr_r5 = round(tr_r5.item(), 2)
    tr_r10 = round(tr_r10.item(), 2)

    print(
        "ir_top1: {}%".format(ir_r1),
        ", ir_top5: {}%".format(ir_r5),
        ", ir_top10: {}%".format(ir_r10),
        ", tr_top1: {}%".format(tr_r1),
        ", tr_top5: {}%".format(tr_r5),
        ", tr_top10: {}%".format(tr_r10),
        " and this is mean: {}%".format(mean),
        pl_module.global_step)

