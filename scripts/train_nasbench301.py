# -*- coding: utf-8 -*-
# pylint: disable-all

import os
import sys
import shutil
import logging
import argparse
import random
import pickle

import yaml
from scipy.stats import stats

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import setproctitle
from torch.utils.data import Dataset, DataLoader

from my_nas import utils
from my_nas.common import get_search_space
from my_nas.evaluator.arch_network import ArchNetwork


class NasBench301Dataset(Dataset):
    def __init__(self, data, minus=None, div=None):
        self.data = data
        self._len = len(self.data)
        self.minus = minus
        self.div = div

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        data = self.data[idx]
        data = (data[0], data[1], data[2])
        if self.minus is not None:
            data = (data[0], data[1], data[2] - self.minus)
        if self.div is not None:
            data = (data[0], data[1], data[2] / self.div)
        return data

def model_update(model, args, archs, zs_p, accs):
    n = len(archs)
    n_max_pairs = int(args.max_compare_ratio * n)
    acc_diff = np.array(accs)[:, None] - np.array(accs)
    acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
    ex_thresh_inds = np.where(acc_abs_diff_matrix > args.compare_threshold)
    ex_thresh_num = len(ex_thresh_inds[0])
    if ex_thresh_num > n_max_pairs:
        if args.choose_pair_criterion == "diff":
            keep_inds = np.argpartition(acc_abs_diff_matrix[ex_thresh_inds], -n_max_pairs)[-n_max_pairs:]
        elif args.choose_pair_criterion == "random":
            keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
        ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
    archs_1, archs_2, better_lst = \
            (archs[ex_thresh_inds[1]], zs_p[ex_thresh_inds[1]]), \
            (archs[ex_thresh_inds[0]], zs_p[ex_thresh_inds[0]]), \
            (acc_diff > 0)[ex_thresh_inds]
        
    n_diff_pairs = len(better_lst)
    loss = model.update_compare(archs_1, archs_2, better_lst)

    return loss, n_diff_pairs

def train(train_loader, model, epoch, args, arch_network_type):
    objs = utils.AverageMeter()
    model.train()
    for step, train_data in enumerate(train_loader):
        train_archs = np.array(train_data[0])
        zs_p = np.array(train_data[1])
        train_accs = np.array(train_data[2])
       
        if args.compare:
            loss, n = model_update(model, args, train_archs, zs_p, train_accs)
        else:
            loss = model.update_predict((train_archs, zs_p), train_accs)
            n = len(train_archs)

        objs.update(loss, n)
        
    return objs.avg

def p_at_tb_k(predict_scores, true_scores, ratios=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]):
    predict_scores = np.array(predict_scores)
    true_scores = np.array(true_scores)
    predict_inds = np.argsort(predict_scores)[::-1]
    num_archs = len(predict_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    patks = []
    for ratio in ratios:
        k = int(num_archs * ratio)
        if k < 1:
            continue
        top_inds = predict_inds[:k]
        bottom_inds = predict_inds[num_archs-k:]
        p_at_topk = len(np.where(true_ranks[top_inds] < k)[0]) / float(k)
        p_at_bottomk = len(np.where(true_ranks[bottom_inds] >= num_archs - k)[0]) / float(k)
        kd_at_topk = stats.kendalltau(predict_scores[top_inds], true_scores[top_inds]).correlation
        kd_at_bottomk = stats.kendalltau(predict_scores[bottom_inds], true_scores[bottom_inds]).correlation
        # [ratio, k, P@topK, P@bottomK, KT in predicted topK, KT in predicted bottomK]
        patks.append((ratio, k, p_at_topk, p_at_bottomk, kd_at_topk, kd_at_bottomk))
    return patks

def minmax_n_at_k(predict_scores, true_scores, ks=[5, 10]):
    true_scores = np.array(true_scores)
    predict_scores = np.array(predict_scores)
    num_archs = len(true_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    predict_best_inds = np.argsort(predict_scores)[::-1]
    minn_at_ks = []
    for k in ks:
        ranks = true_ranks[predict_best_inds[:k]]
        if len(ranks) < 1:
            continue
        minn = int(np.min(ranks)) + 1
        maxn = int(np.max(ranks)) + 1
        minn_at_ks.append((k, minn, float(minn) / num_archs, maxn, float(maxn) / num_archs))
    return minn_at_ks

def valid(val_loader, model, args):
    model.eval()
    all_scores = []
    true_accs = []
    for step, (archs, zs_p, accs) in enumerate(val_loader):
        scores = list(model.predict((archs, zs_p)).cpu().data.numpy())
        all_scores += scores
        true_accs += list(accs)

    if args.save_predict is not None:
        with open(args.save_predict, "wb") as wf:
            pickle.dump((true_accs, all_scores), wf)

    corr = stats.kendalltau(true_accs, all_scores).correlation
    rg_loss = ((np.array(all_scores) - np.array(true_accs)) ** 2).mean()
    patk = p_at_tb_k(true_accs, all_scores)
    natk = minmax_n_at_k(true_accs, all_scores)

    patk = [x[2] for x in patk]
    natk = [x[1] for x in natk]
    return corr, rg_loss, patk, natk

def main(argv):
    parser = argparse.ArgumentParser(prog="train_nasbench301_pkl.py")
    parser.add_argument("cfg_file")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--report_freq", default=200, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--train-dir", default=None, help="Save train log/results into TRAIN_DIR")
    parser.add_argument("--save-every", default=10, type=int)
    parser.add_argument("--test-only", default=False, action="store_true")
    parser.add_argument("--test-funcs", default=None, help="comma-separated list of test funcs")
    parser.add_argument("--load", default=None, help="Load comparator from disk.")
    parser.add_argument("--eval-only-last", default=None, type=int,
                        help=("for pairwise compartor, the evaluation is slow,"
                              " only evaluate in the final epochs"))
    parser.add_argument("--save-predict", default=None, help="Save the predict scores")
    parser.add_argument("--train-pkl", default="", help="Training Datasets pickle")
    parser.add_argument("--valid-pkl", default="", help="Evaluate Datasets pickle")
    parser.add_argument("--train-ratio", default=None, type=float, help="Ratio of Training Datasets")
    args = parser.parse_args(argv)

    setproctitle.setproctitle("python train_nasbench301.py config: {}; train_dir: {}; cwd: {}"\
                              .format(args.cfg_file, args.train_dir, os.getcwd()))

    # log
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %I:%M:%S %p")

    if not args.test_only:
        assert args.train_dir is not None, "Must specificy `--train-dir` when training"
        # if training, setting up log file, backup config file
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        log_file = os.path.join(args.train_dir, "train.log")
        logging.getLogger().addFile(log_file)

        # copy config file
        backup_cfg_file = os.path.join(args.train_dir, "config.yaml")
        shutil.copyfile(args.cfg_file, backup_cfg_file)
    else:
        backup_cfg_file = args.cfg_file

    # cuda
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info("GPU device = %d" % args.gpu)
    else:
        logging.info("no GPU available, use CPU!!")

    if args.seed is not None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    search_space = get_search_space("nb301")
    logging.info("Load pkl cache from nasbench301_train.pkl and nasbench301_valid.pkl")
    with open(args.train_pkl, "rb") as rf:
        train_data = pickle.load(rf)
    with open(args.valid_pkl, "rb") as rf:
        valid_data = pickle.load(rf)

    train_data = [(data[0], data[1], data[2]) for data in train_data]
    valid_data = [(data[0], data[1], data[2]) for data in valid_data]

    with open(backup_cfg_file, "r") as cfg_f:
        cfg = yaml.load(cfg_f)

    logging.info("Config: %s", cfg)

    arch_network_type = cfg.get("arch_network_type", "pointwise_comparator")
    model_cls = ArchNetwork.get_class_(arch_network_type)
    model = model_cls(search_space, **cfg.pop("arch_network_cfg"))
    if args.load is not None:
        logging.info("Load %s from %s", arch_network_type, args.load)
        model.load(args.load)
    model.to(device)

    args.__dict__.update(cfg)
    logging.info("Combined args: %s", args)

    # init nasbench data loaders
    if hasattr(args, "train_ratio") and args.train_ratio is not None:
        _num = len(train_data)
        train_data = train_data[:int(_num * args.train_ratio)]
        logging.info("Train dataset ratio: %.3f", args.train_ratio)
    num_train_archs = len(train_data)
    logging.info("Number of architectures: train: %d; valid: %d", num_train_archs, len(valid_data))
    
    train_data = NasBench301Dataset(train_data, minus=cfg.get("dataset_minus", None),
                                    div=cfg.get("dataset_div", None))
    valid_data = NasBench301Dataset(valid_data, minus=cfg.get("dataset_minus", None),
                                    div=cfg.get("dataset_div", None))
    
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers,
        collate_fn=lambda items: list(zip(*items)))
    val_loader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers,
        collate_fn=lambda items: list(zip(*items)))

    # init test
    if not arch_network_type in {"pairwise_comparator", "random_forest"} or args.test_only:
        if args.test_funcs is not None:
            test_func_names = args.test_funcs.split(",")
        corr, _, _, _ = valid(val_loader, model, args)

    if args.test_only:
        return

    for i_epoch in range(1, args.epochs + 1):
        model.on_epoch_start(i_epoch)
        avg_loss = train(train_loader, model, i_epoch, args, arch_network_type)

        logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))
        if args.eval_only_last is None or (args.epochs - i_epoch < args.eval_only_last):
            corr, rg_loss, _, _ = valid(val_loader, model, args)
            logging.info("Valid: Epoch {:3d}: kendall tau {:.4f}; loss {:.4g}".format(
                                            i_epoch, corr, rg_loss))
        if i_epoch % args.save_every == 0:
            save_path = os.path.join(args.train_dir, "{}.ckpt".format(i_epoch))
            model.save(save_path)
            logging.info("Epoch {:3d}: Save checkpoint to {}".format(i_epoch, save_path))

    corr, rg_loss, patk, natk = valid(val_loader, model, args)
    logging.info(patk)
    logging.info(natk)

if __name__ == "__main__":
    main(sys.argv[1:])
