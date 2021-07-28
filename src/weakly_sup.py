# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import collections
import copy
import dotdict
import json
import numpy as np
import os
import random
import regex
import tempfile
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
from models import LexiconInducer
from evaluate import evaluate


cos = nn.CosineSimilarity(dim=-1)


def setup_configs(configs):
    configs.save_path = configs.save_path.format(src=configs.src_lang, trg=configs.trg_lang)
    configs.stats_path = configs.save_path + '/stats.pt'


def collect_bitext_stats(bitext_path, align_path, save_path, is_reversed=False):
    stats_path = save_path + '/stats.pt'
    freq_path = save_path + '/freqs.pt'
    if os.path.exists(stats_path):
        coocc, semi_matched_coocc, matched_coocc = torch.load(stats_path)
    else:
        coocc = collections.defaultdict(collections.Counter)
        semi_matched_coocc = collections.defaultdict(collections.Counter)
        matched_coocc = collections.defaultdict(collections.Counter)
        tmpdir = tempfile.TemporaryDirectory()
        os.system(f'cat {bitext_path} > {tmpdir.name}/bitext.txt')
        os.system(f'cat {align_path} > {tmpdir.name}/aligns.txt')
        bitext = open(f'{tmpdir.name}/bitext.txt', encoding='utf-8').readlines()
        aligns = open(f'{tmpdir.name}/aligns.txt').readlines()
        tmpdir.cleanup()
        assert len(bitext) == len(aligns)
        bar = tqdm(bitext)
        for i, item in enumerate(bar):
            try:
                src_sent, trg_sent = regex.split(r'\|\|\|', item.strip())
                if is_reversed:
                    src_sent, trg_sent = trg_sent, src_sent
                align = [tuple(x if not is_reversed else reversed(x)) for x in json.loads(aligns[i])['inter']] 
            except:
                continue
            src_words = src_sent.lower().split()
            trg_words = trg_sent.lower().split()
            src_cnt = collections.Counter([x[0] for x in align])
            trg_cnt = collections.Counter([x[1] for x in align])
            for x, sw in enumerate(src_words):
                for y, tw in enumerate(trg_words):
                    if (x, y) in align:
                        semi_matched_coocc[sw][tw] += 1
                        if src_cnt[x] == 1 and trg_cnt[y] == 1:
                            matched_coocc[sw][tw] += 1
                    coocc[sw][tw] += 1
        torch.save((coocc, semi_matched_coocc, matched_coocc), stats_path)
    if os.path.exists(freq_path):
        freq_src, freq_trg = torch.load(freq_path)
    else:
        freq_src = collections.Counter()
        freq_trg = collections.Counter()
        tmpdir = tempfile.TemporaryDirectory()
        os.system(f'cat {bitext_path} > {tmpdir.name}/bitext.txt')
        bitext = open(f'{tmpdir.name}/bitext.txt', encoding='utf-8').readlines()
        tmpdir.cleanup()
        bar = tqdm(bitext)
        for i, item in enumerate(bar):
            try:
                src_sent, trg_sent = regex.split(r'\|\|\|', item.strip())
                if is_reversed:
                    src_sent, trg_sent = trg_sent, src_sent
            except:
                continue
            for w in src_sent.split():
                freq_src[w] += 1
            for w in trg_sent.split():
                freq_trg[w] += 1
        torch.save((freq_src, freq_trg), freq_path)
    return coocc, semi_matched_coocc, matched_coocc, freq_src, freq_trg
    

def load_lexicon(path):
    lexicon = [regex.split(r'\t| ', x.strip()) for x in open(path, encoding='utf-8')]
    return set([tuple(x) for x in lexicon])


def extract_dataset(train_lexicon, coocc):
    test_set = set()
    pos_training_set = set()
    neg_training_set = set()
    for sw in set([x[0] for x in train_lexicon]):
        for tw in coocc[sw]:
            if (sw, tw) in train_lexicon:
                pos_training_set.add((sw, tw))
            else:
                neg_training_set.add((sw, tw))
        if (sw, sw) in train_lexicon:
            pos_training_set.add((sw, sw))
        else:
            neg_training_set.add((sw, sw))
    for sw in coocc:
        added_self = False
        for tw in coocc[sw]:
            test_set.add((sw, tw))
            added_self = added_self | (sw == tw)
        test_set.add((sw, sw))
    pos_training_set = list(pos_training_set)
    neg_training_set = list(neg_training_set)
    test_set = list(test_set)
    return pos_training_set, neg_training_set, test_set


def extract_probs(batch, pretrained_model, lexicon_inducer, info, configs):
    matched_coocc, semi_matched_coocc, coocc, freq_src, freq_trg = info
    tokenizer, model = pretrained_model
    all_probs = list()
    for i in range(0, len(batch), configs.batch_size):
        subbatch = batch[i:i+configs.batch_size]
        src_words, trg_words = zip(*subbatch)
        src_toks = tokenizer(list(src_words), add_special_tokens=False, padding=True, return_tensors='pt')
        trg_toks = tokenizer(list(trg_words), add_special_tokens=False, padding=True, return_tensors='pt')
        src_toks, src_attn_masks = src_toks['input_ids'].to(configs.device), src_toks['attention_mask'].to(configs.device)
        trg_toks, trg_attn_masks = trg_toks['input_ids'].to(configs.device), trg_toks['attention_mask'].to(configs.device)
        src_encodings = model(src_toks, src_attn_masks)[0]
        trg_encodings = model(trg_toks, trg_attn_masks)[0]
        src_encodings = (src_encodings * src_attn_masks.unsqueeze(-1)).sum(1) / src_attn_masks.sum(-1).unsqueeze(-1)
        trg_encodings = (trg_encodings * trg_attn_masks.unsqueeze(-1)).sum(1) / trg_attn_masks.sum(-1).unsqueeze(-1)
        cos_sim = cos(src_encodings, trg_encodings).reshape(-1, 1)
        dot_prod = (src_encodings * trg_encodings).sum(-1).reshape(-1, 1)
        features = torch.tensor(
            [
                [
                    matched_coocc[x[0]][x[1]],
                    semi_matched_coocc[x[0]][x[1]],
                    coocc[x[0]][x[1]],
                    freq_src[x[0]], 
                    freq_trg[x[1]],
                ] for x in subbatch
            ]
        ).float().to(configs.device).reshape(-1, 5)
        features = torch.cat([cos_sim, dot_prod, features], dim=-1)
        probs = lexicon_inducer(features).squeeze(-1)
        all_probs.append(probs)
    return torch.cat(all_probs, dim=0)


def get_test_lexicon(test_set, pretrained_model, lexicon_inducer, info, configs, best_threshold, best_n_cand):
    induced_lexicon = list()
    pred_test_lexicon = collections.defaultdict(collections.Counter)
    with torch.no_grad():
        probs = extract_probs(test_set, pretrained_model, lexicon_inducer, info, configs)
    for i, (x, y) in enumerate(test_set):
        pred_test_lexicon[x][y] = max(pred_test_lexicon[x][y], probs[i].item())
    possible_predictions = list()
    for sw in set([x[0] for x in test_set]):
        for tw in pred_test_lexicon[sw]:
            possible_predictions.append([sw, tw, pred_test_lexicon[sw][tw]])
    possible_predictions = sorted(possible_predictions, key=lambda x:-x[-1])
    word_cnt = collections.Counter()
    for i, item in enumerate(possible_predictions):
        if item[-1] < best_threshold:
            break
        if word_cnt[item[0]] == best_n_cand:
            continue
        word_cnt[item[0]] += 1
        induced_lexicon.append(item[:2])
    return induced_lexicon


def get_optimal_parameters(pos_training_set, neg_training_set, train_lexicon, pretrained_model, lexicon_inducer, info, configs):
    pred_train_lexicon = collections.defaultdict(collections.Counter)
    with torch.no_grad():
        probs = extract_probs(pos_training_set + neg_training_set, pretrained_model, lexicon_inducer, info, configs)
    for i, (x, y) in enumerate(pos_training_set + neg_training_set):
        pred_train_lexicon[x][y] = max(pred_train_lexicon[x][y], probs[i].item())
    possible_predictions = list()
    for sw in set([x[0] for x in train_lexicon]):
        for tw in pred_train_lexicon[sw]:
            pos = 1 if (sw, tw) in train_lexicon else 0
            possible_predictions.append([sw, tw, pred_train_lexicon[sw][tw], pos])
    possible_predictions = sorted(possible_predictions, key=lambda x:-x[-2])
    best_f1 = -1e10
    best_threshold = best_n_cand = 0
    for n_cand in range(1, 6):
        word_cnt = collections.Counter()
        correct_predictions = 0
        bar = tqdm(possible_predictions)
        for i, item in enumerate(bar):
            if word_cnt[item[0]] == n_cand:
                continue
            word_cnt[item[0]] += 1
            if item[-1] == 1:
                correct_predictions += 1
                prec = correct_predictions / (sum(word_cnt.values()) + 1) * 100.0
                rec = correct_predictions / len(train_lexicon) * 100.0
                f1 = 2 * prec * rec / (rec + prec)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = item[-2]
                    best_n_cand = n_cand
                    bar.set_description(f'Best F1={f1:.1f}, Prec={prec:.1f}, Rec={rec:.1f}, NCand={n_cand}, Threshold={item[-2]}')
    return best_threshold, best_n_cand


def train_test(configs, logging_steps=50000):
    setup_configs(configs)
    os.system(f'mkdir -p {configs.save_path}')
    torch.save(configs, configs.save_path + '/configs.pt')
    # prepare feature extractor
    info = collect_bitext_stats(configs.bitext_path, configs.align_path, configs.save_path, configs.reversed)
    # dataset
    train_lexicon = load_lexicon(configs.tuning_set)
    pos_training_set, neg_training_set, test_set = extract_dataset(train_lexicon, info[2])
    training_set_modifier = max(1, len(neg_training_set) // len(pos_training_set))
    training_set = pos_training_set * training_set_modifier + neg_training_set
    print(f'Positive training set is repeated {training_set_modifier} times due to data imbalance.')
    # model and optimizers
    pretrained_tokenizer = AutoTokenizer.from_pretrained(configs.model_name)
    pretrained_model = AutoModel.from_pretrained(configs.model_name).to(configs.device)
    lexicon_inducer = LexiconInducer(7, configs.hiddens, 1, 5).to(configs.device)
    optimizer = torch.optim.Adam(lexicon_inducer.parameters(), lr=.0005)
    # train model
    for epoch in range(configs.epochs):
        model_path = configs.save_path + f'/{epoch}.model.pt'
        if os.path.exists(model_path):
            lexicon_inducer.load_state_dict(torch.load(model_path))
            continue
        random.shuffle(training_set)
        bar = tqdm(range(0, len(training_set), configs.batch_size))
        total_loss = total_cnt = 0
        for i, sid in enumerate(bar):
            batch = training_set[sid:sid+configs.batch_size]
            probs = extract_probs(batch, (pretrained_tokenizer, pretrained_model), lexicon_inducer, info, configs)
            targets = torch.tensor([1 if tuple(x) in train_lexicon else 0 for x in batch]).float().to(configs.device)
            optimizer.zero_grad()
            loss = nn.BCELoss()(probs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            total_cnt += len(batch)
            bar.set_description(f'loss={total_loss / total_cnt:.5f}')
        torch.save(lexicon_inducer.state_dict(), model_path)
        print(f'Epoch {epoch}, loss = {total_loss / total_cnt:.5f}', flush=True)
    torch.save(lexicon_inducer.state_dict(), configs.save_path + f'/model.pt')
    best_threshold, best_n_cand = get_optimal_parameters(
        pos_training_set, neg_training_set, train_lexicon, (pretrained_tokenizer, pretrained_model), lexicon_inducer, info, configs
    )
    induced_test_lexicon = get_test_lexicon(
        test_set, (pretrained_tokenizer, pretrained_model), lexicon_inducer, info, configs, best_threshold, best_n_cand
    )
    with open(configs.save_path + '/induced.weaklysup.dict', 'w', encoding='utf-8') as fout:
        for item in induced_test_lexicon:
            fout.write('\t'.join([str(x) for x in item]) + '\n')
        fout.close()
    return induced_test_lexicon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--align', type=str, help='path to word alignment')
    parser.add_argument('-b', '--bitext', type=str, help='path to bitext')
    parser.add_argument('-src', '--source', type=str, help='source language code')
    parser.add_argument('-trg', '--target', type=str, help='target language code')
    parser.add_argument('-tr', '--train', type=str, help='path to training lexicon')
    parser.add_argument('-o', '--output', type=str, default='./model/', help='path to output folder')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device for training [cuda|cpu]')
    parser.add_argument('-m', '--model', type=str, default='xlm-roberta-large', help='model name')
    args = parser.parse_args()

    configs = dotdict.DotDict(
        {
            'tuning_set': args.train,
            'align_path': args.align,
            'bitext_path': args.bitext,
            'save_path': args.output,
            'batch_size': 64,
            'epochs': 1,
            'device': args.device,
            'hiddens': [8],
            'model_name': args.model
        }
    )
    
    res = train_test(configs)
    print(res[-1])
