"""C28 step 2: fine-tune a multilingual cross-encoder (intfloat/multilingual-e5-small, MIT, 118M) on C28 train pairs,
then score the C04 winner pairs of validation and test.
  python3 matcher/scripts/c28_train.py train            # 1 epoch, bf16, checkpoints every 2000 steps
  python3 matcher/scripts/c28_train.py score val|test   # writes matcher/runs/c28/scores_{val,test}.parquet
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'matcher/runs/c28'
MODEL = 'intfloat/multilingual-e5-small'
MAXLEN, BS = 128, 128
LR = float(os.environ.get('C28_LR', '4e-5'))
TRAIN = os.environ.get('C28_TRAIN', 'train.parquet')   # extra shard: train_b1.parquet
INIT = os.environ.get('C28_INIT', MODEL)               # warm start: matcher/runs/c28/model
OUTM = os.environ.get('C28_OUT', 'model')              # model folder under matcher/runs/c28
torch.backends.cuda.matmul.allow_tf32 = True


def encode(tok, a, b):
    return tok(a, b, truncation='longest_first', max_length=MAXLEN, padding=True, return_tensors='pt')


def train():
    d = pl.read_parquet(RUN / 'data' / TRAIN)
    tok = AutoTokenizer.from_pretrained(INIT)
    model = AutoModelForSequenceClassification.from_pretrained(INIT, num_labels=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = math.ceil(d.height / BS)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    a, b, y = d['a'].to_list(), d['b'].to_list(), torch.tensor(d['label'].to_numpy())
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    run = 0.0
    bar = tqdm(range(steps), desc='train', unit='step', dynamic_ncols=True)
    for s in bar:
        i = slice(s * BS, (s + 1) * BS)
        x = {k: v.cuda(non_blocking=True) for k, v in encode(tok, a[i], b[i]).items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(**x).logits.squeeze(-1)
        loss = lossf(out.float(), y[i].cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item()
        if s % 50 == 0:
            bar.set_postfix(loss=f'{run:.4f}')
        if (s + 1) % 2000 == 0 or s + 1 == steps:
            model.save_pretrained(RUN / OUTM); tok.save_pretrained(RUN / OUTM)
    print('trained', steps, 'steps; final running loss', round(run, 4), flush=True)


@torch.no_grad()
def score(which):
    d = pl.read_parquet(RUN / f'data/{which}.parquet')
    tok = AutoTokenizer.from_pretrained(RUN / OUTM)
    model = AutoModelForSequenceClassification.from_pretrained(RUN / OUTM).cuda().eval()
    lens = (d['a'].str.len_chars() + d['b'].str.len_chars()).to_numpy()
    order = np.argsort(lens)
    a, b = d['a'].to_list(), d['b'].to_list()
    out = np.empty(d.height, np.float32)
    bs = 1024
    for s in tqdm(range(0, d.height, bs), desc=f'score {which}', unit='batch', dynamic_ncols=True):
        idx = order[s:s + bs]
        x = {k: v.cuda(non_blocking=True) for k, v in encode(tok, [a[i] for i in idx], [b[i] for i in idx]).items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out[idx] = torch.sigmoid(model(**x).logits.squeeze(-1).float()).cpu().numpy()
    d.drop('a', 'b').with_columns(pl.Series('tf', out)).write_parquet(RUN / f'scores_{which}{"" if OUTM == "model" else "_" + OUTM}.parquet')
    print('scored', which, d.height, flush=True)


if __name__ == '__main__':
    t0 = time.time()
    train() if sys.argv[1] == 'train' else score(sys.argv[2])
    print('seconds', round(time.time() - t0), flush=True)
