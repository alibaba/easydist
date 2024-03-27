# torchrun --nproc_per_node 4 examples/torch/pipeline_parallelism/bert_train.py
import os
import random

from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader

from easydist.torch.experimental.pp.api import _compile_pp
from easydist.torch.experimental.pp.compile_pipeline import (annotate_split_points)
from easydist.torch.init_helper import SetParaInitHelper
from easydist.torch.experimental.pp.PipelineStage import Schedule1F1B

from torchtext.datasets import IMDB, AG_NEWS

from transformers import BertForSequenceClassification, BertTokenizer


def seed(seed=42):
    # Set seed for PyTorch
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for built-in Python
    random.seed(seed)
    # Set(seed) for each of the random number generators in python:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_step(model, input_ids, attn_mask, labels, opt):
    outputs = model(input_ids, attention_mask=attn_mask)
    loss = F.cross_entropy(outputs.logits, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return outputs.logits, loss

def train_bert():
    seed()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(rank=rank, world_size=world_size)

    compile_device = torch.device('cpu')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10).to(compile_device).train()
    split_points = {
        "bert.embeddings.position_embeddings",
        "bert.encoder.layer.0",
        "bert.encoder.layer.8"
    }
    assert len(split_points) + 1 == world_size
    annotate_split_points(model, split_points)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def tokenize(x):
        encodings =  tokenizer(x, return_tensors='pt', padding="max_length", max_length=128, truncation=True)
        return encodings['input_ids'], encodings['attention_mask']
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=True, capturable=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, foreach=True)

    batch_size = 64 # ~ 50 iters
    num_chunks = world_size * 4

    def get_iters():
        return iter(DataLoader(AG_NEWS(split="train"), batch_size=batch_size, shuffle=True)),\
              iter(DataLoader(AG_NEWS(split="test"), batch_size=batch_size, shuffle=True))

    train_iter, test_iter = get_iters()

    y, x = next(train_iter)    
    input_ids, attn_mask = tokenize(x)
    args = [model, input_ids.to(compile_device), attn_mask.to(compile_device), y.to(compile_device), optimizer]
    kwargs = {}

    compiled_fn = _compile_pp(train_step, "fake", SetParaInitHelper(), None,
                              args, kwargs, Schedule1F1B, None, None, None,
                              num_chunks)

    def test(module, valid_dataloader, epoch, outputs):
        device = torch.device('cuda:0')
        params, buffers, _, _, _ = outputs
        buffers.pop('bert.embeddings.position_ids')
        buffers.pop('bert.embeddings.token_type_ids')
        module.load_state_dict({**params, **buffers})
        module.eval()
        module.to(device)
        correct_cnt = 0
        all_cnt = 0
        for y_batch, x_batch in (tqdm(valid_dataloader, dynamic_ncols=True)):
            y_batch = y_batch.to(device)
            input_ids, attn_mask = tokenize(x_batch)
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            out = module(input_ids, attention_mask=attn_mask).logits
            preds = out.argmax(-1)
            correct_cnt += (preds == y_batch).sum()
            all_cnt += len(y_batch)
        print(f'epoch {epoch} valid accuracy: {correct_cnt / all_cnt}')

    epochs = 5
    for epoch in range(epochs):
        all_cnt = 0
        correct_cnt = 0 
        loss_sum = 0
        for y_batch, x_batch in (tqdm(train_iter, dynamic_ncols=True)
                                 if rank == 0 else train_iter):
            if len(x_batch) != batch_size:  # TODO need to solve this
                continue
            input_ids, attn_mask = tokenize(x_batch)
            ret = compiled_fn(model, input_ids, attn_mask, y_batch, optimizer)
            if rank == world_size - 1:
                out, loss = ret
                all_cnt += len(out)
                preds = out.argmax(-1)
                correct_cnt += (preds == y_batch.to(f'cuda:{rank}')).sum()
                loss_sum += loss.sum()

        if rank == world_size - 1:
            print(f'epoch {epoch} train accuracy: {correct_cnt / all_cnt}, loss sum {loss_sum}, avg loss: {loss_sum / all_cnt}')

        valid_rank = 0
        outputs = compiled_fn.gather_outputs(valid_rank)
        if rank == valid_rank:
            test(model, test_iter, epoch, outputs)
        
        # iter somehow doesn't reload the data
        train_iter, test_iter = get_iters()


    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(cur_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = compiled_fn.state_dict()
    opt_state_dict = compiled_fn.optimizer_state_dict()

    torch.save(state_dict, os.path.join(ckpt_dir, f'state_dict_{rank}.pt'))
    torch.save(opt_state_dict, os.path.join(ckpt_dir, f'opt_state_dict_{rank}.pt'))


if __name__ == '__main__':
    train_bert()