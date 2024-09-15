# Copyright (c) 2023, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ENABLE_COMPILE_CACHE=1 torchrun --nproc_per_node 4 examples/torch/bert_train.py
import os
import random

from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributed._tensor import DeviceMesh

from easydist import easydist_setup
from easydist.torch.api import easydist_compile
from easydist.torch.utils import seed
from easydist.torch.device_mesh import set_device_mesh
from easydist.torch.experimental.pp.runtime import ScheduleDAPPLE
from easydist.torch.experimental.pp.compile_pipeline import (
    annotate_split_points,
    split_into_equal_size)

from torchtext.datasets import IMDB, AG_NEWS  # pip install torchtext torchdata portalocker

from transformers import BertForSequenceClassification, BertTokenizer


def train_bert():
    seed(42)
    easydist_setup(backend="torch", device="cuda", allow_tf32=False)

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)

    device = torch.device('cuda')
    set_device_mesh(
        DeviceMesh("cuda", [0, 1, 2, 3],
                   mesh_dim_names=["pp"]))

    @easydist_compile(
                  parallel_mode='pp',
                  tracing_mode="fake",
                  cuda_graph=False,
                  schedule_cls=ScheduleDAPPLE,
                  num_chunks=16)
    def train_step(model, input_ids, attn_mask, labels, opt):
        outputs = model(input_ids, attention_mask=attn_mask)
        loss = F.cross_entropy(outputs.logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return outputs.logits, loss


    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10).train().to(device)
    annotate_split_points(model, {
        'bert.embeddings',
        'bert.encoder.layer.4',
        'bert.encoder.layer.8'
    })
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def tokenize(x):
        encodings =  tokenizer(x, return_tensors='pt', padding="max_length", max_length=256, truncation=True)
        return encodings['input_ids'].to(device), encodings['attention_mask'].to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=True, capturable=True)

    batch_size = 128

    def get_iters():
        return iter(DataLoader(AG_NEWS(split="train"), batch_size=batch_size, shuffle=True)),\
              iter(DataLoader(AG_NEWS(split="test"), batch_size=batch_size, shuffle=True))

    def test(module, valid_dataloader, epoch, state_dict):
        module.load_state_dict(state_dict, strict=False)
        module.eval()
        module.to(device)
        correct_cnt = 0
        all_cnt = 0
        for y_batch, x_batch in (tqdm(valid_dataloader, dynamic_ncols=True)):
            y_batch = y_batch.to(device)
            input_ids, attn_mask = tokenize(x_batch)
            out = module(input_ids, attention_mask=attn_mask).logits
            preds = out.argmax(-1)
            correct_cnt += (preds == y_batch).sum()
            all_cnt += len(y_batch)
        print(f'epoch {epoch} valid accuracy: {correct_cnt / all_cnt}')

    epochs = 5
    train_iter, test_iter = get_iters()
    for epoch in range(epochs):
        all_cnt = 0
        correct_cnt = 0 
        loss_sum = 0
        for y_batch, x_batch in (tqdm(train_iter, dynamic_ncols=True)
                                 if rank == 0 else train_iter):
            if y_batch.size(0) != batch_size:
                continue
            input_ids, attn_mask = tokenize(x_batch)
            y_batch = y_batch.to(device)
            ret = train_step(model, input_ids, attn_mask, y_batch, optimizer)
            out, loss = ret
            all_cnt += len(out)
            preds = out.argmax(-1)
            correct_cnt += (preds.to(device) == y_batch.to(device)).sum()
            loss_sum += loss.sum()
        state_dict = train_step.compiled_func.state_dict()
        if rank == 0:
            print(f'epoch {epoch} train accuracy: {correct_cnt / all_cnt}, loss sum {loss_sum}, avg loss: {loss_sum / all_cnt}')
            test(model, test_iter, epoch, state_dict)
        train_iter, test_iter = get_iters()

    print(f"rank {rank} peek memory: {torch.cuda.max_memory_allocated()}")


if __name__ == '__main__':
    train_bert()