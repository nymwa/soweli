import contextlib
import random as rd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from soweli.ponart.util import make_tokenizer
from soweli.ponart.sampler import Sampler
from soweli.ponart.ponart import Ponart
from soweli.ponart.scheduler import WarmupScheduler
from soweli.ponart.log import init_logging
from soweli.kipisi.sentence import Sentence
from soweli.kipisi.dataset import Dataset
from soweli.kipisi.model import KipisiClassifier
from logging import getLogger
init_logging()
logger = getLogger(__name__)

def make_dataset(tokenizer):
    Sentence.tokenizer = tokenizer
    data = []
    tmp = set()
    with open('../../soweli/corpus/tatoeba/tatoeba.txt') as f:
        for sent in f:
            sent = Sentence(sent)
            joined = sent.joined()
            if joined not in tmp:
                data.append(sent)
                tmp.add(joined)
    rd.seed(0)
    rd.shuffle(data)
    valid_size = 1000
    train_data = data[:-valid_size]
    valid_data = data[-valid_size:]
    train_dataset = Dataset(train_data, tokenizer.vocab)
    valid_dataset = Dataset(train_data, tokenizer.vocab)
    return train_dataset, valid_dataset

def calculate_loss(model, batch, criterion, requires_grad = True):
    batch.cuda()
    context = contextlib.suppress if requires_grad else torch.no_grad
    with context():
        pred = model(batch.encoder_inputs)
    pred = pred.view(-1, pred.size(-1))
    loss = criterion(pred, batch.encoder_outputs.view(-1))
    return loss

def main():
    tokenizer = make_tokenizer()
    train_dataset, valid_dataset = make_dataset(tokenizer)
    train_sampler = Sampler(train_dataset, 20000)
    valid_sampler = Sampler(valid_dataset, 20000)
    train_loader = DataLoader(train_dataset, batch_sampler = train_sampler, collate_fn = train_dataset.collate)
    valid_loader = DataLoader(valid_dataset, batch_sampler = valid_sampler, collate_fn = valid_dataset.collate)
    model = KipisiClassifier(len(tokenizer.vocab))
    model.load_checkpoint('../../../ponart_soweli/pretrain.pt')
    model = model.cuda()
    print('#params (to train): {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('#params (total): {}'.format(sum(p.numel() for p in model.parameters())))
    optimizer = optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.01)
    scheduler = WarmupScheduler(optimizer, 100)
    criterion = nn.CrossEntropyLoss(ignore_index = -100)

    clip_norm = 1.0
    num_steps = 0
    model.train()
    for epoch in range(30):
        model.train()
        train_accum = 0.0
        train_example = 0
        for batch in train_loader:
            loss = calculate_loss(model, batch, criterion, requires_grad = True)
            train_accum += loss.item() * len(batch)
            train_example += len(batch)
            optimizer.zero_grad()
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            scheduler.step()
            num_steps += 1

        model.eval()
        valid_accum = 0.0
        valid_example = 0
        for batch in valid_loader:
            loss = calculate_loss(model, batch, criterion, requires_grad = False)
            valid_accum += loss.item() * len(batch)
            valid_example += len(batch)

        train_loss = train_accum / train_example
        valid_loss = valid_accum / valid_example
        lr = scheduler.get_last_lr()[0]
        log = 'epoch {} | train {:.5f} | valid {:.5f} | lr {:.4e} | steps {}'.format(epoch, train_loss, valid_loss, lr, num_steps)
        logger.info(log)
    torch.save(model.state_dict(), 'kipisi.pt')

if __name__ == '__main__':
    main()

