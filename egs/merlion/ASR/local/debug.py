import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from lhotse import RecordingSet, Recording, AudioSource
from lhotse import SupervisionSet, SupervisionSegment
from lhotse import FeatureSet, CutSet
from lhotse.dataset import ZipSampler, SimpleCutSampler, K2SpeechRecognitionDataset
# from model import SpeechRecognitionModel

# from langdetect import detect
def prepare_data():

    path="/home/cxiao7/research/merlion_ASR"
    os.chdir(path)
    
    cuts_libri_train = CutSet.from_file("/home/cxiao7/research/merlion_ASR/data/fbank/cuts_LibriSpeech.jsonl.gz")
    cuts_aishell_train = CutSet.from_file("/home/cxiao7/research/merlion_ASR/data/fbank/cuts_AISHELL.jsonl.gz")

    sampler_libri = SimpleCutSampler(cuts_libri_train, max_duration=140, shuffle=True)
    sampler_aishell = SimpleCutSampler(cuts_aishell_train, max_duration=140, shuffle=True)
    cuts_dev = CutSet.from_file("/home/cxiao7/research/merlion_ASR/data/fbank/cuts_dev.jsonl.gz")
    dev_sampler = SimpleCutSampler(cuts_dev, max_duration=160, shuffle=False)
    dev_dataset = K2SpeechRecognitionDataset()
    dev_loader = torch.utils.data.DataLoader(dev_dataset, sampler=dev_sampler, batch_size=None)
        
    sampler = ZipSampler(sampler_libri, sampler_aishell)
    dataset = K2SpeechRecognitionDataset()
    train_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None)
    
    try:
        for idx, batch in enumerate(dev_loader):
            if idx == 2:
                assert False
            print(batch['inputs'].shape)
    except:
        breakpoint()
    
    return train_loader, dev_loader

def train(train_loader, dev_loader, epoch, model, optimizer, device):
    
    model.train()
    start = time.time()
    loss_fn = nn.NLLLoss()
    for idx, batch in enumerate(train_loader):
        
        if idx >= 20:
            break
        batch['supervisions']['language'] = [0 if detect(text) == 'en' else 1 for text in batch['supervisions']['text']]
        # b, t, c -> b, 1, t, c
        x = torch.unsqueeze(batch['inputs'], dim=1).to(device)
        # b, 1, t, c -> b, 1, c, t
        x = x.transpose(2, 3).contiguous() 
        y_hat = model(x)
        y = torch.tensor(batch['supervisions']['language']).to(device)
        y_hat = F.log_softmax(y_hat, dim=1)
        print(y_hat.shape)
        loss = loss_fn(y_hat, y)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch: {epoch}, Iteration: {idx}, Loss:  {loss.item()}')
    
    end = time.time()
    print(f'Epoch: {epoch} used {end-start} seconds')
    #validating
    model.eval()
    preds = []
    acts = []
    for idx, batch in enumerate(dev_loader):
        batch['supervisions']['language'] = [0 if detect(text) == 'en' else 1 for text in batch['supervisions']['text']]
        # b, t, c -> b, 1, t, c
        x = torch.unsqueeze(batch['inputs'], dim=1).to(device)
        # b, 1, t, c -> b, 1, c, t
        x = x.transpose(2, 3).contiguous() 
        y_hat = model(x)
        y = torch.tensor(batch['supervisions']['language']).to(device)
        
        y_hat = F.log_softmax(y_hat, dim=1)
        y_hat = torch.argmax(y_hat, dim=1)
        
        preds.extend(y_hat.items())
        acts.extend(y.items())
    acc = sum(1 for a,b in zip(preds, acts) if a == b) / len(preds)
    print(f"Epoch: {epoch}, accuracy = {accuracy}")

def main(args): 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 2,
        "n_feats": 80,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": args.lr,
        "epochs": args.epochs
    }
    
    train_loader, dev_loader = prepare_data()
    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        train(train_loader, dev_loader, epoch, model, optimizer, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #basic training arguments 
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    
    args = parser.parse_args()
        
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    main(args)
