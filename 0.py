from __future__ import annotations
import pickle
import deepspeed

import time
import numpy as np
import torch
from x_transformers import Decoder

from mgt.models import utils
from mgt.models.compound_word_transformer.compound_word_autoregressive_wrapper import CompoundWordAutoregressiveWrapper
from mgt.models.compound_word_transformer.compound_word_transformer_utils import COMPOUND_WORD_BAR, get_batch
from mgt.models.compound_word_transformer.compound_word_transformer_wrapper import CompoundWordTransformerWrapper
from mgt.models.utils import get_device
from mgt.datamanagers.compound_word_data_manager import CompoundWordDataManager
from mgt.datamanagers.data_helper import DataHelper

import random
import tqdm
import gzip

import torch.optim as optim
from einops import rearrange
from torch import einsum, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse

datamanager = CompoundWordDataManager()

from typing import List
import random

COMPOUND_WORD_PADDING = [0, 0, 0, 0, 0, 0, 0, 0]
COMPOUND_WORD_BAR = [2, 0, 0, 0, 0, 0, 0, 0]

def pad(array: np.ndarray, max_sequence_length: int, padding_compound_word: np.ndarray = None) -> np.ndarray:
    if padding_compound_word is None:
        padding_compound_word = COMPOUND_WORD_PADDING
    if len(array) >= max_sequence_length:
        return array[:max_sequence_length]
    else:
        number_of_padding_elements_to_add = max_sequence_length - len(array)
        padding = np.tile(padding_compound_word, (number_of_padding_elements_to_add, 1))
        return np.vstack((array, padding))

class Dataset(Dataset):
    def __init__(self, data, max_length=1024):
        self.data = data
        self.max_length = max_length
        
    def __len__(self):
        return 512

    def __getitem__(self, idx):
        song_index = random.randint(0, len(self.data) - 1)
        if len(self.data[song_index]) <= self.max_length:
          padded_song = pad(self.data[song_index], self.max_length)
          a = padded_song
        else:
          starting_index = random.randint(0, len(self.data[song_index]) - self.max_length)
          a = self.data[song_index][starting_index: starting_index + self.max_length]
            
        return torch.tensor(a).long()

def add_argument():
    parser=argparse.ArgumentParser(description='enwik8')

    parser.add_argument('--with_cuda', default=True, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    parser.add_argument('-b', '--batch_size', default=12, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=4, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()
    return args

# constants

EPOCHS = 2
GRADIENT_ACCUMULATE_EVERY = 3
GENERATE_EVERY = 1800
GENERATE_LENGTH = 1024
yes = None
yes1 = "a"

# instantiate GPT-like decoder model

defaults = {
    'num_tokens': [
        4,    # Type
        17,   # Bar / Beat
        6912,  # Tempo
        6912,  # Instrument
        6912,   # Note name
        6912,    # Octave
        6912,   # Duration
        6912    # Velocity
    ],
    'emb_sizes': [
        32,   # Type
        96,   # Bar / Beat
        512,  # Tempo
        512,  # Instrument
        512,  # Note Name
        512,  # Octave
        512,  # Duration
        512   # Velocity
    ],
    'max_sequence_length': 1024,
    'learning_rate': 1e-4,
    'dropout': 0.1,
    'dim': 512,
    'depth': 24,
    'heads': 8
}

model = CompoundWordAutoregressiveWrapper(CompoundWordTransformerWrapper(
    num_tokens=defaults["num_tokens"],
    emb_sizes=defaults["emb_sizes"],
    max_seq_len=defaults["max_sequence_length"],
    attn_layers=Decoder(
        dim=defaults["dim"],
        depth=defaults["depth"],
        heads=defaults["heads"],
        ff_glu = True,
        ff_swish = True,
        rel_pos_bias = True,
        attn_dropout=defaults["dropout"],  
        ff_dropout=defaults["dropout"],  
    ))).cuda()

# setup deepspeed
data_train = DataHelper.load('/content/drive/MyDrive/test_dataset')
train_dataset = Dataset(data_train.data)

cmd_args = add_argument()
model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters(), training_data=train_dataset)
if yes:
    _, client_sd = model_engine.load_checkpoint("/")

for _ in range(EPOCHS):
    for i, data in enumerate(trainloader):
        model_engine.train()
        data = data.to(model_engine.local_rank)
        loss = model_engine.train_step(data)
        loss = sum(loss) / len(loss)
        model_engine.backward(loss)
        torch.nn.utils.clip_grad_norm_(model_engine.parameters(), 0.5)
        model_engine.step()
        print(loss.item())

model.eval()    
if yes1:
    model_engine.save_checkpoint("/")


prompt = [COMPOUND_WORD_BAR] 
sample = model.generate(output_length=256, prompt=prompt)
datamanager = CompoundWordDataManager()
midi = datamanager.to_midi(sample)
midi.save("1.midi")




