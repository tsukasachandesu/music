import pickle
import deepspeed
from block_recurrent_transformer_pytorch import BlockRecurrentTransformer, RecurrentTrainerWrapper
from mgt.datamanagers.remi_data_manager import RemiDataManager
from mgt.datamanagers.data_helper import DataHelper
from mgt.datamanagers.remi.efficient_remi_config import EfficientRemiConfig

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from einops import rearrange
from torch import einsum, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse

datamanager = RemiDataManager(
    efficient_remi_config=EfficientRemiConfig(enabled=True, remove_velocity=True)
)

class Dataset(Dataset):
    def __init__(self, data, max_length=1024):
        self.data = data
        self.max_length = max_length
        
    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        song_index = random.randint(0, len(self.data) - 1)
        if len(self.data[song_index]) <= self.max_length:
          starting_index = random.randint(0, len(self.data[song_index]) - 1)
          padded_song = list(np.repeat(0, self.max_length))+self.data[song_index]
          a = padded_song[0:self.max_length]
        else:
          starting_index = random.randint(0, len(self.data[song_index]) - self.max_length)
          a = self.data[song_index][starting_index: starting_index + self.max_length]
        return torch.tensor(a).long().cuda()
  
def add_argument():
    parser=argparse.ArgumentParser(description='enwik8')

    parser.add_argument('--with_cuda', default=True, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()
    return args

# constants

EPOCHS = 7
GRADIENT_ACCUMULATE_EVERY = 1
GENERATE_EVERY = 1800
GENERATE_LENGTH = 1024
SEQ_LEN = 1024

# instantiate GPT-like decoder model

model = BlockRecurrentTransformer(
    num_tokens = 7700,
    dim = 512,
    depth = 16,
    dim_head = 64,
    heads = 12,
    max_seq_len = 1024,
    block_width = 512,
    num_state_vectors = 512,
    recurrent_layers = (4,),
    use_flash_attn = True
)
model = RecurrentTrainerWrapper(
    model,
    xl_memories_dropout = 0.1,
    state_dropout = 0.1,
).cuda()

data_train = DataHelper.load('/content/drive/MyDrive/set')
data_train = data_train.data
train_dataset = Dataset(data_train, SEQ_LEN)

# setup deepspeed

cmd_args = add_argument()
model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters(), training_data=train_dataset)
_, client_sd = model_engine.load_checkpoint("/content/1")
# training
         
for _ in range(EPOCHS):
    for i, data in enumerate(trainloader):
        model_engine.train()
        data = data.to(model_engine.local_rank)
        loss = model_engine(data)

        model_engine.backward(loss)
        torch.nn.utils.clip_grad_norm_(model_engine.parameters(), 0.5)
        model_engine.step()
        print(loss.item() * GRADIENT_ACCUMULATE_EVERY)

        if i % GENERATE_EVERY == 0:
            model.eval()          
            prompt = [2]
            initial = torch.tensor([prompt]).long().cuda() 
            sample = model.generate(initial,GENERATE_LENGTH)
            sample = sample.cpu().detach().numpy()[0]
            midi = datamanager.to_midi(sample)
            midi.save("1.midi")
    model_engine.save_checkpoint("/content/1") 
