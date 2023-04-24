import gzip
import random
import tqdm
import numpy as np

import torch
from lion_pytorch import Lion
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from palm_rlhf_pytorch import PaLM
from accelerate import Accelerator

import pickle
from mgt.datamanagers.remi_data_manager import RemiDataManager
from mgt.datamanagers.data_helper import DataHelper
from mgt.datamanagers.remi.efficient_remi_config import EfficientRemiConfig

datamanager = RemiDataManager(
    efficient_remi_config=EfficientRemiConfig(enabled=True, remove_velocity=True)
)

class TextSampleDataset1(Dataset):
    def __init__(self, split, max_length=1024):
        f = open('/content/music/1.pickle','rb')
        self.data  = pickle.load(f)
        self.max_length = max_length
        
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        song_index = random.randint(0, len(self.data) - 1)
        if len(self.data[song_index]) <= self.max_length:
          starting_index = random.randint(0, len(self.data[song_index]) - 1)
          padded_song = self.data[song_index] + list(np.repeat(0, self.max_length))
          a = padded_song[0:self.max_length]
        else:
          starting_index = random.randint(0, len(self.data[song_index]) - self.max_length)
          a = self.data[song_index][starting_index: starting_index + self.max_length]
        return torch.tensor(a).long()
 
class TextSampleDataset2(Dataset):
    def __init__(self, split, max_length=1024):
        f = open('/content/music/1.pickle','rb')
        self.data  = pickle.load(f)
        self.max_length = max_length
        
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        song_index = random.randint(0, len(self.data) - 1)
        if len(self.data[song_index]) <= self.max_length:
          starting_index = random.randint(0, len(self.data[song_index]) - 1)
          padded_song = self.data[song_index] + list(np.repeat(0, self.max_length))
          a = padded_song[0:self.max_length]
        else:
          starting_index = random.randint(0, len(self.data[song_index]) - self.max_length)
          a = self.data[song_index][starting_index: starting_index + self.max_length]
        return torch.tensor(a).long()

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

# accelerator

accelerator = Accelerator()
device = accelerator.device

# instantiate palm

model = PaLM(
    num_tokens=256,
    dim=512,
    depth=8,
    flash_attn=True
).to(device)

# prepare enwik8 data

train_dataset = TextSamplerDataset1(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset2(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# optimizer

optim = Lion(model.palm_parameters(), lr = LEARNING_RATE)

model, optim, train_loader, val_loader = accelerator.prepare(
    model, optim, train_loader, val_loader
)
accelerator.save_state(output_dir="my_checkpoint")

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)
        accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

    accelerator.print(f"training loss: {loss.item()}")
    accelerator.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            accelerator.print(f"validation loss: {loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        prompt = [2]
        initial = torch.tensor([prompt]).long().cuda() 

        sample = model.generate(GENERATE_LENGTH, initial)
        sample = sample.cpu().detach().numpy()[0]
        midi = datamanager.to_midi(sample)
        midi.save("1.midi")
