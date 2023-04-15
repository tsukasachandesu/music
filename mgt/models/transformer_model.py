import pickle
import deepspeed
from palm_pytorch import PaLM
from palm_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from mgt.datamanagers.data_helper import DataHelper
from mgt.datamanagers.remi_data_manager import RemiDataManager
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

def add_argument():
    parser=argparse.ArgumentParser(description='enwik8')

    parser.add_argument('--with_cuda', default=True, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    parser.add_argument('-b', '--batch_size', default=48, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()
    return args

def pad(array, max_sequence_length, padding_character=0):
    return list(np.repeat(padding_character, max_sequence_length)) + array

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):

        song_index = random.randint(0, len(self.data) - 1)
        starting_index = random.randint(0, len(self.data[song_index]) - 1)
        padded_song = pad(self.data[song_index], self.seq_len)
        bat = padded_song[starting_index: starting_index + self.seq_len + 1]
        full = torch.tensor(bat).long()
        return full

    def __len__(self):
        return 4000

# constants

EPOCHS = 10
GRADIENT_ACCUMULATE_EVERY = 4
VALIDATE_EVERY = 4000
GENERATE_EVERY = 4000
GENERATE_LENGTH = 1024
SEQ_LEN = 3000

model = PaLM(num_tokens = 568, dim = 512, depth = 12)
model = AutoregressiveWrapper(model, max_seq_len=SEQ_LEN)
model.cuda()

x_train = DataHelper.load('/content/drive/MyDrive/yuno')
x_train = x_train.data

train_dataset = TextSamplerDataset(x_train, SEQ_LEN)
val_dataset = TextSamplerDataset(x_train, SEQ_LEN)

# setup deepspeed

cmd_args = add_argument()

model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=model.parameters(), training_data=train_dataset)
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

        if i % VALIDATE_EVERY == 0:
            model_engine.save_checkpoint("/content")
            model.eval()
            with torch.no_grad():
                inp = random.choice(val_dataset)[:-1]
                loss = model(inp[None, :].cuda())
                print(f'validation loss: {loss.item()}')

        if i % GENERATE_EVERY == 0:
            model.eval()

            prompt = [2]
            initial = torch.tensor([prompt]).long().cuda() # assume 0 is start token
            sample = model.generate(initial, GENERATE_LENGTH)
            datamanager = RemiDataManager(
              efficient_remi_config=EfficientRemiConfig(enabled=True, remove_velocity=True)
              )
            midi = datamanager.to_midi(sample.cpu().detach().numpy()[0])
            midi.save("1.midi")

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = self.create_optimizer()

    def train(self, x_train, epochs, batch_size=4, stop_loss=None, batches_per_epoch=100, report_per_x_batches=20,
              gradient_accumulation_steps=1):
        self.model.train()
        start_time = time.time()
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}.")

            epoch_losses = []
            batch_losses = []
            nr_of_batches_processed = 0
            for _ in range(batches_per_epoch):

                for _ in range(gradient_accumulation_steps):
                    batch = utils.get_batch(
                        x_train,
                        batch_size=batch_size,
                        max_sequence_length=self.max_sequence_length)

                    torch_batch = torch.tensor(batch).long().to(utils.get_device())

                    loss = self.model(torch_batch, return_loss=True)
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

                nr_of_batches_processed += 1

                loss_item = loss.item()

                batch_losses.append(loss_item)
                epoch_losses.append(loss_item)

                if nr_of_batches_processed % report_per_x_batches == 0:
                    print(
                        f"Processed {nr_of_batches_processed} / {batches_per_epoch} with loss {np.mean(batch_losses)}.")
                    batch_losses = []

            epoch_loss = np.mean(epoch_losses)
            if stop_loss is not None and epoch_loss <= stop_loss:
                print(f"Loss of {epoch_loss} was lower than stop loss of {stop_loss}. Stopping training.")
                return

            running_time = (time.time() - start_time)
            print(f"Loss after epoch {epoch + 1} is {epoch_loss}. Running time: {running_time}")

    def generate(self, output_length=100, temperature=1., filter_treshold=0.9, prompt=None):
        print(f"Generating a new song with {output_length} characters.")
        if prompt is None:
            prompt = [0]

        self.model.eval()
        initial = torch.tensor([prompt]).long().to(utils.get_device())  # assume 0 is start token

        sample = self.model.generate(2048, initial)
        return sample.cpu().detach().numpy()[0]

    def create_model(self):
        model = PaLM(num_tokens=568,dim=512,depth=8,flash_attn=False).to(utils.get_device())

        return model

    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'dictionary': self.dictionary,
            'max_sequence_length': self.max_sequence_length,
            'learning_rate': self.learning_rate,
            'dropout': self.dropout,
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    @staticmethod
    def load_checkpoint(path) -> TransformerModel:
        checkpoint = torch.load(path)
        model = TransformerModel(
            dictionary=checkpoint['dictionary'],
            max_sequence_length=utils.get_or_default(checkpoint, 'max_sequence_length', defaults),
            learning_rate=utils.get_or_default(checkpoint, 'learning_rate', defaults),
            dropout=utils.get_or_default(checkpoint, 'dropout', defaults),
            dim=utils.get_or_default(checkpoint, 'dim', defaults),
            depth=utils.get_or_default(checkpoint, 'depth', defaults),
            heads=utils.get_or_default(checkpoint, 'heads', defaults)
        )

        model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        return model
