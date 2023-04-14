import pickle
import deepspeed
from palm_pytorch import PaLM
from palm_pytorch.autoregressive_wrapper import AutoregressiveWrapper

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

def pad(array, max_sequence_length, padding_character=0):
    return list(np.repeat(padding_character, max_sequence_length)) + array

def get_batch(training_data, batch_size, max_sequence_length):
    indices = []
    for i in range(batch_size):
        song_index = random.randint(0, len(training_data) - 1)
        starting_index = random.randint(0, len(training_data[song_index]) - 1)
        indices.append((song_index, starting_index))

    sequences = []
    for selection in indices:
        padded_song = pad(training_data[selection[0]], max_sequence_length)
        sequences.append(padded_song[selection[1]: selection[1] + max_sequence_length + 1])

    return sequences
  
def add_argument():
    parser=argparse.ArgumentParser(description='enwik8')

    parser.add_argument('--with_cuda', default=False, action='store_true',
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

EPOCHS = 20
GRADIENT_ACCUMULATE_EVERY = 4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024

# helpers

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model

model = PaLM(num_tokens = 256, dim = 512, depth = 8)

model = AutoregressiveWrapper(model, max_seq_len=2048)
model.cuda()

with open('pickle.pkl', 'rb') as f:
    x_train = pickle.load(f)  

batch = utils.get_batch(x_train, batch_size=100000, max_sequence_length=SEQ_LEN)
train_dataset = torch.tensor(batch).long()
batch = utils.get_batch(x_train, batch_size=10, max_sequence_length=SEQ_LEN)
val_dataset = torch.tensor(batch).long()

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
            model.eval()
            with torch.no_grad():
                inp = random.choice(val_dataset)[:-1]
                loss = model(inp[None, :].cuda())
                print(f'validation loss: {loss.item()}')

        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            print(f'%s \n\n %s', (prime, '*' * 100))

            sample = model.generate(inp[None, ...].cuda(), GENERATE_LENGTH)
            print(sample[0])
