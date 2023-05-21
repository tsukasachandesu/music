from __future__ import annotations

import time

import numpy as np
import torch
from MEGABYTE_pytorch import MEGABYTE
from x_transformers import Decoder
from mgt.models import utils
from mgt.models.compound_word_transformer.compound_word_autoregressive_wrapper import CompoundWordAutoregressiveWrapper
from mgt.models.compound_word_transformer.compound_word_transformer_utils import COMPOUND_WORD_BAR, get_batch
from mgt.models.compound_word_transformer.compound_word_transformer_wrapper import CompoundWordTransformerWrapper
from mgt.models.utils import get_device



defaults = {
    'num_tokens': 6912+16+4,
    'emb_sizes': 512,
    'max_seq_len': 512,
    'learning_rate': 1e-4,
    'dropout': 0.1,
    'dim': 512,
    'depth': (6, 2),
    'heads': 8
}


class CompoundWordTransformerModel(object):

    def __init__(self,
                 num_tokens=defaults['num_tokens'],
                 emb_sizes=defaults['emb_sizes'],
                 max_seq_len=defaults['max_seq_len'],
                 learning_rate=defaults['learning_rate'],
                 dropout=defaults['dropout'],
                 dim=defaults['dim'],
                 depth=defaults['depth'],
                 heads=defaults['heads']
                 ):
        self.num_tokens = num_tokens
        self.emb_sizes = emb_sizes
        self.learning_rate = learning_rate
        self.max_seq_len  = max_seq_len
        self.dropout = dropout
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = self.create_optimizer()

    def train(self,
              x_train,
              epochs,
              batch_size=4,
              stop_loss=None,
              batches_per_epoch=100,
              report_per_x_batches=20,
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
                    batch = get_batch(
                        x_train,
                        batch_size=batch_size,
                        max_sequence_length=self.max_seq_len)

                    torch_batch = torch.tensor(np.array(batch)).long().to(utils.get_device())
                    loss = self.model(torch_batch, return_loss = True)
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

    def generate(self, output_length=100, prompt=None):
        print(f"Generating a new song with {output_length} characters.")
        sample = self.model.generate()
        return sample

    def create_model(self):
        
        model = MEGABYTE(
            num_tokens = 6912 + 16 + 4,
            dim = 512,
            depth = (6, 2),
            max_seq_len = (1024, 8),
            flash_attn = False,
            pad_id = 6929
        ).to(get_device())

        return model

    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'num_tokens': self.num_tokens,
            'emb_sizes': self.emb_sizes,
            'max_seq_len': self.max_seq_len,
            'learning_rate': self.learning_rate,
            'dropout': self.dropout,
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    @staticmethod
    def load_checkpoint(path) -> CompoundWordTransformerModel:
        checkpoint = torch.load(path)
        model = CompoundWordTransformerModel(
            num_tokens=utils.get_or_default(checkpoint, 'num_tokens', defaults),
            emb_sizes=utils.get_or_default(checkpoint, 'emb_sizes', defaults),
            max_seq_len=utils.get_or_default(checkpoint, 'max_seq_len', defaults),
            learning_rate=utils.get_or_default(checkpoint, 'learning_rate', defaults),
            dropout=utils.get_or_default(checkpoint, 'dropout', defaults),
            dim=utils.get_or_default(checkpoint, 'dim', defaults),
            depth=utils.get_or_default(checkpoint, 'depth', defaults),
            heads=utils.get_or_default(checkpoint, 'heads', defaults)
        )

        model.model.load_state_dict(checkpoint['model_state_dict'])

        return model
