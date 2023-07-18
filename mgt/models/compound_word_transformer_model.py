from __future__ import annotations

import time

import numpy as np
import torch
from x_transformers import Decoder
from x_transformers import Encoder
.
from x_transformers import CrossAttender

from mgt.models import utils
from mgt.models.compound_word_transformer.compound_word_autoregressive_wrapper import CompoundWordAutoregressiveWrapper
from mgt.models.compound_word_transformer.compound_word_transformer_utils import COMPOUND_WORD_BAR, get_batch
from mgt.models.compound_word_transformer.compound_word_transformer_wrapper import CompoundWordTransformerWrapper
from mgt.models.utils import get_device


defaults = {
    'num_tokens': [
        18,   # Bar / Beat
        6913,  # Tempo
        6913,  # Instrument
        6913,   # Note name
        6913,    # Octave
        6913, 
        6913,# Duration
    ],
    'emb_sizes': [
        512,   # Bar / Beat
        512,  # Tempo
        512,  # Instrument
        512,  # Note Name
        512,  # Octave
        512,
        512,# Duration
    ],
    'max_sequence_length': 756,
    'learning_rate': 1e-4,
    'dropout': 0.1,
    'dim': 512,
    'depth': 24,
    'heads': 8
}


class CompoundWordTransformerModel(object):

    def __init__(self,
                 num_tokens=defaults['num_tokens'],
                 emb_sizes=defaults['emb_sizes'],
                 max_sequence_length=defaults['max_sequence_length'],
                 learning_rate=defaults['learning_rate'],
                 dropout=defaults['dropout'],
                 dim=defaults['dim'],
                 depth=defaults['depth'],
                 heads=defaults['heads']
                 ):
        self.num_tokens = num_tokens
        self.emb_sizes = emb_sizes
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
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
              batch_size=6,
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
                        max_sequence_length=self.max_sequence_length)

                    torch_batch = torch.tensor(np.array(batch)).long().to(utils.get_device())

                    losses = self.model.train_step(torch_batch)
                    loss = sum(losses) 
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

        if prompt is None:
            prompt = [COMPOUND_WORD_BAR]  # Bar

        self.model.eval()
        sample = self.model.generate(output_length=output_length, prompt=prompt)
        return sample

    def create_model(self):
        model = CompoundWordAutoregressiveWrapper(CompoundWordTransformerWrapper(
            num_tokens=self.num_tokens,
            emb_sizes=self.emb_sizes,
            max_seq_len=self.max_sequence_length,
            attn_layers=Decoder(
                dim=self.dim,
                depth=24,
                heads=self.heads,
                ff_glu = True,
                ff_swish = True,
                use_rmsnorm = True,
                alibi_pos_bias = True,
                alibi_num_heads = 4,   
                layer_dropout = self.dropout,
                attn_dropout=self.dropout,  # dropout post-attention
                ff_dropout=self.dropout,  # feedforward dropout
                ff_no_bias = True,
                attn_one_kv_head = True,
                shift_tokens = 1
            ),
            attn_layers1=CrossAttender(
                dim=512,
                depth=12,
                heads=8,
                ff_glu = True,
                ff_swish = True,
                use_rmsnorm = True,            
                layer_dropout = self.dropout,
                attn_dropout=self.dropout,  
                ff_dropout=self.dropout,
                ff_no_bias = True,
                attn_one_kv_head = True,
                dynamic_pos_bias = True,                # set this to True
                dynamic_pos_bias_log_distance = False   # whether to use log distance, as in SwinV2
            )
        )).to(get_device())

        return model

    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'num_tokens': self.num_tokens,
            'emb_sizes': self.emb_sizes,
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
    def load_checkpoint(path) -> CompoundWordTransformerModel:
        checkpoint = torch.load(path)
        model = CompoundWordTransformerModel(
            num_tokens=utils.get_or_default(checkpoint, 'num_tokens', defaults),
            emb_sizes=utils.get_or_default(checkpoint, 'emb_sizes', defaults),
            max_sequence_length=utils.get_or_default(checkpoint, 'max_sequence_length', defaults),
            learning_rate=utils.get_or_default(checkpoint, 'learning_rate', defaults),
            dropout=utils.get_or_default(checkpoint, 'dropout', defaults),
            dim=utils.get_or_default(checkpoint, 'dim', defaults),
            depth=utils.get_or_default(checkpoint, 'depth', defaults),
            heads=utils.get_or_default(checkpoint, 'heads', defaults)
        )

        model.model.load_state_dict(checkpoint['model_state_dict'])

        return model
