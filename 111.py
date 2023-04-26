import math
import multiprocessing
import os
from itertools import chain
import random
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from datasets import load_dataset
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    default_data_collator,
    get_linear_schedule_with_warmup
)
import numpy as np
from palm_rlhf_pytorch import PaLM
import pickle

from mgt.datamanagers.remi_data_manager import RemiDataManager
from mgt.datamanagers.data_helper import DataHelper
from mgt.datamanagers.remi.efficient_remi_config import EfficientRemiConfig

datamanager = RemiDataManager(
    efficient_remi_config=EfficientRemiConfig(enabled=True, remove_velocity=True)
)

# constants


class CFG:
    BATCH_SIZE: int = 12
    GRADIENT_ACCUMULATE_EVERY: int = 3
    SEED: int = 42
    LEARNING_RATE: float = 3e-4
    SEQ_LEN: int = 1024
    NUM_CPU: int = multiprocessing.cpu_count()
    RESUME_FROM_CHECKPOINT: str = "palm1"
    CHECKPOINTING_STEPS: int = 600
    OUTPUT_DIR: str = "palm"
    VALIDATION_STEPS: int = 500
    ENTITY_NAME: str = "a_man_chooses"


# helpers


def print_num_params(model, accelerator: Accelerator):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of parameters in model: {n_params}")


# dataloaders

class TextSampleDataset1(Dataset):
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
        return torch.tensor(a).long()
 
class TextSampleDataset2(Dataset):
    def __init__(self, data, max_length=1024):
        self.data = data
        self.max_length = max_length
        
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        song_index = random.randint(0, len(self.data) - 1)
        if len(self.data[song_index]) <= self.max_length:
          starting_index = random.randint(0, len(self.data[song_index]) - 1)
          padded_song = list(np.repeat(0, self.max_length))+self.data[song_index]   
          a = padded_song[0:self.max_length]
        else:
          starting_index = random.randint(0, len(self.data[song_index]) - self.max_length)
          a = self.data[song_index][starting_index: starting_index + self.max_length]
        return torch.tensor(a).long()

# helpers
data_train = DataHelper.load('/content/drive/MyDrive/set')
data_train = data_train.data
train_dataset = TextSampleDataset1(data_train, CFG.SEQ_LEN)
val_dataset = TextSampleDataset1(data_train, CFG.SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE)
val_loader= DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE)

def main():

    # accelerator

    accelerator = Accelerator(
        gradient_accumulation_steps=CFG.GRADIENT_ACCUMULATE_EVERY
    )

    accelerator.init_trackers(
        project_name="palm",
        config={
            "batch_size": CFG.BATCH_SIZE,
            "gradient_accumulate_every": CFG.GRADIENT_ACCUMULATE_EVERY,
            "learning_rate": CFG.LEARNING_RATE,
            "seq_len": CFG.SEQ_LEN,
            "validation_steps": CFG.VALIDATION_STEPS,
        },
        init_kwargs={"wandb": {"entity": CFG.ENTITY_NAME}},
    )

    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    # instantiate palm

    model = PaLM(
        num_tokens=7700, dim=512, depth=12, dim_head=128, heads=8, flash_attn=False
    )

    model = model.to(accelerator.device)

    print_num_params(model, accelerator)

    # optimizer

    optim = AdamW(
        model.parameters(), 
        lr=CFG.LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Determine number of training steps
    
    max_train_steps = math.ceil(len(train_loader) / CFG.GRADIENT_ACCUMULATE_EVERY)
    accelerator.print(f"Max train steps: {max_train_steps}")

    # lr scheduler
    # We cant decide on an actual number
    NUM_WARMUP_STEPS = int(max_train_steps * 0.069420)
    accelerator.print(f"Num warmup steps: {NUM_WARMUP_STEPS}")

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=NUM_WARMUP_STEPS * CFG.GRADIENT_ACCUMULATE_EVERY,
        num_training_steps=max_train_steps * CFG.GRADIENT_ACCUMULATE_EVERY,
    )

    # prepare

    model, optim, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optim, train_loader, val_loader, lr_scheduler
    )

    # checkpoint scheduler
    accelerator.register_for_checkpointing(lr_scheduler)

    # I do not know why Huggingface recommends recalculation of max_train_steps
    num_epochs = 3
    max_train_steps = math.ceil(len(train_loader) / CFG.GRADIENT_ACCUMULATE_EVERY * num_epochs)
    accelerator.print(f"Max train steps recalculated: {max_train_steps}")

    # Total batch size for logging

    total_batch_size = (
        CFG.BATCH_SIZE * accelerator.num_processes * CFG.GRADIENT_ACCUMULATE_EVERY
    )
    accelerator.print(f"Total batch size: {total_batch_size}")

    # resume training

    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    if CFG.RESUME_FROM_CHECKPOINT:
        if CFG.RESUME_FROM_CHECKPOINT is not None or CFG.RESUME_FROM_CHECKPOINT != "":
            accelerator.print(f"Resuming from checkpoint {CFG.RESUME_FROM_CHECKPOINT}")
            accelerator.load_state(CFG.RESUME_FROM_CHECKPOINT)
            path = os.path.basename(CFG.RESUME_FROM_CHECKPOINT)

    # training

    model.train()
    for epoch in range(num_epochs):
      for step, batch in enumerate(train_loader):
          with accelerator.accumulate(model):
              loss = model(batch, return_loss=True)
              accelerator.backward(loss)

              if accelerator.sync_gradients:
                  accelerator.clip_grad_norm_(model.parameters(), 1.0)

              optim.step()
              lr_scheduler.step()
              optim.zero_grad()

          if accelerator.sync_gradients:
              progress_bar.update(1)
              completed_steps += 1

          if isinstance(CFG.CHECKPOINTING_STEPS, int):
              if completed_steps % CFG.CHECKPOINTING_STEPS == 0:
                  output_dir = f"step_{completed_steps }"
                  if CFG.OUTPUT_DIR is not None:
                      output_dir = os.path.join(CFG.OUTPUT_DIR, output_dir)
                  accelerator.save_state(output_dir)

        # validation - I was following Lucidrains validation here...

          if step % CFG.VALIDATION_STEPS == 0:
              model.eval()
              with torch.no_grad():
                  for batch in val_loader:
                      loss = model(batch, return_loss=True)
                      accelerator.print(step)
                      accelerator.print(loss.item())

          if completed_steps >= max_train_steps:
              break
    
    model.eval()
    prompt = [2]
    initial = torch.tensor([prompt]).long().to(accelerator.device)
    sample = model.generate(1024, initial)
    sample = sample.cpu().detach().numpy()[0]
    midi = datamanager.to_midi(sample)
    midi.save("1.midi")
    accelerator.save_state(output_dir="palm1")
    accelerator.end_training()

if __name__ == "__main__":
    main()
