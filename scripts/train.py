from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

import os
import json
import logging

from models.crrn import CRNN
from data.dataloader import create_dataloaders

class Trainer:
    def __init__(self, config_file, **kwargs):
        ## take configs
        self.config = self._load_config(config_file)
        
        # take the jeyword args
        for key in ['cehckpoints_path']:
            if key in kwargs:
                self.config[key] = kwargs[key]
        
        self.device = self.config['device']
        self.batch_size = self.config['batch_size']
        self.checkpoints_path = self.config['checkpoints_path']
        self.rho = self.config['rho']
        self.epoch_num = self.config['epoch_num']
        self.log_per_epoch = self.config['log_per_epoch']
        self.patience_limit = self.config['patience_limit']
        self.curr_patience = 0
        self.best_val_loss = 0.0

        # create checkpoint dir
        os.makedirs(self.checkpoints_path, exist_ok=True)
        
        # initalize model
        tokens = [''] + list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.model = CRNN(num_classes=53, lexicon_path=None, tokens=tokens)
        self.model = self.model.to(self.device)
        
        # create dataloders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(self.batch_size)
        
        # get the lost_step
        self.log_step = len(self.train_loader) // self.log_per_epoch
        
        # loss function and optimizer
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        self.opt = optim.Adadelta(self.model.parameters(), rho=self.rho)
        
        # setup logging
        self.train_log_file = open(os.path.join(self.checkpoints_path, 'training.log'), 'a')
        self.val_log_file = open(os.path.join(self.checkpoints_path, 'val.log'), 'a')
        
        
        
    @staticmethod
    def _load_config(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
        
    
    def evaluate_val_loss(self, epoch, step):
        # do the evaluation on validation set
        self.model.eval()
        avg_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for val_batch in self.val_loader:
                img, label, label_lengths = val_batch
                
                img = img.to(self.device)
                label = label.to(self.device)
                label_lengths = label_lengths.to(self.device)
                
                logits = self.model(img)
            
                T = logits.size(1) 
                N = logits.size(0) 
                logits = logits.permute(1, 0, 2)
                input_lengths = torch.full((N,), T, dtype=torch.long).to(self.device)
                
                avg_val_loss += self.ctc_loss(logits, label, input_lengths, label_lengths)
            
            val_steps += 1
        
        
        avg_val_loss = avg_val_loss / val_steps
        avg_val_loss = round(avg_val_loss, 4)
        self.val_log_file.write(f"{avg_val_loss}\n")
        self.val_log_file.flush()
        tqdm.write(f"Validation Loss at epoch: {epoch} - step {step}: {avg_val_loss}")
        self.model.train()
        return avg_val_loss
    
    
    def save_checkpoint(self, epoch, step, val_loss):
        checkpoint_path = os.path.join(self.checkpoints_path, f"epoch{epoch}_step{step}_loss{val_loss}.pt")
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }, checkpoint_path)
        tqdm.write(f"Checkpoint saved at {checkpoint_path}")
    
    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.epoch_num), desc="Epochs"):
            total_loss = 0
            batch_progress_bar = tqdm(enumerate(self.train_dataloader),
                                    total=len(self.train_dataloader),
                                    desc=f"Epoch {epoch+1}/{self.epoch_num}",
                                    leave=False
                                    ) 
            
        
        for step, batch in batch_progress_bar:
            img, label, label_lengths = batch

            # move tensors to device
            img = img.to(self.device)
            label = label.to(self.device)
            label_lengths = label_lengths.to(self.device)
            
            # forward propagation
            logits = self.model(img)
            
            T = logits.size(1) 
            N = logits.size(0) 
            logits = logits.permute(1, 0, 2)  # (T, N, C)
            input_lengths = torch.full((N,), T, dtype=torch.long).to(self.device)
            
            # calculate loss
            loss = self.ctc_loss(logits, label, input_lengths, label_lengths)
            
            # backward propagation
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            self.train_log_file.write(f"{loss.item():.4f}\n")
            self.train_log_file.flush()
            batch_progress_bar.set_postfix(Step=step+1, Loss=round(loss.item(), 4))
            
            # get the loss on validation set and save checkpoit
            if (step+1) * self.log_step == 0 or step == len(self.train_loader) - 1:
                val_loss = self.evaluate_val_loss(epoch+1, step+1)
                self.save_checkpoint(epoch+1, step+1, val_loss)
                
                # check patience
                if val_loss < self.best_val_loss:
                    self.curr_patience = 0
                    self.best_val_loss = val_loss
                else:
                    self.curr_patience += 1
                    if self.curr_patience >= self.patience_limit:
                        # stop the training
                        return