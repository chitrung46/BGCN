import os
from numpy import copy
import torch
import time
import pynvml
from lib.logger import get_logger
from lib.metric import Metrics

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

class BasicTrainer:
    def __init__(self, args, optimizer, loss, scaler, model, train_loader, val_loader, test_loader, device):
        super(BasicTrainer, self).__init__()
        self.args = args
        self.optimizer = optimizer
        self.loss = loss
        self.scaler = scaler
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_train_batches = len(self.train_loader)
        self.num_val_batches = len(self.val_loader) if self.val_loader != None else len(self.test_loader)
        self.num_test_batches = len(self.test_loader)
        self.device = device
        self.best_path = os.path.join(self.args.log_dir, f'{self.args.model_name}.pth')
        
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info(args)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            batch_size = batch[0].size(0)
            input = [x.to(self.device) for x in batch]
            
            self.optimizer.zero_grad()

            seqs, labels, mask = batch
            logits = self.model(seqs)  # B x T x V
            logits = logits.view(-1, logits.size(-1))  # (B*T) x V
            labels = labels.view(-1)  # B*T
            loss = self.loss(logits, labels)

            loss.backward()

            # max grad clipping
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            # log information
            if (batch_idx + 1) % self.args.log_step == 0:
                self.logger.info('Train Epoch {} [{}/{}]: Loss: {:.6f}'.format(
                    epoch, (batch_idx + 1), self.num_train_batches, loss.item()))
            
        train_epoch_loss = total_loss / self.num_train_batches
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

        epoch_time = time.time() - start_time
        self.logger.info('====<>==== Train Epoch {}: Average loss: {:.4f} | GPU Memory Usage: {:.2f} GB | Training time: {:.2f} s'.format(
            epoch, train_epoch_loss, (meminfo.used - self.meminfo.used) / 1024**3, epoch_time))

        # learning rate decay
        if self.args.lr_decay:
            self.scheduler.step()

        accum_iter += batch_size

            
        return accum_iter
    
    def _val_epoch(self, epoch, val_loader):
        self.model.eval()
        metrics = []
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = [x.to(self.device) for x in batch]
                seqs, candidates, labels = batch
                scores = self.model(seqs) # B x T x V
                scores = scores[:, -1, :] # B x V
                scores = scores.gather(1, candidates) # B x C
                metrics.append(Metrics(scores, labels, self.args.ks))

                val_time = time.time() - start_time
                avg_metrics = {}
                for key in metrics[0].keys():   
                    avg_metrics[key] = sum([m[key] for m in metrics]) / len(metrics)

            self.logger.info('Average Validation Metrics: {}\n Validation time: {:.2f} s'.format(avg_metrics, val_time))
            
    def save_checkpoint(self):
        state = {
            'state_dict': copy.deepcopy(self.model.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
            'config': copy.deepcopy(self.args)
        }
        torch.save(state, self.best_path)
        self.logger.info('Checkpoint saved to {}'.format(self.best_path))

    def train(self):
        self.meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

        best_loss = float('inf')
        not_improved_count = 0

        self.logger.info('Training started...')

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self._train_epoch(epoch)          
            if train_loss > 1e6:
                self.logger.warning('Gradient exposion detected. Ending training... ')
                break

            if self.val_loader != None:
                val_dataloader = self.val_loader
            else:
                val_dataloader = self.test_loader   

            val_loss = self._val_epoch(epoch, val_dataloader)

            # Save best checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                not_improved_count = 0
                self.save_checkpoint()
                self.logger.info('Best model updated at epoch {}'.format(epoch))    
            else:
                not_improved_count += 1

            # Early stopping
            if self.args.early_stop:
                if not_improved_count >= self.args.early_stop_patience:
                    self.logger.info('No improvement for {} epochs. Training stops.'.format(self.args.early_stop_patience))
                    break
        self.logger.info('Training finished.')

    def test(self):
        if os.path.isfile(self.best_path):
            checkpoint = torch.load(self.best_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info('Loaded best checkpoint from {}'.format(self.best_path))
        else:
            self.logger.error('No checkpoint found at {}'.format(self.best_path))
        if self.test_loader != None:
            test_loader = self.test_loader
        else:
            self.logger.error('No test loader provided.')

        self.logger.info('Testing started...')
        self.model.eval()
        metrics = []
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch = [x.to(self.device) for x in batch]

                metrics.append(Metrics(batch))

        inference_time = time.time() - start_time
        avg_metrics = {}
        for key in metrics[0].keys():   
            avg_metrics[key] = sum([m[key] for m in metrics]) / len(metrics)

        self.logger.info('Testing finished.')
        self.logger.info('Average Test Metrics: {}\n Inference time: {:.2f} s'.format(avg_metrics, inference_time))