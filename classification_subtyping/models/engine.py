import os
import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import torch.nn.functional as F
import pandas as pd
import torch.optim
import torch.nn.parallel


class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold

        # tensorboard
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        self.best_auc = 0
        self.best_epoch = 0
        self.filename_best = None
        self.best_metrics = {}

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()

        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_auc = checkpoint['best_auc']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (auc: {})".format(checkpoint['best_auc']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.validate(val_loader, model, criterion)
            return
        
        
        
        
        early_stop_patience = getattr(self.args, 'early_stop_patience', 50)
        early_stop_counter = 0
        min_delta = 1e-6

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer)
            # evaluate on validation set
            metrics = self.validate(val_loader, model, criterion)
            # remember best c-index and save checkpoint
            is_best = metrics['auc'] > self.best_auc
            if is_best:
                self.best_auc = metrics['auc']
                self.best_epoch = self.epoch
                self.best_metrics = metrics
                early_stop_counter=0
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_auc': self.best_auc})
            else:
                early_stop_counter += 1

            print(' *** best auc={:.4f} at epoch {}'.format(self.best_auc, self.best_epoch))
            print(' *** early stop counter: {}/{}'.format(early_stop_counter, early_stop_patience))

            if scheduler is not None:
                if self.args.scheduler == 'plateau':
                    scheduler.step(metrics['auc'])
                else:
                    scheduler.step()

            if early_stop_patience is not None and early_stop_patience > 0 and early_stop_counter >= early_stop_patience:
                print('Early stopping triggered at epoch {}. Best auc={:.4f} at epoch {}'.format(
                    epoch, self.best_auc, self.best_epoch
                ))
                break

        return self.best_metrics, self.best_epoch

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        train_loss = 0.0
        all_labels = []
        all_preds = []
        dataloader = tqdm(data_loader, desc='Train Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI, label, case_id) in enumerate(dataloader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                label = label.type(torch.LongTensor).cuda()

            Y_hat = model(x_path=data_WSI)
            loss = criterion(Y_hat, label)
            
            train_loss += loss.item()
            
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(F.softmax(Y_hat, dim=1).detach().cpu().numpy()[:, 1])

            loss.backward()
            optimizer.step()

        # calculate loss and error for epoch
        train_loss /= len(dataloader)
        
        # Calculate metrics
        auc_value = roc_auc_score(all_labels, all_preds)
        pred_labels = (np.array(all_preds) > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, pred_labels)
        precision = precision_score(all_labels, pred_labels)
        recall = recall_score(all_labels, pred_labels)
        fscore = f1_score(all_labels, pred_labels)
        tn, fp, fn, tp = confusion_matrix(all_labels, pred_labels).ravel()
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0


        print('loss: {:.4f}, accuracy: {:.4f}, auc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, specificity: {:.4f}, npv: {:.4f}'.format(
            train_loss, accuracy, auc_value, precision, recall, fscore, specificity, npv))

        if self.writer:
            self.writer.add_scalar('train/loss', round(train_loss, 4), self.epoch)
            self.writer.add_scalar('train/accuracy', round(accuracy, 4), self.epoch)
            self.writer.add_scalar('train/auc', round(auc_value, 4), self.epoch)
            self.writer.add_scalar('train/precision', round(precision, 4), self.epoch)
            self.writer.add_scalar('train/recall', round(recall, 4), self.epoch)
            self.writer.add_scalar('train/f1', round(fscore, 4), self.epoch)
            self.writer.add_scalar('train/specificity', round(specificity, 4), self.epoch)
            self.writer.add_scalar('train/npv', round(npv, 4), self.epoch)


    def validate(self, data_loader, model, criterion):
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        case_id_list = []

        dataloader = tqdm(data_loader, desc='Test Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI, label, case_id) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                label = label.type(torch.LongTensor).cuda()

            with torch.no_grad():
                Y_hat = model(x_path=data_WSI)

            loss = criterion(Y_hat, label)
            val_loss += loss.item()

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(F.softmax(Y_hat, dim=1).detach().cpu().numpy()[:, 1])
            case_id_list.extend(case_id)

        val_loss /= len(dataloader)
        
        # Calculate metrics
        auc_value = roc_auc_score(all_labels, all_preds)
        pred_labels = (np.array(all_preds) > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, pred_labels)
        precision = precision_score(all_labels, pred_labels)
        recall = recall_score(all_labels, pred_labels)
        fscore = f1_score(all_labels, pred_labels)
        tn, fp, fn, tp = confusion_matrix(all_labels, pred_labels).ravel()
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        print('loss: {:.4f}, accuracy: {:.4f}, auc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, specificity: {:.4f}, npv: {:.4f}'.format(
            val_loss, accuracy, auc_value, precision, recall, fscore, specificity, npv))

        if self.writer:
            self.writer.add_scalar('val/loss', round(val_loss, 4), self.epoch)
            self.writer.add_scalar('val/accuracy', round(accuracy, 4), self.epoch)
            self.writer.add_scalar('val/auc', round(auc_value, 4), self.epoch)
            self.writer.add_scalar('val/precision', round(precision, 4), self.epoch)
            self.writer.add_scalar('val/recall', round(recall, 4), self.epoch)
            self.writer.add_scalar('val/f1', round(fscore, 4), self.epoch)
            self.writer.add_scalar('val/specificity', round(specificity, 4), self.epoch)
            self.writer.add_scalar('val/npv', round(npv, 4), self.epoch)

        metrics = {
            'auc': round(auc_value, 4),
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'fscore': round(fscore, 4),
            'specificity': round(specificity, 4),
            'npv': round(npv, 4)
        }
        return metrics

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir, 'fold_' + str(self.fold),
                                          'model_best_{auc:.4f}_{epoch}.pth.tar'.format(auc=state['best_auc'],
                                                                                          epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
 