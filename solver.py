import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from model import *
import logging
# import wandb
from modules.loss import CosineAlignLoss, FocalLoss
from tqdm import tqdm
from dataset import class_weights

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params  # args
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.is_train = is_train
        self.model = model = MMA(hp)
        model.print_trainable_parameters()
        # Training hyperarams

        self.update_batch = hp.update_batch  

        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            
        else:
            self.device = torch.device("cpu")

        # self.criterion = nn.L1Loss(reduction="mean")  
        # self.criterion = nn.HuberLoss(reduction='mean')
        self.crit_sa = nn.L1Loss(reduction="mean")  
        # self.crit_er = nn.CrossEntropyLoss(weight=class_weights)
        num_cls = None
        if hp.use_cross_iemocap_labels or hp.dataset == 'iemocap':
            num_cls = 6
        elif hp.use_cross_meld_labels or hp.dataset == 'meld':
            num_cls = 7
        self.crit_er = FocalLoss(alpha=class_weights, reduction='mean', num_classes=num_cls)
        self.align_crit = CosineAlignLoss()

        # optimizer
        self.optimizer={}

        if self.is_train:
            # mmilb_param = []
            main_param = []
            bert_param = []
            lora_param = []
            adapter_param = []

            for name, p in model.named_parameters():
                if p.requires_grad:
                    if 'lora_' in name: 
                        lora_param.append(p)
                    elif 'adapter' in name:
                        adapter_param.append(p)
                    else: 
                        main_param.append(p)
                
            for p in main_param:
                if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                    nn.init.xavier_normal_(p) 

        
        optimizer_main_group = [
            {'params': lora_param, 'weight_decay': hp.weight_decay_lora, 'lr': hp.lr_lora},
            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main},
            {'params': adapter_param, 'weight_decay': hp.weight_decay_adapter, 'lr': hp.lr_adapter}
        ]


        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
            optimizer_main_group
        )

        self.scheduler_main = StepLR(self.optimizer_main, step_size=self.hp.when, gamma=0.1)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        model = model.cuda()
        optimizer_main = self.optimizer_main
        scheduler_main = self.scheduler_main

        def train(model, optimizer):
            epoch_loss = 0

            model.train()
            proc_loss, proc_size = 0, 0
            multi_con_loss = 0.0

            left_batch = self.update_batch
            expert_distribution = []
            for batch_data in tqdm(self.train_loader, desc="Train", leave=False):
                vision = batch_data['vision']
                audio = batch_data['audio']
                text = batch_data['text']
                labels = batch_data['labels']
                y_sa = labels.get('M', None)
                y_er = labels.get('ER', None)

                model.zero_grad()
                with torch.cuda.device(0):
                    vision, audio, text, y_sa, y_er = vision.cuda(), audio.cuda(), text.cuda(), y_sa.cuda(), y_er.cuda()
                
                if (y_sa != None):
                    batch_size = y_sa.size(0)
                else:
                    batch_size = y_er.size(0)
                pred_sa, pred_er, LBLoss, pooled_sa, pooled_er = model(vision, audio, text)

                if y_sa is not None:
                    loss_sa = self.crit_sa(pred_sa, y_sa)
                else:
                    loss_sa = 0.0
                if y_er is not None:
                    loss_er = self.crit_er(pred_er, y_er.view(-1).long())
                else:
                    loss_er = 0.0

                loss_align = self.align_crit(pooled_sa, pooled_er)

                lambda_align = self.hp.lambda_align
                lambda_lb = self.hp.lambda_lb

                loss = (loss_sa if isinstance(loss_sa, torch.Tensor) else torch.tensor(
                    float(loss_sa)).to(self.device)) + (loss_er if isinstance(loss_er, torch.Tensor) else torch.tensor(
                        float(loss_er)).to(self.device)) + lambda_lb*LBLoss
                loss.backward()
                
                # -------------------------------------------------------- #
                left_batch -= 1  
                if left_batch == 0: 
                    left_batch = self.update_batch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)  
                    optimizer.step()
                # -------------------------------------------------------- #

                proc_loss += loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += loss.item() * batch_size
                
                    
            return epoch_loss

        def evaluate(model, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            main_loss = 0.0        
            results_sa = []
            results_er = []
            truths_sa = []
            truths_er = []
            # expert_output = torch.zeros((12,1,769))
            # expert_distribution = []
            with torch.no_grad():
                for batch_data in tqdm(loader, desc='eval', leave=False):
                    vision = batch_data['vision']
                    audio = batch_data['audio']
                    text = batch_data['text']
                    labels = batch_data['labels']
                    y_sa = labels.get('M', None)
                    y_er = labels.get('ER', None)

                    with torch.cuda.device(0):
                        vision, audio, text, y_sa, y_er = vision.cuda(), audio.cuda(), text.cuda(), y_sa.cuda(), y_er.cuda()

                    pred_sa, pred_er, _, _, _ = model(vision, audio, text)
                    if y_sa != None:
                        results_sa.append(pred_sa)
                        truths_sa.append(y_sa)
                    if y_er != None:
                        results_er.append(pred_er)
                        truths_er.append(y_er)
            
            avg_main_loss_sa = None
            avg_main_loss_er = None
            if len(results_sa) > 0:
                results_sa = torch.cat(results_sa)
                truths_sa = torch.cat(truths_sa)
                test_preds_sa = results_sa.view(-1).cpu().detach().numpy()
                test_truth_sa = truths_sa.view(-1).cpu().detach().numpy()
                avg_main_loss_sa = np.mean(np.absolute(test_preds_sa - test_truth_sa))
            if len(results_er) > 0:
                results_er = torch.cat(results_er)
                truths_er = torch.cat(truths_er)
                test_preds_er = results_er
                test_truth_er = truths_er.view(-1).long().to(test_preds_er.device)
                avg_main_loss_er = self.crit_er(test_preds_er, test_truth_er)

            return avg_main_loss_sa, avg_main_loss_er, results_sa, truths_sa, results_er, truths_er

        best_valid = 1e8
        patience = self.hp.patience

        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()
            logging.info(f'epoch {epoch}:')

            self.epoch = epoch

            train_main_loss = train(model, optimizer_main)
            val_loss, loss_er, _, _, _, _ = evaluate(model, test=False) 
            test_loss, test_loss_er, results, truths, test_results_er, test_truths_er = evaluate(model, test=True)     
            
            end = time.time()
            duration = end-start
            # scheduler_main.step(val_loss)    # Decay learning rate by validation loss
            
            scheduler_main.step()
            learning_rate = optimizer_main.state_dict()['param_groups'][0]['lr']

            # print(f'training on epoch {epoch}')
            # print(f'learnng rate: {learning_rate}')
            print("-"*50)
            print(
                'Epoch {:2d} | Time {:5.4f} sec | Valid Loss MSA {:5.4f} | Test Loss MSA {:5.4f} | Valid Loss MER {:5.4f} | Test Loss MER {:5.4f}'.format(
                    epoch, duration, val_loss, test_loss, loss_er, test_loss_er))
            print("-"*50)
            

            if val_loss + loss_er < best_valid:
                # model.model.save()
               
                patience = self.hp.patience
                best_valid = val_loss + loss_er
                best_epoch = epoch
                if self.hp.dataset in ["mosei_senti", "mosei"]:
                    eval_mosei_senti(results, truths, True)
                    eval_emotionlines(test_results_er.argmax(dim=1).cpu().numpy() if torch.is_tensor(test_results_er) else test_results_er,
                                      test_truths_er.cpu().numpy() if torch.is_tensor(test_truths_er) else test_truths_er)
                elif self.hp.dataset == 'mosi':
                    eval_mosi(results, truths, True)
                    eval_emotionlines(test_results_er.argmax(dim=1).cpu().numpy() if torch.is_tensor(test_results_er) else test_results_er,
                                      test_truths_er.cpu().numpy() if torch.is_tensor(test_truths_er) else test_truths_er)
                best_results = results
                best_truths = truths
                best_results_er = test_results_er.argmax(dim=1).cpu().numpy() if torch.is_tensor(test_results_er) else test_results_er
                best_truths_er = test_truths_er.cpu().numpy() if torch.is_tensor(test_truths_er) else test_truths_er

            else:
                patience -= 1
                if patience == 0:
                    break

        # print(f'Best epoch: {best_epoch}')
        logging.info(f'Best epoch: {best_epoch}')

        if self.hp.dataset in ["mosei_senti", "mosei"]:
            best_dict = eval_mosei_senti(best_results, best_truths, True)
            best_dict_er = eval_emotionlines(best_results_er, best_truths_er)
        elif self.hp.dataset == 'mosi':
            best_dict = eval_mosi(best_results, best_truths, True)
            best_dict_er = eval_emotionlines(best_results_er, best_truths_er)
        sys.stdout.flush() 
        return best_dict, best_dict_er