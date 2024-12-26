from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, transfer_weights, show_matrix, cal_accuracy
from utils.augmentations import masked_data
from utils.metrics import metric
from utils.losses import MaskedMSELoss
from utils.masking import mask_function
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import shutil
from tensorboardX import SummaryWriter
import random
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
from collections import OrderedDict

warnings.filterwarnings('ignore')


class Exp_Pretrain_PatchTST(Exp_Basic):

    def __init__(self, args):
        super(Exp_Pretrain_PatchTST, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))
            model = transfer_weights(self.args.load_checkpoints, model)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):

        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.task == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def pretrain(self):

        print("{}>\t mask_rule: patch_masking\tmask_rate: {}".format('-'*50, self.args.mask_rate))

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim, T_max=self.args.train_epochs)
        min_vali_loss = None

        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            train_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler, epoch)
            vali_loss = self.valid_one_epoch(vali_loader, epoch)

            end_time = time.time()
            print("Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f} Val Loss: {4:.4f}"
                  .format(epoch, model_scheduler.get_lr()[0], end_time-start_time, train_loss, vali_loss))

            loss_scalar_dict = {
                'train_loss': train_loss,
                'vali_loss': vali_loss,
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print("Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, vali_loss, epoch))

                min_vali_loss = vali_loss
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
                torch.save(encoder_ckpt, os.path.join(path, "ckpt_best.pth"))

            if (epoch + 1) % 5 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler, epoch):

        train_loss = []

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()

            # To device
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            # model
            loss, _, _, _ = self.model(batch_x, batch_x_mark) # pred/target: [bs x num_patch x n_vars x patch_len] mask: [bs x num_patch x n_vars]

            # Backward
            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())

        model_scheduler.step()
        train_loss = np.average(train_loss)

        return train_loss

    def valid_one_epoch(self, vali_loader, epoch):
        valid_loss = []

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

            # To device
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            # model
            loss, _, _, _ = self.model(batch_x, batch_x_mark) # pred/target: [bs x num_patch x n_vars x patch_len] mask: [bs x num_patch x n_vars]
            valid_loss.append(loss.item())

        vali_loss = np.average(valid_loss)

        self.model.train()
        return vali_loss


class Exp_Pretrain_TimeSiam(Exp_Basic):

    def __init__(self, args):
        super(Exp_Pretrain_TimeSiam, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")
        self.loss_fn = self._select_criterion()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))
            model = transfer_weights(self.args.load_checkpoints, model, device=self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        if self.args.task == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = MaskedMSELoss()
        return criterion

    def pretrain(self):

        print("{}> pretrain seq len: {} \t sampling_range: {}*{} \t lineage_tokens: {} \t mask_rule: {} \t mask_rate: {} \t tokens_using: {} \t representation_using: {}<{}"
              .format('-'*50, self.args.seq_len, self.args.sampling_range, self.args.seq_len, self.args.lineage_tokens, self.args.masked_rule, self.args.mask_rate, self.args.tokens_using, self.args.representation_using, '-'*50))

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim, T_max=self.args.train_epochs)
        min_vali_loss = None

        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            train_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
            vali_loss = self.valid_one_epoch(vali_loader)

            end_time = time.time()
            print("Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f} Val Loss: {4:.4f}"
                  .format(epoch, model_scheduler.get_lr()[0], end_time-start_time, train_loss, vali_loss))

            loss_scalar_dict = {
                'train_loss': train_loss,
                'vali_loss': vali_loss,
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print("Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, vali_loss, epoch))
                min_vali_loss = vali_loss
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
                torch.save(encoder_ckpt, os.path.join(path, "ckpt_best.pth"))

            if (epoch + 1) % 5 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.model.state_dict()}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):

        train_loss = []

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, segment) in enumerate(train_loader):
            model_optim.zero_grad()

            past_x, cur_x = batch_x, batch_y

            _, cur_x_, mask = mask_function(cur_x, self.args)

            # to device
            past_x = past_x.float().to(self.device)
            cur_x = cur_x.float().to(self.device)
            cur_x_ = cur_x_.float().to(self.device)
            mask = mask.to(self.device)

            # Encoder
            pred_cur = self.model(past_x, None, cur_x_, None, segment=segment, mask=mask)
            
            # import pdb; pdb.set_trace()

            if self.args.mask_rate == 0 or self.args.task == 'classification':
                loss = self.loss_fn(pred_cur, cur_x)
            else:
                loss = self.loss_fn(pred_cur, cur_x, ~mask)

            # Backward
            loss.backward()
            model_optim.step()
            train_loss.append(loss.item())

        model_scheduler.step()
        train_loss = np.average(train_loss)

        return train_loss

    def valid_one_epoch(self, vali_loader):
        valid_loss = []

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, segment) in enumerate(vali_loader):

            past_x, cur_x = batch_x, batch_y

            _, cur_x_, mask = mask_function(cur_x, self.args)

            # to device
            past_x = past_x.float().to(self.device)
            cur_x = cur_x.float().to(self.device)
            cur_x_ = cur_x_.float().to(self.device)
            mask = mask.to(self.device)

            # Encoder
            pred_cur = self.model(past_x, None, cur_x_, None, segment=segment, mask=mask)

            if self.args.mask_rate == 0 or self.args.task == 'classification':
                loss = self.loss_fn(pred_cur, cur_x)
            else:
                loss = self.loss_fn(pred_cur, cur_x, ~mask)

            valid_loss.append(loss.item())

        vali_loss = np.average(valid_loss)

        self.model.train()
        return vali_loss

class Exp_Pretrain_SimMTM(Exp_Basic):
    def __init__(self, args):
        super(Exp_Pretrain_SimMTM, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))

            model = transfer_weights(self.args.load_checkpoints, model, device=self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.task == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def pretrain(self):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # show cases
        self.train_show = next(iter(train_loader))
        self.valid_show = next(iter(vali_loader))

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        #model_optim.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,
                                                                     T_max=self.args.train_epochs)

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            train_loss, train_cl_loss, train_rb_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
            vali_loss, valid_cl_loss, valid_rb_loss = self.valid_one_epoch(vali_loader)

            # log and Loss
            end_time = time.time()
            print(
                "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f}/{4:.4f}/{5:.4f} Val Loss: {6:.4f}/{7:.4f}/{8:.4f}"
                .format(epoch, model_scheduler.get_lr()[0], end_time - start_time, train_loss, train_cl_loss,
                        train_rb_loss, vali_loss, valid_cl_loss, valid_rb_loss))

            loss_scalar_dict = {
                'train_loss': train_loss,
                'train_cl_loss': train_cl_loss,
                'train_rb_loss': train_rb_loss,
                'vali_loss': vali_loss,
                'valid_cl_loss': valid_cl_loss,
                'valid_rb_loss': valid_rb_loss,
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, vali_loss, epoch))

                min_vali_loss = vali_loss
                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'enc_embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')  # multi-gpu
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'enc_embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

                self.show(5, epoch + 1, 'train')
                self.show(5, epoch + 1, 'valid')

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):

        train_loss = []
        train_cl_loss = []
        train_rb_loss = []

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()

            if self.args.select_channels < 1:

                # random select channels
                B, S, C = batch_x.shape
                random_c = int(C * self.args.select_channels)
                if random_c < 1:
                    random_c = 1

                index = torch.LongTensor(random.sample(range(C), random_c))
                batch_x = torch.index_select(batch_x, 2, index)

            # data augumentation
            batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, self.args.mask_rate, self.args.lm, self.args.positive_nums)
            batch_x_om = torch.cat([batch_x, batch_x_m], 0)

            batch_x = batch_x.float().to(self.device)
            # batch_x_mark = batch_x_mark.float().to(self.device)

            # masking matrix
            mask = mask.to(self.device)
            mask_o = torch.ones(size=batch_x.shape).to(self.device)
            mask_om = torch.cat([mask_o, mask], 0).to(self.device)

            # to device
            batch_x = batch_x.float().to(self.device)
            batch_x_om = batch_x_om.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            # encoder
            loss, loss_cl, loss_rb, _, _, _, _ = self.model(batch_x_om, batch_x_mark, batch_x, mask=mask_om)

            # backward
            loss.backward()
            model_optim.step()

            # record
            train_loss.append(loss.item())
            train_cl_loss.append(loss_cl.item())
            train_rb_loss.append(loss_rb.item())

        model_scheduler.step()

        train_loss = np.average(train_loss)
        train_cl_loss = np.average(train_cl_loss)
        train_rb_loss = np.average(train_rb_loss)

        return train_loss, train_cl_loss, train_rb_loss

    def valid_one_epoch(self, vali_loader):
        valid_loss = []
        valid_cl_loss = []
        valid_rb_loss = []

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            # data augumentation
            batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, self.args.mask_rate, self.args.lm,
                                                          self.args.positive_nums)
            batch_x_om = torch.cat([batch_x, batch_x_m], 0)

            # masking matrix
            mask = mask.to(self.device)
            mask_o = torch.ones(size=batch_x.shape).to(self.device)
            mask_om = torch.cat([mask_o, mask], 0).to(self.device)

            # to device
            batch_x = batch_x.float().to(self.device)
            batch_x_om = batch_x_om.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            # encoder
            loss, loss_cl, loss_rb, _, _, _, _ = self.model(batch_x_om, batch_x_mark, batch_x, mask=mask_om)

            # Record
            valid_loss.append(loss.item())
            valid_cl_loss.append(loss_cl.item())
            valid_rb_loss.append(loss_rb.item())

        vali_loss = np.average(valid_loss)
        valid_cl_loss = np.average(valid_cl_loss)
        valid_rb_loss = np.average(valid_rb_loss)

        self.model.train()
        return vali_loss, valid_cl_loss, valid_rb_loss

    def train(self, setting):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Optimizer
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            start_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                if self.args.select_channels < 1:
                    # Random select channels
                    B, S, C = batch_x.shape
                    random_c = int(C * self.args.select_channels)
                    if random_c < 1:
                        random_c = 1

                    index = torch.LongTensor(random.sample(range(C), random_c))
                    batch_x = torch.index_select(batch_x, 2, index)
                    batch_y = torch.index_select(batch_y, 2, index)

                # to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # loss
                loss = criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()

                # record
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            end_time = time.time()
            print(
            "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        self.lr = model_optim.param_groups[0]['lr']

        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark)

                # loss
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)

                # record
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './outputs/test_results/{}'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('{0}->{1}, mse:{2:.3f}, mae:{3:.3f}'.format(self.args.seq_len, self.args.pred_len, mse, mae))
        f = open("./outputs/score.txt", 'a')
        f.write('{0}->{1}, {2:.3f}, {3:.3f} \n'.format(self.args.seq_len, self.args.pred_len, mse, mae))
        f.close()

    def show(self, num, epoch, type='valid'):

        # show cases
        if type == 'valid':
            batch_x, batch_y, batch_x_mark, batch_y_mark = self.valid_show
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = self.train_show

        # data augumentation
        batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, self.args.mask_rate, self.args.lm,
                                                      self.args.positive_nums)
        batch_x_om = torch.cat([batch_x, batch_x_m], 0)

        # masking matrix
        mask = mask.to(self.device)
        mask_o = torch.ones(size=batch_x.shape).to(self.device)
        mask_om = torch.cat([mask_o, mask], 0).to(self.device)

        # to device
        batch_x = batch_x.float().to(self.device)
        batch_x_om = batch_x_om.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)

        # Encoder
        with torch.no_grad():
            loss, loss_cl, loss_rb, positives_mask, logits, rebuild_weight_matrix, pred_batch_x = self.model(batch_x_om, batch_x_mark, batch_x, mask=mask_om)

        for i in range(num):

            if i >= batch_x.shape[0]:
                continue

        fig_logits, fig_positive_matrix, fig_rebuild_weight_matrix = show_matrix(logits, positives_mask, rebuild_weight_matrix)
        self.writer.add_figure(f"/{type} show logits_matrix", fig_logits, global_step=epoch)
        self.writer.add_figure(f"/{type} show positive_matrix", fig_positive_matrix, global_step=epoch)
        self.writer.add_figure(f"/{type} show rebuild_weight_matrix", fig_rebuild_weight_matrix, global_step=epoch)

class Exp_Train(Exp_Basic):
    def __init__(self, args):
        super(Exp_Train, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")
        self.iters = 0

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))

            model = transfer_weights(self.args.load_checkpoints, model)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):

        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.task == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            start_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                if self.args.select_channels < 1:
                    # Random
                    B, S, C = batch_x.shape
                    random_C = int(C * self.args.select_channels)

                    if random_C < 1:
                        random_C = 1

                    index = torch.LongTensor(random.sample(range(C), random_C))
                    batch_x = torch.index_select(batch_x, 2, index)
                    batch_y = torch.index_select(batch_y, 2, index)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                self.iters += 1

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            end_time = time.time()
            print("Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, end_time-start_time, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            loss_scalar_dict = {
                'train_loss': train_loss,
                'valid_loss': vali_loss,
                'test_loss': test_loss,
            }
            self.writer.add_scalars(f"/epochs_loss", loss_scalar_dict, epoch + 1)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.lr = model_optim.param_groups[0]['lr']

        return self.model

    def vali(self, vali_data, vali_loader, criterion, index=None):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                if index is not None:
                    batch_x = torch.index_select(batch_x, 2, index)
                    batch_y = torch.index_select(batch_y, 2, index)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting=None, test=0, log=1, iters=None):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './outputs/test_results/{}'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        if log:
            if iters is not None:
                log = 'iter, {0}, {1}->{2}, {3:.3f}, {4:.3f}'.format(iters, self.args.seq_len, self.args.pred_len, mse, mae)
            else:
                log = 'epoch: {0}, seq_len: {1}, pred_len: {2}, mse: {3:.6f}, mae, {4:.6f}'.format(self.args.train_epochs, self.args.seq_len, self.args.pred_len, mse, mae)

            print(log)
            f = open(f"{folder_path}/{self.args.task}_results.txt", 'a')
            f.write(log + '\n')
            f.close()

    def freeze(self):
        """
        freeze the model head
        require the model to have head attribute
        """

        if hasattr(get_model(self.model), 'head'):
            for name, param in get_model(self.model).named_parameters():
                param.requires_grad = False
            for name, param in get_model(self.model).named_parameters():
                if 'head' in name:
                    param.requires_grad = True
                    # print('unfreeze:', name)
            print('model is frozen except the head!')

    def freeze_part(self):
        """
        freeze the model head
        require the model to have head attribute
        """

        if hasattr(get_model(self.model), 'head'):
            for name, param in get_model(self.model).named_parameters():
                param.requires_grad = False
            for name, param in get_model(self.model).named_parameters():
                if 'enc_embedding' in name or 'norm' in name or 'head' in name:
                    param.requires_grad = True
                    print('unfreeze:', name)
                else:
                    print('freeze:', name)

    def unfreeze(self):
        for name, param in get_model(self.model).named_parameters():
            if 'token' in name:
                param.requires_grad = False
                # print('freeze:', name)
                continue
            param.requires_grad = True
            # print('unfreeze:', name)

    def fine_tune(self, setting):
        """
        Finetune the entire network
        """
        if self.args.operation == 'train':
            print('Training the entire network!')
            self.unfreeze()
        elif self.args.operation == 'fine_tune':
            print('Fine-tuning the entire network!')
            self.unfreeze()
        elif self.args.operation == 'fine_tune_part':
            print('Fine-tuning part network!')
            self.freeze_part()
        elif self.args.operation == 'linear_probe':
            print('Fine-tuning the head!')
            self.freeze()
        else:
            raise ValueError("Wrong task_name {}!".format(self.args.task_name))

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        self.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        self.test(setting)


def get_model(model):
    "Return the model maybe wrapped inside `model`."
    return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model

class Exp_Regression(Exp_Basic):
    def __init__(self, args):
        super(Exp_Regression, self).__init__(args)

    def _build_model(self):
        # model init    
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))

            model = transfer_weights(self.args.load_checkpoints, model)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            start_time = time.time()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)
                
                # 使用batch_y对outputs反归一化 [bs x target_window x 1]
                stdev = batch_y.std(dim=1, keepdim=True)
                mean = batch_y.mean(dim=1, keepdim=True)
                outputs = outputs * stdev + mean

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            end_time = time.time()
            print("Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f}".format(
                epoch + 1, train_steps, end_time-start_time, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # self.lr = model_optim.param_groups[0]['lr']
        
        return self.model
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)
                
                stdev = batch_y.std(dim=1, keepdim=True)
                mean = batch_y.mean(dim=1, keepdim=True)
                outputs = outputs * stdev + mean

                pred = outputs.detach()
                loss = criterion(pred, batch_y)
                total_loss.append(loss.item())
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './outputs/test_results/{}'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)
                
                stdev = batch_y.std(dim=1, keepdim=True)
                mean = batch_y.mean(dim=1, keepdim=True)
                outputs = outputs * stdev + mean

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                if i%self.args.seq_len == 0:
                    preds.append(pred[0])
                    trues.append(true[0])

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.flatten()
        trues = trues.flatten()

        print(preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('{0}->{1}, mse:{2:.3f}, mae:{3:.3f}'.format(self.args.seq_len, self.args.pred_len, mse, mae))
        f = open("./outputs/score.txt", 'a')
        f.write('{0}->{1}, {2:.3f}, {3:.3f} \n'.format(self.args.seq_len, self.args.pred_len, mse, mae))
        f.close()
        
        # import matplotlib.pyplot as plt

        # def plot_predictions(preds, trues, folder_path):
        #     plt.figure(figsize=(12, 6))
        #     plt.plot(preds, label='Predictions')
        #     plt.plot(trues, label='True Values')
        #     plt.xlabel('Time')
        #     plt.ylabel('Values')
        #     plt.title('Predictions vs True Values')
        #     plt.legend()
        #     plt.grid(True)
        #     # plt.savefig(os.path.join(folder_path, 'predictions_vs_true_values.png'))
        #     plt.savefig('predictions_vs_true_values.png')
        #     plt.close()

        # # Call the plot function after testing
        # plot_predictions(preds, trues, folder_path)
        
    def unfreeze(self):
        for name, param in get_model(self.model).named_parameters():
            if 'token' in name:
                param.requires_grad = False
                # print('freeze:', name)
                continue
            param.requires_grad = True
            # print('unfreeze:', name)
        
    def fine_tune(self, setting):
        """
        Finetune the entire network
        """
        if self.args.operation == 'train':
            print('Training the entire network!')
            self.unfreeze()
        elif self.args.operation == 'fine_tune':
            print('Fine-tuning the entire network!')
            self.unfreeze()
        # elif self.args.operation == 'fine_tune_part':
        #     print('Fine-tuning part network!')
        #     self.freeze_part()
        # elif self.args.operation == 'linear_probe':
        #     print('Fine-tuning the head!')
        #     self.freeze()
        else:
            raise ValueError("Wrong task_name {}!".format(self.args.task_name))

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        self.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        self.test(setting)

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model init    
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))

            model = transfer_weights(self.args.load_checkpoints, model)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, None, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, None, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting=None, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        preds = []
        trues = []
        folder_path = './outputs/test_results/{}'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, None, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        print('accuracy:{}'.format(accuracy))
        f = open(f"{folder_path}/{self.args.task}_results.txt", 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.close()
        return
    
    def unfreeze(self):
        for name, param in get_model(self.model).named_parameters():
            if 'token' in name:
                param.requires_grad = False
                # print('freeze:', name)
                continue
            param.requires_grad = True
            # print('unfreeze:', name)
    
    def fine_tune(self, setting):
        if self.args.task == 'classification':
            print('Training the entire network!')
            self.unfreeze()
            
        else:
            raise ValueError("Wrong task_name {}!".format(self.args.task_name))

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        self.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        self.test(setting)
