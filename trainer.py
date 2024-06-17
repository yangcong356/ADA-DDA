import os
import torch
import numpy as np
import math
class Trainer():
    def __init__(self, model, model_type, optimizer, lr_schedule, cuda_stat, source_loader, \
                target_train_loader, target_test_loader, args):
        self.model = model
        self.model_type = model_type
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.cuda_stat = cuda_stat
        self.source_loader = source_loader
        self.target_train_loader = target_train_loader
        self.target_test_loader = target_test_loader
        self.args = args
        self.savepath = './record/' + self.model_type + '/' + self.args.source_dir + '->' + self.args.test_dir + '/'

        self.best_acc_num = 0
        self.D_M = 0
        self.D_C = 0
        self.MU = 0
        # self.best_loss = sys.float_info.max
        # self.logger = logger

    def fit(self):
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        with open(os.path.join(self.savepath,'Loss_result.txt'), 'w') as f:
            f.write('Epoch\tLoss_all\tCls_loss\tMmd_oss\tCmmd_loss\tJoint_loss\n')
            for epoch in range(1, self.args.epochs):
                print('Epoch {}/{}'.format(epoch, self.args.epochs))
                # self._dump_infos()
                loss, cls_loss, mmd_loss, cmmd_loss, joint_loss = self._train(epoch)
                f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (epoch, loss, cls_loss, mmd_loss, cmmd_loss, joint_loss))
                correct = self._test()
                self.lr_schedule.step()
                self._save_best_model(correct)
                # print()
            f.close()


    def _train(self, current_epoch):
        self.model.train()  # Set model to training mode

        len_source_dataset = len(self.source_loader.dataset)
        len_source_loader = len(self.source_loader)
        len_target_loader = len(self.target_train_loader)
        iter_source = iter(self.source_loader)
        iter_target = iter(self.target_train_loader)
        num_iter = len_source_loader
        criterion = torch.nn.CrossEntropyLoss()

        d_m = 0
        d_c = 0

        ''' update mu per epoch '''
        if self.D_M==0 and self.D_C==0 and self.MU==0:
            self.MU = 0.5
        else:
            self.D_M = self.D_M/len_source_loader
            self.D_C = self.D_C/len_source_loader
            self.MU = 1 - self.D_M/(self.D_M + self.D_C)

        for i in range(1, num_iter):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            if i % len_target_loader == 0:
                iter_target = iter(self.target_train_loader)
            if self.cuda_stat:
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target = data_target.cuda()

            self.optimizer.zero_grad()

            s_output, cmmd_loss, mmd_loss = self.model(data_source, data_target, label_source)
            d_c = d_c + cmmd_loss.mean().cpu().item()  # Local A-distance
            d_m = d_m + mmd_loss.mean().cpu().item()   # Global A-distance

            joint_loss = (1 - self.MU) * mmd_loss + self.MU * cmmd_loss
            cls_loss = criterion(s_output, label_source)
            gamma = 2 / (1 + math.exp(-10 * (current_epoch) / self.args.epochs)) - 1
            loss = cls_loss + gamma * joint_loss
            loss.mean().backward()
            self.optimizer.step()
            if i % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMU:{:.4f}\tLoss: {:.6f}\tLabel_Loss: {:.6f}\tMMD_Loss: {:.6f}\tCMMD_Loss: {:.6f}\tJoint_Loss: {:.6f}'.format(
                    current_epoch, i * len(data_source), len_source_dataset,
                    100. * i / len_source_loader, self.MU, loss.mean().item(), cls_loss.item(), mmd_loss.mean().item(), cmmd_loss.mean().item(), joint_loss.mean().item()))
        self.D_M = np.copy(d_m).item()
        self.D_C = np.copy(d_c).item()
        return loss.mean().item(), cls_loss.item(), mmd_loss.mean().item(), cmmd_loss.mean().item(), joint_loss.mean().item()


    def _test(self):
        # self.model.eval()
        test_loss = 0
        correct = 0
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        len_target_dataset = len(self.target_test_loader.dataset)
        with torch.no_grad():
            for data, target in self.target_test_loader:
                if self.cuda_stat:
                    data, target = data.cuda(), target.cuda()
                s_output, _, _ = self.model(data, data, target)
                test_loss += criterion(s_output, target)# sum up batch loss
                pred = torch.max(s_output, 1)[1]  # get the index of the max log-probability
                correct += torch.sum(pred == target)
            test_loss /= len_target_dataset
            print(self.args.test_dir, '  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len_target_dataset,
                100. * correct / len_target_dataset))
        return correct

    def _save_best_model(self, pred_acc_num):
        # Save Model
        if pred_acc_num > self.best_acc_num:
            self.best_acc_num = pred_acc_num
            best_model = os.path.join(self.savepath, 'best_dict.pkl')
            torch.save(self.model.state_dict(), best_model)
        print("%s max correct:" % self.args.test_dir, self.best_acc_num.item())
        print(self.args.source_dir, "to", self.args.test_dir)
