from model import Lstm, SelfAttentionLstm, Lstm1
# from data_loader_seq import DataLoader
# from data_loader_seq_sync import DataLoader
from data_loader_seq_sync import DataLoader
import torch
import numpy as np
import random
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from util import ModelType
import pandas as pd

# torch.backends.cudnn.deterministic = True
# torch.manual_seed(999)


class Run:
    def __init__(self, args, device, is_debug):

        self.data_loader = DataLoader(args.feature_name, args.data_id, args.label_id, data_type='seq', normalize=True)

        # self.model = Lstm(input_size=self.data_loader.feature_size, hidden_size=args.hidden_size)
        # self.model = Lstm1(input_size=self.data_loader.feature_size, hidden_size=args.hidden_size)
        self.model_type = args.model_type
        if ModelType(args.model_type).name is 'lstm':
            self.model = Lstm(input_size=self.data_loader.feature_size, hidden_size=args.hidden_size)
        else:
            self.model = SelfAttentionLstm(input_size=self.data_loader.feature_size, hidden_size=args.hidden_size, model_type=args.model_type)
        # self.model = attention(input_size=30, hidden_size=args.hidden_size)
        self.feature_name = args.feature_name
        self.lr = args.lr
        self.epoch = args.epoch_num
        self.is_debug = is_debug
        self.device = device
        self.data_id = args.data_id

    def train(self):
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss().to(device=self.device)
        # criterion = torch.nn.BCEWithLogitsLoss().to(device=self.device)
        mse_min = 100000
        mae_min = 100000
        r2_max = 0
        best_pred = None
        beta_good = None
        beta_good_2 = None
        y_score_test = None
        test_ids = self.data_loader.get_test_keys()
        for epoch in range(self.epoch):
            self.model.train()
            print("In the epoch ", epoch)
            # y_pred = []
            # y_score = []
            cost_total = 0
            cost_total_time = 0
            cost_total_time_2 = 0
            for idx in range(self.data_loader.split.train_length()):
                train_data, data_id = self.data_loader.get_train_item()
                feature = train_data[self.feature_name]
                score = train_data['label']
                feature = torch.from_numpy(feature).unsqueeze(0)
                score = torch.from_numpy(score)
                feature, score = feature.float().cuda(), score.float().cuda()
                epochs = train_data['epoch']
                # if feature.shape[2]<30:
                #     print(data_id)
                y, beta, beta_1 = self.model(feature)
                # y = torch.squeeze(y)
                # y_pred.append(torch.squeeze(y))
                # y_score.append(score[0])
                # score = score.long()
                optimizer.zero_grad()
                # cross CrossEntropyLoss
                # cost = criterion(y, score)
                # cross BCEWithLogitsLoss
                cost = criterion(y, score.unsqueeze(0))
                cost_all = cost

                standard_scalar = (feature.shape[1] / 10)
                # standard_scalar = (56.9 / 10)
                if self.model_type == 10 or self.model_type == 7:
                    # time_loss = self.time_att_loss(beta)
                    time_loss = self.time_att_loss_2(beta, epochs)
                    cost_all += standard_scalar * time_loss
                    cost_total_time += time_loss
                if self.model_type == 11 or self.model_type == 8:
                    time_loss = self.time_att_loss(beta)
                    time_loss_2 = self.time_att_loss(beta_1)
                    time_2_scalar = time_loss.data / time_loss_2.data
                    if time_2_scalar>6:time_2_scalar = 6
                    if time_2_scalar<0.5:time_2_scalar = 0.5
                    cost_all += standard_scalar * time_loss + standard_scalar*time_2_scalar * time_loss_2
                    cost_total_time += time_loss
                    cost_total_time_2 += time_loss_2

                cost_all.backward()
                cost_total += cost
                if self.model_type is 11 or self.model_type is 8 or self.model_type is 10 or self.model_type is 7:
                    cost_total_time += time_loss
                if self.model_type is 11 or self.model_type is 8:
                    cost_total_time_2 += time_loss_2
                optimizer.step()

            # y_pred = torch.stack(y_pred, 0)
            # y_score = torch.stack(y_score, 0)

            # mse = mean_squared_error(y_score.cpu(), y_pred.detach().cpu())
            # mse_rg = mean_squared_error(y_score.cpu(), np.array([5]*y_score.shape[0]))
            print('cost = ', cost_total/self.data_loader.split.train_length())
            print('cost_time = ', cost_total_time / self.data_loader.split.train_length())

            print('cost_time_2 = ', cost_total_time_2 / self.data_loader.split.train_length())
            # print('train mse = ', mse)
            # print('random guess train', mse_rg)

            y_score, y_pred, beta, beta_2 = self.eval(test_ids)
            current_mse = mean_squared_error(y_score, y_pred)
            current_mae = mean_absolute_error(y_score, y_pred)
            r2 = r2_score(y_score, y_pred)
            if current_mse < mse_min:
                mse_min = current_mse
                r2_max = r2
                beta_good = beta
                beta_good_2 = beta_2
                best_pred = y_pred
                y_score_test = y_score
                if self.is_debug and epoch > 5:
                    self.plot_attention(beta_good, epoch, r2_max)
                    # torch.save(self.model.state_dict(),
                    #            'models/{}_{}_{:.2f}'.format(self.data_id, self.model_type, current_mae))
                    print("save model ".format(current_mae))

            if current_mae < mae_min:
                mae_min = current_mae
            if epoch % 10 == 0:
                print('current_square = ', current_mse)
                print('min_square = ', mse_min)
                print('current_absolute = ', current_mae)
                print('min_absolute = ', mae_min)
                print('r2_max = ', r2_max)
                # print('random guess test', mse_rg_t)
                # print('random guess test ture', mse_rg_true)

        return y_score_test, best_pred, test_ids

    def plot_attention(self, beta_good, epoch, r2_max):
        beta_current = beta_good.detach().cpu()
        if len(beta_current.shape) is 3:
            beta_current = beta_current.squeeze().numpy()
        else:
            beta_current = beta_current.unsqueeze(1).numpy()
        sns.set()
        # np.save()
        np.save('./models/big_lambda', beta_current)
        # fig, ax = plt.subplots(figsize=(10, 15))  # Sample figsize in inches
        fig, ax = plt.subplots(figsize=(11, 2))
        # sns.heatmap(np.swapaxes(beta_current, 0, 1), ax=ax, cmap="YlGnBu", xticklabels=4, yticklabels=False)
        plt.plot(np.array([i for i in range(beta_current.shape[0])]), beta_current, color='g', label='ground true')
        # ax = sns.heatmap(np.swapaxes(beta_current, 0, 1))

        # plt.title('{}_{:.2f}_{}'.format(ModelType(self.model_type).name, r2_max, epoch))
        plt.title('LSTM with self-attention')
        plt.show()

    @staticmethod
    def sensor_att_loss(beta):
        beta = beta.squeeze()
        beta_diff = torch.abs(beta[1:] - beta[:-1])
        beta_diff_sum = torch.sum(beta_diff)
        return beta_diff_sum

    @staticmethod
    def time_att_loss(weight):
        weight_diff = torch.abs(weight[1:] - weight[:-1])
        weight_diff_sum = torch.sum(weight_diff)
        return weight_diff_sum

    # @staticmethod
    def time_att_loss_2(self, weight, epochs):
        weight_diff = torch.abs(weight[1:] - weight[:-1])
        epoch_diff = 1./np.diff(epochs)
        epoch_diff = torch.from_numpy(epoch_diff).float().to(self.device)
        weight_diff_sum = torch.dot(weight_diff, epoch_diff)
        return weight_diff_sum



    def eval(self, test_ids):
        self.model.eval()
        with torch.no_grad():
            y_pred = []
            y_score = []
            for test_id in test_ids:
                test_data = self.data_loader.get_data_with_id(test_id)
                feature = test_data[self.feature_name]
                score = test_data['label']
                feature = torch.from_numpy(feature).unsqueeze(0)
                score = torch.from_numpy(score)
                feature, score = feature.float().cuda(), score.float().cuda()
                y, beta, beta_2 = self.model(feature)
                y = torch.squeeze(y)

                y_pred.append(y)
                y_score.append(score[0])

            y_pred = torch.stack(y_pred, 0)
            y_score = torch.stack(y_score, 0)
            randomlist = []
            for i in range(y_score.shape[0]):
                n = random.randint(0, 10)
                randomlist.append(n)
            y_score = y_score.cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()

        return y_score, y_pred, beta, beta_2

    def eval_weights_observe(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        y_pred = []
        y_score = []
        att_all = []
        epoch_all = []
        test_ids = self.data_loader.get_test_keys()
        with torch.no_grad():
            for test_id in test_ids:
                test_data = self.data_loader.get_data_with_id(test_id)
                feature = test_data[self.feature_name]
                score = test_data['label']
                epoch = test_data['epoch']
                feature = torch.from_numpy(feature).unsqueeze(0)
                score = torch.from_numpy(score)
                feature, score = feature.float().cuda(), score.float().cuda()

                y, beta, beta_2 = self.model(feature)
                y = torch.squeeze(y)
                epoch_all.append(epoch)
                att_all.append(beta.detach().cpu().numpy())
                y_pred.append(y)
                y_score.append(score[0])

            y_pred = torch.stack(y_pred, 0)
            y_score = torch.stack(y_score, 0)
            y_score = y_score.cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()

        return test_ids, y_pred, y_score, att_all, epoch_all



