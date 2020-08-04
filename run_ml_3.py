from data_loader_summary import DataLoader
import random
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import re
from util import ModelType, LabelName
import os


class RunML:
    def __init__(self, args, is_debug):
        self.data_loader = DataLoader(args.feature_name, args.data_id, args.label_id, 'summary', False)
        print(args.feature_name)
        self.model = Ridge(alpha=1)
        self.model_type = args.model_type
        self.feature_name = args.feature_name
        self.label_id = args.label_id
        self.data_id = args.data_id
        self.is_debug = is_debug
        print(args.feature_name)
        print(LabelName(args.label_id).name)
        # if args.model_type is 'ridge':
        #     self.model = Ridge(alpha=.5)
        #     # self.model = LinearRegression()
        # else:
        #     self.model = Lasso()

    def train(self):
        train_x, train_y, test_x, test_y = self.data_loader.get_data_summary()
        print('feature size', train_x.shape[1])

        # -------------- feature correlation
        corr_feature = train_x.corr()
        corr_feature = corr_feature.abs()

        # --------------- feature-score ranking
        train_x['score'] = train_y
        corr_rank = train_x.corr()
        corr_rank = corr_rank.abs()
        corr_rank = corr_rank.sort_values(by=['score'], ascending=False)
        # corr_rank['name'] = corr_rank.index
        # corr_rank = corr_rank['score']
        if self.is_debug:
            sns.heatmap(corr_feature.abs(), xticklabels=False, yticklabels=False, cbar=False)
            plt.show()

        columns = np.full((corr_feature.shape[0],), True, dtype=bool)

        corr_group = []
        for i in range(corr_feature.shape[0]):
            if columns[i]:
                corr_group_sub = [train_x.columns[i]]
                columns[i] = False
                for j in range(i + 1, corr_feature.shape[0]):
                    if corr_feature.iloc[i, j] >= 0.80:
                        if columns[j]:
                            corr_group_sub.append(train_x.columns[j])
                            columns[j] = False
                corr_group.append(corr_group_sub)

        len_total = 0
        selected_columns = []
        for corr_group_sub in corr_group:
            sub_rank = corr_rank.loc[corr_group_sub, :]
            sub_rank = sub_rank['score']
            sub_rank = sub_rank.dropna()
            if sub_rank.shape[0]>0:
                sub_rank = sub_rank.sort_values(ascending=False)
                selected_columns.append(sub_rank.index[0])
            else:
                for name in corr_group_sub:
                    print(name)
            len_total += len(corr_group_sub)

        print(len_total)

        # selected_columns = train_x.columns[columns]

        train_x = train_x[selected_columns]
        test_x = test_x[selected_columns]

        print('feature size after selection', train_x.shape[1])

        if self.is_debug:
            corr_new = np.corrcoef(train_x, rowvar=False)
            sns.heatmap(corr_new, xticklabels=False, yticklabels=False, cbar=False)
            plt.show()
        if self.model_type is 3:
            # alpha=0.05
            # 0.15 best before reduce feature size
            selector = SelectFromModel(estimator=Lasso(alpha=5),
                                       threshold=-np.inf,
                                       max_features=3).fit(train_x, train_y)
            # selector = SelectFromModel(estimator=Lasso(alpha=5)).fit(train_x, train_y)
            if not self.is_debug:
                coef = pd.Series(selector.estimator_.coef_, index=train_x.columns)
                coef.to_csv('./results_feature/{}_{}_{}.csv'.format(LabelName(self.label_id).name, self.feature_name, self.data_id))


            train_x = selector.transform(train_x)
            test_x = selector.transform(test_x)
            if self.is_debug:
                corr_new = np.corrcoef(train_x, rowvar=False)
                sns.heatmap(corr_new, xticklabels=False, yticklabels=False, cbar=False)
                plt.show()

            # sns.heatmap(train_x, xticklabels=False, yticklabels=False, cbar=False)
            # plt.show()
            print('feature size after Lasso selection', train_x.shape[1])

        #
        model = Ridge(alpha=0.5)
        model.fit(train_x, train_y)

        test_keys = self.data_loader.get_test_keys_summary()
        y_pred = model.predict(test_x)
        y_pred_train = model.predict(train_x)
        mse_trin = mean_squared_error(train_y, y_pred_train)
        print('mse_trin = ', mse_trin)
        return test_y, y_pred, test_keys


