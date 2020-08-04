import os
import torch
from args import Args
from run_ml_2 import RunML as RunML4
from run_ml_3 import RunML
import pandas as pd
from enum import Enum
from util import ModelType, LabelName
import re
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from scipy import stats


corr_selection = True
lasso_selection = True
is_debug = False
if __name__ == "__main__":
    args = Args().parse()
    # data_ids = [data_id for data_id in range(6)]
    feature_types = ['feature_acc']
    label_ids = [1]
    data_ids = [data_id for data_id in range(5)]
    model_types = [3]

    # debug ----------------------------------
    if is_debug:
        feature_types = ['feature_all']
        model_types = [3]
        data_ids = [0,1,2,3]
        label_ids = [1]
    # debug ----------------------------------s

    for model_type in model_types:
        args.model_type = model_type
        for label_id in label_ids:
            current_path = './results/{}/{}'.format(ModelType(model_type).name, LabelName(label_id).name)
            os.makedirs(current_path, exist_ok=True)
            args.label_id = label_id
            for feature_type in feature_types:
                args.feature_name = feature_type
                print('model type', ModelType(model_type).name)
                print('label id', LabelName(label_id).name)
                print('feature_type', feature_type)
                for data_id in data_ids:
                    args.data_id = data_id
                    print('data id', data_id)
                    saving_name = "{}_{}".format(data_id, feature_type)
                    saving_path_in_list = any(re.match(saving_name + r'(_-?0\.[0-9]{1,2})?\.csv', string) for string in
                                              os.listdir(current_path))
                    if saving_path_in_list and not is_debug:
                        print('path exist: ', saving_name)
                        continue
                    if model_type is 4:
                        print('heheh')
                        run = RunML4(args, is_debug)
                    else:
                        print('not hehe')
                        run = RunML(args, is_debug)

                    y_score, y_pred, test_keys = run.train()
                    scores = pd.DataFrame(np.stack([np.array(test_keys), y_score, y_pred], axis=-1),
                                          columns=['idx', 'score', 'pred'])

                    current_mse = mean_squared_error(y_score, y_pred)
                    current_mae = mean_absolute_error(y_score, y_pred)
                    current_r2 = r2_score(y_score, y_pred)
                    r_tmp, _ = stats.pearsonr(y_score, y_pred)
                    print('model type', ModelType(model_type).name)
                    print('label id', LabelName(label_id).name)
                    print('data id', data_id)
                    print('feature_type', feature_type)
                    print('final_r = ', r_tmp)
                    print('final_r2 = ', current_r2)
                    print('final abs = ', current_mae)
                    print('final square root = ', current_mse**0.5)
                    if not is_debug:
                        saving_path = "{}/{}_{:.2f}.csv".format(current_path, saving_name, r_tmp)
                        scores.to_csv(saving_path)
    # mae_avg = 0
    # r2_avg = 0
    # for data_id in data_ids:
    #     args.data_id = data_id
    #     print(data_id)
    #     run = RunML(args)
    #     mse, mae, r2 = run.train()
    #     mae_avg += mae
    #     r2_avg +=r2
    # r2_avg /= len(data_ids)
    # mae_avg /= len(data_ids)
    # print('r2 = ', r2_avg)
    # print('mae = ', mae_avg)
