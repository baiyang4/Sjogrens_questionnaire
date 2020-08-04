import os
import torch
from args import Args
from run_dl import Run
import pandas as pd
import numpy as np
import re
from enum import Enum
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, r2_score
from util import LabelName, ModelType
from scipy import stats

is_debug = False
if __name__ == "__main__":
    print('-----------------------------')
    args = Args().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    device = torch.device("cuda:" + str(args.cuda_device) if torch.cuda.is_available() else "cpu")
    # feature_types = ['feature_fai', 'feature_hr', 'feature_acc', 'feature_all_fai', 'feature_all']
    feature_types = ['feature_all', 'feature_hr', 'feature_acc']
    # label_ids = [0, 1, 2, 3, 4, 5]
    label_ids = [1]
    data_ids = [data_id for data_id in range(5)]
    # model_types = [0, 1]
    model_types = [0, 9, 10]

    # debug ----------------------------------
    if is_debug:
        feature_types = ['feature_all']
        model_types = [10]
        data_ids = [0]
        label_ids = [1]
    # debug ----------------------------------
    for model_type in model_types:
        args.model_type = model_type
        if model_type == 0:
            args.epoch_num = 140
        else:
            args.epoch_num = 250
        # if is_debug:
        #     args.epoch_num = 140
        for label_id in label_ids:
            current_path = './results/{}/{}'.format(ModelType(model_type).name, LabelName(label_id).name)
            os.makedirs(current_path, exist_ok=True)
            # result_table = pd.DataFramnp.random.random(200)e(columns=data_ids)
            # result_table_mae = pd.DataFrame(columns=data_ids)
            args.label_id = label_id
            for feature_type in feature_types:
                print('model type', ModelType(model_type).name)
                print('label id', LabelName(label_id).name)
                for data_id in data_ids:
                    args.feature_name = feature_type
                    args.data_id = data_id
                    print('data id', data_id)
                    print('feature_type', feature_type)

                    saving_name = "{}_{}".format(data_id, feature_type)
                    saving_path_in_list = any(re.match(saving_name + r'(_-?0\.[0-9]{1,2})?\.csv', string) for string in os.listdir(current_path))

                    if saving_path_in_list and not is_debug:
                        print('path exist: ', saving_name)
                        continue

                    run = Run(args, device, is_debug)
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
                    print('final square = ', current_mse)
                    if not is_debug:
                        saving_path = "{}/{}_{:.2f}.csv".format(current_path, saving_name, r_tmp)
                        scores.to_csv(saving_path)
                    # feature_results.append(float('%.3f' % min_mse))
                    # feature_results_mae.append(float('%.3f' % min_mae))
                # result_table.loc[str(feature_type)] = feature_results
                # result_table_mae.loc[str(feature_type)] = feature_results_mae
            # result_table['avg'] = result_table.mean(axis=1)
            # result_table_mae['avg'] = result_table_mae.mean(axis=1)
            # result_table.to_csv("./results/results_2_{}.csv".format(LabelName(label_id).name))
            # result_table_mae.to_csv("./results/mae_results_2_{}.csv".format(LabelName(label_id).name))
