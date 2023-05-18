import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, mean_squared_error


def squeeze_tensor(tensor):
    return tensor.squeeze().cpu()


def update_csv_col_name(all_datas):
    df = all_datas.copy()
    df.columns = [0, 1, 2, 3]

    return df


def tensor2allcsv(visuals, col_num):
    df = pd.DataFrame()
    sr_df = pd.DataFrame(squeeze_tensor(visuals['SR']))
    ori_df = pd.DataFrame(squeeze_tensor(visuals['ORI']))
    lr_df = pd.DataFrame(squeeze_tensor(visuals['LR']))
    inf_df = pd.DataFrame(squeeze_tensor(visuals['INF']))

    if col_num != 1:
        for i in range(col_num, sr_df.shape[1]):
            sr_df.drop(labels=i, axis=1, inplace=True)
            ori_df.drop(labels=i, axis=1, inplace=True)
            lr_df.drop(labels=i, axis=1, inplace=True)
            inf_df.drop(labels=i, axis=1, inplace=True)

    df['SR'] = sr_df.mean(axis=1)
    df['ORI'] = ori_df.mean(axis=1)
    df['LR'] = lr_df.mean(axis=1)
    df['INF'] = inf_df.mean(axis=1)

    df['differ'] = (ori_df - sr_df).abs().mean(axis=1)
    df['label'] = squeeze_tensor(visuals['label'])

    differ_df = (sr_df - ori_df)

    return df, sr_df, differ_df


def merge_all_csv(all_datas, all_data):
    all_datas = pd.concat([all_datas, all_data])
    return all_datas


def save_csv(data, data_path):
    data.to_csv(data_path, index=False)


def get_mean(df):
    mean = df['value'].astype('float32').mean()
    normal_mean = df['value'][df['label'] == 0].astype('float32').mean()
    anomaly_mean = df['value'][df['label'] == 1].astype('float32').mean()

    return mean, normal_mean, anomaly_mean


def get_val_mean(df):
    mean_dict = {}

    ori = 'ORI'
    ori_mean = df[ori].astype('float32').mean()
    ori_normal_mean = df[ori][df['label'] == 0].astype('float32').mean()
    ori_anomaly_mean = df[ori][df['label'] == 1].astype('float32').mean()

    gen_mean = df['SR'].astype('float32').mean()
    gen_normal_mean = df['SR'][df['label'] == 0].astype('float32').mean()
    gen_anomaly_mean = df['SR'][df['label'] == 1].astype('float32').mean()

    mean_dict['MSE'] = mean_squared_error(df[ori], df['SR'])

    mean_dict['ori_mean'] = ori_mean
    mean_dict['ori_normal_mean'] = ori_normal_mean
    mean_dict['ori_anomaly_mean'] = ori_anomaly_mean

    mean_dict['gen_mean'] = gen_mean
    mean_dict['gen_normal_mean'] = gen_normal_mean
    mean_dict['gen_anomaly_mean'] = gen_anomaly_mean

    mean_dict['mean_differ'] = ori_mean - gen_mean
    mean_dict['normal_mean_differ'] = ori_normal_mean - gen_normal_mean
    mean_dict['anomaly_mean_differ'] = ori_anomaly_mean - gen_anomaly_mean

    mean_dict['ori_no-ano_differ'] = ori_normal_mean - ori_anomaly_mean
    mean_dict['ori_mean-no_differ'] = ori_mean - ori_normal_mean
    mean_dict['ori_mean-ano_differ'] = ori_mean - ori_anomaly_mean

    mean_dict['gen_no-ano_differ'] = gen_normal_mean - gen_anomaly_mean
    mean_dict['gen_mean-no_differ'] = gen_mean - gen_normal_mean
    mean_dict['gen_mean-ano_differ'] = gen_mean - gen_anomaly_mean

    return mean_dict


def relabeling_strategy(df, params):
    y_true = []
    best_N = 0
    best_f1 = -1
    best_thred = 0
    best_predictions = []
    thresholds = np.arange(params['start_label'], params['end_label'], params['step_label'])

    df_sort = df.sort_values(by="differ", ascending=False)
    df_sort = df_sort.reset_index(drop=False)

    for t in thresholds:
        # if (t - 1) % params['step_t'] == 0:
        #     print("t: ", t)
        y_true, y_pred, thred = predict_labels(df_sort, t)
        for i in range(len(y_true)):
            if y_pred[i] == 1 and y_true[i] == 1:
                j = i - 1
                while j >= 0 and y_true[j] == 1 and y_pred[j] == 0:
                    y_pred[j] = 1
                    j -= 1
                j = i + 1
                while j < len(y_pred) and y_true[j] == 1 and y_pred[j] == 0:
                    y_pred[j] = 1
                    j += 1

        f1 = calculate_f1(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_N = t
            best_thred = thred
            best_predictions = y_pred

    accuracy = calculate_accuracy(y_true, best_predictions)
    precision = calculate_precision(y_true, best_predictions)
    recall = calculate_recall(y_true, best_predictions)

    return best_f1


def predict_labels(df_sort, num):
    df_sort['pred_label'] = 0
    df_sort.loc[0:num - 1, 'pred_label'] = 1
    thred = df_sort.loc[num - 1, 'differ']

    df_sort = df_sort.set_index('index')
    df_sort = df_sort.sort_index()

    y_true = df_sort['label'].tolist()
    y_pred = df_sort['pred_label'].tolist()

    return y_true, y_pred, thred


def calculate_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def calculate_precision(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    return precision


def calculate_recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    return recall


def calculate_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    return f1
