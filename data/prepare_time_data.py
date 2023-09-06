import math
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")


class PrepareTimeData:
    def __init__(self, data_path, phase, base, size):
        self.data_path = data_path
        self.phase = phase
        self.base = base
        self.size = size

        self.data_name = self.data_path.split('/')[-1].split('_')[0]
        self.read_dataset(self.data_path, self.data_name)
        self.df = self.ori_df.copy()
        self.row_num = self.ori_df.shape[0]
        self.col_num = self.ori_df.shape[1]
        self.mean = self.df.mean(axis=1)

        self.df = self.get_mean_df(self.df)
        self.df = self.vertical_merge_df(self.df)
        self.df = self.join_together_labels(self.df)
        self.df = self.fill_data(self.df)
        self.df = self.standardize_data(self.df)

    def get_hr_data(self):
        df = self.df.copy()
        ori_values, values, labels, pre_labels = self.get_data_by_interval(df)

        return ori_values, values, labels, pre_labels

    def get_sr_data(self):
        df = self.df.copy()
        ori_values, values, labels, pre_labels = self.get_data_by_insert_normal()

        return ori_values, values, labels, pre_labels

    def get_mean_df(self, df):
        df = df.copy()
        for col in df.columns:
            df[col] = self.mean
        return df

    def vertical_merge_df(self, df):
        df = df.copy()
        two_power = 2

        if self.col_num < 16:
            two_power = 16
            df_temp = pd.DataFrame()
            col_count = 0
            for i in range(two_power - self.col_num):
                if col_count >= self.col_num:
                    col_count = 0
                df_temp[i] = df.iloc[:, col_count]
                col_count = col_count + 1
        else:
            while self.col_num > two_power:
                two_power = two_power * 2
            df_temp = df.iloc[:, 0:(two_power - self.col_num)]

        col_name = []
        for i in range(self.col_num):
            col_name.append('value_' + str(i))

        df.columns = col_name
        col_name = []
        for i in range(self.col_num, two_power):
            col_name.append('value_' + str(i))

        df_temp.columns = col_name
        df = pd.concat([df, df_temp], axis=1)
        return df

    def join_together_labels(self, df):
        df = df.copy()

        if self.phase == 'train':
            df['label'] = 0
        else:
            df['label'] = self.test_labels
        return df

    def fill_data(self, df):
        df = df.copy()
        data_end = math.ceil(self.row_num / self.size) * self.size

        for i in range(self.row_num, data_end):
            df = df.append(pd.Series(), ignore_index=True)

        df.fillna(0, inplace=True)
        return df

    def read_dataset(self, data_path, data_name):
        if data_name.upper().find('MSL') != -1:
            cols = [-1]
            self.get_dataset(data_path, cols)
        elif data_name.upper().find('PSM') != -1:
            if self.phase == 'train':
                cols = [-1]
                self.get_dataset(data_path, cols)
                if self.ori_df.columns.__contains__('timestamp_(min)'):
                    self.ori_df.drop(columns=['timestamp_(min)'], inplace=True)
            else:
                cols = [-1]
                self.get_dataset(data_path, cols)
                if self.ori_df.columns.__contains__('timestamp_(min)'):
                    self.ori_df.drop(columns=['timestamp_(min)'], inplace=True)
                    self.test_labels.drop(columns=['timestamp_(min)'], inplace=True)
        elif data_name.upper().find('SMAP') != -1:
            cols = [0, 1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 15, 16, 19, 20]
            self.get_dataset(data_path, cols)
        elif data_name.upper().find('SMD') != -1:
            cols = [0, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 28, 33, 35, 36, 37]
            self.get_dataset(data_path, cols)

    def get_dataset(self, data_path, cols):
        if self.phase == 'train':
            if -1 in cols:
                self.ori_df = pd.read_csv(data_path)
            else:
                self.ori_df = pd.read_csv(data_path, usecols=cols)
        else:
            if -1 in cols:
                self.ori_df = pd.read_csv(data_path)
            else:
                self.ori_df = pd.read_csv(data_path, usecols=cols)

            test_label_path = self.data_path.replace('_test.csv', '_test_label.csv')
            self.test_labels = pd.read_csv(test_label_path)

    def get_data_by_insert_normal(self):
        df = pd.DataFrame(columns=['value', 'label'])
        df['value'] = self.df['value_0']
        df['label'] = self.df['label']

        df_pre_label = self.mutation_point(df)
        insert_datas = self.insert_normal(df_pre_label)

        ori_values = []
        values = []
        labels = []
        pre_labels = []

        start_index = 0
        end_index = self.size

        for col in self.df.columns:
            if col == 'label':
                continue
            self.df[col] = insert_datas['value']
        self.df['pre_label'] = insert_datas['pre_label']

        ori_df = self.vertical_merge_df(self.ori_df)
        ori_df = self.fill_data(ori_df)

        for i in range(0, self.df.shape[0], self.size):
            insert_data = pd.DataFrame()
            ori_value = pd.DataFrame()

            insert_data = pd.concat([insert_data, self.df[start_index: end_index]])
            ori_value = pd.concat([ori_value, ori_df[start_index: end_index]])
            start_index += self.size
            end_index += self.size

            value = insert_data.copy().drop(['label', 'pre_label'], axis=1)
            label = insert_data['label']
            pre_label = insert_data['pre_label']

            value = torch.tensor(np.array(value).astype(np.float32))
            label = torch.tensor(np.array(label).astype(np.int64))
            pre_label = torch.tensor(np.array(pre_label).astype(np.int64))
            ori_value = torch.tensor(np.array(ori_value).astype(np.float32))

            values.append(value.unsqueeze(0))
            labels.append(label)
            pre_labels.append(pre_label)
            ori_values.append(ori_value.unsqueeze(0))

        return ori_values, values, labels, pre_labels

    def standardize_data(self, df):
        df = df.copy()
        name = self.data_path.split('.csv')[0]
        print(name, "Points: {}".format(self.row_num))
        df = self.complete_value(df)

        if self.phase != 'train':
            anomaly_len = len(df[df['label'] == 1].index.tolist())
            print("Labeled anomalies: {}".format(anomaly_len))

        return df

    def complete_value(self, df):

        df.fillna(0, inplace=True)
        return df

    def get_mutation_point(self, df_pre_label, start_index, end_index, last_size_var):
        size_var = df_pre_label['value'][start_index: end_index].var()
        label_count = len(df_pre_label[start_index: end_index][df_pre_label['label'] == 1].index.tolist())

        if last_size_var == 0:
            times = 'Nan'
        else:
            times = size_var / last_size_var
            if times < 1 and times != 0:
                times = 1 / times

        if times != "Nan" and times >= 10:
            df_pre_label['pre_label'][start_index: end_index] = 1
        else:
            df_pre_label['pre_label'][start_index: end_index] = 0

        return size_var

    def mutation_point(self, df):
        df_pre_label = df.copy()
        df_pre_label['pre_label'] = 0

        size = 128
        start_index = 0
        end_index = size
        all_var = df_pre_label['value'].var()

        last_size_var = 0
        for i in range(int(self.row_num / size)):
            last_size_var = self.get_mutation_point(df_pre_label, start_index, end_index, last_size_var)

            start_index += size
            end_index += size

        self.get_mutation_point(df_pre_label, start_index, self.row_num - 1, last_size_var)
        return df_pre_label

    def get_index(self, indexes):
        count = 0
        start_indexes = []
        end_indexes = []

        if len(indexes) != 0:
            count = count + 1
            start_indexes.append(indexes[0])

            for i in range(1, len(indexes)):
                if indexes[i - 1] + 1 != indexes[i]:
                    count = count + 1
                    end_indexes.append(indexes[i - 1])
                    start_indexes.append(indexes[i])

            end_indexes.append(indexes[len(indexes) - 1])

        return start_indexes, end_indexes, count

    def insert_normal(self, data):
        pre_labels = 'pre_label'

        nor_indexes = data[0:self.row_num][data[pre_labels] == 0].index.tolist()
        ano_indexes = data[0:self.row_num][data[pre_labels] == 1].index.tolist()

        nor_start_indexes, nor_end_indexes, nor_count = self.get_index(nor_indexes)
        ano_start_indexes, ano_end_indexes, ano_count = self.get_index(ano_indexes)

        interval = int(self.size / self.base)
        ano_len = 2

        df = pd.DataFrame(columns=['ind', 'value', 'label', 'pre_label'])

        for i in range(nor_count):

            if nor_end_indexes[i] - nor_start_indexes[i] + 1 < interval:
                temp_df = pd.DataFrame(columns=['ind', 'value', 'label', 'pre_label'])

                x = range(nor_start_indexes[i], nor_end_indexes[i] + 1)
                xp = [nor_start_indexes[i], nor_end_indexes[i]]
                fp = [data['value'][nor_start_indexes[i]], data['value'][nor_end_indexes[i]]]
                z = np.interp(x, xp, fp)

                temp_df['ind'] = x
                temp_df['value'] = z
                temp_df['pre_label'] = 0
                df = pd.concat([df, temp_df])
            else:
                last_start_x = -1
                start_xs = range(nor_start_indexes[i], nor_end_indexes[i] + 1, interval)
                xp = []
                fp = []
                for start_x in start_xs:
                    if start_x + interval > nor_end_indexes[i]:
                        last_start_x = start_x
                        break

                    xp.append(start_x)
                    xp.append(start_x + interval - 1)

                    fp.append(data['value'][start_x])
                    fp.append(data['value'][start_x + interval - 1])

                x = range(nor_start_indexes[i], last_start_x)
                z = np.interp(x, xp, fp)

                temp_df = pd.DataFrame(columns=['ind', 'value', 'label', 'pre_label'])
                temp_df['ind'] = x
                temp_df['value'] = z
                temp_df['pre_label'] = 0
                df = pd.concat([df, temp_df])

                if last_start_x != -1:
                    temp_df = pd.DataFrame(columns=['ind', 'value', 'label', 'pre_label'])

                    x = range(last_start_x, nor_end_indexes[i] + 1)
                    xp = [last_start_x, nor_end_indexes[i]]
                    fp = [data['value'][last_start_x], data['value'][nor_end_indexes[i]]]
                    z = np.interp(x, xp, fp)

                    temp_df['ind'] = x
                    temp_df['value'] = z
                    temp_df['pre_label'] = 0
                    df = pd.concat([df, temp_df])

        for i in range(ano_count):
            temp_df = pd.DataFrame(columns=['ind', 'value', 'label', 'pre_label'])

            x = range(ano_start_indexes[i] - 1, ano_end_indexes[i] + 2)
            xp = [ano_start_indexes[i] - 1, ano_end_indexes[i] + 1]
            fp = [data['value'][ano_start_indexes[i] - 1], data['value'][ano_end_indexes[i] + 1]]
            z = np.interp(x, xp, fp)

            for j in range(len(x)):
                if j == 0 or j == len(x) - 1:
                    continue

                temp_df.loc[x[j], 'ind'] = x[j]
                temp_df.loc[x[j], 'value'] = z[j]
                temp_df.loc[x[j], 'pre_label'] = 1

            df = pd.concat([df, temp_df])

        df = df.set_index(['ind'], inplace=False).sort_index()
        df['label'] = data['label']
        for i in range(self.row_num, data.shape[0]):
            df = df.append(pd.Series(), ignore_index=True)
        df.fillna(0, inplace=True)
        return df

    def get_row_num(self):
        return self.row_num

    def get_col_num(self):
        return self.col_num
