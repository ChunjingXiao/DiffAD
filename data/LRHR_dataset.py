from torch.utils.data import Dataset

from data.prepare_time_data import PrepareTimeData


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, phase, l_resolution=16, r_resolution=128,
                 split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.phase = phase
        self.pre_data = PrepareTimeData(data_path=dataroot, phase=phase, base=l_resolution, size=r_resolution)
        self.row_num = self.pre_data.get_row_num()
        self.col_num = self.pre_data.get_col_num()

        if datatype == 'time':
            self.hr_path, self.sr_path, self.labels, self.pre_labels = self.pre_data.get_sr_data()
            self.dataset_len = len(self.sr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        data_LR = None
        data_ORI = self.hr_path[index]
        data_HR = self.hr_path[index]
        data_SR = self.sr_path[index]
        data_label = self.labels[index]

        if self.phase == 'train':
            return {'HR': data_HR, 'SR': data_SR, 'Index': index}
        else:
            return {'ORI': data_ORI, 'HR': data_HR, 'SR': data_SR, 'label': data_label, 'Index': index}
