import torch
import torch.utils.data as data
class DatasetRCD(data.Dataset):


    def __init__(self, data,label,label_map_class):
        self.data=data.values
        self.label=label.values
        self.label_map_class=label_map_class
        print(label_map_class)
        print(len(self.data))




    def __getitem__(self, index):
        data_this=self.data[index]
        # print(data_this)
        data_this=torch.tensor(data_this)
        # print(data_this)


        if self.label_map_class!=-1:
            label_this=self.label[index][0]
            # print(label_this)
            label_class=self.label_map_class[label_this]
        else:
           label_class=-1


        return data_this, label_class,index

    def __len__(self):
        return len(self.data)


