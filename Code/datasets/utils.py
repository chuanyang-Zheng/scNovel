import copy
import os

import torchvision.transforms as trn
import torchvision.datasets as dset

import scanpy as sc
from sklearn.model_selection import train_test_split
import numpy as np
import os

import torch.utils.data as data
# from datasets.DeepRCDDataset import DatasetRCD
class DatasetRCD(data.Dataset):


    def __init__(self, data,label,label_map_class):
        self.data=data
        self.label=label
        self.label_map_class=label_map_class




    def __getitem__(self, index):
        data_this=self.data[index]

        if self.label_map_class!=-1:
            label_this=self.label[index]
            label_class=self.label_map_class[label_this]
        else:
           label_class=-1


        return data_this, label_class

    def __len__(self):
        return len(self.data)


# def build_dataset(dataset, mode="train"):
#
#     mean = (0.492, 0.482, 0.446)
#     std = (0.247, 0.244, 0.262)
#
#     train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
#                                        trn.ToTensor(), trn.Normalize(mean, std)])
#     test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
#
#
#     if dataset == 'cifar10':
#         if mode == "train":
#             data = CIFAR10(root='./data/',
#                                     download=True,
#                                     dataset_type="train",
#                                     transform=train_transform
#                                     )
#         else:
#             data = CIFAR10(root='./data/',
#                                    download=True,
#                                    dataset_type="test",
#                                    transform=test_transform
#                                    )
#         num_classes = 10
#     elif dataset == 'cifar100':
#         if mode == "train":
#             data = CIFAR100(root='./data/',
#                                      download=True,
#                                      dataset_type="train",
#                                      transform=train_transform
#                                      )
#         else:
#             data = CIFAR100(root='./data/',
#                                     download=True,
#                                     dataset_type="test",
#                                     transform=test_transform
#                                     )
#         num_classes = 100
#     elif dataset == "Textures":
#         data = dset.ImageFolder(root="/data/ood_test/dtd/images",
#                                     transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
#                                                            trn.ToTensor(), trn.Normalize(mean, std)]))
#         num_classes = 10
#     elif dataset == "SVHN":
#         if mode == "train":
#             data = svhn.SVHN(root='/data/ood_test/svhn/', split="train",
#                              transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
#                              download=False)
#         else:
#             data = svhn.SVHN(root='/data/ood_test/svhn/', split="test",
#                              transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
#                              download=True)
#         num_classes = 10
#
#     elif dataset == "Places365":
#         data = dset.ImageFolder(root="/data/ood_test/places365/test_subset",
#                                 transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
#                                                        trn.ToTensor(), trn.Normalize(mean, std)]))
#         num_classes = 10
#     elif dataset == "LSUN-C":
#         data = dset.ImageFolder(root="/data/ood_test/LSUN_C",
#                                     transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
#         num_classes = 10
#     elif dataset == "LSUN-R":
#         data = dset.ImageFolder(root="/data/ood_test/LSUN_R",
#                                     transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
#         num_classes = 10
#     elif dataset == "iSUN":
#         data = dset.ImageFolder(root="/data/ood_test/iSUN",
#                                     transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
#         num_classes = 10
#     return data, num_classes


dataset_location={"deng":["Deng_counts.csv","Deng_label.csv"],
                  "campbell":["campbell_counts.csv", "campbell_label.csv"],
                  "campLiver":["campLiver_counts_log.csv", "campLiver_label.csv"],
                  "darmanis": ["darmanis_counts.csv", "darmanis_label.csv"],
                  "Goolam": ["Goolam_counts.csv", "Goolam_label.csv"],
                  "lake": ["lake_counts_normed.csv", "lake_label.csv"],
                  "patel": ["patel_counts_log.csv", "patel_label.csv"],
                  "usoskin": ["usoskin_counts_normed.csv", "usoskin_label.csv"],
                  "zillionis": ["zillionis_counts_norm-001.csv", "zillionis_label.csv"],
                  "BaronMouse": ["Baron Mouse/Filtered_MousePancreas_data.csv", "Baron Mouse/Labels.csv"],
                  "BaronHuman": ["Baron Human/Filtered_Baron_HumanPancreas_data.csv", "Baron Human/Labels.csv"],

"10Xv2": ["scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/10Xv2/10Xv2_pbmc1.csv", "scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/10Xv2/10Xv2_pbmc1Labels.csv"],
"10Xv3": ["scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/10Xv3/10Xv3_pbmc1.csv", "scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/10Xv3/10Xv3_pbmc1Labels.csv"],
"celseq": ["scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/CEL-Seq/CL_pbmc1.csv", "scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/CEL-Seq/CL_pbmc1Labels.csv"],
"dropseq": ["scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/Drop-Seq/DR_pbmc1.csv", "scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/Drop-Seq/DR_pbmc1Labels.csv"],
"indrop": ["scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/inDrop/iD_pbmc1.csv", "scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/inDrop/iD_pbmc1Labels.csv"],
"seqwell": ["scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/Seq-Well/SW_pbmc1.csv", "scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/Seq-Well/SW_pbmc1Labels.csv"],
# "smartseq": ["scRNAseq_Benchmark_datasets/Inter-dataset/PbmcBench/Smart-Seq2/SM2_pbmc1.csv", "scRNAseq_Benchmark_datasets/scRNAseq_Benchmark_datasetsInter-dataset/PbmcBench/Smart-Seq2/SM2_pbmc1Labels.csv"],

                  }

dataset_drop_index=["BaronMouse","BaronHuman","10Xv2","10Xv3","celseq","dropseq","indrop","seqwell","smartseq"]

main_path="/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset"
process_path="/research/dept8/gds/cyzheng21/data/DeepRCD/scBalance_dataset_process"


# def build_dataset(dataset,drop_number=1,save=False):
#     data_location_ori,label_location_ori=dataset_location[dataset][0],dataset_location[dataset][1]
#     data_location=os.path.join(main_path,data_location_ori)
#     label_location = os.path.join(main_path, label_location_ori)
#     adata=sc.read_csv(data_location)
#
#     if (not "_log" in data_location_ori) and save==False:
#         sc.pp.normalize_total(adata, target_sum=1e4)
#         sc.pp.log1p(adata)
#
#     label=sc.read_csv(label_location,dtype="str")
#
#
#     adata=adata.to_df()
#     label=label.to_df()
#     # print(adata.values)
#     # print(adata.shape[1])
#     # print(len(adata.values[0]))
#     # print(adata.values[0])
#     input_dim=adata.shape[1]
#     # adata_copy=copy.deepcopy(adata)
#
#
#
#
#
#     print("Load data From: {}".format(data_location))
#     print("Load Label From: {}".format(label_location))
#     assert len(adata)==len(label),"Data Length {} Does not Match Label Length {}".format(len(adata),len(label))
#
#     if dataset in dataset_drop_index:
#         label=label.drop(columns=["Index"])
#         print("Drop")
#     # print(label)
#     # c=input("CCC")
#     # print(label.values)
#     # c = input("CCC")
#     print(label.values[0])
#     # c = input("CCC")
#
#
#
#
#     set_label=np.unique(label.values)
#     print(set_label)
#
#     #Find the rare cell class
#     count=[]
#     for label_index in range(len(set_label)):
#         # print(label.values==set_label[label_index])
#         sum_number=sum(label.values==set_label[label_index])
#         count.append(sum_number)
#         print("Label {}: {}".format(set_label[label_index],sum_number))
#     min_index=np.argmin(count)
#     min_label=set_label[min_index]
#     print("Rare Cell Name {}: {}".format(min_label,count[min_index]))
#
#     # Build Class Mapping for cell class
#     label_map={}
#     count_map=0
#     for label_index in range(len(set_label)):
#         if set_label[label_index]!=min_label:
#             label_map[set_label[label_index]]=count_map
#             count_map+=1
#     print(label_map)
#     label_map[min_label]=count_map
#     print(label_map)
#
#     test_extract_ood = label["Label"] == min_label
#     label_ood = label.loc[test_extract_ood.values]
#     X_test_ood=adata.loc[test_extract_ood.values]
#     # print(label_ood.values[0])
#     adata_id = adata.loc[~test_extract_ood.values]
#     label_id = label.loc[~test_extract_ood.values]
#
#
#
#
#
#
#     X_train, X_test, y_train, y_test = train_test_split(adata_id,label_id,random_state=42,stratify=label_id)
#     # print(X_train)
#     # print(X_test)
#     # print(X_test.values[0])
#     # print(np.max(X_train.values))
#
#     # train_extract=y_train["Label"]!=min_label
#     # print(train_extract)
#     # print(X_train)
#     # X_train=X_train[train_extract.values]
#     # print(X_train)
#     # y_train=y_train.loc[train_extract]
#     #
#     # print("Train Data Length {}".format(len(X_train)))
#     # print(y_train)
#     #
#     # test_extract=y_test["Label"]!=min_label
#     # X_test_id = X_test.loc[test_extract.values]
#     # y_test_id=y_test.loc[test_extract]
#     if save:
#         base_path=os.path.join(main_path,data_location_ori)
#         print(base_path)
#         base_path=base_path.replace("/scBalance_dataset","/scBalance_dataset_process")
#         base_path=os.path.dirname(base_path)
#         base_path=os.path.join(base_path,dataset)
#         print(base_path)
#         if not os.path.exists(base_path):
#             os.makedirs(base_path)
#         X_train.to_csv(os.path.join(base_path,"train_data.csv"),index=True)
#         y_train.to_csv(os.path.join(base_path,"train_label.csv"),index=True)
#         #
#         X_test.to_csv(os.path.join(base_path,"test_data.csv"),index=True)
#         y_test.to_csv(os.path.join(base_path,"test_label.csv"),index=True)
#
#         X_test_ood.to_csv(os.path.join(base_path,"test_ood_data.csv"),index=True)
#         label_ood.to_csv(os.path.join(base_path,"test_ood_label.csv"),index=True)
#
#
#     train_set=DatasetRCD(X_train,y_train,label_map)
#     test_set = DatasetRCD(X_test, y_test, label_map)
#     test_set_ood = DatasetRCD(X_test_ood, label_ood, label_map)
#
#
#
#
#
#
#
#
#
#     return train_set,test_set,test_set_ood,count_map,input_dim



def build_dataset(dataset,drop_number=1,save=False):
    data_location_ori,label_location_ori=dataset_location[dataset][0],dataset_location[dataset][1]
    data_location=os.path.join(main_path,data_location_ori)
    label_location = os.path.join(main_path, label_location_ori)
    adata=sc.read_csv(data_location)

    if (not "_log" in data_location_ori) and save==False:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    label=sc.read_csv(label_location,dtype="str")


    adata=adata.to_df()
    label=label.to_df()
    # print(adata.values)
    # print(adata.shape[1])
    # print(len(adata.values[0]))
    # print(adata.values[0])
    input_dim=adata.shape[1]
    # adata_copy=copy.deepcopy(adata)





    print("Load data From: {}".format(data_location))
    print("Load Label From: {}".format(label_location))
    assert len(adata)==len(label),"Data Length {} Does not Match Label Length {}".format(len(adata),len(label))

    if dataset in dataset_drop_index:
        label=label.drop(columns=["Index"])
        print("Drop")
    # print(label)
    # c=input("CCC")
    # print(label.values)
    # c = input("CCC")
    print(label.values[0])
    # c = input("CCC")




    set_label=np.unique(label.values)
    print(set_label)

    #Find the rare cell class
    count=[]
    for label_index in range(len(set_label)):
        # print(label.values==set_label[label_index])
        sum_number=sum(label.values==set_label[label_index])
        count.append(sum_number[0])
        print("Label {}: {}".format(set_label[label_index],sum_number))
    print(count)
    print(np.array(count))
    print(np.argmin(count))
    min_index=np.argsort(count)[:drop_number]
    print(min_index)
    min_label=[]
    for i in min_index:
        min_label.append(set_label[i])
    print(min_label)
    print("Rare Cell Name {}: {}".format(min_label,[count[count_i] for count_i in min_index]))

    # Build Class Mapping for cell class
    label_map={}
    count_map=0
    for label_index in range(len(set_label)):
        if not label_index in min_index:
            label_map[set_label[label_index]]=count_map
            count_map+=1
    print(label_map)
    count_map_temp=count_map
    for label_index in range(len(set_label)):
        if label_index in min_index:
            label_map[set_label[label_index]]=count_map_temp
            count_map_temp+=1
    # label_map[min_label]=count_map
    print(label_map)

    test_extract_ood = label["Label"].isin(min_label)
    label_ood = label.loc[test_extract_ood.values]
    X_test_ood=adata.loc[test_extract_ood.values]
    # print(label_ood.values[0])
    adata_id = adata.loc[~test_extract_ood.values]
    label_id = label.loc[~test_extract_ood.values]






    X_train, X_test, y_train, y_test = train_test_split(adata_id,label_id,random_state=42,stratify=label_id)
    # print(X_train)
    # print(X_test)
    # print(X_test.values[0])
    # print(np.max(X_train.values))

    # train_extract=y_train["Label"]!=min_label
    # print(train_extract)
    # print(X_train)
    # X_train=X_train[train_extract.values]
    # print(X_train)
    # y_train=y_train.loc[train_extract]
    #
    # print("Train Data Length {}".format(len(X_train)))
    # print(y_train)
    #
    # test_extract=y_test["Label"]!=min_label
    # X_test_id = X_test.loc[test_extract.values]
    # y_test_id=y_test.loc[test_extract]
    if save:
        base_path=os.path.join(main_path,data_location_ori)
        print(base_path)
        base_path=base_path.replace("/scBalance_dataset","/scBalance_dataset_process_{}".format(drop_number))
        base_path=os.path.dirname(base_path)
        base_path=os.path.join(base_path,dataset)
        print(base_path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        X_train.to_csv(os.path.join(base_path,"train_data.csv"),index=True)
        y_train.to_csv(os.path.join(base_path,"train_label.csv"),index=True)
        #
        X_test.to_csv(os.path.join(base_path,"test_data.csv"),index=True)
        y_test.to_csv(os.path.join(base_path,"test_label.csv"),index=True)

        X_test_ood.to_csv(os.path.join(base_path,"test_ood_data.csv"),index=True)
        label_ood.to_csv(os.path.join(base_path,"test_ood_label.csv"),index=True)


    train_set=DatasetRCD(X_train,y_train,label_map)
    test_set = DatasetRCD(X_test, y_test, label_map)
    test_set_ood = DatasetRCD(X_test_ood, label_ood, label_map)









    return train_set,test_set,test_set_ood,count_map,input_dim

# build_dataset("deng")

# for k,v in dataset_location.items():
#     build_dataset(k,save=True,drop_number=2)
# for k,v in dataset_location.items():
#     build_dataset(k,save=True,drop_number=3)
# for k,v in dataset_location.items():
#     try:
#         build_dataset(k,save=True,drop_number=4)
#     except :
#         print("Error: 没有找到文件或读取文件失败")



