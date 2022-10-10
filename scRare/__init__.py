import torch
import numpy as np
import torch.nn as nn
import time
import torch.utils.data as Data
from scRare.scRare_classifier import classifier
from scRare.weightsampling import Weighted_Sampling
from scRare.score_functions import get_ood_scores





def scRare(test = None, reference = None, label = None, processing_unit = 'cuda',score_function="sim",iteration_number=3000):

    '''

    :param test: Input your test dataset to annotate
    :param reference: Input your reference dataset
    :param label: Input label set
    :param processing_unit: Choose which processing unit you would like to use,
    if you choose 'gpu', than the software will try to access CUDA,
    if you choose 'cpu', the software will use CPU as default.
    :return:
    '''

    label.columns = ['Label']
    label.Label.value_counts()
    status_dict = label['Label'].unique().tolist()
    int_label = label['Label'].apply(lambda x: status_dict.index(x))
    label.insert(1,'transformed',int_label)

    X_train = reference.values
    X_test = test.values
    y_train = label['transformed'].values

    dtype = torch.float
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)

    sampler = Weighted_Sampling(y_train)

    #construct pytorch object
    train_data = Data.TensorDataset(X_train, y_train)
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=32,
                                   sampler = sampler,
                                   num_workers=1)


    test_data = Data.TensorDataset(X_test, torch.zeros(len(X_test)))
    test_loader = Data.DataLoader(dataset=test_data,
                                   batch_size=32,
                                   num_workers=1)


    input_size = X_train.shape[1]
    num_class = len(status_dict)
    iteration=iteration_number
    count_iteration=0

    model = classifier(input_size, num_class)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # total_step = len(train_loader)

    print("--------Start annotating----------")
    start = time.perf_counter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if processing_unit == 'cpu':
        device = torch.device('cpu')
    elif processing_unit == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print("No GPUs are available on your server.")

    print("Computational unit be used is:", device)

    model.to(device)

    model.train()
    while count_iteration<iteration:
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            train_loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            count_iteration+=1
            # print(count_iteration)
            if count_iteration>=iteration:
                break

        print("Finish {}/{}".format(float(count_iteration), iteration))
    # print("Finish {}/{}".format(float(count_iteration), iteration))
    print("\n--------Annotation Finished----------")

    model.eval()

    train_score=get_ood_scores(train_loader,model,score_function,device=device)
    test_score = get_ood_scores(test_loader, model, score_function, device=device)

    max_socre=np.max(train_score)

    test_score=(test_score-1.0/num_class)/(max_socre-1.0/num_class)

    # test_score[test_score<=threshold]=0
    # test_score[test_score > threshold] = 0

    # detection_result=test_score


    return test_score