import torch
import numpy as np
import torch.nn.functional as F
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_localoutlierfactor_scores(val, test, out_scores):
    import sklearn.neighbors
    scorer = sklearn.neighbors.LocalOutlierFactor(novelty=True)
    print("fitting validation set")
    start = time.time()
    scorer.fit(np.vstack((test, out_scores)))
    end = time.time()
    print("fitting took ", end - start)
    val = np.asarray(val)
    test = np.asarray(test)
    out_scores = np.asarray(out_scores)
    print(val.shape, test.shape, out_scores.shape)
    return scorer.score_samples(np.vstack((test, out_scores)))\

def get_onceclass_scores(val, test, out_scores):
    import sklearn.neighbors
    rng = np.random.RandomState(42)
    scorer = sklearn.svm.OneClassSVM()
    print("fitting validation set")
    start = time.time()
    # scorer.fit(np.vstack((test, out_scores)))
    scorer.fit(np.vstack((test, out_scores)))
    end = time.time()
    print("fitting took ", end - start)
    val = np.asarray(val)
    test = np.asarray(test)
    out_scores = np.asarray(out_scores)
    print(val.shape, test.shape, out_scores.shape)
    return scorer.score_samples(np.vstack((test, out_scores)))

def get_EllipticEnvelope_scores(val, test, out_scores):
    from sklearn.covariance import EllipticEnvelope
    rng = np.random.RandomState(42)
    scorer = EllipticEnvelope(random_state=rng)
    print("fitting validation set")
    start = time.time()
    scorer.fit(np.vstack((test, out_scores)))
    end = time.time()
    print("fitting took ", end - start)
    val = np.asarray(val)
    test = np.asarray(test)
    out_scores = np.asarray(out_scores)
    print(val.shape, test.shape, out_scores.shape)
    return scorer.score_samples(np.vstack((test, out_scores)))



def get_isolationforest_scores(val, test, out_scores):
    import sklearn.ensemble
    rng = np.random.RandomState(42)
    scorer = sklearn.ensemble.IsolationForest(random_state = rng)
    print("fitting validation set")
    start = time.time()
    scorer.fit(np.vstack((test, out_scores)))
    end = time.time()
    print("fitting took ", end - start)
    val = np.asarray(val)
    test = np.asarray(test)
    out_scores = np.asarray(out_scores)
    print(val.shape, test.shape, out_scores.shape)
    return scorer.score_samples(np.vstack((test, out_scores)))


def get_logit( net, loader, device):
    _score = []
    _right_score = []
    _wrong_score = []
    _target=[]

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]
            # if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            #     break

            data = data.to(device)
            output = net.forward_feature_list(data)
            _target.extend(target)


            _score.append(to_np(output))



    return concat(_score).copy(),np.array(_target)


def get_logit_split( net, loader, device,class_number):
    _score = []
    _right_score = []
    _wrong_score = []
    _target=[]

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]
            # if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            #     break

            data = data.to(device)
            output = net.forward_feature_list(data)
            _target.extend(target)


            _score.append(to_np(output))


    test_data,test_label=[],[]
    test_ood, test_ood_label = [], []
    _score=concat(_score)
    print("Length ALL Test {} OOD {}".format(len(_target),len(_score)))
    print("Shape {}".format(_score.shape))
    # _score=_score.tolist()

    for i in range(len(_target)):
        if _target[i]<class_number:
            test_data.append([_score[i]])
            test_label.append(_target[i])
        else:
            test_ood.append([_score[i]])
            test_ood_label.append(_target[i])
    print("Length Test {} OOD {}".format(len(test_data),len(test_ood)))
    print("Shape {}".format(concat(test_data).shape))

    return concat(test_data).copy(),np.array(test_label),concat(test_ood).copy(),np.array(test_ood_label)


def get_PCA( net, loader,test_loader,test_ood_loader, device,args):
    _score = []
    _score_test = []
    _score_test_ood = []

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]
            # if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            #     break

            data = data.to(device)


            _score.append(to_np(data))
        for batch_idx, examples in enumerate(test_loader):
            data, target = examples[0], examples[1]
            # if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            #     break

            data = data.to(device)


            _score_test.append(to_np(data))
        for batch_idx, examples in enumerate(test_ood_loader):
            data, target = examples[0], examples[1]
            # if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            #     break

            data = data.to(device)


            _score_test_ood.append(to_np(data))


    _score=concat(_score).copy()
    _score_test = concat(_score_test).copy()
    _score_test_ood = concat(_score_test_ood).copy()

    # _score_all=np.concatenate((_score,_score_test,_score_test_ood), axis=0)
    # pca_analysis=PCA(n_components=args.num_classes)
    # pca_analysis.fit(_score_all)
    # _score_all=pca_analysis.transform(_score_all)

    _score_all=_score
    pca_analysis=PCA(n_components=args.num_classes)
    pca_analysis.fit(_score_all)
    _score_all=pca_analysis.transform(_score_all)

    _score_test_testood=np.concatenate((_score_test,_score_test_ood), axis=0)
    pca_analysis=PCA(n_components=args.num_classes)
    pca_analysis.fit(_score_test_testood)
    _score_test_testood=pca_analysis.transform(_score_test_testood)

    # _score_copy=_score_all[:len(_score)]
    # _score_test_copy = _score_all[len(_score):-len(_score_test_ood)]
    # _score_test_ood_copy = _score_all[-len(_score_test_ood):]

    _score_copy=_score_all
    _score_test_copy = _score_test_testood[:len(_score_test)]
    _score_test_ood_copy = _score_test_testood[len(_score_test):]



    return _score_copy.copy(),_score_test_copy.copy(),_score_test_ood_copy.copy()

def get_ood_scores(args, net, loader, ood_num_examples, device, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]
            # if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            #     break

            data = data.to(device)
            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.score == 'energy':
                all_score = -to_np(args.T * torch.logsumexp(output / args.T, dim=1))
                # print(all_score.shape)
                # all_score = np.sum(all_score, axis=1)
            else:
                all_score = -np.max(to_np(F.softmax(output/args.T, dim=1)), axis=1)

            _score.append(all_score)
            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(all_score[right_indices])
                _wrong_score.append(all_score[wrong_indices])

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score).copy()


def get_ood_gradnorm(args, net, loader, ood_num_examples, device, in_dist=False, print_norm=False):
    _score = []
    _right_score = []
    _wrong_score = []

    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)
    for batch_idx, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        # if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
        #     break
        data = data.to(device)
        net.zero_grad()
        output = net(data)
        num_classes = output.shape[-1]
        targets = torch.ones((data.shape[0], num_classes)).to(device)
        output = output / args.T
        loss = torch.mean(torch.sum(-targets * logsoftmax(output), dim=-1))

        loss.backward()
        layer_grad = net.layer4.weight.grad.data
        layer_grad_bias = net.layer4.bias.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()+torch.sum(torch.abs(layer_grad_bias)).cpu().numpy()
        print(layer_grad_norm)
        all_score = -layer_grad_norm
        _score.append(all_score)

    if in_dist:
        return np.array(_score).copy()
    else:
        return np.array(_score).copy()

def get_calibration_scores(T, net, loader, device):
    logits_list = []
    labels_list = []

    from common.loss_function import _ECELoss
    ece_criterion = _ECELoss(n_bins=15)
    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]

            data = data.to(device)
            label = target.to(device)
            logits = net(data)

            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
    ece_error = ece_criterion(logits, labels, T)
    return ece_error
