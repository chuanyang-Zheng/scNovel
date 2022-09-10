
# Mitigating Neural Network Overconfidence with Logit Normalization

Propose your own dataset:
```
1) Go to datasets/utils.py.
2) Modify the function def build_dataset
```

## Install
```
Please check environment.yml for installation.
```

To train the model(s) in the paper, run this command:



## Training

To train the model(s) in the paper, run this command:

```train

Check the code in bash9.sh

```


## Evaluation

To evaluate the model on CIFAR-10, run:

```eval
python test.py cifar10 --method_name cifar10_wrn_${exp_name}_standard --num_to_avg 10 --gpu 0 --seed 1 --prefetch 0

# Example with pretrained model
python test.py cifar10 --method_name cifar10_wrn_logitnorm_standard --num_to_avg 10 --gpu 0 --seed 1 --prefetch 0

```



