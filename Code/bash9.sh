drop_number=1
expName="newModelWithoutReWeight_${drop_number}"
epochs=200
sample_ratio=1

python train_rcd.py --dataset deng --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset BaronMouse --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset BaronHuman --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset darmanis --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset campbell --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
##
#python train_rcd.py --dataset campLiver --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset Goolam --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset lake --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset patel --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset usoskin --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset zillionis --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
##
##
#python train_rcd.py --dataset 10Xv2 --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset 10Xv2 --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset 10Xv3 --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1  --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset celseq --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset indrop --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd.py --dataset dropseq --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#
#python train_rcd.py --dataset seqwell --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number







python train_rcd_test.py --dataset deng --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 0.04 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset BaronMouse --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset BaronHuman --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset darmanis --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset campbell --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
##
#python train_rcd_test.py --dataset campLiver --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset Goolam --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset lake --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset patel --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset usoskin --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset zillionis --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
##
##
#python train_rcd_test.py --dataset 10Xv2 --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset 10Xv2 --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset 10Xv3 --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1  --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset celseq --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset indrop --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#python train_rcd_test.py --dataset dropseq --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number
#
#python train_rcd_test.py --dataset seqwell --alg standard -m wrn --exp_name $expName --gpu 0 --loss ce --temp 1 --epochs $epochs --sample_ratio $sample_ratio --drop_number $drop_number






