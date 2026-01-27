# CE
python main_CE.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4
python main_CE.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --supervised_dataset
# uPU
python main_nnPU.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --risk_estimator "uPU" --class_prior 0.4781
# nnPU
python main_nnPU.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --risk_estimator "nnPU" --class_prior 0.4781
# ImbPU
python main_nnPU.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --risk_estimator "ImbPU" --class_prior 0.4781
# vPU
python main_vPU.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --val_unlabeled_size 4000 --true_class_prior 0.4 --lr 3e-5 --lam 0.3
# TEDn
python main_TEDn.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --val_unlabeled_size 4000 --true_class_prior 0.4 --lr 0.1
# PUET
python main_PUET.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --class_prior 0.4781
# HolisticPU
python main_HolisticPU.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --batch_size 64 --lr 0.0015
# DistPU
python main_DistPU.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --class_prior 0.4781
# PiCO
CUDA_VISIBLE_DEVICES=0 python main_PiCO.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4
# LaGAM
python main_LaGAM.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --val_negative_size 100 --batch_size 64 --num_cluster 5 --num_neighbors 10 --epochs 400
# WSC
python main_WSC.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --lr 0.001 --epochs 250 --average_entropy_loss --noise_matrix_scale 0.5 --vol_lambda 0.01 --alpha 1 --beta 12 --lam 1 --lam_consist 3 --balance_lam 0.1
# NoiCPU
python main_NoiCPU.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4