# CE
python main_CE.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5
python main_CE.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5  --supervised_dataset
# uPU
python main_nnPU.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --risk_estimator "uPU" --class_prior 0.7188
# nnPU
python main_nnPU.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --risk_estimator "nnPU" --class_prior 0.7188
# ImbPU
python main_nnPU.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --risk_estimator "ImbPU" --class_prior 0.7188
# vPU
python main_vPU.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --val_unlabeled_size 4000 --true_class_prior 0.5 --lr 3e-5 --lam 0.3
# TEDn
python main_TEDn.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --val_unlabeled_size 4000 --true_class_prior 0.5 --lr 0.1
# PUET
python main_PUET.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --class_prior 0.7188
# HolisticPU
python main_HolisticPU.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --batch_size 64 --lr 0.0015
# DistPU
python main_DistPU.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --class_prior 0.7188
# PiCO
CUDA_VISIBLE_DEVICES=0 python main_PiCO.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5
# LaGAM
python main_LaGAM.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --val_negative_size 100 --batch_size 64 --num_cluster 5 --num_neighbors 10 --epochs 400
# WSC
python main_WSC.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --lr 0.001 --epochs 250 --average_entropy_loss --noise_matrix_scale 0.5 --vol_lambda 0.01 --alpha 1 --beta 12 --lam 1 --lam_consist 3 --balance_lam 0.1
# NoiCPU
python main_NoiCPU.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --ent_loss_weight 0.5