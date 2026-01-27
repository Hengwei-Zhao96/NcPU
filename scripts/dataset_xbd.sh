# CE
python main_CE.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --true_class_prior 0.4 --batch_size 128 --lr 0.0001
python main_CE.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --true_class_prior 0.4 --batch_size 128 --lr 0.0001 --supervised_dataset
# uPU
python main_nnPU.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --true_class_prior 0.4 --batch_size 128 --lr 0.0001 --risk_estimator "uPU" --class_prior 0.4547
# nnPU
python main_nnPU.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --true_class_prior 0.4 --batch_size 128 --lr 0.0001 --risk_estimator "nnPU" --class_prior 0.4547
# ImbPU
python main_nnPU.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --true_class_prior 0.4 --batch_size 128 --lr 0.0001 --risk_estimator "ImbPU" --class_prior 0.4547
# vPU
python main_vPU.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --val_unlabeled_size 2000 --true_class_prior 0.4 --lr 3e-5 --lam 0.3 --batch_size 250
# TEDn
python main_TEDn.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --val_unlabeled_size 2000 --true_class_prior 0.4 --lr 0.1 --batch_size 250
# PUET
python main_PUET.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --true_class_prior 0.4 --class_prior 0.4547
# HolisticPU
python main_HolisticPU.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --true_class_prior 0.4 --batch_size 64 --lr 0.0015
# DistPU
python main_DistPU.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --true_class_prior 0.4 --batch_size 128 --class_prior 0.4547
# PiCO
CUDA_VISIBLE_DEVICES=0 python main_PiCO.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --true_class_prior 0.4 --batch_size 128 --lr 0.0001
# LaGAM
python main_LaGAM.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --true_class_prior 0.4 --val_negative_size 50 --batch_size 32 --num_cluster 5 --num_neighbors 10 --epochs 400
# WSC
python main_WSC.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --true_class_prior 0.4 --batch_size 128 --lr 0.0001 --epochs 250 --average_entropy_loss --noise_matrix_scale 0.5 --vol_lambda 0.01 --alpha 1 --beta 12 --lam 1 --lam_consist 3
# NoiCPU
python main_NoiCPU.py --dataset "xbd-all" --positive_class_index "0,1" --positive_size 500 --unlabeled_size 20000 --true_class_prior 0.4 --batch_size 128 --lr 0.0001