# CE
python main_CE.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512
python main_CE.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512 --supervised_dataset
# uPU
python main_nnPU.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512 --risk_estimator "uPU" --class_prior 0.4854
# nnPU
python main_nnPU.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512 --risk_estimator "nnPU" --class_prior 0.4854
# ImbPU
python main_nnPU.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512 --risk_estimator "ImbPU" --class_prior 0.4854
# vPU
python main_vPU.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --val_unlabeled_size 9000 --true_class_prior 0 --lr 1e-4 --lam 0.03 --batch_size 250
# TEDn
python main_TEDn.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --val_unlabeled_size 9000 --true_class_prior 0 --lr 0.1 --batch_size 250
# PUET
python main_PUET.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --class_prior 0.4854
# HolisticPU
python main_HolisticPU.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 64 --lr 0.001
# DistPU
python main_DistPU.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 32 --class_prior 0.4854
# PiCO
CUDA_VISIBLE_DEVICES=0 python main_PiCO.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512 --lr 0.01
# LaGAM
python main_LaGAM.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --val_negative_size 100 --batch_size 64 --num_cluster 100 --num_neighbors 10 --epochs 400
# WSC
python main_WSC.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 256 --lr 0.001 --epochs 250 --average_entropy_loss --noise_matrix_scale 0.5 --vol_lambda 0.01 --alpha 1 --beta 12 --lam 1 --lam_consist 3
# NoiCPU
python main_NoiCPU.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512 --ent_loss_weight 0.5 --lr 0.01