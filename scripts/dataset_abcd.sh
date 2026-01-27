# CE
python main_CE.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --true_class_prior 0.5 --batch_size 32 --lr 0.001
python main_CE.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --true_class_prior 0.5 --batch_size 32 --lr 0.001 --supervised_dataset
# uPU
python main_nnPU.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --true_class_prior 0.5 --batch_size 32 --lr 0.001 --risk_estimator "uPU" --class_prior 0.5751
# nnPU
python main_nnPU.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --true_class_prior 0.5 --batch_size 32 --lr 0.001 --risk_estimator "nnPU" --class_prior 0.5751
# ImbPU
python main_nnPU.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --true_class_prior 0.5 --batch_size 32 --lr 0.001 --risk_estimator "ImbPU" --class_prior 0.5751
# vPU
python main_vPU.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --val_unlabeled_size 400 --true_class_prior 0.5 --lr 3e-5 --lam 0.3 --batch_size 60
# TEDn
python main_TEDn.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --val_unlabeled_size 400 --true_class_prior 0.5 --lr 0.1 --batch_size 60
# PUET
python main_PUET.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --true_class_prior 0.5 --class_prior 0.5751
# HolisticPU
python main_HolisticPU.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --true_class_prior 0.5 --batch_size 64 --lr 0.0015
# DistPU
python main_DistPU.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --true_class_prior 0.5 --batch_size 32 --class_prior 0.5751
# PiCO
CUDA_VISIBLE_DEVICES=0 python main_PiCO.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --true_class_prior 0.5 --batch_size 32 --lr 0.001
# LaGAM
python main_LaGAM.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --true_class_prior 0.5 --val_negative_size 30 --batch_size 32 --num_cluster 5 --num_neighbors 10 --epochs 400
# WSC
python main_WSC.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --true_class_prior 0.5 --batch_size 32 --lr 0.001 --epochs 250 --average_entropy_loss --noise_matrix_scale 0.5 --vol_lambda 0.01 --alpha 1 --beta 12 --lam 1 --lam_consist 3
# NoiCPU
python main_NoiCPU.py --dataset "abcd" --positive_class_index "1" --positive_size 300 --unlabeled_size 4000 --true_class_prior 0.5 --batch_size 32 --lr 0.001