# CE
python main_CE.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5
python main_CE.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5  --supervised_dataset
# uPU
python main_nnPU.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --risk_estimator "uPU" --class_prior 0.7188
# nnPU
python main_nnPU.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --risk_estimator "nnPU" --class_prior 0.7188
# ImbPU
python main_nnPU.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --risk_estimator "ImbPU" --class_prior 0.7188
# NcPU
python main_NcPU.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --ent_loss_weight 0.5