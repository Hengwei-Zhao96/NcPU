# CE
python main_CE.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4
python main_CE.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --supervised_dataset
# uPU
python main_nnPU.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --risk_estimator "uPU" --class_prior 0.4781
# nnPU
python main_nnPU.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --risk_estimator "nnPU" --class_prior 0.4781
# ImbPU
python main_nnPU.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4 --risk_estimator "ImbPU" --class_prior 0.4781
# NcPU
python main_NcPU.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4