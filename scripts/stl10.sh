# CE
python main_CE.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512
python main_CE.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512 --supervised_dataset
# uPU
python main_nnPU.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512 --risk_estimator "uPU" --class_prior 0.4854
# nnPU
python main_nnPU.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512 --risk_estimator "nnPU" --class_prior 0.4854
# ImbPU
python main_nnPU.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512 --risk_estimator "ImbPU" --class_prior 0.4854
# NcPU
python main_NcPU.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512 --ent_loss_weight 0.5 --lr 0.01