<h2 align="center">Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning</h2>


<h5 align="right">
by <a href="https://hengwei-zhao96.github.io">Hengwei Zhao</a>,
<a href="https://taco-group.github.io">Zhengzhong Tu</a>,
<a href="https://zhuozheng.top">Zhuo Zheng</a>,
<a href="https://wwangwitsel.github.io">Wei Wang</a>,
<a href="https://junjuewang.top">Junjue Wang</a>,
<a href="https://www.rustyfeagin.com">Rusty Feagin</a>,
and <a href="https://scholar.google.com/citations?hl=en&user=1v9ooFUAAAAJ">Wenzhe Jiao</a>
</h5>

[[`arXiv`](https://arxiv.org/abs/2510.01278)]
[[`Paper(ICLR 2026)`](https://arxiv.org/abs/2510.01278)]

---------------------

This is an official implementation of _NcPU_ in our ICLR 2026 paper.

## Highlights:
1. Representation learning under unreliable supervision
2. _NcPU_ outperforms SOTA PU methods, even supervised counterparts, across diverse datasets, without requiring class priors

## Requirements:
- python == 3.10.18
- pytorch == 2.7.1

## Running:
CIFAR-10
```bash
python main_NcPU.py --dataset "cifar10" --positive_class_index "0,1,8,9" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.4
```
CIFAR-100
```bash
python main_NcPU.py --dataset "cifar100" --positive_class_index "4,30,55,72,95,1,32,67,73,91,6,7,14,18,24,3,42,43,88,97,15,19,21,31,38,34,63,64,66,75,26,45,77,79,99,2,11,35,46,98,27,29,44,78,93,36,50,65,74,80" --positive_size 1000 --unlabeled_size 40000 --true_class_prior 0.5 --ent_loss_weight 0.5
```
STL-10
```bash
python main_NcPU.py --dataset "stl10" --positive_class_index "0,2,3,8,9" --positive_size 1000 --unlabeled_size 90000 --true_class_prior 0 --batch_size 512 --ent_loss_weight 0.5 --lr 0.01
```

## Citation:
If you use _NcPU_ in your research, please cite the following paper:
```text
@article{zhao2025noisy,
  title={Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning},
  author={Zhao, Hengwei and Tu, Zhengzhong and Zheng, Zhuo and Wang, Wei and Wang, Junjue and Feagin, Rusty and Jiao, Wenzhe},
  journal={arXiv preprint arXiv:2510.01278},
  year={2025}
}
```
_NcPU_ can be used for academic purposes only, and any commercial use is prohibited.
<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">

<img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>