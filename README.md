<h2 align="center">Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning</h2>


<h5 align="right">
by <a href="https://hengwei-zhao96.github.io">Hengwei Zhao</a>,
<a href="https://taco-group.github.io">Zhengzhong Tu</a>,
<a href="https://zhuozheng.top">Zhuo Zheng</a>,
<a href="https://wwangwitsel.github.io">Wei Wang</a>,
<a href="https://junjuewang.top">Junjue Wang</a>,
Rusty Feagin,
and Wenzhe Jiao
</h5>

[[`arXiv`](https://arxiv.org/abs/2510.01278)]
[[`Paper(ICLR 2026)`](https://arxiv.org/abs/2510.01278)]

---------------------

This is an official implementation of _NcPU_ in our ICLR 2026 paper.

## Highlights:
1. Representation learning with unreliable supervision
2. NcPU achieves improvements over SOTA PU methods (even supervised counterparts) across diverse datasets without class prior

## Requirements:
- python == 3.10.18
- pytorch == 2.7.1

## Running
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

## Citation
If you use _T-HOneCls_ in your research, please cite the following paper:
```text
@InProceedings{Zhao_2023_ICCV,
    author    = {Zhao, Hengwei and Wang, Xinyu and Li, Jingtao and Zhong, Yanfei},
    title     = {Class Prior-Free Positive-Unlabeled Learning with Taylor Variational Loss for Hyperspectral Remote Sensing Imagery},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {16827-16836}}

@article{ZHAO2022328,
    title = {Mapping the distribution of invasive tree species using deep one-class classification in the tropical montane landscape of Kenya},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {187},
    pages = {328-344},
    year = {2022},
    issn = {0924-2716},
    doi = {https://doi.org/10.1016/j.isprsjprs.2022.03.005},
    url = {https://www.sciencedirect.com/science/article/pii/S0924271622000715},
    author = {Hengwei Zhao and Yanfei Zhong and Xinyu Wang and Xin Hu and Chang Luo and Mark Boitt and Rami Piiroinen and Liangpei Zhang and Janne Heiskanen and Petri Pellikka}}

@ARTICLE{10174705,
    author={Zhao, Hengwei and Zhong, Yanfei and Wang, Xinyu and Shu, Hong},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={One-Class Risk Estimation for One-Class Hyperspectral Image Classification}, 
    year={2023},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TGRS.2023.3292929}}
```
_NcPU_ can be used for academic purposes only, and any commercial use is prohibited.
<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">

<img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>