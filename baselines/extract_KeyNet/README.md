# Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters
This repository contains the PyTorch implementation of Key.Net keypoint detector:

```text
"Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters".
Axel Barroso-Laguna, Edgar Riba, Daniel Ponsa, Krystian Mikolajczyk. ICCV 2019.
```
[[Paper on arxiv](https://arxiv.org/abs/1904.00889)]

The training code will be soon published, in the meantime, please check our official [TensorFlow implementation](https://github.com/axelBarroso/Key.Net) for notes about training Key.Net.


## Prerequisite

Python 3.7 is required for running Key.Net code. Use Conda to install the dependencies:

```bash
conda create --name keyNet_torch 
conda activate keyNet_torch 
conda install pytorch==1.2.0 -c pytorch
conda install -c conda-forge opencv tqdm 
conda install -c anaconda pandas 
conda install -c pytorch torchvision 
pip install kornia==0.1.4
```

## Feature Extraction

`extract_kpts_dsc.py` can be used to extract Key.Net + HyNet features for a given list of images. The list of images must contain the relative path to them, and you must provide the root path of images separately. 

The script generates two numpy files, one '.kpt' for keypoints, and a '.dsc' for descriptors. The descriptor used together with Key.Net is [HyNet](https://github.com/yuruntian/HyNet). The output format of the keypoints is as follow:

- `keypoints` [`N x 4`] array containing the positions of keypoints `x, y`, scales `s` and their scores `sc`. 


Arguments:

  * list-images: File containing the image paths for extracting features.
  * root-images: The output path to save the extracted features.
  * results-dir: Indicates the root of the directory containing the images.
  * num-points: The number of desired features to extract. Default: 5000.

## BibTeX

If you use this code in your research, please cite our paper:

```bibtex
@InProceedings{Barroso-Laguna2019ICCV,
    author = {Barroso-Laguna, Axel and Riba, Edgar and Ponsa, Daniel and Mikolajczyk, Krystian},
    title = {{Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters}},
    booktitle = {Proceedings of the 2019 IEEE/CVF International Conference on Computer Vision},
    year = {2019},
}
```

In addition, if you also use the descriptors extracted by HyNet, please consider citing: 
```bibtex
@inproceedings{hynet2020,
 author = {Tian, Yurun and Barroso Laguna, Axel and Ng, Tony and Balntas, Vassileios and Mikolajczyk, Krystian},
 title = {HyNet: Learning Local Descriptor with Hybrid Similarity Measure and Triplet Loss},
 booktitle = {NeurIPS},
 year      = {2020}
}
