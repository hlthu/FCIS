# Fully Convolutional Instance-aware Semantic Segmentation

(These codes are mainly from https://github.com/msracver/FCIS, you can see it for maore details).

The major contributors of this repository include [Haozhi Qi](https://github.com/Oh233), [Yi Li](https://github.com/liyi14), [Guodong Zhang](https://github.com/gd-zhang), [Haochen Zhang](https://github.com/Braininvat), [Jifeng Dai](https://github.com/daijifeng001), and [Yichen Wei](https://github.com/YichenWei).

## Introduction

**FCIS** is a fully convolutional end-to-end solution for instance segmentation, which won the first place in COCO segmentation challenge 2016. 

FCIS is initially described in a [CVPR 2017 spotlight paper](https://arxiv.org/abs/1611.07709). It is worth noticing that:
* FCIS provides a simple, fast and accurate framework for instance segmentation.
* Different from [MNC](https://github.com/daijifeng001/MNC), FCIS performs instance mask estimation and categorization jointly and simultanously, and estimates class-specific masks.
* We did not exploit the various techniques & tricks in the Mask RCNN system, like increasing RPN anchor numbers (from 12 to 15), training on anchors out of image boundary, enlarging the image (shorter side from 600 to 800 pixels), utilizing FPN features and aligned ROI pooling. These techniques & tricks should be orthogonal to our simple baseline.


## Disclaimer

This is an official implementation for [Fully Convolutional Instance-aware Semantic Segmentation](https://arxiv.org/abs/1611.07709) (FCIS) based on MXNet. It is worth noticing that:

  * The original implementation is based on our internal Caffe version on Windows. There are slight differences in the final accuracy and running time due to the plenty details in platform switch.
  * The code is tested on official [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60) with the extra operators for FCIS.
  * We trained our model based on the ImageNet pre-trained [ResNet-v1-101](https://github.com/KaimingHe/deep-residual-networks) using a [model converter](https://github.com/dmlc/mxnet/tree/430ea7bfbbda67d993996d81c7fd44d3a20ef846/tools/caffe_converter). The converted model produces slightly lower accuracy (Top-1 Error on ImageNet val: 24.0% v.s. 23.6%).
  * This repository used code from [MXNet rcnn example](https://github.com/dmlc/mxnet/tree/master/example/rcnn) and [mx-rfcn](https://github.com/giorking/mx-rfcn).


## Results

|model | <sub>training data</sub> | <sub>testing data</sub>  | <sub>mAP^r</sub>  | <sub>mAP^r@0.5</sub> | <sub>mAP^r@0.75</sub>| <sub>mAP^r@S</sub> | <sub>mAP^r@M</sub> | <sub>mAP^r@L</sub> |
|:---------------------------------:|:---------------:|:---------------:|:------:|:---------:|:---------:|:-------:|:-------:|:-------:|
| <sub>FCIS, ResNet-v1-101, OHEM </sub> | <sub>train+valminusminival2014(~115k)</sub> | <sub>minival</sub> | 29.0 | 50.5 | 29.6 | 7.2 | 31.5 | 51.0 |

## Requirements 

### Software

1. MXNet from [the offical repository](https://github.com/dmlc/mxnet). We tested our code on [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60). Due to the rapid development of MXNet, it is recommended to checkout this version if you encounter any issues. We may maintain this repository periodically if MXNet adds important feature in future release. And please install it to your python environment.

2. Python2 packages might missing: cython, opencv-python >= 3.2.0, easydict. If `pip` is set up on your system, those packages should be able to be fetched and installed by running
	```
	pip install Cython
	pip install opencv-python==3.2.0.6
	pip install easydict==1.6
	pip install hickle
	```

### Hardware

Any NVIDIA GPUs with at least 5GB memory should be OK, we use a single machine with 8 Tesla K80 GPUs.

### Installation

1. Clone the FCIS repository, and we'll call the directory that you cloned FCIS as `${FCIS_ROOT}`.
	```
	git clone https://github.com/msracver/FCIS.git
	```

2. For Linux user, run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.

3. Install MXNet:
	
	**Note: The MXNet's Custom Op cannot execute parallelly using multi-gpus after this [PR](https://github.com/apache/incubator-mxnet/pull/6928). We strongly suggest the user rollback to version [MXNet@(commit 998378a)](https://github.com/dmlc/mxnet/tree/998378a) for training (following Section 3.2 - 3.6).**

	3.1 Clone MXNet and checkout to [MXNet@(commit 998378a)](https://github.com/dmlc/mxnet/tree/998378a) by
	```
	git clone --recursive https://github.com/dmlc/mxnet.git
	git checkout 998378a
	git submodule update
	```
	3.2 Copy channel operators in `$(FCIS_ROOT)/fcis/operator_cxx` to `$(YOUR_MXNET_FOLDER)/src/operator/contrib` by
	```
	cp -r $(FCIS_ROOT)/fcis/operator_cxx/channel_operator* $(MXNET_ROOT)/src/operator/contrib/
    ```
	3.3 Compile MXNet
	```
	cd ${MXNET_ROOT}
	make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
	```
	3.4 Install the MXNet Python binding by
	
	```
	cd $(YOUR_MXNET_FOLDER)/python
	sudo python setup.py install
	```

## Training and Testing

### Preparation 

1. Please download [COCO dataset](http://mscoco.org/dataset/#download) and annotations for the 5k image [minival](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0) subset and [val2014 minus minival (val35k)](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0). Make sure it looks like this:
	```
	.data/coco/
	.data/coco/annotations/instances_valminusminival2014.json
	.data/coco/annotations/instances_minival2014.json
	```

2. Please download ImageNet-pretrained ResNet-v1-101 model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMEtxf1Ciym8uZ8sg), and put it under folder `./model`. Make sure it looks like this:
	```
	./model/pretrained_model/resnet_v1_101-0000.params
	```

### Training and Testing

1. Please see `experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml` for experiment settings (GPU #, dataset, etc.).

2. To perform experiments(Both training and testing), run the python scripts with the corresponding config file as input. For example, to train and test FCIS on COCO with ResNet-v1-101, use the following command

    ```
    python experiments/fcis/fcis_end2end_train_test.py --cfg experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml
    ```
    
	Then a cache folder would be created automatically to save the model and the log under `output/fcis/coco/`. Also there will be a cache folder under `data/cache/`.

### Testing Alone

Using the following command:

```
python experiments/fcis/fcis_end2end_test.py --cfg experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml
```

## Demo

1. To run the demo with our trained model (on COCO `train2014+valminusminival2014`), you can see `model/README.md` to download them, and put it under folder `model/`. After downloading, please use a soft link `fcis_coco-0000.params` to point to the model you want to used, just like:

	```
	-rw-r--r-- 1 huanglu huanglu 223M Jan 18 00:23 e2e-0008-finetune.params
	-rw-r--r-- 1 huanglu huanglu 223M Jan 17 13:13 e2e-0008-train.params
	lrwxrwxrwx 1 huanglu huanglu   24 Jan 18 00:24 fcis_coco-0000.params -> e2e-0008-finetune.params
	```

2. Run.
	```
	python ./fcis/demo.py
	```
	And you can change line 58 in `./fcis/demo.py` to use different images, remember to put the images in directory `demo/`.

	If you can not see the shown figure(as I use a server without Desktop), please uncomment the codes in line 54 of `lib/utils/show_masks.py`

	```
	# if show:
    #     plt.show()
	```

	And add a line of `plt.savefig(im_name, dvi=5000)` to save it to a image, where `im_name` is the name of saved image.

## Citing FCIS

If you find FCIS useful in your research, please consider citing:
```
@inproceedings{li2016fully,
  Author = {Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji and Yichen Wei}
  Title = {Fully Convolutional Instance-aware Semantic Segmentation},
  Conference = {CVPR},
  year = {2017}
}
```


## License

© Microsoft, 2017. Licensed under an Apache-2.0 license.