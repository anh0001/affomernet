# affomernet

Build deep learning network for affordances concept using transformer

## Setup
python -m venv affomernet_env
source affomernet_env/bin/activate

## Install
pip install -r requirements.txt

## Download and extract COCO 2017 train and val images.
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images

cd configs/dataset/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip
unzip val2017.zip
rm train2017.zip
rm val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

Modify config img_folder, ann_file

## Training on a Single GPU:

```
# training on single-gpu
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

```
# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

```
# val on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint --test-only
```

## Export
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint --check

## Train custom data
set remap_mscoco_category: False. This variable only works for ms-coco dataset. If you want to use remap_mscoco_category logic on your dataset, please modify variable mscoco_category2name based on your dataset.

add -t path/to/checkpoint (optinal) to tuning rtdetr based on pretrained checkpoint. see training script details.