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

Modify config img_folder and ann_file in the coco_detection.yml

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

## Export to onnx
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint --check
example:
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r101vd_6x_coco.yml -r output/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth --check -f output/rtdetr_r101vd_coco_objects365.onnx

## Inference
python tools/export_onnx.py --inference --file-name output/model.onnx --image path/to/your/image.jpg
example:
python tools/export_onnx.py --inference --file-name output/rtdetr_r101vd_coco_objects365.onnx --image dataset/coco/val2017/000000000139.jpg

## Train custom data
set remap_mscoco_category: False. This variable only works for ms-coco dataset. If you want to use remap_mscoco_category logic on your dataset, please modify variable mscoco_category2name based on your dataset.

add -t path/to/checkpoint (optinal) to tuning rtdetr based on pretrained checkpoint. see training script details.


## Train/test script examples
- `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=8989 tools/train.py -c path/to/config &> train.log 2>&1 &`
- `-r path/to/checkpoint`
- `--amp`
- `--test-only` 


Tuning script examples
- `torchrun --master_port=8844 --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth` 


Export script examples
- `python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check`


GPU do not release memory
- `ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9`


Save all logs
- Appending `&> train.log 2>&1 &` or `&> train.log 2>&1`
