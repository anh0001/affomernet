# affomernet

Build deep learning network for affordances concept using transformer

## Setup

```bash
python -m venv affomernet_env
source affomernet_env/bin/activate
```

## Install

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download and extract COCO 2017 train and val images:

```bash
cd configs/dataset/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip
unzip val2017.zip
rm train2017.zip val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
```

Modify `img_folder` and `ann_file` in `coco_detection.yml`.

## Training

### Single GPU

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_iit.yml
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_iit.yml -t output/rtdetr_r50vd_6x_coco/checkpoint0069.pth &> train.log 2>&1
```

### Multi-GPU

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_iit.yml
```

### Validation

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_iit.yml -r path/to/checkpoint --test-only
```

## Export to ONNX

```bash
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_iit.yml -r path/to/checkpoint --check
```

Example:
```bash
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r101vd_6x_coco.yml -r output/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth --check -f output/rtdetr_r101vd_coco_objects365.onnx
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r50vd_6x_iit.yml -r output/rtdetr_r50vd_6x_iit/checkpoint0000.pth --check -f output/rtdetr_r50vd_6x_iit_1.onnx
```

## Inference

```bash
python tools/export_onnx.py --inference --file-name output/model.onnx --image path/to/your/image.jpg
```

Example:
```bash
python tools/export_onnx.py --inference --file-name output/rtdetr_r101vd_coco_objects365.onnx --image dataset/coco/val2017/000000000139.jpg
python tools/export_onnx.py --inference --file-name output/rtdetr_r50vd_6x_iit_1.onnx --image dataset/iit/data/VOCdevkit2012/VOC2012/JPEGImages/0.jpg
```

## Custom Data Training

- Set `remap_mscoco_category: False` in the config file.
- Modify `mscoco_category2name` based on your dataset if needed.
- Add `-t path/to/checkpoint` (optional) to fine-tune based on a pretrained checkpoint.

## Additional Training/Testing Scripts

```bash
# Train on multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=8989 tools/train.py -c path/to/config &> train.log 2>&1 &

# Additional options
-r path/to/checkpoint  # Resume from checkpoint
--amp                  # Use Automatic Mixed Precision
--test-only            # Run evaluation only

# Fine-tuning example
torchrun --master_port=8844 --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth
```

## Troubleshooting

If GPUs do not release memory:
```bash
ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9
```

To save all logs:
Append `&> train.log 2>&1 &` or `&> train.log 2>&1` to your command.
