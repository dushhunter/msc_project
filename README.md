# StoneVolMain — Stone Depth Estimation

## Quick Start

# 1. Self-supervised training (with GT depth supervision)
python train.py ./configs/train_args.txt

# 2. Inference / testing
python test_simple_SQL_config.py ./configs/infer_args.txt

# 3. Fine-tuning (metric depth)
python finetune/train_ft_SQLdepth.py ./configs/model_cvnXt.txt ./configs/finetune_args.txt

# 4. EXR → lossless float32 PNG conversion
python convert_exr_to_lossless_float32_png.py \
    --input_dir stone_syn_dataset/data_depth_annotated/train/groundtruth \
    --output_dir stone_syn_dataset/data_depth_annotated/train/groundtruth_float32png \
    --recursive --verify
```

## Project Structure

```
StoneVolMain/
├── train.py                    # Self-supervised training entry point
├── test_simple_SQL_config.py   # Inference entry point
├── convert_exr_to_lossless_float32_png.py  # EXR→PNG converter
├── SQLdepth.py                 # Core model wrapper
├── trainer.py                  # Training loop
├── options.py                  # CLI argument definitions
├── layers.py                   # Depth/pose layer utilities
├── utils.py                    # General utilities
├── kitti_utils.py              # KITTI dataset utilities
├── requirements.txt            # Python dependencies
│
├── configs/                    # All configuration files
│   ├── train_args.txt          # train.py arguments
│   ├── infer_args.txt          # Inference arguments
│   ├── model_cvnXt.txt         # ConvNeXt-Large model config
│   └── finetune_args.txt       # Fine-tuning arguments
│
├── datasets/                   # Dataset loaders
├── networks/                   # Neural network modules
│
├── finetune/                   # Fine-tuning pipeline
│   ├── train_ft_SQLdepth.py    # Fine-tune entry point
│   ├── dataloader.py           # Data loading
│   ├── loss.py                 # Loss functions (SILog, L2)
│   ├── model_io.py             # Model I/O
│   ├── utils.py                # Fine-tune utilities
│   └── file_lists/             # Train/eval split files
│
├── splits/stone/               # Dataset split definitions
├── stone_syn_dataset/          # Training data (12 stones × 120 frames)
├── stone_weights/              # Pre-trained model weights
├── test_results/               # Inference output
└── docs/                       # Reference documents
```

## Architecture

- **Backbone:** ConvNeXt-Large (timm Unet encoder)
- **Decoder:** Depth_Decoder_QueryTr (transformer)
- **Resolution:** 576 × 1024
- **Depth range:** 0.01 – 1.0 m
- **GT encoding:** Float32 RGBA PNG (lossless, bit-exact)
