# darkHUB RunningHub LoRA Training Nodes

Custom ComfyUI nodes for LoRA training workflows on restricted cloud platforms such as RunningHub, where terminal access and direct filesystem paths may not be available.

## Purpose

This extension provides a self-contained LoRA training workflow that can:

- upload/select a dataset `.zip`;
- read paired image and `.txt` caption files;
- encode images and captions for training;
- train a LoRA with intermediate checkpoints;
- export the final LoRA and checkpoints into a downloadable `.zip` bundle.

## Nodes

### Load Image Set From Folder

Node id: `LoadImageSetFromFolderNode`

Despite the legacy name, this node is intended to load a dataset ZIP.

Inputs:

- `clip`: CLIP model used to encode captions.
- `vae`: VAE model used to encode images into latents.
- `dataset_zip`: ZIP file from ComfyUI `input`.
- `resize_mode`: `Stretch`, `Center Crop`, or `Pad`.
- `width`: training image width.
- `height`: training image height.
- `vae_batch_size`: number of images to VAE encode at once. Default is `1` for lower VRAM pressure.

Outputs:

- `LATENT`: training latents.
- `CONDITIONING`: one encoded caption per image.
- `IMAGE`: loaded image batch for inspection/debugging.
- `count`: number of loaded images.

Dataset ZIP format:

```text
dataset.zip
  image_001.png
  image_001.txt
  image_002.png
  image_002.txt
```

Caption matching:

- `image_001.png` -> `image_001.txt`
- `image_001.png` -> `image_001.png.txt`
- folder-level fallback: `caption.txt`

### Train LoRA Checkpointed

Node id: `TrainLoraCheckpointedNode`

This is a LoRA training node with intermediate safetensors checkpoints.

Additional checkpoint inputs:

- `checkpoint_every_steps`: default `1000`. Set `0` to disable intermediate checkpoint saving.
- `checkpoint_prefix`: default `runninghub_lora/checkpoints/my_zimage_lora`.

Intermediate checkpoints are written to:

```text
ComfyUI/output/runninghub_lora/checkpoints/
```

Example files:

```text
my_zimage_lora_1000_steps_00001.safetensors
my_zimage_lora_2000_steps_00001.safetensors
my_zimage_lora_3000_steps_00001.safetensors
```

### RunningHub LoRA Export

Node id: `RunningHubLoRAExport`

Exports the final training result into ComfyUI `output`.

Outputs saved:

- final `.safetensors`;
- loss graph `.png`;
- metadata `.json`;
- final `*_download.zip` bundle;
- intermediate checkpoints inside the bundle under `checkpoints/`.

Example final bundle:

```text
ComfyUI/output/runninghub_lora/my_zimage_lora_00001_download.zip
```

The ZIP contains:

```text
my_zimage_lora_00001.safetensors
my_zimage_lora_00001_loss.png
my_zimage_lora_00001_info.json
checkpoints/
  my_zimage_lora_1000_steps_00001.safetensors
  my_zimage_lora_2000_steps_00001.safetensors
```

## Frontend Helpers

The extension includes a small frontend helper in `web/load_image_set_zip_upload.js`.

It adds:

- `Choose .zip` button on `Load Image Set From Folder`;
- `Download latest ZIP` button on `RunningHub LoRA Export`.

If a cloud platform blocks custom frontend JavaScript, the Python nodes still save files to ComfyUI `output`.

## Installation

Place the folder in:

```text
ComfyUI/custom_nodes/LoadImageSetFromFolderNode
```

Restart ComfyUI and hard-refresh the browser.

## Dependencies

No extra Python packages are required beyond a standard ComfyUI training environment.

Uses existing ComfyUI dependencies:

- `torch`
- `numpy`
- `Pillow`
- `safetensors`

## Recommended Low-VRAM Settings

For Z-Image/Lumina LoRA training on a 48 GB GPU:

```text
batch_size: 1
vae_batch_size: 1
training_dtype: bf16
lora_dtype: bf16
gradient_checkpointing: true
offloading: true
```

If VRAM is still insufficient, reduce image resolution before reducing rank.

## License

MIT License. See [LICENSE](LICENSE).
