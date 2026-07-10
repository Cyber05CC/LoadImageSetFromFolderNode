Subject: Request to Add darkHUB RunningHub LoRA Training Nodes to RunningHub

Hello RunningHub Team,

I would like to request that you review and add the following custom ComfyUI node package to RunningHub:

**darkHUB RunningHub LoRA Training Nodes**

This custom node package is designed specifically for LoRA training workflows on restricted cloud platforms where users may not have terminal access or reliable direct filesystem path access.

The package includes:

- a dataset ZIP loader that reads image files and matching `.txt` captions;
- automatic image and caption preparation for LoRA training;
- a LoRA training node with intermediate `.safetensors` checkpoint saving;
- an export node that packages the final LoRA, loss graph, metadata, and intermediate checkpoints into a downloadable ZIP file;
- optional frontend helper buttons for ZIP upload and final ZIP download.

The main reason for this request is that standard folder-path based dataset loaders are difficult to use on RunningHub, because users often cannot access or manage internal paths directly. This package avoids that problem by using ZIP-based datasets and exporting all final results into ComfyUI's standard output directory.

The dataset node does not require a user-entered absolute folder path. If the selected placeholder ZIP is not present, it falls back to the newest uploaded ZIP in ComfyUI `input`. If no ZIP is available, it can read directly uploaded image and matching caption TXT files from ComfyUI `input`.

Expected user workflow:

1. Upload/select a dataset ZIP.
2. Train the LoRA.
3. Save intermediate checkpoints every 1000 steps.
4. Export the final result as a single downloadable ZIP bundle.

The final ZIP bundle contains:

- the final `.safetensors` LoRA file;
- intermediate checkpoint `.safetensors` files;
- a loss graph PNG;
- metadata JSON.

The extension is released under the MIT License and does not require additional Python dependencies beyond a standard ComfyUI training environment.

Please consider adding this node package to RunningHub so users can train LoRAs more reliably without needing terminal access or manual filesystem path setup.

Thank you for your time and consideration.

Best regards,
darkHUB
