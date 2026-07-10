import json
import logging
import os
import re
import zipfile
from io import BytesIO

import numpy as np
import safetensors.torch
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

import folder_paths


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ZIP_EXTENSIONS = {".zip"}


def walk_files(root_dir):
    try:
        for root, _, files in os.walk(root_dir, onerror=lambda err: logging.warning("Cannot read %s: %s", err.filename, err)):
            for filename in files:
                yield root, filename
    except OSError as err:
        logging.warning("Cannot scan %s: %s", root_dir, err)


def find_input_zips():
    input_dir = folder_paths.get_input_directory()
    zip_paths = []
    for root, filename in walk_files(input_dir):
        if os.path.splitext(filename)[1].lower() in ZIP_EXTENSIONS:
            zip_paths.append(os.path.join(root, filename))
    zip_paths.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return zip_paths


def list_input_zips():
    input_dir = folder_paths.get_input_directory()
    zip_files = [
        os.path.relpath(path, input_dir).replace("\\", "/")
        for path in find_input_zips()
    ]
    zip_files.sort(key=str.lower)
    return zip_files or ["upload_dataset.zip"]


def get_input_file_path(filename):
    filename = str(filename).strip().strip('"')
    if os.path.isabs(filename) and os.path.isfile(filename):
        return filename
    annotated = folder_paths.get_annotated_filepath(filename)
    if os.path.isfile(annotated):
        return annotated
    candidate = os.path.join(folder_paths.get_input_directory(), filename)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"Dataset zip not found: {filename}. Upload a .zip to ComfyUI/input and refresh the node."
    )


def resolve_dataset_source(dataset_zip):
    try:
        return "zip", get_input_file_path(dataset_zip)
    except FileNotFoundError as first_error:
        zip_paths = find_input_zips()
        if zip_paths:
            fallback_zip = zip_paths[0]
            logging.warning(
                "Selected dataset zip was not found (%s). Using newest uploaded zip instead: %s",
                dataset_zip,
                fallback_zip,
            )
            return "zip", fallback_zip

        input_dir = folder_paths.get_input_directory()
        if folder_has_images(input_dir):
            logging.warning(
                "No dataset zip found. Falling back to image/caption files uploaded directly in ComfyUI/input: %s",
                input_dir,
            )
            return "folder", input_dir

        available = list_input_zips()
        raise FileNotFoundError(
            f"{first_error}\nNo fallback dataset zip or uploaded image/caption set was found in ComfyUI/input. "
            f"Visible dataset zip choices: {available}"
        )


def safe_zip_names(zip_file):
    names = []
    for info in zip_file.infolist():
        if info.is_dir():
            continue
        normalized = info.filename.replace("\\", "/").lstrip("/")
        parts = normalized.split("/")
        if any(part in ("", ".", "..") for part in parts):
            continue
        names.append(normalized)
    return names


def safe_relpath(path, base_dir):
    return os.path.relpath(path, base_dir).replace("\\", "/")


def caption_key(path):
    directory, filename = os.path.split(path.replace("\\", "/"))
    stem, _ = os.path.splitext(filename)
    return f"{directory}/{stem}".strip("/")


def folder_has_images(dataset_dir):
    for _, filename in walk_files(dataset_dir):
        if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
            return True
    return False


def read_dataset_zip(zip_path, width, height, resize_mode):
    tensors = []
    captions = []
    with zipfile.ZipFile(zip_path) as zf:
        names = safe_zip_names(zf)
        image_names = [
            name for name in names
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        ]
        image_names.sort(key=str.lower)
        if not image_names:
            raise ValueError(f"No images found inside dataset zip: {zip_path}")

        text_by_path = {}
        for name in names:
            if os.path.splitext(name)[1].lower() == ".txt":
                with zf.open(name) as f:
                    text_by_path[name.lower()] = f.read().decode("utf-8", errors="replace").strip()

        for image_name in image_names:
            with zf.open(image_name) as f:
                image = Image.open(BytesIO(f.read()))
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = resize_dataset_image(image, width, height, resize_mode)
                array = np.asarray(image).astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(array))

            key = caption_key(image_name)
            caption_candidates = [
                f"{key}.txt".lower(),
                f"{image_name}.txt".lower(),
                f"{os.path.dirname(image_name)}/caption.txt".strip("/").lower(),
                "caption.txt",
            ]
            caption = None
            for candidate in caption_candidates:
                if candidate in text_by_path:
                    caption = text_by_path[candidate]
                    break
            if caption is None:
                raise FileNotFoundError(
                    f"Missing caption for {image_name}. Expected {key}.txt inside the zip."
                )
            captions.append(caption)

    return tensors, captions


def read_dataset_folder(dataset_dir, width, height, resize_mode):
    text_by_path = {}
    image_paths = []

    for root, filename in walk_files(dataset_dir):
        full_path = os.path.join(root, filename)
        rel_path = safe_relpath(full_path, dataset_dir)
        extension = os.path.splitext(filename)[1].lower()
        if extension in IMAGE_EXTENSIONS:
            image_paths.append(full_path)
        elif extension == ".txt":
            text_by_path[rel_path.lower()] = full_path

    image_paths.sort(key=lambda path: safe_relpath(path, dataset_dir).lower())
    if not image_paths:
        raise ValueError(f"No images found in uploaded dataset folder: {dataset_dir}")

    tensors = []
    captions = []
    for image_path in image_paths:
        rel_image = safe_relpath(image_path, dataset_dir)
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = resize_dataset_image(image, width, height, resize_mode)
            array = np.asarray(image).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(array))

        key = caption_key(rel_image)
        caption_candidates = [
            f"{key}.txt".lower(),
            f"{rel_image}.txt".lower(),
            f"{os.path.dirname(rel_image)}/caption.txt".strip("/").lower(),
            "caption.txt",
        ]
        caption_path = None
        for candidate in caption_candidates:
            if candidate in text_by_path:
                caption_path = text_by_path[candidate]
                break
        if caption_path is None:
            raise FileNotFoundError(
                f"Missing caption for {rel_image}. Expected {key}.txt in the uploaded dataset."
            )
        with open(caption_path, "r", encoding="utf-8", errors="replace") as f:
            captions.append(f.read().strip())

    return tensors, captions


def read_dataset_source(source_type, source_path, width, height, resize_mode):
    if source_type == "zip":
        return read_dataset_zip(source_path, width, height, resize_mode)
    if source_type == "folder":
        return read_dataset_folder(source_path, width, height, resize_mode)
    raise ValueError(f"Unsupported dataset source type: {source_type}")


def resize_dataset_image(image, width, height, resize_mode):
    if resize_mode == "Stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)
    if resize_mode == "Center Crop":
        return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    if resize_mode == "Pad":
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        return canvas
    return image


def encode_caption_list(clip, captions):
    conditioning = []
    for caption in captions:
        tokens = clip.tokenize(caption)
        if hasattr(clip, "encode_from_tokens_scheduled"):
            encoded = clip.encode_from_tokens_scheduled(tokens)
            if isinstance(encoded, list):
                conditioning.extend(encoded)
            else:
                conditioning.append(encoded)
        else:
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conditioning.append([cond, {"pooled_output": pooled}])
    return conditioning


def encode_images_in_batches(vae, images, batch_size):
    batch_size = max(1, int(batch_size))
    latent_batches = []
    with torch.inference_mode():
        for start in range(0, images.shape[0], batch_size):
            latent_batches.append(vae.encode(images[start:start + batch_size]))
    return torch.cat(latent_batches, dim=0)


def clean_filename(value, fallback):
    value = str(value or fallback).strip()
    value = re.sub(r"[^\w\-. ]+", "_", value)
    value = value.replace(" ", "_").strip("._")
    return value or fallback


def find_checkpoint_files(output_dir, filename):
    checkpoint_dir = os.path.join(output_dir, "runninghub_lora", "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        return []

    filename = clean_filename(filename, "trained_lora")
    matches = []
    for root, _, files in os.walk(checkpoint_dir):
        for file in files:
            if file.lower().endswith(".safetensors") and filename.lower() in file.lower():
                matches.append(os.path.join(root, file))
    matches.sort(key=lambda path: os.path.getmtime(path))
    return matches


def make_loss_graph(loss_values):
    width, height = 800, 480
    margin = 46
    img = Image.new("RGB", (width + margin, height + margin), "white")
    draw = ImageDraw.Draw(img)

    if not loss_values:
        loss_values = [0.0]
    loss_values = [float(v) for v in loss_values]
    min_loss, max_loss = min(loss_values), max(loss_values)
    span = max(max_loss - min_loss, 1e-8)
    scaled_loss = [(value - min_loss) / span for value in loss_values]

    if len(scaled_loss) == 1:
        scaled_loss = scaled_loss * 2

    prev_point = (margin, height - int(scaled_loss[0] * height))
    steps = len(scaled_loss)
    for i, value in enumerate(scaled_loss[1:], start=1):
        x = margin + int(i / (steps - 1) * width)
        y = height - int(value * height)
        draw.line([prev_point, (x, y)], fill="blue", width=2)
        prev_point = (x, y)

    draw.line([(margin, 0), (margin, height)], fill="black", width=2)
    draw.line([(margin, height), (width + margin, height)], fill="black", width=2)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    draw.text((8, height // 2), "Loss", font=font, fill="black")
    draw.text((width // 2, height + 14), "Steps", font=font, fill="black")
    draw.text((margin - 42, 0), f"{max_loss:.4f}", font=font, fill="black")
    draw.text((margin - 42, height - 14), f"{min_loss:.4f}", font=font, fill="black")
    return img


class RunningHubLoRAExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": ("LORA_MODEL",),
                "filename": (
                    "STRING",
                    {
                        "default": "my_zimage_lora",
                        "tooltip": "Base filename only. Files are always saved into ComfyUI/output.",
                    },
                ),
                "save_zip_bundle": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "loss": ("LOSS_MAP",),
                "steps": ("INT",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_files",)
    FUNCTION = "save_bundle"
    CATEGORY = "model/training"
    OUTPUT_NODE = True

    def save_bundle(self, lora, filename, save_zip_bundle, loss=None, steps=None):
        output_dir = folder_paths.get_output_directory()
        filename = clean_filename(filename, "trained_lora")
        step_suffix = f"_{int(steps)}_steps" if steps is not None else ""
        full_output_folder, base_name, counter, subfolder, _ = folder_paths.get_save_image_path(
            f"runninghub_lora/{filename}{step_suffix}",
            output_dir,
        )

        lora_file = f"{base_name}_{counter:05}.safetensors"
        lora_path = os.path.join(full_output_folder, lora_file)
        safetensors.torch.save_file(lora, lora_path)

        saved_paths = [lora_path]
        metadata = {
            "lora_file": lora_file,
            "steps": int(steps) if steps is not None else None,
            "loss_points": 0,
        }

        graph_file = None
        if loss is not None and isinstance(loss, dict):
            loss_values = loss.get("loss", [])
            metadata["loss_points"] = len(loss_values)
            graph = make_loss_graph(loss_values)
            graph_file = f"{base_name}_{counter:05}_loss.png"
            graph_path = os.path.join(full_output_folder, graph_file)
            graph.save(graph_path)
            saved_paths.append(graph_path)

        json_file = f"{base_name}_{counter:05}_info.json"
        json_path = os.path.join(full_output_folder, json_file)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        saved_paths.append(json_path)

        zip_file = None
        if save_zip_bundle:
            zip_file = f"{base_name}_{counter:05}_download.zip"
            zip_path = os.path.join(full_output_folder, zip_file)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for path in saved_paths:
                    zf.write(path, os.path.basename(path))
                for path in find_checkpoint_files(output_dir, filename):
                    zf.write(path, f"checkpoints/{os.path.basename(path)}")
            saved_paths.append(zip_path)

        ui_images = []
        if graph_file is not None:
            ui_images.append({"filename": graph_file, "subfolder": subfolder, "type": "output"})

        downloads = []
        if zip_file is not None:
            downloads.append({"filename": zip_file, "subfolder": subfolder, "type": "output"})
        else:
            downloads.append({"filename": lora_file, "subfolder": subfolder, "type": "output"})

        saved_text = "\n".join(saved_paths)
        logging.info("Saved RunningHub LoRA bundle:\n%s", saved_text)
        return {
            "ui": {
                "images": ui_images,
                "text": [saved_text],
                "runninghub_downloads": downloads,
            },
            "result": (saved_text,),
        }


class LoadImageSetFromFolderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "dataset_zip": (
                    list_input_zips(),
                    {
                        "tooltip": "Select a .zip from ComfyUI/input. The zip must contain matching image + .txt caption files.",
                    },
                ),
                "resize_mode": (
                    ["Stretch", "Center Crop", "Pad"],
                    {"default": "Stretch"},
                ),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "vae_batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "How many images to VAE encode at once. Keep 1 for low VRAM pressure.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "IMAGE", "INT")
    RETURN_NAMES = ("LATENT", "CONDITIONING", "IMAGE", "count")
    FUNCTION = "load_dataset"
    CATEGORY = "image/training"

    def load_dataset(
        self,
        clip,
        vae,
        dataset_zip,
        resize_mode,
        width,
        height,
        vae_batch_size,
    ):
        source_type, source_path = resolve_dataset_source(dataset_zip)
        tensors, captions = read_dataset_source(source_type, source_path, width, height, resize_mode)

        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise ValueError(
                "All images must have the same size for one batch. Use Stretch, Center Crop, or Pad."
            )

        conditioning = encode_caption_list(clip, captions)
        images = torch.stack(tensors, dim=0)
        latents = {"samples": encode_images_in_batches(vae, images, vae_batch_size)}
        logging.info("Loaded dataset %s (%s): %s images", source_path, source_type, len(tensors))
        return latents, conditioning, images, len(tensors)


NODE_CLASS_MAPPINGS = {
    "LoadImageSetFromFolderNode": LoadImageSetFromFolderNode,
    "RunningHubLoRAExport": RunningHubLoRAExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageSetFromFolderNode": "Load Image Set From Folder",
    "RunningHubLoRAExport": "RunningHub LoRA Export",
}
