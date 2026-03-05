import cv2
from deepface import DeepFace
import numpy as np
import os
from PIL import Image, ImageOps
import torch
import torchvision.transforms.functional as TF
import torchvision.io as tvio
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings

# ---------------------------------------------------------------------------
# GPU configuration — maximise VRAM usage and throughput
# ---------------------------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PINNED_BUF   : torch.Tensor | None = None
_CUDA_STREAM  : torch.cuda.Stream | None = None
_VRAM_WORKSPACE: torch.Tensor | None = None

if _DEVICE.type == "cuda":
    # ── allocator / cuDNN flags ────────────────────────────────────────────
    torch.cuda.set_per_process_memory_fraction(1.0, device=0)
    torch.backends.cudnn.benchmark            = True
    torch.backends.cudnn.enabled              = True
    torch.backends.cuda.matmul.allow_tf32     = True
    torch.backends.cudnn.allow_tf32           = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

    props      = torch.cuda.get_device_properties(0)
    vram_total = props.total_memory

    _CUDA_STREAM = torch.cuda.Stream(device=_DEVICE)

    _PIN_ELEMS  = 4096 * 4096 * 3
    _PINNED_BUF = torch.empty(_PIN_ELEMS, dtype=torch.float16).pin_memory()

    workspace_bytes    = int(vram_total * 0.60)
    workspace_elements = workspace_bytes // 2
    try:
        _VRAM_WORKSPACE = torch.zeros(workspace_elements, dtype=torch.float16,
                                      device=_DEVICE)
        claimed_gb = workspace_bytes / 1024 ** 3
        print(f"GPU : {props.name}")
        print(f"VRAM: {vram_total / 1024**3:.1f} GB total | "
              f"{claimed_gb:.2f} GB pre-allocated workspace")
    except torch.cuda.OutOfMemoryError:
        print("Warning: Could not pre-allocate full GPU workspace.")

else:
    print("No CUDA GPU found — running on CPU")

# ---------------------------------------------------------------------------
# Pre-load models (Facenet embedding + RetinaFace detector)
# ---------------------------------------------------------------------------
print("Pre-loading DeepFace models...")

# RetinaFace detector — GPU-accelerated via PyTorch (much faster than opencv Haar)
_DETECTOR_BACKEND = "retinaface"

try:
    DeepFace.build_model("Facenet")
    print("  Facenet embedding model loaded.")
except Exception as e:
    print(f"  Warning: Could not pre-load Facenet: {e}")

try:
    DeepFace.build_model("retinaface")
    print("  RetinaFace detector model loaded (GPU).")
except Exception as e:
    print(f"  Warning: Could not pre-load RetinaFace ({e}), will fall back to opencv detector.")
    _DETECTOR_BACKEND = "opencv"


# ---------------------------------------------------------------------------
# Image load + resize helpers — return numpy array, NO disk round-trip
# ---------------------------------------------------------------------------

def _load_and_resize_gpu(image_path, max_dimension=1200):
    """
    Load image with torchvision.io (libjpeg-turbo), resize on GPU in fp16,
    return an (H,W,3) BGR uint8 numpy array (DeepFace/OpenCV convention).
    Does NOT write to disk.
    """
    try:
        img_tensor = tvio.read_image(image_path,
                                     mode=tvio.ImageReadMode.RGB,
                                     apply_exif_orientation=True)  # uint8 CHW

        c, h, w = img_tensor.shape

        with torch.cuda.stream(_CUDA_STREAM):
            n_elem = c * h * w
            global _PINNED_BUF
            if _PINNED_BUF is None or _PINNED_BUF.numel() < n_elem:
                _PINNED_BUF = torch.empty(n_elem, dtype=torch.float16).pin_memory()

            _PINNED_BUF[:n_elem].copy_(img_tensor.reshape(-1).to(torch.float16) / 255.0)
            gpu_tensor = _PINNED_BUF[:n_elem].to(_DEVICE, non_blocking=True).reshape(c, h, w)

            was_resized = False
            if max(h, w) > max_dimension:
                ratio  = max_dimension / max(h, w)
                new_h  = int(h * ratio)
                new_w  = int(w * ratio)
                gpu_tensor = TF.resize(gpu_tensor, [new_h, new_w],
                                       interpolation=TF.InterpolationMode.BILINEAR,
                                       antialias=True)
                was_resized = True

            _CUDA_STREAM.synchronize()
            result_uint8 = (gpu_tensor.clamp(0.0, 1.0) * 255).byte().cpu()

        # CHW RGB tensor → HWC BGR numpy (DeepFace expects BGR like cv2.imread)
        np_bgr = cv2.cvtColor(result_uint8.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        return np_bgr, was_resized

    except Exception as e:
        print(f"Warning: GPU load/resize failed ({e}), falling back to CPU.")
        return None, False


def _load_and_resize_cpu(image_path, max_dimension=1200):
    """CPU fallback: load with PIL, EXIF-transpose, resize, return BGR numpy."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        was_resized = False
        if max(img.width, img.height) > max_dimension:
            ratio    = max_dimension / max(img.width, img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            was_resized = True
        np_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return np_bgr, was_resized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_faces_from_image(image_path):
    """
    Detect faces and return their Facenet embeddings.

    Optimised pipeline:
      1. Load + GPU-resize into an in-memory numpy array (NO disk re-read)
      2. Pass the BGR array directly to DeepFace.represent (skips file I/O)
      3. Use RetinaFace detector (GPU-accelerated) instead of opencv Haar
      4. Write the resized image back to disk only once (for serving later)
    """
    t0 = time.perf_counter()
    try:
        # ── Step 1: Load & resize in memory ───────────────────────────────
        if _DEVICE.type == "cuda":
            np_bgr, was_resized = _load_and_resize_gpu(image_path)
            if np_bgr is None:  # GPU path failed — fallback
                np_bgr, was_resized = _load_and_resize_cpu(image_path)
        else:
            np_bgr, was_resized = _load_and_resize_cpu(image_path)

        # ── Step 2: Save resized file to disk (for serving) ───────────────
        if was_resized:
            cv2.imwrite(image_path, np_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # ── Step 3: Extract embeddings from in-memory numpy array ─────────
        representations = DeepFace.represent(
            img_path=np_bgr,                     # numpy array — no disk read
            model_name="Facenet",
            detector_backend=_DETECTOR_BACKEND,   # retinaface (GPU) or opencv
            enforce_detection=True,
        )

        elapsed = time.perf_counter() - t0
        print(f"[{'GPU' if _DEVICE.type == 'cuda' else 'CPU'}] "
              f"Extracted {len(representations)} face(s) in {elapsed:.2f}s")
        return [rep['embedding'] for rep in representations]

    except ValueError:
        # No face detected
        return []
    except Exception as e:
        print(f"Error extracting faces from {image_path}: {e}")
        return []

def find_matching_photos(user_encoding_list, event_encodings_data, tolerance=0.40):
    """
    Compares the user's face encoding against all face encodings in an event
    and returns the paths of matching photos using Cosine Distance.

    When a CUDA GPU is available the entire comparison is done in a single
    batched matrix operation on the GPU, loading all embeddings into VRAM at
    once for maximum throughput.  Falls back to NumPy on CPU otherwise.

    :param user_encoding_list: The user's encoding as a list (128-d Facenet vector)
    :param event_encodings_data: List of dicts [{'photo_id': int, 'file_path': str, 'encoding': list}, ...]
    :param tolerance: Cosine Distance threshold for Facenet. Default 0.40.
    :return: List of unique photo file paths that match
    """
    if not user_encoding_list or not event_encodings_data:
        return []

    file_paths = [item['file_path'] for item in event_encodings_data]
    embeddings  = [item['encoding']  for item in event_encodings_data]

    if _DEVICE.type == "cuda":
        # -----------------------------------------------------------------
        # GPU batched cosine similarity — all embeddings loaded into VRAM
        # at once, comparison done in a single matrix-vector multiply.
        # float16 halves VRAM usage so we can hold more embeddings at once.
        # -----------------------------------------------------------------
        with torch.no_grad():
            # Shape: (N, D)  — entire embedding matrix on GPU
            gallery = torch.tensor(embeddings, dtype=torch.float16, device=_DEVICE)
            query   = torch.tensor(user_encoding_list, dtype=torch.float16, device=_DEVICE)  # (D,)

            # L2-normalise both sides → dot product == cosine similarity
            gallery_norm = gallery / gallery.norm(dim=1, keepdim=True).clamp(min=1e-8)
            query_norm   = query   / query.norm().clamp(min=1e-8)

            # (N,) cosine similarities, then convert to distances
            cosine_sims      = gallery_norm @ query_norm          # (N,)
            cosine_distances = (1.0 - cosine_sims).cpu().tolist() # back to CPU list

        matched = set()
        for path, dist in zip(file_paths, cosine_distances):
            print(f"DEBUG [GPU]: {path} -> Cosine Distance: {dist:.4f} (Threshold: {tolerance})")
            if dist <= tolerance:
                matched.add(path)

        return list(matched)

    else:
        # -----------------------------------------------------------------
        # CPU NumPy fallback
        # -----------------------------------------------------------------
        user_vec = np.array(user_encoding_list, dtype=np.float32)
        gallery  = np.array(embeddings,          dtype=np.float32)  # (N, D)

        user_norm    = np.linalg.norm(user_vec)
        gallery_norm = np.linalg.norm(gallery, axis=1)              # (N,)

        valid = gallery_norm > 0
        cosine_distances = np.ones(len(embeddings))                  # default distance = 1
        cosine_distances[valid] = 1.0 - (gallery[valid] @ user_vec) / (gallery_norm[valid] * user_norm)

        matched = set()
        for path, dist in zip(file_paths, cosine_distances):
            print(f"DEBUG [CPU]: {path} -> Cosine Distance: {dist:.4f} (Threshold: {tolerance})")
            if dist <= tolerance:
                matched.add(path)

        return list(matched)
