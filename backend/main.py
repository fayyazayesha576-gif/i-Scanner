import os
import io
import cv2
import torch
import joblib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
LABEL_NAMES = {
    'N': 'Normal',
    'D': 'Diabetic Retinopathy',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'Age-related Macular Degeneration',
    'H': 'Hypertensive Retinopathy',
    'M': 'Myopia',
    'O': 'Other Abnormalities',
}
NUM_LABELS = len(LABELS)
IMG_SIZE = (224, 224)

# ── Skip rollout on CPU to avoid timeout (set SKIP_ROLLOUT=true in env) ───────
SKIP_ROLLOUT = os.environ.get("SKIP_ROLLOUT", "false").lower() == "true"


# ── Model Definition ──────────────────────────────────────────────────────────
class AttentionFusionVit(nn.Module):
    def __init__(self, meta_dim=2, num_labels=NUM_LABELS, attn_heads=4, attn_layers=2):
        super().__init__()
        vit = vit_b_16(weights=None)
        vit.heads = nn.Identity()
        self.backbone = vit
        self.feat_dim = 768

        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_dim, nhead=attn_heads,
            dim_feedforward=self.feat_dim * 2,
            activation='gelu', dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=attn_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(0.4),
            nn.Linear(self.feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def forward(self, left_img, right_img, meta):
        left_feat  = self.backbone(left_img)
        right_feat = self.backbone(right_img)
        meta_feat  = self.meta_proj(meta)
        seq    = torch.stack([left_feat, right_feat, meta_feat], dim=1)
        out    = self.transformer(seq)
        pooled = out.mean(dim=1)
        return self.classifier(pooled)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n=3):
        for block in list(self.backbone.encoder.layers)[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.backbone.encoder.ln.parameters():
            p.requires_grad = True


# ── Attention Rollout ─────────────────────────────────────────────────────────
def get_attention_rollout(model, img_tensor):
    attentions = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            attn_weights = output[1]
            if attn_weights is not None:
                attentions.append(attn_weights.detach().cpu())
                logger.debug(f"Captured attention: {attn_weights.shape}")

    original_forwards = {}
    hooks = []

    for block in model.backbone.encoder.layers:
        mha = block.self_attention
        original_forwards[mha] = mha.forward

        def make_patched(orig_fn):
            def patched(query, key, value, **kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False
                return orig_fn(query, key, value, **kwargs)
            return patched

        mha.forward = make_patched(original_forwards[mha])
        hooks.append(mha.register_forward_hook(hook_fn))

    try:
        with torch.no_grad():
            _ = model.backbone(img_tensor)
    finally:
        for h in hooks:
            h.remove()
        for mha, orig in original_forwards.items():
            mha.forward = orig

    if not attentions:
        logger.error("No attention weights captured! Check ViT block structure.")
        return None

    logger.info(f"Captured {len(attentions)} attention layers, shape: {attentions[0].shape}")

    seq_len = attentions[0].shape[-1]
    result  = torch.eye(seq_len)

    for attn in attentions:
        attn_avg = attn[0].mean(dim=0)
        attn_aug = attn_avg + torch.eye(seq_len)
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
        result   = torch.matmul(attn_aug, result)

    mask = result[1:, 0]
    mask = mask - mask.min()
    mask = mask / (mask.max() + 1e-8)

    logger.info(f"Rollout mask: min={mask.min():.4f} max={mask.max():.4f} mean={mask.mean():.4f}")
    return mask.numpy()


def overlay_heatmap_on_image(img_np, heatmap, alpha=0.5):
    n_patches = len(heatmap)
    patch_grid = int(round(n_patches ** 0.5))
    heatmap_2d = heatmap.reshape(patch_grid, patch_grid)

    img_h, img_w = img_np.shape[:2]
    heatmap_resized = cv2.resize(
        heatmap_2d.astype(np.float32), (img_w, img_h),
        interpolation=cv2.INTER_CUBIC
    )
    heatmap_resized = (heatmap_resized - heatmap_resized.min())
    heatmap_resized = heatmap_resized / (heatmap_resized.max() + 1e-8)

    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).numpy()
    img  = img * std + mean
    return np.clip(img, 0, 1)


def image_to_base64(img_np):
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="iScanner API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")
logger.info(f"SKIP_ROLLOUT: {SKIP_ROLLOUT}")

model      = None
scaler     = None
thresholds = None

MODEL_PATH     = os.environ.get("MODEL_PATH",     "attention_fusion_best.pth")
SCALER_PATH    = os.environ.get("SCALER_PATH",    "meta_scaler.pkl")
THRESHOLD_PATH = os.environ.get("THRESHOLD_PATH", "best_thresholds.npy")


@app.on_event("startup")
async def load_model():
    global model, scaler, thresholds

    try:
        model = AttentionFusionVit().to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state', checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("✅ Model loaded successfully")

        dummy = torch.zeros(1, 3, 224, 224).to(device)
        dummy_meta = torch.zeros(1, 2).to(device)
        with torch.no_grad():
            _ = model(dummy, dummy, dummy_meta)
        logger.info("✅ Model forward pass OK")

    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        model = None

    try:
        scaler = joblib.load(SCALER_PATH)
        logger.info("✅ Scaler loaded")
    except Exception as e:
        logger.warning(f"⚠️  Scaler not found ({e}) — using raw metadata values")
        scaler = None

    try:
        thresholds = np.load(THRESHOLD_PATH)
        logger.info(f"✅ Thresholds loaded: {np.round(thresholds, 3)}")
    except Exception as e:
        logger.warning(f"⚠️  Threshold file not found ({e}) — using 0.5 for all classes")
        thresholds = np.full(NUM_LABELS, 0.5)

    # Only test rollout if not skipped
    if model is not None and not SKIP_ROLLOUT:
        try:
            dummy_img = torch.zeros(1, 3, 224, 224).to(device)
            mask = get_attention_rollout(model, dummy_img)
            if mask is not None:
                logger.info(f"✅ Attention rollout test passed — mask shape: {mask.shape}")
            else:
                logger.warning("⚠️  Attention rollout returned None on test")
        except Exception as e:
            logger.error(f"❌ Attention rollout test failed: {e}")
    elif SKIP_ROLLOUT:
        logger.info("⏭️  Attention rollout skipped (SKIP_ROLLOUT=true) — faster CPU inference")


# ── Transforms ────────────────────────────────────────────────────────────────
eval_tfms = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    return eval_tfms(img)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "device":       str(device),
        "skip_rollout": SKIP_ROLLOUT,
        "thresholds":   thresholds.tolist() if thresholds is not None else None,
    }


# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(
    left_eye:  UploadFile = File(...),
    right_eye: UploadFile = File(...),
    age:       float      = Form(...),
    gender:    str        = Form(...),
):
    if model is None:
        raise HTTPException(503, "Model not loaded — check backend logs.")

    try:
        left_bytes  = await left_eye.read()
        right_bytes = await right_eye.read()

        left_tensor  = preprocess_image(left_bytes).unsqueeze(0).to(device)
        right_tensor = preprocess_image(right_bytes).unsqueeze(0).to(device)

        logger.info(f"Input tensors — left: {left_tensor.shape}, right: {right_tensor.shape}")

        # Metadata
        sex_bin  = 0.0 if gender.lower() in ('male', 'm') else 1.0
        meta_raw = np.array([[sex_bin, float(age)]])
        if scaler is not None:
            meta_scaled = scaler.transform(meta_raw)
        else:
            meta_scaled = meta_raw
        meta_tensor = torch.tensor(meta_scaled, dtype=torch.float32).to(device)

        # Prediction
        with torch.no_grad():
            logits = model(left_tensor, right_tensor, meta_tensor)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]

        preds = (probs >= thresholds).astype(int)
        logger.info(f"Probs: {np.round(probs, 3)}")

        results = []
        for i, lbl in enumerate(LABELS):
            results.append({
                "label":       lbl,
                "name":        LABEL_NAMES[lbl],
                "probability": float(round(float(probs[i]), 4)),
                "threshold":   float(round(float(thresholds[i]), 4)),
                "detected":    bool(preds[i]),
            })
        results.sort(key=lambda x: x["probability"], reverse=True)

        # Denormalised images for overlay
        left_np  = denormalize(left_tensor[0].cpu())
        right_np = denormalize(right_tensor[0].cpu())

        # ── Attention rollout (skipped if SKIP_ROLLOUT=true) ──────────────────
        left_rollout  = None
        right_rollout = None

        if not SKIP_ROLLOUT:
            logger.info("Computing attention rollout for left eye…")
            left_rollout = get_attention_rollout(model, left_tensor)
            logger.info("Computing attention rollout for right eye…")
            right_rollout = get_attention_rollout(model, right_tensor)
        else:
            logger.info("⏭️  Skipping attention rollout (SKIP_ROLLOUT=true)")

        left_overlay_b64  = None
        right_overlay_b64 = None

        if left_rollout is not None:
            left_img_uint8 = (left_np * 255).astype(np.uint8)
            lo = overlay_heatmap_on_image(left_img_uint8, left_rollout)
            left_overlay_b64 = image_to_base64(lo)
            logger.info(f"✅ Left heatmap generated, b64 length: {len(left_overlay_b64)}")

        if right_rollout is not None:
            right_img_uint8 = (right_np * 255).astype(np.uint8)
            ro = overlay_heatmap_on_image(right_img_uint8, right_rollout)
            right_overlay_b64 = image_to_base64(ro)
            logger.info(f"✅ Right heatmap generated, b64 length: {len(right_overlay_b64)}")

        # Original images
        left_orig_b64  = image_to_base64((left_np  * 255).astype(np.uint8))
        right_orig_b64 = image_to_base64((right_np * 255).astype(np.uint8))

        detected_diseases = [r for r in results if r["detected"] and r["label"] != "N"]
        is_normal = len(detected_diseases) == 0

        return JSONResponse({
            "success":        True,
            "results":        results,
            "detected":       detected_diseases,
            "is_normal":      bool(is_normal),
            "left_original":  left_orig_b64,
            "right_original": right_orig_b64,
            "left_heatmap":   left_overlay_b64,
            "right_heatmap":  right_overlay_b64,
        })

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(500, f"Prediction error: {str(e)}")
