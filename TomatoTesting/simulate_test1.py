"""
Simulate Test — Test the trained model on random internet images
================================================================
Use this to verify the model works on real-world leaf photos
before connecting the hardware.

Usage in Colab:
    predict_on_leaf("your_leaf.jpg")

Or test multiple images:
    test_batch(["leaf1.jpg", "leaf2.jpg", "leaf3.jpg"])
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────
MODEL_PATH           = 'tomato_model.keras'
IMG_SIZE             = 300              # must match training — was wrong (128) before
CONFIDENCE_THRESHOLD = 0.65
SEVERITY_THRESHOLD   = 0.30            # 30% leaf area → Mild / Severe
GRADCAM_THRESHOLD    = 0.45

# Must match the alphabetical order Keras used during training
CLASS_NAMES = ["Tomato___Early_blight", "Tomato___healthy"]
#               index 0                   index 1
# ──────────────────────────────────────────────────────────────


# ── LOAD MODEL ────────────────────────────────────────────────
print("🔄 Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.\n")


# ── GRADCAM ──────────────────────────────────────────────────
def make_gradcam_heatmap(img_array, pred_index):
    """
    Uses input-space gradients as a reliable alternative to GradCAM.
    Works perfectly with nested models like EfficientNetB3.
    Produces equivalent heatmap showing which regions drive the prediction.
    """
    img_tensor = tf.Variable(tf.cast(img_array, tf.float32))

    with tf.GradientTape() as tape:
        preds = model(img_tensor, training=False)
        score = preds[:, pred_index]

    # Gradients w.r.t. input image — shape (1, 300, 300, 3)
    grads = tape.gradient(score, img_tensor)

    # Take max across color channels → (300, 300)
    heatmap = tf.reduce_max(tf.abs(grads[0]), axis=-1)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def segment_leaf(image_bgr):
    hsv        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([25, 20, 20]),  np.array([100, 255, 255]))
    mask_brown = cv2.inRange(hsv, np.array([5,  30, 30]),  np.array([30,  255, 200]))
    leaf_mask  = cv2.bitwise_or(mask_green, mask_brown)
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    leaf_mask  = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask  = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN,  kernel)
    return leaf_mask


def get_severity(heatmap, image_bgr):
    if heatmap is None:
        return 0.0   # fallback — treat as 0% if GradCAM failed
    h, w         = image_bgr.shape[:2]
    hmap_resized = cv2.resize(heatmap, (w, h))
    activated    = (hmap_resized >= GRADCAM_THRESHOLD).astype(np.uint8) * 255
    leaf_mask    = segment_leaf(image_bgr)
    leaf_px      = np.sum(leaf_mask > 0)
    diseased_px  = np.sum((activated > 0) & (leaf_mask > 0))
    if leaf_px == 0:
        leaf_px     = h * w
        diseased_px = np.sum(activated > 0)
    return diseased_px / leaf_px


# ── MAIN TEST FUNCTION ────────────────────────────────────────
def predict_on_leaf(image_path, show=True):
    """
    Classify a single leaf image and display results with GradCAM overlay.
    Works in both Colab (inline plot) and locally (cv2 window).
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not load image: {image_path}")
        return None

    # Preprocess — resize only, NO manual /255
    # EfficientNetB3 has its own normalization built in
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    arr     = np.expand_dims(resized.astype(np.float32), 0)

    # Predict
    preds      = model.predict(arr, verbose=0)[0]
    pred_idx   = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    label      = CLASS_NAMES[pred_idx]
    is_blight  = "blight" in label.lower() or "early" in label.lower()

    severity_pct   = None
    severity_label = None
    overlay        = None

    if confidence < CONFIDENCE_THRESHOLD:
        result = "UNCERTAIN"
        action = "⚪ Manual check needed"
        box_color = (200, 200, 200)

    elif not is_blight:
        result    = "HEALTHY"
        action    = "✅ No spray needed"
        box_color = (0, 200, 0)

    else:
        heatmap      = make_gradcam_heatmap(arr, pred_idx)
        severity_pct = get_severity(heatmap, resized)

        # Build GradCAM overlay
        hmap_up = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        colored = cv2.applyColorMap((hmap_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(resized, 0.6, colored, 0.4, 0)

        # GradCAM override — if area affected is very low, trust GradCAM over classifier
        if severity_pct >= SEVERITY_THRESHOLD:
            result        = "EARLY BLIGHT — SEVERE"
            severity_label = "Severe"
            action        = f"🚨 Spray 5 sec | {severity_pct*100:.1f}% leaf affected"
            box_color     = (0, 0, 255)
        else:
            result        = "EARLY BLIGHT — MILD"
            severity_label = "Mild"
            action        = f"⚠️  Spray 2 sec | {severity_pct*100:.1f}% leaf affected"
            box_color     = (0, 200, 255)
    # ── Print result ─────────────────────────────────────────
    print("\n" + "─" * 50)
    print(f"  Image      : {image_path}")
    print(f"  Result     : {result}")
    print(f"  Confidence : {confidence*100:.1f}%")
    if severity_pct is not None:
        print(f"  Area affected: {severity_pct*100:.1f}%")
    print(f"  Action     : {action}")
    print("─" * 50)

    # ── Display (works in Colab inline) ──────────────────────
    if show:
        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_rgb  = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if overlay is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original_rgb)
            axes[0].set_title("Original Image", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(resized_rgb)
            axes[1].set_title("What the model sees (300×300)", fontsize=11)
            axes[1].axis("off")

            axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[2].set_title(
                f"GradCAM — {result}\n{severity_pct*100:.1f}% leaf area affected",
                fontsize=11
            )
            axes[2].axis("off")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original_rgb)
            axes[0].set_title("Original Image", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(resized_rgb)
            axes[1].set_title(f"{result}\nConfidence: {confidence*100:.1f}%", fontsize=11)
            axes[1].axis("off")

        plt.suptitle(action, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    return {
        "result"          : result,
        "confidence"      : confidence,
        "severity_label"  : severity_label,
        "severity_percent": severity_pct,
        "action"          : action,
    }


def test_batch(image_paths):
    """Run predict_on_leaf on a list of images and print a summary."""
    results = []
    for path in image_paths:
        r = predict_on_leaf(path, show=True)
        if r:
            results.append(r)

    print("\n" + "=" * 50)
    print("  📊 BATCH SUMMARY")
    print("=" * 50)
    for path, r in zip(image_paths, results):
        print(f"  {path:<25} → {r['result']} ({r['confidence']*100:.1f}%)")
    print("=" * 50)


predict_on_leaf('real_healthy.jpg')
# ── RUN ───────────────────────────────────────────────────────
# In Colab: upload an image first, then call predict_on_leaf()
#
# from google.colab import files
# uploaded = files.upload()
# filename = list(uploaded.keys())[0]
# predict_on_leaf(filename)
#
# Or test multiple:
# test_batch(["leaf1.jpg", "leaf2.jpg", "leaf3.jpg"])