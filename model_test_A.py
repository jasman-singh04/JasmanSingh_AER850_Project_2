# Project 2 model_test_A
# Jasman Singh, 501180039
# Due November 5, 2025

import os
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# SETTINGS
OUT_DIR = "./model_outputs"
IMG_SIZE = (500, 500)

# Explicit test images (one per class)
TEST_FILES = {
    "crack": os.path.join("Project 2 Data", "Data", "test", "crack", "test_crack.jpg"),
    "missing-head": os.path.join("Project 2 Data", "Data", "test", "missing-head", "test_missinghead.jpg"),
    "paint-off": os.path.join("Project 2 Data", "Data", "test", "paint-off", "test_paintoff.jpg"),
}


# HELPERS
def load_class_mapping():
    """Load the saved class indices from training and reverse them."""
    path = os.path.join(OUT_DIR, "class_indices.json")
    with open(path, "r") as f:
        mapping = json.load(f)
    return {v: k for k, v in mapping.items()}


def preprocess(img_path):
    """Load image, resize, normalize, add batch dimension."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def annotate(predictions, name, class_map):
    """
    Annotate test images with class probabilities and show/save.
    """
    imgs = []
    font_size = 14
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for true_cls, path, pred_label, conf, p_array in predictions:
        img = Image.open(path).convert("RGB").resize((300, 300))
        draw = ImageDraw.Draw(img)
        color = (0, 255, 0) if true_cls == pred_label else (255, 0, 0)

        text_lines = [f"T: {true_cls}"]
        for i, prob in enumerate(p_array):
            text_lines.append(f"{class_map[i]}: {prob*100:.1f}%")
        draw.multiline_text((8, 8), "\n".join(text_lines), fill=color, font=font, spacing=2)
        imgs.append(img)

    # Combine images horizontally
    w = sum(i.width for i in imgs)
    h = max(i.height for i in imgs)
    canvas = Image.new("RGB", (w, h), (40, 40, 40))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width

    out_path = os.path.join(OUT_DIR, f"test_results_{name}.png")
    canvas.save(out_path)
    print(f"Saved annotated results for {name} → {out_path}")

    # Display in notebook
    plt.figure(figsize=(12, 6))
    plt.imshow(canvas)
    plt.axis("off")
    plt.show()



# MAIN
if __name__ == "__main__":
    class_map = load_class_mapping()
    model_path = os.path.join(OUT_DIR, "model_A_best.h5")
    model = load_model(model_path)

    preds, correct = [], 0
    for true_cls, path in TEST_FILES.items():
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue
        arr = preprocess(path)
        p = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(p))
        pred_label = class_map[idx]
        conf = float(p[idx])
        preds.append((true_cls, path, pred_label, conf, p))
        if pred_label == true_cls:
            correct += 1

        print(f"{os.path.basename(path)} | True: {true_cls} | Pred: {pred_label} ({conf:.2f})")

    total = len(preds)
    acc = 100 * correct / total if total > 0 else 0
    print(f"Model A Test Accuracy: {acc:.1f}%")

    annotate(preds, "model_A", class_map)