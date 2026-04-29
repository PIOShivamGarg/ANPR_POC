from ultralytics import YOLO
import yaml, os

# ──────────────────────────────────────────
# CONFIG — change these to match your setup
# ──────────────────────────────────────────
DATASET_PATH = "D:\projects\ANPR_POC\dataset"   # root folder
DATA_YAML    = f"{DATASET_PATH}/data.yaml"
MODEL        = "yolov8s.pt"    # s = small, good balance of speed vs accuracy
EPOCHS       = 20
IMG_SIZE     = 640
BATCH        = 16              # reduce to 8 if you get out-of-memory errors
DEVICE       = 'cpu'           # 0 = GPU, 'cpu' = CPU only
PROJECT      = "runs/train"
RUN_NAME     = "ilpd_plate_v1"

# ──────────────────────────────────────────
# STEP 1: Verify data.yaml
# ──────────────────────────────────────────
with open(DATA_YAML, "r") as f:
    config = yaml.safe_load(f)

print("── data.yaml contents ──")
print(yaml.dump(config))

# Fix paths if Roboflow wrote absolute paths
config["path"]  = os.path.abspath(DATASET_PATH)
config["train"] = "train/images"
config["val"]   = "valid/images"
config["test"]  = "test/images"

with open(DATA_YAML, "w") as f:
    yaml.dump(config, f)

print("✅ data.yaml verified and fixed")

# ──────────────────────────────────────────
# STEP 2: Check dataset stats
# ──────────────────────────────────────────
for split in ["train", "valid", "test"]:
    img_dir = os.path.join(DATASET_PATH, split, "images")
    lbl_dir = os.path.join(DATASET_PATH, split, "labels")
    imgs    = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
    lbls    = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0
    print(f"{split:8s} → images: {imgs:4d}  |  labels: {lbls:4d}")

# ──────────────────────────────────────────
# STEP 3: Load model & train
# ──────────────────────────────────────────
model = YOLO(MODEL)   # downloads yolov8s.pt automatically

results = model.train(
    data        = DATA_YAML,
    epochs      = EPOCHS,
    imgsz       = IMG_SIZE,
    batch       = BATCH,
    device      = DEVICE,
    project     = PROJECT,
    name        = RUN_NAME,

    # Optimizer
    optimizer   = "AdamW",
    lr0         = 0.001,        # initial learning rate
    lrf         = 0.01,         # final lr = lr0 * lrf
    momentum    = 0.937,
    weight_decay= 0.0005,

    # Early stopping
    patience    = 15,           # stop if no improvement for 15 epochs

    # Augmentation (built-in, great for plates)
    augment     = True,
    hsv_h       = 0.015,        # hue variation
    hsv_s       = 0.7,          # saturation variation
    hsv_v       = 0.4,          # brightness variation
    flipud      = 0.0,          # no vertical flip (plates aren't upside down)
    fliplr      = 0.5,          # horizontal flip OK
    mosaic      = 1.0,          # mosaic augmentation (very helpful)
    translate   = 0.1,
    scale       = 0.5,

    # Misc
    cache       = True,         # cache images in RAM (faster training)
    workers     = 4,            # dataloader workers
    verbose     = True,
    save        = True,
    save_period = 10,           # save checkpoint every 10 epochs
)

print("\n✅ Training complete!")
print(f"📁 Best weights saved at: {PROJECT}/{RUN_NAME}/weights/best.pt")