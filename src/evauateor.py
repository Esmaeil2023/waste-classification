import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Type

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.metrics import balanced_accuracy_score, f1_score

from datasets import get_transforms
from models import get_model


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
@dataclass
class Config:
    checkpoint_path: str
    model_name: str = "resnet50"
    num_classes: int = 6
    batch_size: int = 32

    realwaste_root: str = ""
    taco_root: str = ""
    taco_annotations: str = ""


# ─────────────────────────────────────────────
# DOMAIN SPACE
# ─────────────────────────────────────────────
UNIFIED_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(UNIFIED_CLASSES)}


# ─────────────────────────────────────────────
# MAPPERS (stratgy pattern)
# ─────────────────────────────────────────────
class BaseMapper:
    def map(self, label: str) -> str:
        raise NotImplementedError


class RealWasteMapper(BaseMapper):
    MAP = {
        "Cardboard": "cardboard",
        "Glass": "glass",
        "Metal": "metal",
        "Paper": "paper",
        "Plastic": "plastic",
        "Miscellaneous Trash": "trash",
        "Food Organics": None,
        "Textile Trash": None,
        "Vegetation": None,
    }

    def map(self, label: str):
        return self.MAP.get(label)


class TACOMapper(BaseMapper):
    MAP = {
        #cardboard
        "Toilet tube": "cardboard",
        "Other carton": "cardboard",
        "Egg carton": "cardboard",
        "Drink carton": "cardboard",
        "Corrugated carton": "cardboard",
        "Meal carton": "cardboard",
        "Pizza box": "cardboard",

        # plastic
        "Other plastic bottle": "plastic",
        "Clear plastic bottle": "plastic",
        "Plastic bottle cap": "plastic",
        "Other plastic cup": "plastic",
        "Disposable plastic cup": "plastic",
        "Foam cup": "plastic",
        "Plastic lid": "plastic",
        "Other plastic": "plastic",
        "Plastic film": "plastic",
        "Six pack rings": "plastic",
        "Garbage bag": "plastic",
        "Other plastic wrapper": "plastic",
        "Single-use carrier bag": "plastic",
        "Polypropylene bag": "plastic",
        "Crisp packet": "plastic",
        "Spread tub": "plastic",
        "Tupperware": "plastic",
        "Disposable food container": "plastic",
        "Foam food container": "plastic",
        "Other plastic container": "plastic",
        "Plastic glooves": "plastic",
        "Plastic utensils": "plastic",
        "Plastic straw": "plastic",
        "Squeezable tube": "plastic",

        #glass
        "Glass bottle": "glass",
        "Glass cup": "glass",
        "Glass jar": "glass",

        #metal
         "Food Can": "metal",
        "Drink can": "metal",
        "Aerosol": "metal",
        "Scrap metal": "metal",
        "Metal bottle cap": "metal",
        "Pop tab": "metal",

        # paper
        "Magazine paper": "paper",
        "Tissues": "paper",
        "Wrapping paper": "paper",
        "Normal paper": "paper",
        "Paper bag": "paper",
        "Plastified paper bag": "paper",
        "Paper cup": "paper",
        "Paper straw": "paper",

        # trash
        "Battery": "trash",
        "Aluminium foil": "trash",
        "Aluminium blister pack": "trash",
        "Carded blister pack": "trash",
        "Food waste": "trash",
        "Rope & strings": "trash",
        "Shoe": "trash",
        "Styrofoam piece": "trash",
        "Unlabeled litter": "trash",
        "Cigarette": "trash"

    }

    def map(self, label: str):
        label = label.lower().strip()
        return self.MAP.get(label, "trash")


# ─────────────────────────────────────────────
# BASE DATASET
# ─────────────────────────────────────────────
class BaseDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self._load_image(path)

        if self.transform:
            img = self.transform(img)

        return img, label


# ─────────────────────────────────────────────
# REALWASTE DATASET
# ─────────────────────────────────────────────
class RealWasteDataset(BaseDataset):
    def __init__(self, root, mapper: RealWasteMapper, transform=None):
        super().__init__(transform)
        self.root = Path(root)
        self.mapper = mapper
        self._index()

    def _index(self):
        for folder in self.root.iterdir():

            mapped = self.mapper.map(folder.name)
            if mapped is None:
                continue

            label = CLASS_TO_IDX[mapped]

            for img in folder.glob("*.*"):
                if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((str(img), label))


# ─────────────────────────────────────────────
# TACO DATASET (COCO STYLE)
# ─────────────────────────────────────────────
class TACODataset(BaseDataset):
    def __init__(self, root, annotations, mapper: TACOMapper, transform=None):
        super().__init__(transform)
        self.root = Path(root)
        self.mapper = mapper
        self._index(annotations)

    def _index(self, ann_path):

        with open(ann_path, "r") as f:
            data = json.load(f)

        images = {img["id"]: img["file_name"] for img in data["images"]}
        categories = {c["id"]: c["name"] for c in data["categories"]}

        for ann in data["annotations"]:

            raw = categories[ann["category_id"]]
            mapped = self.mapper.map(raw)

            if mapped is None:
                continue

            label = CLASS_TO_IDX[mapped]
            img_path = self.root / images[ann["image_id"]]

            self.samples.append((str(img_path), label))


# ─────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────
class DatasetFactory:
    def __init__(self):
        self.registry: Dict[str, Type[BaseDataset]] = {}

    def register(self, name: str, cls):
        self.registry[name] = cls

    def create(self, name: str, **kwargs):
        return self.registry[name](**kwargs)


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
def load_model(checkpoint_path, model_name, num_classes, device):

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model, checkpoint


# ─────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────
class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(self, loader, name: str):

        preds, labels, losses = [], [], []

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            out = self.model(x)
            loss = self.criterion(out, y)

            losses.append(loss.item())

            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())

        acc = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")

        return {
            "dataset": name,
            "loss": sum(losses) / len(losses),
            "balanced_acc": acc,
            "macro_f1": f1
        }



# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
def run(cfg: Config):

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print(f"Device: {device}")

    model, ckpt = load_model(
        cfg.checkpoint_path,
        cfg.model_name,
        cfg.num_classes,
        device
    )

    print("Checkpoint epoch:", ckpt.get("epoch", "N/A"))

    factory = {
        "realwaste": RealWasteDataset,
        "taco": TACODataset
    }

    datasets = {
        "realwaste": {
            "root": cfg.realwaste_root,
            "mapper": RealWasteMapper(),
        },
        "taco": {
            "root": cfg.taco_root,
            "annotations": cfg.taco_annotations,
            "mapper": TACOMapper(),
        }
    }

    evaluator = Evaluator(model, device)

    results = {}

    # ─────────────────────────────
    # IN-DOMAIN EVAL
    # ─────────────────────────────
    for name in datasets:

        ds = factory[name](transform= get_transforms("TEST"), **datasets[name])

        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

        res = evaluator.evaluate(loader, name)
        results[name] = res

        print(f"{name}: acc={res['balanced_acc']:.4f} | f1={res['macro_f1']:.4f}")

    # ─────────────────────────────
    # CROSS DOMAIN EVAL
    # ─────────────────────────────
    print("\n===== CROSS DOMAIN =====")

    cross_pairs = [
        ("taco", "realwaste"),
        ("realwaste", "taco")
    ]

    for src, tgt in cross_pairs:

        ds = factory[src](transform=get_transforms("TEST"), **datasets[src])
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

        res = evaluator.evaluate(loader, f"{src}->{tgt}")

        print(f"{src}->{tgt}: acc={res['balanced_acc']:.4f} | f1={res['macro_f1']:.4f}")

        results[f"{src}->{tgt}"] = res

    # ─────────────────────────────
    # GENERALIZATION GAP
    # ─────────────────────────────
    gap = abs(
        results["realwaste"]["balanced_acc"] -
        results["taco"]["balanced_acc"]
    )

    print("\n===== SUMMARY =====")
    print("RealWaste:", results["realwaste"])
    print("TACO:", results["taco"])
    print("Generalization Gap:", gap)

    return results
# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":

    cfg = Config(
        checkpoint_path="/content/drive/MyDrive/ColabNotebooks/resnet50_best.pth",
        realwaste_root="/content/drive/MyDrive/ColabNotebooks/dataset",
        taco_root="/content/drive/MyDrive/ColabNotebooks/Taco/data/TACO/data",
        taco_annotations="/content/drive/MyDrive/ColabNotebooks/Taco/data/TACO/data/annotations.json",
    )

    run(cfg)