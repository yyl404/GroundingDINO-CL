import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

import groundingdino.datasets.transforms as T

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def build_caption(classes: List[str]) -> str:
    return " . ".join(classes) + " ."


def _normalize_split_name(split: str) -> str:
    split = split.strip().lower()
    if split not in {"train", "val", "test", "trainval"}:
        raise ValueError(f"Unsupported split: {split}")
    return split


def _read_split_ids(voc_root: str, split: str) -> List[str]:
    split = _normalize_split_name(split)
    split_file = os.path.join(voc_root, "ImageSets", "Main", f"{split}.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file does not exist: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f.readlines() if line.strip()]
    return ids


def _parse_voc_annotation(xml_path: str, class_to_id: Dict[str, int], keep_difficult: bool):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []
    difficult_flags = []

    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None:
            continue
        class_name = name_node.text.strip()
        if class_name not in class_to_id:
            continue

        difficult_node = obj.find("difficult")
        difficult = int(difficult_node.text) if difficult_node is not None else 0
        if difficult and not keep_difficult:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_to_id[class_name])
        difficult_flags.append(difficult)

    return boxes, labels, difficult_flags


def build_class_token_map(tokenizer, classes: List[str]) -> Dict[int, List[int]]:
    caption = build_caption(classes)
    caption_ids = tokenizer(caption, add_special_tokens=True)["input_ids"]
    token_map = {}

    for class_idx, class_name in enumerate(classes):
        class_ids = tokenizer(class_name, add_special_tokens=False)["input_ids"]
        matches = []
        if class_ids:
            for i in range(len(caption_ids) - len(class_ids) + 1):
                if caption_ids[i : i + len(class_ids)] == class_ids:
                    matches = list(range(i, i + len(class_ids)))
                    break
        if not matches:
            raise RuntimeError(f"Unable to map class '{class_name}' into caption tokens.")
        token_map[class_idx] = matches
    return token_map


def build_train_transform():
    return T.Compose(
        [
            T.RandomResize([640, 672, 704, 736, 768, 800], max_size=1333),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform():
    return T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class VOCDataset(Dataset):
    def __init__(
        self,
        voc_root: str,
        split: str,
        transforms=None,
        classes: List[str] = None,
        keep_difficult: bool = False,
    ):
        self.voc_root = voc_root
        self.split = _normalize_split_name(split)
        self.transforms = transforms
        self.classes = classes if classes is not None else VOC_CLASSES
        self.class_to_id = {name: i for i, name in enumerate(self.classes)}
        self.keep_difficult = keep_difficult
        self.image_ids = _read_split_ids(voc_root, self.split)

        self.images_dir = os.path.join(voc_root, "JPEGImages")
        self.annotations_dir = os.path.join(voc_root, "Annotations")
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Image directory does not exist: {self.images_dir}")
        if not os.path.isdir(self.annotations_dir):
            raise FileNotFoundError(f"Annotation directory does not exist: {self.annotations_dir}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        anno_path = os.path.join(self.annotations_dir, f"{image_id}.xml")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes, labels, difficult_flags = _parse_voc_annotation(
            anno_path, self.class_to_id, self.keep_difficult
        )

        target = {
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "boxes_abs": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "difficult": torch.tensor(difficult_flags, dtype=torch.int64),
            "image_name": image_id,
        }
        if len(boxes) > 0:
            areas = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (
                target["boxes"][:, 3] - target["boxes"][:, 1]
            )
        else:
            areas = torch.zeros((0,), dtype=torch.float32)
        target["area"] = areas

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
