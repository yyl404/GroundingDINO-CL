import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Union

from PIL import Image
import torch
import torchvision.transforms.functional as TF

from torch import Tensor
from torch.utils.data import Dataset

from finetune.datasets.yolo import collate_fn


def _load_coco_json(path: Union[str, Path]) -> Dict:
    """Load COCO annotation json and build class/index mappings."""
    json_path = Path(path).expanduser().resolve()
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = sorted(data["categories"], key=lambda c: c["id"])
    class_names = [c["name"] for c in categories]
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}

    images = data["images"]
    annotations = data["annotations"]

    image_id_to_info = {img["id"]: img for img in images}
    image_id_to_anns: Dict[int, list] = defaultdict(list)
    for ann in annotations:
        image_id_to_anns[ann["image_id"]].append(ann)

    return {
        "json_path": str(json_path),
        "class_names": class_names,
        "cat_id_to_idx": cat_id_to_idx,
        "images": images,
        "image_id_to_info": image_id_to_info,
        "image_id_to_anns": dict(image_id_to_anns),
    }


def _coco_bbox_to_cxcywh(bbox: List[float], width: int, height: int) -> List[float]:
    """Convert COCO [x, y, w, h] pixel bbox to normalized cxcywh."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    return [cx, cy, w / width, h / height]


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        annotation_path: Union[str, Path],
        image_dir: Union[str, Path] = None,
        transform: Callable = None,
        target_transform: Callable = None,
        joint_transform: Callable = None,
    ):
        """Build a dataset from one COCO annotation json file."""
        super().__init__()
        self.cfg = _load_coco_json(annotation_path)
        self.image_dir = Path(
            image_dir if image_dir is not None else Path(annotation_path).parent
        ).expanduser().resolve()

        self._transform = transform
        self._target_transform = target_transform
        self._joint_transform = joint_transform

        self.samples: List[tuple[str, int]] = []
        for img_info in self.cfg["images"]:
            image_path = str(self.image_dir / img_info["file_name"])
            self.samples.append((image_path, img_info["id"]))

    @property
    def class_names(self) -> List[str]:
        return list(self.cfg["class_names"])

    @property
    def num_classes(self) -> int:
        return len(self.cfg["class_names"])

    def __len__(self) -> int:
        return len(self.samples)

    def _read_annotations(self, image_id: int, width: int, height: int) -> Dict[str, Tensor]:
        anns = self.cfg["image_id_to_anns"].get(image_id, [])
        labels = []
        boxes = []
        cat_id_to_idx = self.cfg["cat_id_to_idx"]

        for ann in anns:
            cls_idx = cat_id_to_idx[ann["category_id"]]
            cxcywh = _coco_bbox_to_cxcywh(ann["bbox"], width, height)
            labels.append(cls_idx)
            boxes.append(cxcywh)

        if not labels:
            return {
                "labels": torch.zeros(0, dtype=torch.long),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            }

        return {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }

    def __getitem__(self, idx: int):
        image_path, image_id = self.samples[idx]
        img_info = self.cfg["image_id_to_info"][image_id]
        width = img_info["width"]
        height = img_info["height"]

        image = Image.open(image_path).convert("RGB")
        target = self._read_annotations(image_id, width, height)
        target["info"] = {"ori_size": (width, height)}

        if self._joint_transform is not None:
            image, target = self._joint_transform(image, target)

        if self._target_transform is not None:
            target = self._target_transform(target)

        if self._transform is not None:
            image = self._transform(image)
        else:
            image = TF.to_tensor(image)

        return image, target
