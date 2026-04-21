from abc import abstractmethod
from typing import Union, Callable, Dict, List
from pathlib import Path
import warnings

from PIL import Image
import torch
import yaml

from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from groundingdino.util.misc import nested_tensor_from_tensor_list

IMG_EXTS = ["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff", "gif"]


def collate_fn(batch):
    """Collate (image, target) pairs into a padded NestedTensor + targets list."""
    images, targets = zip(*batch)
    images = nested_tensor_from_tensor_list(list(images))
    return images, list(targets)


def _load_yolo_yaml(path: Union[str, Path]):
    """Load YOLO dataset yaml and normalize split paths/class names."""
    yaml_path = Path(path).expanduser().resolve()

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    root = Path(data.get("path", yaml_path.parent))

    names = data.get("names", {})
    if isinstance(names, dict):
        keys = sorted(names.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
        class_names = [str(names[k]) for k in keys]
    elif isinstance(names, list):
        class_names = [str(x) for x in names]
    else:
        raise ValueError(f"Unsupported names type in yaml: {type(names)}")

    splits = {}
    for split in ("train", "val", "test"):
        split_paths = data.get(split)
        if split_paths is None:
            continue
        if not isinstance(split_paths, list):
            split_paths = [split_paths]
        splits[split] = [
            str(p if p.is_absolute() else (root / p).resolve())
            for p in (Path(e) for e in split_paths)
        ]

    return {
        "yaml_path": str(yaml_path),
        "root": str(root),
        "class_names": class_names,
        "splits": splits,
        "raw": data,
    }


def _collect_images_from_dir(dir:  Union[str, Path]) -> List[str]:
    """Recursively collect image file absolute paths under a directory."""
    image_dir = Path(dir).expanduser().resolve()
    if not image_dir.is_dir():
        return []
    exts = {f".{ext.lower().lstrip('.')}" for ext in IMG_EXTS}
    images = [
        str(p.resolve())
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
    images.sort()
    return images


def _label_path_for_image(image_path: Union[str, Path],
                          images_dir: Union[str, Path],
                          labels_dir: Union[str, Path]
                         ) -> Union[str, Path]:
    """Map one image path to its corresponding YOLO txt label path."""
    image_path = Path(image_path).expanduser().resolve()
    images_dir = Path(images_dir).expanduser().resolve()
    labels_dir = Path(labels_dir).expanduser().resolve()
    try:
        rel = image_path.relative_to(images_dir)
        label_path = labels_dir / rel.parent / f"{image_path.stem}.txt"
    except ValueError:
        label_path = labels_dir / f"{image_path.stem}.txt"
    return str(label_path)


class YoloBaseDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        split: str = "train",
        transform: Callable = None,
        target_transform: Callable = None,
        joint_transform: Callable = None,
    ):
        """Build a dataset for one split from a YOLO yaml config."""
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got {split!r}")

        self.cfg = _load_yolo_yaml(path)
        if split not in self.cfg["splits"]:
            raise KeyError(
                f"Split {split!r} not in YAML. Available: {list(self.cfg['splits'].keys())}"
            )

        self.split = split
        self.images_dirs = self.cfg["splits"][split]
        if isinstance(self.images_dirs, str):
            self.images_dirs = [self.images_dirs]
        self.label_dirs = [image_dir.replace("images", "labels") for image_dir in self.images_dirs]
        
        self._transform = transform
        self._target_transform = target_transform
        self._joint_transform = joint_transform

        self.images_and_labels = []
        for image_dir, labels_dir in zip(self.images_dirs, self.label_dirs):
            images = _collect_images_from_dir(image_dir)
            for img in images:
                lbl = _label_path_for_image(img, image_dir, labels_dir)
                self.images_and_labels.append((img, str(Path(lbl).resolve())))
        if not self.images_and_labels:
            warnings.warn(f"No images found for split '{split}' in {self.images_dirs}")

    @property
    def class_names(self) -> List[str]:
        """Return class name list from yaml."""
        return list(self.cfg["class_names"])

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(self.cfg["class_names"])

    def __len__(self) -> int:
        """Return number of image-label pairs."""
        return len(self.images_and_labels)

    @abstractmethod
    def _read_labels(self, label_path: Union[str, Path]) -> Dict[str, Tensor]:
        raise NotImplementedError(
            f"{YoloBaseDataset.__name__} is an abstract base class: "
            f"the abstract method _read_labels() must be implemented in a concrete subclass, "
            f"not on {YoloBaseDataset.__name__} itself. "
            f"(instance class: {type(self).__name__!r}, label_path={label_path!r})"
        )

    def __getitem__(self, idx: int):
        """Load one sample and apply optional joint/target/image transforms."""
        image_path, label_path = self.images_and_labels[idx]

        image = Image.open(image_path).convert("RGB")
        target = self._read_labels(label_path)
        info = {
            "ori_size": image.size
        }

        target["info"] = info

        if self._joint_transform is not None:
            image, target = self._joint_transform(image, target)

        if self._target_transform is not None:
            target = self._target_transform(target)

        if self._transform is not None:
            image = self._transform(image)
        else:
            image = TF.to_tensor(image)

        return image, target


class YoloDetectionDataset(YoloBaseDataset):
    def _read_labels(self, label_path: Union[str, Path]) -> Dict[str, Tensor]:
        """Read one YOLO txt label file into tensor dict: labels[N], boxes[N,4] in cxcywh."""
        label_path = Path(label_path).expanduser().resolve()
        if not label_path.is_file():
            warnings.warn(f"Label file not found: {label_path}")
            return {
                "labels": torch.zeros(0, dtype=torch.long),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            }

        labels = []
        boxes = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                labels.append(cls_id)
                boxes.append([x, y, w, h])

        if not labels:
            return {
                "labels": torch.zeros(0, dtype=torch.long),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            }

        return {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }


class YoloOBBDataset(YoloDetectionDataset):
    def _read_labels(self, label_path: Union[str, Path]) -> Dict[str, Tensor]:
        raise NotImplementedError("Not implemented yet.")