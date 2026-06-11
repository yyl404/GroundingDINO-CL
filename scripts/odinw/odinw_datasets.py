"""OdinW-13 dataset path configuration (dictionary order)."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
ODINW_ROOT = REPO_ROOT / "data" / "OdinW-13"
FEWSHOT_SEED = 30


@dataclass(frozen=True)
class OdinWDataset:
    name: str
    subpath: str = ""
    train_dir: str = "train"
    val_dir: str = "valid"
    test_dir: str = "test"
    pistols_layout: bool = False


ODINW_DATASETS: List[OdinWDataset] = [
    OdinWDataset("AerialMaritimeDrone", "tiled"),
    OdinWDataset("Aquarium", "Aquarium Combined.v2-raw-1024.coco"),
    OdinWDataset("CottontailRabbits"),
    OdinWDataset("EgoHands", "generic"),
    OdinWDataset("NorthAmericaMushrooms", "North American Mushrooms.v1-416x416.coco"),
    OdinWDataset("Packages", "Raw"),
    OdinWDataset("PascalVOC", test_dir="minival"),
    OdinWDataset("Raccoon", "Raccoon.v2-raw.coco"),
    OdinWDataset("ShellfishOpenImages", "raw"),
    OdinWDataset("VehiclesOpenImages", "416x416"),
    OdinWDataset("pistols", "export", pistols_layout=True),
    OdinWDataset("pothole"),
    OdinWDataset("thermalDogsAndPeople"),
]


def dataset_root(ds: OdinWDataset) -> Path:
    base = ODINW_ROOT / ds.name
    if ds.subpath:
        return base / ds.subpath
    return base


def normalize_shot_mode(shot_mode: str) -> str:
    if shot_mode == "full":
        return "full"
    if shot_mode.startswith("shot"):
        return shot_mode
    return f"shot{shot_mode}"


def train_json(ds: OdinWDataset, shot_mode: str) -> Path:
    root = dataset_root(ds)
    shot_mode = normalize_shot_mode(shot_mode)
    if shot_mode == "full":
        if ds.pistols_layout:
            return root / "train_annotations_without_background.json"
        return root / ds.train_dir / "_annotations.coco.json"
    shot_num = shot_mode.replace("shot", "")
    if ds.pistols_layout:
        return root / f"fewshot_train_shot{shot_num}_seed{FEWSHOT_SEED}.json"
    return root / ds.train_dir / f"fewshot_train_shot{shot_num}_seed{FEWSHOT_SEED}.json"


def val_json(ds: OdinWDataset) -> Path:
    root = dataset_root(ds)
    if ds.pistols_layout:
        return root / "val_annotations_without_background.json"
    return root / ds.val_dir / "annotations_without_background.json"


def test_json(ds: OdinWDataset) -> Path:
    root = dataset_root(ds)
    if ds.pistols_layout:
        return root / "test_annotations_without_background.json"
    return root / ds.test_dir / "annotations_without_background.json"


def image_dir_for_json(json_path: Path) -> Path:
    return json_path.parent
