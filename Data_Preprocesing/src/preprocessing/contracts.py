"""
Data Preprocessing Contracts & Data Models.
Defines strongly typed dataclasses used across pipeline stage boundaries.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class BoundingBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox":
        return cls(
            class_id=int(data["class_id"]),
            x_center=float(data["x_center"]),
            y_center=float(data["y_center"]),
            width=float(data["width"]),
            height=float(data["height"]),
        )


@dataclass
class ImageRecord:
    path: str
    subfolder: str
    filename: str
    width: int = 0
    height: int = 0
    channels: int = 3
    format: str = ""
    corrupted: bool = False
    annotation_path: Optional[str] = None
    detected_annotation_format: str = "none"
    modality: str = "rgb"
    provenance_group: Optional[str] = None
    status: str = "valid"  # valid, unannotated, corrupted, invalid_annotation, duplicate, low_quality
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "subfolder": self.subfolder,
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "format": self.format,
            "corrupted": self.corrupted,
            "annotation_path": self.annotation_path,
            "detected_annotation_format": self.detected_annotation_format,
            "modality": self.modality,
            "provenance_group": self.provenance_group,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class AnnotationRecord:
    image_path: str
    annotation_path: Optional[str]
    source_format: str
    boxes: List[BoundingBox] = field(default_factory=list)
    valid: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "annotation_path": self.annotation_path,
            "source_format": self.source_format,
            "boxes": [b.to_dict() for b in self.boxes],
            "valid": self.valid,
            "error_message": self.error_message,
        }


@dataclass
class DuplicateGroup:
    group_id: str
    representative_path: str
    duplicate_paths: List[str] = field(default_factory=list)
    subfolder: str = ""
    similarity_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "representative_path": self.representative_path,
            "duplicate_paths": self.duplicate_paths,
            "subfolder": self.subfolder,
            "similarity_scores": self.similarity_scores,
        }


@dataclass
class SplitAssignment:
    image_path: str
    split: str  # train, val, test
    group_id: str
    subfolder: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "split": self.split,
            "group_id": self.group_id,
            "subfolder": self.subfolder,
        }
