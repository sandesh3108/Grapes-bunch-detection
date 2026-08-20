"""
Custom Annotation Converter Placeholder.
Raises clear, actionable errors when an unsupported or custom annotation format is encountered.
"""

from src.preprocessing.contracts import AnnotationRecord


def handle_custom_format(
    annotation_path: str,
    image_path: str,
    format_name: str,
) -> AnnotationRecord:
    """Handles custom or unrecognized annotation formats by raising an error/logging."""
    msg = (
        f"Unsupported custom annotation format '{format_name}' at {annotation_path}. "
        "Please add a converter under src/preprocessing/annotation/custom_to_yolo.py."
    )
    return AnnotationRecord(
        image_path=image_path,
        annotation_path=annotation_path,
        source_format=format_name,
        valid=False,
        error_message=msg,
    )
