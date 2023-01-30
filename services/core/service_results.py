from __future__ import annotations
from typing import *


class FaceRecognitionResult:
    def __init__(self, label: str, certainty: float):
        self.label = label
        self.certainty = certainty

    @staticmethod
    def from_dto(dto: dict[str, any]) -> FaceRecognitionResult:
        label = dto['label']
        certainty = float(dto['certainty'])
        return FaceRecognitionResult(label, certainty)
