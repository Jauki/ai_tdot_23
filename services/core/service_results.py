from __future__ import annotations
from typing import *
from enum import Enum
import numpy as np


class FaceRecognitionResult:
    def __init__(self, label: str, certainty: float):
        self.label = label
        self.certainty = certainty

    @staticmethod
    def from_dto(dto: dict[str, any]) -> FaceRecognitionResult:
        label = dto['label']
        certainty = float(dto['certainty'])
        return FaceRecognitionResult(label, certainty)


class GlassesRecognitionResultType(Enum):
    NO_FACE = 'NO_FACE'
    NO_GLASSES = 'NO_GLASSES'
    HAS_GLASSES = 'HAS_GLASSES'


class GlassesRecognitionResult:
    def __init__(self, result: GlassesRecognitionResultType, landmarks: np.array):
        self.result = result
        self.landmarks = landmarks

    def to_dto(self) -> dict[str, any]:
        return {
            'result': self.result.value,
            'landmarks': self.landmarks.tolist(),
        }

    @staticmethod
    def from_dto(dto: dict[str, any]) -> GlassesRecognitionResult:
        return GlassesRecognitionResult(
            result=GlassesRecognitionResultType[dto['result']],
            landmarks=np.array(dto['landmarks']),
        )


class GenderEstimationResult:
    def __init__(self, gender: str):
        self.gender = gender

    def to_dto(self) -> dict[str, any]:
        return {
            'gender': self.gender,
        }

    @staticmethod
    def from_dto(dto: dict[str, any]) -> GenderEstimationResult:
        return GenderEstimationResult(
            gender=dto['gender'],
        )
