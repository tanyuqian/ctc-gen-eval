# ***automatically_generated***
# ***source json:../../../../../../CTC_task/ctc-gen-eval/data_utils/ft_onto.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Evaluation pack for creation, transduction and compression
Automatically generated ontology CTC Evaluation. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from typing import Optional

__all__ = [
    "metrics",
]


@dataclass
class metrics(Annotation):
    """
    Metrics of the generated text
    Attributes:
        consistency (Optional[float]):
        relevance (Optional[float]):
        preservation (Optional[float]):
        engagingness (Optional[float]):
        groundedness (Optional[float]):
    """

    consistency: Optional[float]
    relevance: Optional[float]
    preservation: Optional[float]
    engagingness: Optional[float]
    groundedness: Optional[float]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.consistency: Optional[float] = None
        self.relevance: Optional[float] = None
        self.preservation: Optional[float] = None
        self.engagingness: Optional[float] = None
        self.groundedness: Optional[float] = None
