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
from forte.data.ontology.top import Generics
from typing import Optional

__all__ = [
    "Metric",
]


@dataclass
class Metric(Generics):
    """
    CTC Metrics for generation tasks
    Attributes:
        metric_name (Optional[str]):
        metric_value (Optional[float]):
    """

    metric_name: Optional[str]
    metric_value: Optional[float]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.metric_name: Optional[str] = None
        self.metric_value: Optional[float] = None
