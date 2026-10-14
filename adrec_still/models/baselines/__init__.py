"""Controlled baseline implementations."""

from adrec_still.common.registry import MODEL_REGISTRY, register_model
from adrec_still.models.baselines.base import (
    BASELINE_INTERFACE_VERSION,
    BaselineModel,
    BaselinePrediction,
    masked_log_probabilities,
    validate_batch,
)
from adrec_still.models.baselines.gru4rec import GRU4RecH
from adrec_still.models.baselines.markov import FirstOrderMarkov, MarkovTrainingSource
from adrec_still.models.baselines.popularity import GlobalPop
from adrec_still.models.baselines.repeat_last import RepeatLast
from adrec_still.models.baselines.sasrec import SASRecH
from adrec_still.models.baselines.sasrec_time import (
    SASRecClock,
    SASRecElapsed,
    SASRecHPM,
    SASRecS,
    SASRecTimeControl,
)
from adrec_still.models.baselines.user_history_pop import UserHistoryPop

for _model in (
    GlobalPop,
    UserHistoryPop,
    RepeatLast,
    FirstOrderMarkov,
    GRU4RecH,
    SASRecH,
    SASRecHPM,
    SASRecClock,
    SASRecS,
    SASRecElapsed,
):
    if _model.name not in MODEL_REGISTRY:
        register_model(_model.name, _model)

__all__ = [
    "BASELINE_INTERFACE_VERSION",
    "BaselineModel",
    "BaselinePrediction",
    "FirstOrderMarkov",
    "GRU4RecH",
    "GlobalPop",
    "MarkovTrainingSource",
    "RepeatLast",
    "SASRecClock",
    "SASRecElapsed",
    "SASRecH",
    "SASRecHPM",
    "SASRecS",
    "SASRecTimeControl",
    "UserHistoryPop",
    "masked_log_probabilities",
    "validate_batch",
]

