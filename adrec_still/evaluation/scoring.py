"""Model scorer surfaces for the exact full-catalog evaluator.

Every implemented model exposes one of two surfaces the exact adapter can score
through: a bounded ``exact_candidate_factory`` (the mixture arms' own normalized
marginal) or the ``encode`` / ``score_candidates`` pair of the encoder baselines.
These helpers turn either into the ``features -> (ids -> scores)`` callback that
:func:`adrec_still.evaluation.exact.exact_ranking_contributions` consumes. The callback
sees the forecast fields only (:data:`INFERENCE_FIELDS`); labels never reach it.
"""

from __future__ import annotations

import torch

INFERENCE_FIELDS = (
    "user_id",
    "history_poi",
    "history_timestamp",
    "history_mask",
    "last_timestamp",
    "query_s_sec",
)


def encoder_candidate_factory(model):
    """Adapter for the encode/score_candidates interface (e.g. GRU4Rec-H).

    No dense catalog allocation. Both calls run in eval/no-grad and restore mode.
    Mixture marginal scorers require owner-supplied exact component normalizers;
    this adapter must not be used to softmax arbitrary mixture component scores.
    """
    model.require_fitted()

    def prepare(features):
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                context = model.encode(features)
        finally:
            model.train(was_training)

        def score(ids):
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    return model.score_candidates(
                        context, ids.to(context.device).expand(len(context), -1)
                    )
            finally:
                model.train(was_training)

        return score

    return prepare


def exact_model_candidate_factory(model, chunk_size: int = 4096):
    """Use a model's bounded normalized adapter when it has one, else the encoder pair."""
    if hasattr(model, "exact_candidate_factory"):
        return model.exact_candidate_factory(chunk_size=chunk_size)
    return encoder_candidate_factory(model)


def scorer_surface(model) -> tuple[str | None, str | None]:
    """``(source_label, score_kind)`` for a model, without building the scorer."""
    if callable(getattr(model, "exact_candidate_factory", None)) and isinstance(
        getattr(model, "exact_score_kind", None), str
    ):
        return "model-exact-ranking-scorer", model.exact_score_kind
    if callable(getattr(model, "encode", None)) and callable(
        getattr(model, "score_candidates", None)
    ):
        return "encoder-candidate-factory", "logits"
    return None, None
