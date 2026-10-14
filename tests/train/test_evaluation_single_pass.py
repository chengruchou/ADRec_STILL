"""#103: the final evaluation scores every row once; the horizon breakdown buckets
the same per-row contributions."""

from __future__ import annotations

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import yaml

import adrec_still.training.evaluate as evaluate_module
from adrec_still.training import checkpoint as ckpt
from adrec_still.training.evaluate import EvaluationOptions, evaluate_split
from adrec_still.training.inputs import resolve_data
from adrec_still.training.service import RunSpec, resolve_evaluation, resolve_model_config


def _small_model(canonical_data):
    from adrec_still.models.baselines.gru4rec import GRU4RecH

    number_of_pois = ckpt.number_of_pois(canonical_data / "poi_catalog.parquet")
    model = GRU4RecH(number_of_pois, embedding_dim=8, hidden_dim=8, num_layers=1, dropout=0.0)
    model.is_fitted = True
    return model


def test_breakdown_scores_each_batch_once_and_matches_the_horizon_filtered_passes(
    canonical_data, monkeypatch,
):
    model = _small_model(canonical_data)
    landmarks = canonical_data / "landmark_samples.parquet"
    options = EvaluationOptions(batch_size=16, max_sequence_length=8)
    calls: list[int] = []
    original = evaluate_module.exact_ranking_contributions

    def counting(scorer, targets, *args, **kwargs):
        calls.append(int(targets.shape[0]))
        return original(scorer, targets, *args, **kwargs)

    monkeypatch.setattr(evaluate_module, "exact_ranking_contributions", counting)

    plain = evaluate_split(model, landmarks, split="validation", options=options)
    plain_calls = list(calls)
    calls.clear()
    broken_down = evaluate_split(model, landmarks, split="validation", breakdown=True,
                                 options=options)
    # One catalog pass per loader batch, exactly as without the breakdown.
    assert calls == plain_calls
    for key in ("rows", "eligible_rows", "ndcg@10", "recall@10", "mrr@10", "poi_nll"):
        assert broken_down[key] == plain[key]

    table = pq.read_table(landmarks, columns=["split", "query_s_sec"])
    validation = table.filter(pc.equal(table["split"], "validation"))
    horizons = sorted(int(h) for h in pc.unique(validation["query_s_sec"]).to_pylist())
    assert len(horizons) >= 2
    assert sorted(int(h) for h in broken_down["by_horizon"]) == horizons
    assert sum(bucket["rows"] for bucket in broken_down["by_horizon"].values()) == plain["rows"]
    assert sum(bucket["eligible_rows"] for bucket in broken_down["by_horizon"].values()) \
        == plain["eligible_rows"]
    for horizon in horizons:
        bucket = broken_down["by_horizon"][str(horizon)]
        expected_rows = int(pc.sum(pc.equal(validation["query_s_sec"], horizon)).as_py())
        assert bucket["rows"] == expected_rows
        filtered = evaluate_split(model, landmarks, split="validation", horizon=horizon,
                                  options=options)
        assert filtered["rows"] == bucket["rows"]
        assert filtered["eligible_rows"] == bucket["eligible_rows"]
        # The filtered pass scores the same rows in other batches (buffered, left-padded
        # to the widest history): the float32 forward differs in its last bits, which
        # the exact ordering absorbs and the float64 NLL shows at ~1e-8 relative.
        for key in ("ndcg@10", "recall@10", "mrr@10"):
            assert bucket[key] == filtered[key]
        if filtered["poi_nll"] is None:
            assert bucket["poi_nll"] is None
        else:
            assert bucket["poi_nll"] == pytest.approx(filtered["poi_nll"], rel=1e-6)


def test_experiment_yaml_chunk_size_applies_and_the_flag_overrides_it(raw_gowalla_file,
                                                                     tmp_path):
    data = resolve_data(raw_gowalla_file, cache_dir=tmp_path / "cache")
    (tmp_path / "model.yaml").write_text(yaml.safe_dump({"name": "global_pop"}), encoding="utf-8")
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(yaml.safe_dump({
        "name": "x", "model_config": "model.yaml",
        "evaluation": {"batch_size": 32, "chunk_size": 4096, "num_workers": 0},
    }), encoding="utf-8")
    spec = RunSpec(data=data.processed_root, output=tmp_path / "out", config=experiment)
    _, source = resolve_model_config(spec)
    options = resolve_evaluation(spec, source, data)
    assert (options.batch_size, options.chunk_size) == (32, 4096)
    spec = RunSpec(data=data.processed_root, output=tmp_path / "out", config=experiment,
                   chunk_size=2048)
    _, source = resolve_model_config(spec)
    assert resolve_evaluation(spec, source, data).chunk_size == 2048
