import json
from pathlib import Path

import torch

from adrec_still.evaluation.evaluator import (
    EVALUATION_SCHEMA_VERSION,
    evaluate,
    write_evaluation_json,
)
from adrec_still.time.protocol import LogNormalPrediction


def test_shared_evaluator_and_stable_json(tmp_path: Path):
    observed = torch.tensor([True, False, True])
    batch = {
        "next_poi": torch.tensor([1, 0, 2]),
        "observed_mask": observed,
        "target_in_train_catalog": torch.tensor([False, False, True]),
        "residual_time_sec": torch.tensor([10.0, 0.0, 25.0]),
        "censor_time_sec": torch.tensor([0.0, 15.0, 0.0]),
        "query_s_sec": torch.tensor([0, 21600, 0]),
    }
    prediction = LogNormalPrediction(
        mu=torch.log(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)),
        sigma=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64),
    )
    poi_logits = torch.tensor(
        [
            [0.0, 100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 100.0, 3.0, 2.0],
        ]
    )
    result = evaluate(poi_logits, batch, prediction, split="test")
    assert EVALUATION_SCHEMA_VERSION == "still-evaluation-v0.2"
    assert result["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert result["ranking"]["sample_count"] == 1
    assert result["ranking"]["cold_start_target_count"] == 1
    assert result["ranking"]["ranking_coverage"] == 0.5
    assert set(result["ranking"]["random_floor"]) == {
        "recall@5",
        "ndcg@5",
        "mrr@5",
        "recall@10",
        "ndcg@10",
        "mrr@10",
        "recall@20",
        "ndcg@20",
        "mrr@20",
    }
    assert result["ranking"]["candidate_count"] == 2
    assert result["ranking"]["mrr@5"] == 1.0
    assert result["sample_counts_by_horizon_seconds"] == {"0": 2, "21600": 1}
    output = tmp_path / "nested" / "metrics.json"
    write_evaluation_json(result, output)
    assert (
        json.loads(output.read_text(encoding="utf-8"))["schema_version"]
        == EVALUATION_SCHEMA_VERSION
    )
