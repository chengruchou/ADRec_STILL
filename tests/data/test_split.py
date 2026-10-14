from pathlib import Path

import torch

from adrec_still.data.adapters.gowalla import parse_and_clean_events
from adrec_still.data.pipeline import encode_events, fit_training_vocabularies
from adrec_still.data.split import assign_chronological_splits, build_base_gaps
from adrec_still.evaluation.ranking import ranking_metrics

FIXTURE = Path(__file__).parents[1] / "fixtures" / "gowalla_small.txt"


def test_split_is_chronological_and_gap_safe():
    events, _ = parse_and_clean_events(FIXTURE)
    gaps = build_base_gaps(events)
    split = assign_chronological_splits(gaps, train_ratio=0.34, validation_ratio=0.34)
    assert not split.groupby("gap_id")["split"].nunique().gt(1).any()
    ordered = split.sort_values(["next_event_time_s", "gap_id"])
    assert ordered["split"].astype(str).tolist() == ["train", "validation", "test"]


def test_held_out_pois_do_not_enter_training_vocabulary():
    events, _ = parse_and_clean_events(FIXTURE)
    split = assign_chronological_splits(
        build_base_gaps(events), train_ratio=0.34, validation_ratio=0.34
    )
    user_vocab, poi_vocab, _, _ = fit_training_vocabularies(events, split)
    encoded = encode_events(events, user_vocab, poi_vocab)
    assert 1 in encoded["poi_id"].to_numpy()
    unseen = encoded.loc[~encoded["raw_poi_id"].isin(poi_vocab)]
    assert unseen["raw_poi_id"].nunique() >= 2
    assert set(unseen["poi_id"]) == {1}
    distinct_unseen_targets = torch.tensor(
        unseen.drop_duplicates("raw_poi_id")["poi_id"].head(2).to_numpy(), dtype=torch.long
    )
    metrics = ranking_metrics(
        torch.tensor([[0.0, 100.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0]]),
        distinct_unseen_targets,
        target_in_train_catalog=torch.tensor([False, False]),
        observed_mask=torch.tensor([True, True]),
        ks=(1,),
    )
    assert metrics["cold_start_target_count"] == 2
    assert metrics["sample_count"] == 0
    assert metrics["recall@1"] is None


def test_equal_next_event_times_use_gap_id_tie_breaker():
    events, _ = parse_and_clean_events(FIXTURE)
    gaps = build_base_gaps(events)
    gaps.loc[:, "next_event_time_s"] = 123
    ordered = assign_chronological_splits(gaps)
    assert ordered["gap_id"].tolist() == sorted(ordered["gap_id"].tolist())
