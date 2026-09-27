import pandas as pd
import pytest

from str_suitability.modeling.train import GROUPED_SCHEME, RANDOM_SCHEME, split_observations


def make_observations(groups_count: int = 8, per_group: int = 3):
    cells = [f"c{group}" for group in range(groups_count) for _ in range(per_group)]
    features = pd.DataFrame({"f": range(len(cells))})
    target = pd.Series(range(len(cells)))
    groups = pd.Series(cells)
    return features, target, groups


def test_grouped_split_keeps_cells_disjoint_between_train_and_test():
    features, target, groups = make_observations()
    features_train, features_test, _, _ = split_observations(features, target, groups)
    assert set(groups.loc[features_train.index]).isdisjoint(set(groups.loc[features_test.index]))


def test_grouped_split_is_deterministic():
    features, target, groups = make_observations()
    first = split_observations(features, target, groups)
    second = split_observations(features, target, groups)
    pd.testing.assert_index_equal(first[1].index, second[1].index)


def test_random_split_leaves_groups_intact_across_the_partition():
    features, target, _ = make_observations()
    frames = split_observations(features, target)
    assert len(frames[0]) + len(frames[1]) == len(features)
