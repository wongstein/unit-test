import pytest

import ml_pipeline


@pytest.fixture
def iris_dataset():
    return datasets.load_iris()


def describe_data():
    def it_has_expected_features():
        pass


def describe_training_test_split():
    def it_splits_on_70_30():
        pass
