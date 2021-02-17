import pytest

import ml_pipeline


@pytest.fixture
def iris_dataset():
    return datasets.load_iris()


def describe_data():
    def it_has_expected_features():
        assert True == True

def describe_data_transformation():
    pass

def describe_training_test_split():
    def it_splits_on_80_20():
        assert True == False

def describe_model():
    def it_has_5_layers():
        pass

    def its_layers_have_weights():
        pass

