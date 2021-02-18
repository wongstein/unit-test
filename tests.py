import pytest

import ml_pipeline




# def describe_data():
    # def it_has_expected_features():
    #     data = ml_pipeline.load_data()
    #     expected_columns = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    #     for column in expected_columns:
    #         assert column in data.columns

    # def it_has_target():
    #     data = ml_pipeline.load_data()
    #     assert "target" in data.columns


def describe_data_transformation():
    @pytest.fixture
    def data():
        return [[i] for i in range(0, 10)]


    @pytest.fixture
    def normalized_data(data):
        return [[i/10.0] for i in range(0, 10)]


    def it_normalizes_x_data(data, normalized_data):
        assert ml_pipeline.normalize(data) == normalized_data



# def describe_training_test_split():
#     @pytest.fixture
#     def iris_dataset():
#         return ml_pipeline.load_data()

#     def it_splits_on_80_20(iris_dataset):
#         x_train, x_test, y_train, y_test = ml_pipeline.split_data(iris_dataset)
#         assert len(y_train)/iris_dataset.shape[0] == 0.8

# def describe_model():
#     def it_has_5_layers():
#         pass

#     def its_layers_have_weights():
#         pass

