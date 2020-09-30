import pipeline
import pytest
import numpy as np


@pytest.mark.parametrize("id,test_size,expected",
                         [
                             [1, 1, True],
                             [1, 0, False],
                         ])
def test_hashed_id(id, test_size, expected):
    in_test_set = pipeline.belongs_to_test(id, test_size)
    assert in_test_set is expected


@pytest.fixture()
def scaler_data():
    return np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])


def test_scaler_fit(scaler_data):
    scaler = pipeline.MinMaxScaler()
    scaler.fit(scaler_data)
    assert np.all(scaler.data_max == np.array([1., 18.]))


def test_scalar_transform(scaler_data):
    scaler = pipeline.MinMaxScaler()
    scaler.fit(scaler_data)
    assert np.all(scaler.transform(scaler_data) == np.array([[0., 0.],
                                                             [0.25, 0.25],
                                                             [0.5, 0.5],
                                                             [1., 1.]]))


def test_scalar_transform_new_data(scaler_data, new_data=np.array([2., 2.])):
    scaler = pipeline.MinMaxScaler()
    scaler.fit(scaler_data)
    assert np.all(scaler.transform(new_data) == np.array([1.5, 0.]))