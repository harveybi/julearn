# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from julearn.transformers import ConfoundRemover, get_transformer
from julearn.pipeline import (ExtendedDataFramePipeline,
                              create_dataframe_pipeline,
                              _create_extended_pipeline)
X = pd.DataFrame(dict(A=np.arange(10),
                      B=np.arange(10, 20),
                      C=np.arange(30, 40)
                      ))
X_cont_only = ['continuous'] * 3
y = pd.Series(np.arange(50, 60))

X_with_types = pd.DataFrame({
    'a__:type:__continuous': np.arange(10),
    'b__:type:__continuous': np.arange(10, 20),
    'c__:type:__confound': np.arange(30, 40),
    'd__:type:__confound': np.arange(40, 50),
    'e__:type:__categorical': np.arange(40, 50),
    'f__:type:__categorical': np.arange(40, 50),
})

X_with_types = [
    'continuous', 'continuous',
    'confound', 'confound',
    'categorical', 'categorical',
]


def test_create_dataframe_pipeline_steps_added_correctly():
    # test whether the steps are added
    # and whether all the hyperparameters were transferred correctly
    scaler = StandardScaler(with_mean=False)
    pca = PCA(n_components=3)
    lr = LinearRegression()
    steps = [('zscore', scaler),
             ('pca', pca),
             ('linear_reg', lr)]

    my_pipeline = create_dataframe_pipeline(steps)

    for my_step, original_estimator in zip(
            my_pipeline.steps, [scaler, pca, lr]):
        for est_param in original_estimator.get_params():
            est_val = getattr(original_estimator, est_param)
            assert my_step[1].get_params().get(est_param) == est_val


def test_create_dataframe_pipeline_returned_features_same():

    steps = [('zscore', StandardScaler()), ]
    sklearn_pipe = make_pipeline(StandardScaler())

    my_pipe = create_dataframe_pipeline(steps)
    X_trans = my_pipe.fit_transform(
        X.values, X_types=X_cont_only, X_names=["A", "B", "C"])
    X_trans_sklearn = sklearn_pipe.fit_transform(X)

    assert (X_trans.columns == X.columns).all()
    assert_array_equal(X_trans.values, X_trans_sklearn)


def test_ExtendedDataFramePipeline_transform_with_categorical():

    steps = [('zscore', StandardScaler())]
    sklearn_pipe = make_pipeline(StandardScaler())

    my_pipe = create_dataframe_pipeline(steps)
    my_pipe = ExtendedDataFramePipeline(my_pipe, categorical_features=['C'])
    X_trans = my_pipe.fit_transform(X)
    X_trans_sklearn = sklearn_pipe.fit_transform(X.loc[:, ['A', 'B']])

    assert_array_equal(X_trans.loc[:, ['A', 'B']].values, X_trans_sklearn)
    assert_array_equal(X_trans.loc[:, ['C__:type:__categorical']].values,
                       X.loc[:, ['C']].values)


def test_tune_params():
    params = {'svm__kernel': 'linear',
              'zscore__with_mean': True,
              'confounds__zscore__with_mean': True,
              'target__with_mean': True}

    extended_pipe = _create_extended_pipeline(
        preprocess_steps_features=[('zscore', get_transformer('zscore'))],
        preprocess_steps_confounds=[('zscore', get_transformer('zscore'))],
        preprocess_transformer_target=get_transformer('zscore', target=True),
        model=('svm', SVR()),
        confounds=None,
        categorical_features=None
    )

    extended_pipe.set_params(**params)
    for param, val in params.items():
        assert extended_pipe.get_params()[param] == val

    with pytest.raises(ValueError, match='Each element of the'):
        extended_pipe.set_params(cOnFouunds__zscore__with_mean=True)


def test_ExtendedDataFramePipeline___rpr__():
    extended_pipe = _create_extended_pipeline(
        preprocess_steps_features=[('zscore', get_transformer('zscore'))],
        preprocess_steps_confounds=[('zscore', get_transformer('zscore'))],
        preprocess_transformer_target=get_transformer('zscore', target=True),
        model=('svm', SVR()),
        confounds=None,
        categorical_features=None
    )
    extended_pipe.__repr__()


def test_extended_pipeline_get_wrapped_transformer_params():
    steps = [('zscore', StandardScaler(with_mean=False))]

    my_pipe = create_dataframe_pipeline(steps)
    extended_pipe = ExtendedDataFramePipeline(my_pipe)
    extended_pipe.fit(X)
    assert extended_pipe['zscore'].with_mean is False
