# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression
from joblib import Parallel
from sklearn.utils.fixes import delayed

from . base import juwrap_fit, juwrap_transform, JuTransformerMixin


def wrap_class(cls):
    class WrappedClass(cls):
        pass

    WrappedClass.fit = juwrap_fit(cls.fit, cls.transform)
    WrappedClass.transform = juwrap_transform(cls.transform)

    return WrappedClass


@wrap_class
class ConfoundRemover(BaseEstimator, TransformerMixin, JuTransformerMixin):

    def __init__(self,
                 model_confound=None,
                 threshold=None,
                 confounds_match="confound",
                 n_jobs=None,
                 verbose=0,
                 apply_to_column_names=None,
                 apply_to_column_types=None,
                 returned_features='same'):
        self.model_confound = model_confound
        self.threshold = threshold
        self.confounds_match = confounds_match
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.apply_to_column_names = apply_to_column_names
        self.apply_to_column_types = apply_to_column_types
        self.returned_features = returned_features

    def fit(self, X, y=None, **fit_params):

        X_conf_mask = self.get_features_confounds_mask()
        self.model_confound = (
            LinearRegression()
            if self.model_confound is None
            else self.model_confound)
        self.apply_to_column_types = (
            ["continuous", "confound"]
            if self.apply_to_column_types is None
            else self.apply_to_column_types)

        features = X[:, X_conf_mask]
        confounds = X[:, ~X_conf_mask]

        def fit_confound_models(t_X, confounds):
            _model = clone(self.model_confound)
            _model.fit(confounds, t_X)
            return _model

        self.models_confound_ = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose
        )(delayed(fit_confound_models)(features[:, i], confounds)
            for i in range(features.shape[1])
          )

        return self

    def transform(self, X):

        X_conf_mask = self.get_features_confounds_mask()

        features = X[:, X_conf_mask]
        confounds = X[:, ~X_conf_mask]

        for i, model in enumerate(self.models_confound_):
            t_pred = model.predict(confounds)
            residuals = features[:, i] - t_pred
            if self.threshold is not None:
                residuals[np.abs(residuals) < self.threshold] = 0
            features[:, i] = residuals

        X[:, X_conf_mask] = features
        return X

    def get_features_confounds_mask(self):
        # determining what is a confound and a feature
        # in the already subsetted X
        X_conf_types = self.X_types_[self.mask_columns_]
        X_conf_mask = self.confounds_match != X_conf_types
        return X_conf_mask
