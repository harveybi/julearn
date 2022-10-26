# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL

from copy import deepcopy
from ..utils.types import JuTransformerLike
from .. utils import raise_error
from .available_transformers import (_get_returned_features,
                                     _get_apply_to)
import numpy as np
from . base import JuTransformerMixin
from sklearn.base import TransformerMixin


class JuTransformerWrapper(JuTransformerMixin):

    def __init__(self,
                 transformer,
                 apply_to_column_names=None,
                 apply_to_column_types=None,
                 returned_features=None,
                 **params
                 ):

        self.transformer = transformer
        self.apply_to_column_names = apply_to_column_names
        self.apply_to_column_types = apply_to_column_types
        self.returned_features = returned_features
        self.set_params(**params)

    def fit(self, X, y=None,
            X_names=None,
            X_types=None, **fit_params):

        self.apply_to_column_types = (_get_apply_to(self.transformer)
                                      if self.apply_to_column_types is None
                                      else self.apply_to_column_types)
        self.returned_features = (_get_returned_features(self.transformer)
                                  if self.returned_features is None
                                  else self.returned_features)
        if hasattr(self, '_validate_data'):
            X, y = self._validate_data(X, y)

        self.X_names_ = None if X_names is None else np.array(X_names)
        self.X_types_ = None if X_types is None else np.array(X_types)

        if ((self.X_names_ is None) or
                (self.X_types_ is None)):
            raise_error("no feature names")  # TODO: add message

        self._set_columns_to_transform(X)
        X_transform = X[:, self.mask_columns_]
        self.transformer.fit(X_transform, y, **fit_params)

        X_transform = self.transformer.transform(X_transform)
        self._set_features_out(X_transform)

        return self

    def transform(self, X):

        if hasattr(self, '_validate_data'):
            X = self._validate_data(X)

        if self.mask_columns_ is None:
            X_out = self.transformer.transform(X)

        else:

            X_transform = X[:, self.mask_columns_]
            X_transform = self.transformer.transform(X_transform)
            X_rest = X[:, ~self.mask_columns_]
            X_out = self.combine_trans_rest(X_transform, X_rest)

        return X_out

    def fit_transform(self, X, y=None, X_names=None, X_types=None, **fit_params):
        self.fit(X, y=y, X_names=X_names, X_types=X_types, **fit_params)
        return self.transform(X)

    def get_params(self, deep=True):
        params = dict(
            transformer=self.transformer,
            apply_to_column_names=self.apply_to_column_names,
            apply_to_column_types=self.apply_to_column_types,
            returned_features=self.returned_features,
        )

        transformer_params = self.transformer.get_params(deep=deep)
        for param, val in transformer_params.items():
            params[param] = val
        return deepcopy(params) if deep else params

    def set_params(self, **params):
        for param in ['transformer',
                      'apply_to_column_names',
                      'apply_to_column_types',
                      'returned_features']:
            if params.get(param) is not None:
                setattr(self, param, params.pop(param))
        self.transformer.set_params(**params)
        return self


def wrap_transformer(trans,
                     apply_to_column_names=None,
                     apply_to_column_types=None,
                     returned_features=None,
                     ):

    return JuTransformerWrapper(trans,
                                apply_to_column_names,
                                apply_to_column_types,
                                returned_features)


def check_transformer(trans):
    out = (trans if isinstance(trans, JuTransformerLike)
           else wrap_transformer(trans))
    return out
