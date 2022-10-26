import inspect
import numpy as np
from ..utils import raise_error
from ..utils.column_types import mask_columns


def juwrap_init(init, apply_to_column_names=None,
                apply_to_column_types='continuous',
                returned_features=None):

    def ju_init(self, *args, apply_to_column_names=apply_to_column_names,
                apply_to_column_types=apply_to_column_types,
                returned_features=returned_features, **kwargs):
        # todo use default arguments and raise error when kwargs has fals param
        init(self, *args, **kwargs)
        self.apply_to_column_names = apply_to_column_names
        self.apply_to_column_types = apply_to_column_types
        self.returned_features = returned_features

    original_signiture = inspect.signature(init)
    new_sig = [param for param in original_signiture.parameters.values()]
    new_sig.extend([
        inspect.Parameter(name='apply_to_column_names',
                          default=None,
                          kind=inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter(name='apply_to_column_types',
                          default='continuous',
                          kind=inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter(name='returned_features', default=None,
                          kind=inspect.Parameter.KEYWORD_ONLY),
    ])

    ju_init.__signature__ = inspect.Signature(parameters=new_sig)
    return ju_init


def juwrap_fit(fit, transform, **kwargs):

    y = kwargs.get('y')
    X_types = (kwargs.pop('X_types')
               if 'X_types' in kwargs
               else None
               )

    X_names = (kwargs.pop('X_names')
               if 'X_names' in kwargs
               else None
               )

    def ju_fit(self, X, y=y, X_names=X_names, X_types=X_types, **fit_params):
        if hasattr(self, '_validate_data'):
            X, y = self._validate_data(X, y)

        self.X_names_ = None if X_names is None else np.array(X_names)
        self.X_types_ = None if X_types is None else np.array(X_types)

        if ((self.X_names_ is None) or
                (self.X_types_ is None)):
            raise_error("no feature names")  # TODO: add message

        self._set_columns_to_transform(X)
        X_transform = X[:, self.mask_columns_]
        fit(self, X_transform, y, **fit_params)

        X_transform = transform(self, X_transform)
        self._set_features_out(X_transform)

        return self
    original_docs = "" if fit.__doc__ is None else fit.__doc__
    ju_fit.__doc__ = ('This fit method was wrapped using julearn.\n'
                      'It selects features in X following specifications '
                      'in the init and then applies the '
                      'fit method of the original class. \n'
                      'Therefore, the following reference to X always '
                      'refer to the selected features of X'
                      ) + original_docs
    return ju_fit


def juwrap_transform(transform):

    def ju_transform(self, X):

        if hasattr(self, '_validate_data'):
            X = self._validate_data(X)

        if self.mask_columns_ is None:
            X_out = self.transformer.transform(X)

        else:
            X_transform = X[:, self.mask_columns_]
            X_transform = transform(self, X_transform)
            X_rest = X[:, ~self.mask_columns_]
            X_out = self.combine_trans_rest(X_transform, X_rest)
        return X_out

    original_docs = "" if transform.__doc__ is None else transform.__doc__
    ju_transform.__doc__ = (
        'This transform method was wrapped using julearn.\n'
        'It selects features in X following specifications '
        'in the init and fit then applies the '
        'transform method of the original class. \n'
        'Therefore, the following reference to X always '
        'refer to the selected features of X'
    ) + original_docs
    return ju_transform


class JuTransformerMixin:
    def _set_columns_to_transform(self, X):
        n_cols = len(X[0])

        if self.apply_to_column_names is not None:
            names_mask = mask_columns(
                self.apply_to_column_names,
                self.X_names_)
        else:
            names_mask = np.ones(n_cols, dtype=bool)
        if self.apply_to_column_types is not None:
            types_mask = mask_columns(
                self.apply_to_column_types,
                self.X_types_)
        else:
            types_mask = np.ones(n_cols, dtype=bool)

        self.mask_columns_ = names_mask & types_mask

    def _set_features_out(self, X_transform):
        if self.returned_features == 'same':
            self.feature_names_out_ = self.X_names_
            self.feature_types_out_ = self.X_types_

        elif self.returned_features == 'subset':
            trans_names_in = self.X_names_[self.mask_columns_]
            trans_types_in = self.X_types_[self.mask_columns_]
            rest_names_in = self.X_names_[~self.mask_columns_]
            rest_types_in = self.X_types_[~self.mask_columns_]

            mask_subset = super().get_support(trans_names_in)
            trans_names_in = trans_names_in[mask_subset]
            trans_types_in = trans_types_in[mask_subset]
            self.feature_names_out_ = np.c_[trans_names_in, rest_names_in]
            self.feature_types_out_ = np.c_[trans_types_in, rest_types_in]

        elif self.returned_features == 'unknown':
            n_columns = X_transform.shape[1]
            trans_names_in = np.array([
                f'{super.__name__.lower()}{i}'
                for i in np.arange(n_columns)
            ])
            trans_types_in = np.array(['continuous'] * n_columns)
            rest_names_in = self.X_names_[~self.mask_columns_]
            rest_types_in = self.X_types_[~self.mask_columns_]
            if len(rest_names_in) == 0:
                self.feature_names_out_ = trans_names_in
                self.feature_types_out_ = trans_types_in
            else:
                self.feature_names_out_ = np.concatenate(
                    [trans_names_in, rest_names_in],
                    axis=None
                )
                self.feature_types_out_ = np.concatenate(
                    [trans_types_in, rest_types_in],
                    axis=None
                )

        else:
            raise_error(
                "No supported return_features = "
                f"{self.returned_features}")  # TODO improve

    def combine_trans_rest(self, X_transform, X_rest):
        X_out = np.c_[X_transform, X_rest]
        if self.returned_features == 'unknown':
            return X_out

        idx_all = np.arange(len(self.mask_columns_))
        idx_trans = idx_all[self.mask_columns_]
        idx_rest = idx_all[~self.mask_columns_]

        if self.returned_features == 'subset':
            idx_trans = super().get_support(idx_trans)

        # set index to the original order
        reindexer = np.concatenate([idx_trans, idx_rest])
        X_out = X_out[:, reindexer]
        return X_out

    def get_feature_names_out(self):
        return self.feature_names_out_

    def get_feature_types_out(self):
        return self.feature_types_out_
