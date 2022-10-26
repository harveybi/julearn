from typing import runtime_checkable, Protocol


@runtime_checkable
class EstimatorLike(Protocol):
    def fit(self, X, y=None):
        pass

    def get_params(self, deep=True):
        pass

    def set_params(self, **set_params):
        pass


@runtime_checkable
class TransformerLike(EstimatorLike, Protocol):
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        pass


@runtime_checkable
class JuTransformerLike(TransformerLike, Protocol):

    # def fit(self, X, y=None, X_names=None, X_types=None):
    #     pass
    #
    # def fit_transform(self, X, y=None, X_names=None, X_types=None):
    #     pass

    def get_feature_names_out(self):
        pass

    def get_feature_types_out(self):
        pass
