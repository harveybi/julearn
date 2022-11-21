from sklearn.base import BaseEstimator, TransformerMixin


class JuTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, apply_to, needed_types=None):
        self.apply_to = apply_to
        self.needed_types = needed_types

    def get_needed_types(self):
        return self.needed_types

    def get_apply_to(self):
        return self.apply_to