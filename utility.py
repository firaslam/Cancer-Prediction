from sklearn.base import BaseEstimator, TransformerMixin

class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, pandas_df):
        return pandas_df[self.key]
