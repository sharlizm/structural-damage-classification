"""Menyediakan preprocessing dan transformasi fitur untuk model klasik, CatBoost, dan LightGBM."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from scripts.feature_engineering import building_id_prefix

def build_classical_preprocessor(numeric_cols, categorical_cols):
    """Membangun preprocessing numerik dan kategorikal menggunakan imputasi, scaling, dan one-hot encoding."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

def prepare_lgb_categories(x_fit, x_eval, categorical_features):
    """Menyamakan kategori fitur antara data training dan evaluasi berdasarkan kategori training."""
    x_fit = x_fit.copy()
    x_eval = x_eval.copy()
    for col in categorical_features:
        fit_values = x_fit[col].astype("string").fillna("__MISSING__")
        eval_values = x_eval[col].astype("string").fillna("__MISSING__")
        categories = pd.Index(fit_values.unique())
        x_fit[col] = pd.Categorical(fit_values, categories=categories)
        x_eval[col] = pd.Categorical(eval_values, categories=categories)
    return x_fit, x_eval

class FinalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Mentransformasikan fitur final dengan prefix building_id sepanjang 8 digit dan kategori yang dipelajari saat fit."""

    def __init__(self, feature_cols, categorical_cols, id_col="building_id"):
        """Menyimpan daftar fitur, fitur kategorikal, dan nama kolom ID."""
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols
        self.id_col = id_col

    def fit(self, x, y=None):
        """Mempelajari kategori fitur dan prefix building_id dari data training."""
        self.categories_ = {}
        for col in self.categorical_cols:
            values = x[col].astype("string").fillna("__MISSING__")
            self.categories_[col] = pd.Index(values.unique())
        full_prefix8 = building_id_prefix(x[self.id_col], length=8)
        self.prefix8_full_categories_ = pd.Index(full_prefix8.unique())
        return self

    def transform(self, x):
        """Mentransformasikan fitur sesuai urutan semula dan menambahkan prefix building_id sebagai kategori."""
        x_full = x[self.feature_cols].copy()
        for col in self.categorical_cols:
            values = x_full[col].astype("string").fillna("__MISSING__")
            x_full[col] = pd.Categorical(values, categories=self.categories_[col])
        full_prefix8 = building_id_prefix(x[self.id_col], length=8)
        x_full["building_id_prefix_8"] = pd.Categorical(
            full_prefix8.to_numpy(), categories=self.prefix8_full_categories_)
        return x_full


class CatBoostNativeTransformer(BaseEstimator, TransformerMixin):
    """Mempersiapkan fitur kategorikal dalam format string untuk model CatBoost."""

    def __init__(self, categorical_cols):
        """Simpan nama fitur kategorikal CatBoost."""
        self.categorical_cols = categorical_cols

    def fit(self, x, y=None):
        """Mempertahankan transformer tanpa mempelajari parameter tambahan dari data."""
        return self

    def transform(self, x):
        """Konversi kategori ke string dan missing menjadi penanda asli."""
        result = x.copy()
        for col in self.categorical_cols:
            result[col] = result[col].astype("string").fillna("__MISSING__").astype(str)
        return result


class LightGBMNativeTransformer(BaseEstimator, TransformerMixin):
    """Pertahankan kategori native LightGBM berdasarkan data training saja."""

    def __init__(self, categorical_cols):
        """Simpan nama fitur kategorikal untuk pembentukan kategori native."""
        self.categorical_cols = categorical_cols

    def fit(self, x, y=None):
        """Pelajari urutan kategori dari training tanpa melihat validation."""
        self.categories_ = {}
        for col in self.categorical_cols:
            values = x[col]
            if isinstance(values.dtype, pd.CategoricalDtype):
                self.categories_[col] = pd.Index(values.cat.categories)
            else:
                self.categories_[col] = pd.Index(
                    values.astype("string").fillna("__MISSING__").unique()
                )
        return self

    def transform(self, x):
        """Konversi kategori tanpa mengubah unseen category menjadi kelas baru."""
        result = x.copy()
        for col in self.categorical_cols:
            values = result[col]
            if not isinstance(values.dtype, pd.CategoricalDtype):
                values = values.astype("string").fillna("__MISSING__")
            result[col] = pd.Categorical(values, categories=self.categories_[col])
        return result


def prepare_catboost_features(x_train, x_valid, categorical_cols):
    """Konversi kategori train/validation untuk CatBoost sesuai notebook awal."""
    x_train_cb = x_train.copy()
    x_valid_cb = x_valid.copy()
    for col in categorical_cols:
        x_train_cb[col] = x_train_cb[col].astype("string").fillna("__MISSING__").astype(str)
        x_valid_cb[col] = x_valid_cb[col].astype("string").fillna("__MISSING__").astype(str)
    return x_train_cb, x_valid_cb


def validate_competition_data(train, test, sample_submission, target="damage_grade", id_col="building_id"):
    """Periksa kolom dasar, missing target, duplikat ID, dan kecocokan sample submission."""
    required = {target, id_col}
    if not required.issubset(train.columns):
        raise ValueError(f"Kolom train tidak lengkap: {sorted(required - set(train.columns))}")
    if id_col not in test.columns:
        raise ValueError(f"Kolom {id_col} tidak ada pada test.csv")
    if train[target].isna().any():
        raise ValueError(f"Ada nilai kosong pada {target}")
    if train[id_col].duplicated().any() or test[id_col].duplicated().any():
        raise ValueError(f"Ada {id_col} duplikat pada train atau test")
    if sample_submission.columns.tolist() != [id_col, target]:
        raise ValueError("Format kolom sample_submission.csv tidak sesuai")
    if len(test) != len(sample_submission):
        raise ValueError("Jumlah baris test dan sample_submission berbeda")
    if not sample_submission[id_col].reset_index(drop=True).equals(test[id_col].reset_index(drop=True)):
        raise ValueError("Urutan building_id sample_submission berbeda dengan test")
