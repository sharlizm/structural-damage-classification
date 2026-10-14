"""Training LightGBM Prefix8 final sesuai konfigurasi notebook asli."""

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier

from scripts.preprocess import (
    FinalFeatureTransformer, CatBoostNativeTransformer, LightGBMNativeTransformer,
)

def make_final_pipeline(feature_cols, categorical_cols, random_state=42, final_n_estimators=200):
    """Membangun pipeline Prefix8 dengan kategori native dan model LightGBM multiclass."""
    final_lgb_prefix8 = LGBMClassifier(
        objective="multiclass",
        n_estimators=final_n_estimators,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,)
    return Pipeline([
        ("features", FinalFeatureTransformer(feature_cols, categorical_cols)),
        ("model", final_lgb_prefix8),])

def train_final_lightgbm(train, target, feature_cols, categorical_cols,
                         grade_to_int, random_state=42, final_n_estimators=200):
    """Melatih pipeline LightGBM final menggunakan seluruh data berlabel dengan konfigurasi yang ditentukan."""
    y_full = train[target].map(grade_to_int).astype(int)
    final_categorical_cols = categorical_cols + ["building_id_prefix_8"]
    final_lgb_prefix8 = make_final_pipeline(
        feature_cols, categorical_cols, random_state, final_n_estimators)
    final_lgb_prefix8.fit(
        train,
        y_full,
        model__categorical_feature=final_categorical_cols,
    )
    return final_lgb_prefix8


def make_sklearn_pipeline(preprocessor, estimator):
    """Menggabungkan salinan preprocessing dan estimator ke dalam pipeline scikit-learn."""
    from sklearn.base import clone
    return Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])


def fit_and_evaluate_sklearn(estimator, x_train, y_train, x_valid, y_valid):
    """Melatih estimator dan mengevaluasi Macro F1, akurasi, serta durasi pelatihan pada data validasi."""
    import time
    from sklearn.metrics import accuracy_score, f1_score
    start = time.perf_counter()
    estimator.fit(x_train, y_train)
    pred = estimator.predict(x_valid)
    metrics = {
        "macro_f1": float(f1_score(y_valid, pred, average="macro")),
        "accuracy": float(accuracy_score(y_valid, pred)),
        "train_minutes": (time.perf_counter() - start) / 60,
    }
    return estimator, pred, metrics



class PipelineCatBoostClassifier(ClassifierMixin, CatBoostClassifier, BaseEstimator):
    """Menyediakan classifier CatBoost yang kompatibel dengan pipeline scikit-learn."""


def make_catboost_pipeline(params, categorical_cols):
    """Membangun pipeline CatBoost dengan preprocessing fitur kategorikal."""
    return Pipeline([
        ("preprocess", CatBoostNativeTransformer(categorical_cols)),
        ("model", PipelineCatBoostClassifier(**params)),
    ])


def make_lgb_experiment_pipeline(params, categorical_cols):
    """Membangun pipeline eksperimen LightGBM dengan preprocessing kategori native."""
    return Pipeline([
        ("preprocess", LightGBMNativeTransformer(categorical_cols)),
        ("model", LGBMClassifier(**params)),
    ])
