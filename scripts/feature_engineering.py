"""Feature engineering helpers for building-ID prefixes and categorical interactions."""
from __future__ import annotations
from collections.abc import Sequence
import pandas as pd


def building_id_prefix(ids: pd.Series, length: int = 8, width: int = 12) -> pd.Series:
    """Menyeragamkan panjang building_id menjadi 12 digit, lalu mengambil sejumlah digit awal sebagai fitur prefix."""
    if not 1 <= length <= width:
        raise ValueError("prefix_length must be between 1 and width.")
    return ids.astype("string").str.zfill(width).str[:length]


def add_categorical_prefix_feature(
    frame: pd.DataFrame,
    ids: pd.Series,
    prefix_length: int = 8,
    categories: Sequence[str] | pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.Index]:
    """Menambahkan fitur kategorikal berupa prefix building_id dengan kategori yang dapat disamakan berdasarkan data training."""
    column = f"building_id_prefix_{prefix_length}"
    prefix = building_id_prefix(ids, length=prefix_length)
    learned_categories = pd.Index(prefix.unique()) if categories is None else pd.Index(categories)
    result = frame.copy()
    result[column] = pd.Categorical(prefix.to_numpy(), categories=learned_categories)
    return result, learned_categories


def combine_categorical_features(
    frame: pd.DataFrame,
    left_col: str,
    right_col: str,
    separator: str = "__",
) -> pd.Series:
    """Menggabungkan dua kolom kategorikal menjadi satu string key dengan penanda khusus untuk nilai yang hilang."""
    left = frame[left_col].astype("string").fillna("__MISSING__")
    right = frame[right_col].astype("string").fillna("__MISSING__")
    return left + separator + right


def audit_prefix_groups(
    train_ids: pd.Series,
    train_target: pd.Series,
    valid_ids: pd.Series,
    test_ids: pd.Series,
    label_order: Sequence[str],
    prefix_lengths: Sequence[int] = (2, 4, 6, 8, 10),
) -> pd.DataFrame:
    """Mengevaluasi ukuran grup prefix building_id, variasi proporsi kelas, serta cakupan prefix pada data validasi dan uji."""
    rows = []
    train_ids = train_ids.reset_index(drop=True)
    train_target = train_target.reset_index(drop=True)
    valid_ids = valid_ids.reset_index(drop=True)
    test_ids = test_ids.reset_index(drop=True)
    for length in prefix_lengths:
        train_prefix = building_id_prefix(train_ids, length=length)
        valid_prefix = building_id_prefix(valid_ids, length=length)
        test_prefix = building_id_prefix(test_ids, length=length)
        group_size = train_prefix.value_counts()
        audit_data = pd.DataFrame({"prefix": train_prefix, "target": train_target})
        class_rates = pd.crosstab(audit_data["prefix"], audit_data["target"], normalize="index")
        class_rates = class_rates.reindex(columns=list(label_order), fill_value=0)
        known = group_size.index
        row = {
            "prefix_length": length,
            "num_groups": int(group_size.size),
            "median_group_size": float(group_size.median()),
            "share_groups_20plus": float((group_size >= 20).mean()),
            "validation_prefix_coverage": float(valid_prefix.isin(known).mean()),
            "test_prefix_coverage": float(test_prefix.isin(known).mean()),
        }
        for label in label_order:
            key = label.lower().replace(" ", "_") + "_rate_std"
            row[key] = float(class_rates[label].std())
        rows.append(row)
    return pd.DataFrame(rows)
