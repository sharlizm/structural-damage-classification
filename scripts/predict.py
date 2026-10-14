"""Menghasilkan file submission menggunakan model LightGBM dengan fitur prefix building_id sepanjang 8 digit."""

import numpy as np
import pandas as pd


def validate_submission(submission, test, sample_submission,
                        target="damage_grade", id_col="building_id"):
    """Memvalidasi struktur, jumlah baris, keunikan ID, nilai prediksi, dan urutan ID submission."""
    if submission.columns.tolist() != [id_col, target]:
        raise ValueError("Kolom submission tidak sesuai sample_submission.csv.")
    if len(submission) != len(test) or len(submission) != len(sample_submission):
        raise ValueError("Jumlah baris submission tidak cocok dengan data test.")
    if not submission[id_col].is_unique or submission[target].isna().any():
        raise ValueError("Submission memiliki ID duplikat atau prediksi kosong.")
    if not submission[target].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("damage_grade harus berupa kelas 1 sampai 5.")
    if not np.array_equal(submission[id_col].to_numpy(), sample_submission[id_col].to_numpy()):
        raise ValueError("Urutan building_id tidak sama dengan sample_submission.csv.")


def make_submission(final_lgb_prefix8, test, sample_submission,
                    target="damage_grade", id_col="building_id"):
    """Menghasilkan prediksi kelas damage_grade, menyusun submission sesuai urutan ID sampel, dan memvalidasi hasilnya."""
    test_pred_int = np.asarray(final_lgb_prefix8.predict(test)).astype(int)
    if not np.isin(test_pred_int, range(5)).all():
        raise ValueError("Prediksi model harus berada pada rentang kelas 0 sampai 4.")
    test_pred_grade = test_pred_int + 1

    predictions = pd.DataFrame({
        id_col: test[id_col].to_numpy(),
        target: test_pred_grade,
    })
    submission = sample_submission[[id_col]].merge(
        predictions, on=id_col, how="left", validate="one_to_one",
    )[[id_col, target]]

    validate_submission(submission, test, sample_submission, target, id_col)
    return submission
