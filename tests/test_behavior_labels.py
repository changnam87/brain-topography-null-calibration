from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bt.io as btio


def test_triad_labels_define_ccc_as_all_three_cooperate(monkeypatch):
    choices = {
        1: np.array(["C", "C", "D", "C"], dtype=object),
        2: np.array(["C", "D", "D", "C"], dtype=object),
        3: np.array(["C", "C", "C", "D"], dtype=object),
    }

    def fake_subject_choice_vector(dataset_root, group, participant):
        assert group == 1
        return choices[participant]

    monkeypatch.setattr(
        btio,
        "subject_choice_vector",
        fake_subject_choice_vector,
    )

    labels, matrix = btio.triad_labels("unused", 1)

    expected = np.array([1, 0, 0, 0], dtype=np.uint8)

    assert np.array_equal(labels, expected)
    assert matrix.shape == (4, 3)
    assert matrix[0].tolist() == ["C", "C", "C"]


def test_ccc_label_is_binary_uint8(monkeypatch):
    choices = {
        1: np.array(["D", "C"], dtype=object),
        2: np.array(["D", "C"], dtype=object),
        3: np.array(["D", "C"], dtype=object),
    }

    monkeypatch.setattr(
        btio,
        "subject_choice_vector",
        lambda dataset_root, group, participant: choices[participant],
    )

    labels, _ = btio.triad_labels("unused", 1)

    assert labels.dtype == np.uint8
    assert labels.tolist() == [0, 1]
