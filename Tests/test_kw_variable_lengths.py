import unittest

import numpy as np

from fastDSA.kwdsa import (
    KernelDMD,
    _KOOPLEARN_V2,
    compute_wasserstein_distance,
)
from fastDSA.simdist import FastDSASimilarity, SimDistConfig


def _trajectory(length, phase=0.0):
    time = np.linspace(0.0, 5.0, length)
    return np.column_stack(
        (
            np.sin(time + phase),
            np.cos(0.6 * time - phase),
        )
    ).astype(np.float32)


class KwVariableLengthTests(unittest.TestCase):
    def test_kernel_dmd_fits_variable_length_trajectories(self):
        trajectories = [_trajectory(47), _trajectory(71, phase=0.2)]
        model = KernelDMD(
            trajectories,
            n_delays=3,
            delay_interval=2,
            rank=2,
            n_centers=20,
            eigen_solver="full",
        )

        model.fit()

        if _KOOPLEARN_V2:
            expected_rows = (47 - 4) + (71 - 4)
            self.assertEqual(model.data.shape, (expected_rows, 6))
        else:
            expected_contexts = (47 - 6) + (71 - 6)
            self.assertEqual(model.data.shape, (expected_contexts, 4, 2))
        self.assertEqual(model.A_v.shape, (2, 2))

    def test_kooplearn_v2_embedding_preserves_trial_boundaries(self):
        trajectories = [_trajectory(14), _trajectory(19, phase=0.3)]
        model = KernelDMD(
            trajectories,
            n_delays=3,
            delay_interval=2,
            rank=2,
            n_centers=10,
            eigen_solver="full",
        )

        training_data = model._build_delay_embedded_training_data(trajectories)

        first_rows = 14 - (3 - 1) * 2
        expected_rows = first_rows + (19 - (3 - 1) * 2)
        expected_pairs = (14 - 3 * 2) + (19 - 3 * 2)
        self.assertEqual(training_data.shape, (expected_rows, 6))
        self.assertEqual(len(model._pair_x_indices), expected_pairs)
        self.assertFalse(
            np.any(
                (model._pair_x_indices < first_rows)
                & (model._pair_y_indices >= first_rows)
            )
        )

    def test_kw_similarity_accepts_variable_length_datasets(self):
        dataset_a = [_trajectory(107).T, _trajectory(211, phase=0.2).T]
        dataset_b = [_trajectory(173, phase=0.5).T, _trajectory(1000, phase=0.7).T]

        similarity = FastDSASimilarity(
            SimDistConfig(
                n_delays=3,
                delay_interval=1,
                rank=2,
                method="kw",
                device="cpu",
            )
        )
        score, used_rank = similarity.fit_score(dataset_a, dataset_b)

        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0.0)
        self.assertEqual(used_rank, 2)

    def test_wasserstein_handles_real_and_complex_spectra_together(self):
        score = compute_wasserstein_distance(
            np.array([0.5, 0.8]),
            np.array([0.5 + 0.2j, 0.8 - 0.2j]),
        )

        self.assertTrue(np.isfinite(score))
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
