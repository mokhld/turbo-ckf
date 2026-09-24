"""The Rust backend owns filter state (REVIEW B1, A3, A7).

Covers lazy attribute reads, the in-place edit rule for x/P/Q/R, that
reading attributes never changes results or re-factors TurboSRCKF's P, and
the copy / to_dict / from_dict / pickle / reset surface with its counters.
"""

import copy
import inspect
import pickle
import unittest

import numpy as np

from turbo_ckf import TurboCKF, TurboSRCKF

CLASSES = (TurboCKF, TurboSRCKF)
ZS = np.random.default_rng(3).normal(size=(40, 2))


def fx(s, dt):
    f = np.eye(4)
    f[0, 2] = dt
    f[1, 3] = dt
    return s @ f.T


def hx(s):
    return s[:, :2]


def make(cls):
    kf = cls(4, 2, 0.1, hx=hx, fx=fx)
    kf.x = [0.0, 0.0, 1.0, 0.5]
    kf.P = np.diag([1.0, 1.0, 0.5, 0.5])
    kf.Q = 1e-3 * np.eye(4)
    kf.R = 0.1 * np.eye(2)
    return kf


def make_with_jitter(cls):
    """A filter whose counters are non-zero: a rank-1 P needs jitter to
    factor (on the first predict for TurboCKF, on assignment for SR)."""

    kf = make(cls)
    kf.P = np.ones((4, 4))
    steps(kf, ZS[:3])
    assert kf.jitter_count >= 1 and kf.max_jitter > 0.0
    return kf


def steps(kf, zs=ZS):
    for z in zs:
        kf.predict()
        kf.update(z)


def ill_conditioned_srckf():
    rng = np.random.default_rng(5)
    u = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    p0 = u @ np.diag([1e6, 1e6, 1e-12, 1e-12]) @ u.T
    kf = TurboSRCKF(4, 2, 0.001, hx=hx, fx=fx)
    kf.x = np.zeros(4)
    kf.P = 0.5 * (p0 + p0.T)
    kf.Q = 1e-14 * np.eye(4)
    kf.R = 1e-10 * np.eye(2)
    return kf


class InPlaceEditTests(unittest.TestCase):
    def test_in_place_edits_take_effect_on_next_call(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                edited, assigned = make(cls), make(cls)
                edited.x[0] = 5.0
                edited.P[2:, 2:] *= 1000.0
                edited.Q[0, 0] = 0.5
                edited.R[1, 1] = 2.0

                for name, edit in (
                    ("x", lambda a: a.__setitem__(0, 5.0)),
                    ("P", lambda a: a[2:, 2:].__imul__(1000.0)),
                    ("Q", lambda a: a.__setitem__((0, 0), 0.5)),
                    ("R", lambda a: a.__setitem__((1, 1), 2.0)),
                ):
                    value = getattr(assigned, name).copy()
                    edit(value)
                    setattr(assigned, name, value)

                steps(edited, ZS[:5])
                steps(assigned, ZS[:5])
                np.testing.assert_array_equal(edited.x, assigned.x)
                np.testing.assert_array_equal(edited.P, assigned.P)

    def test_filterpy_scale_idiom_assigns_back_the_live_array(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                kf, ref = make(cls), make(cls)
                live = kf.P
                kf.P *= 1000.0  # getter, in-place multiply, setter with the same array
                self.assertIs(kf.P, live)
                self.assertTrue(live.flags.writeable)
                ref.P = ref.P * 1000.0
                kf.predict()
                ref.predict()
                np.testing.assert_array_equal(kf.P, ref.P)

    def test_arrays_replaced_by_a_step_become_read_only(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                kf = make(cls)
                x, P = kf.x, kf.P
                returned = kf.predict()
                self.assertIs(returned, kf.x)
                with self.assertRaises(ValueError):
                    x[0] = 1.0
                with self.assertRaises(ValueError):
                    P[0, 0] = 1.0
                # The current arrays are writable.
                kf.x[0] = 1.0
                kf.P[0, 0] = 2.0

    def test_arrays_replaced_by_assignment_become_read_only(self):
        kf = make(TurboCKF)
        q = kf.Q
        kf.Q = 2e-3 * np.eye(4)
        with self.assertRaises(ValueError):
            q[0, 0] = 1.0

    def test_q_and_r_stay_live_across_steps(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                kf, ref = make(cls), make(cls)
                q = kf.Q
                steps(kf, ZS[:2])
                steps(ref, ZS[:2])
                q[0, 0] = 0.5  # predict/update do not replace Q
                ref.Q = np.diag([0.5, 1e-3, 1e-3, 1e-3])
                steps(kf, ZS[2:4])
                steps(ref, ZS[2:4])
                np.testing.assert_array_equal(kf.x, ref.x)

    def test_paper_ahrs_update_replaces_r(self):
        kf = TurboCKF(4, 6, 0.01, hx=lambda s: s[:, :6], fx=lambda s, dt: s)
        kf.x = [1.0, 0.0, 0.0, 0.0]
        r = kf.R
        kf.update_paper_ahrs(np.array([0.0, 0.0, 1.0, 0.8, 0.0, 0.6]), 1e-2, 2e-2)
        with self.assertRaises(ValueError):
            r[0, 0] = 1.0
        np.testing.assert_array_equal(np.diag(kf.R), [1e-2] * 3 + [2e-2] * 3)

    def test_assignment_copies_the_callers_array(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                kf = make(cls)
                x0 = np.array([1.0, 2.0, 3.0, 4.0])
                kf.x = x0
                x0[0] = 99.0
                self.assertEqual(kf.x[0], 1.0)
                kf.predict()
                x0[0] = 5.0  # the caller's array is never frozen

    def test_non_finite_in_place_edit_raises_on_next_call(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                kf = make(cls)
                kf.P[0, 0] = np.nan
                with self.assertRaisesRegex(ValueError, "P must contain only finite"):
                    kf.predict()

    def test_pending_edit_is_included_in_copy_and_to_dict(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                kf = make(cls)
                kf.x[0] = 7.0
                self.assertEqual(kf.copy().x[0], 7.0)
                self.assertEqual(kf.to_dict()["x"][0], 7.0)


class ReadsDoNotChangeResultsTests(unittest.TestCase):
    NAMES = (
        "x", "P", "Q", "R", "K", "y", "z", "S", "x_prior", "P_prior", "x_post",
        "P_post", "z_pred", "log_likelihood", "likelihood", "mahalanobis",
        "nis", "last_jitter", "max_jitter", "jitter_count",
        "singular_innovation_count",
    )

    def test_reading_every_attribute_leaves_results_bit_identical(self):
        extra = {TurboCKF: ("SI",), TurboSRCKF: ("chol_P", "chol_Q", "chol_R", "S_innov",
                                                 "downdate_fallback_count")}
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                reader, quiet = make(cls), make(cls)
                for z in ZS:
                    for kf in (reader, quiet):
                        kf.predict()
                    for name in self.NAMES + extra[cls]:
                        getattr(reader, name)
                    for kf in (reader, quiet):
                        kf.update(z)
                    for name in self.NAMES + extra[cls]:
                        getattr(reader, name)
                np.testing.assert_array_equal(reader.x, quiet.x)
                np.testing.assert_array_equal(reader.P, quiet.P)
                self.assertEqual(reader.jitter_count, quiet.jitter_count)
                self.assertEqual(reader.max_jitter, quiet.max_jitter)

    def test_srckf_reading_p_every_step_never_refactors(self):
        kf = ill_conditioned_srckf()
        kf.reset_jitter_counters()
        for _ in range(500):
            kf.predict()
            kf.P
            kf.update(np.zeros(2))
            kf.P
        self.assertEqual(kf.jitter_count, 0)
        self.assertEqual(kf.max_jitter, 0.0)

    def test_outputs_are_cached_until_the_next_call(self):
        kf = make(TurboCKF)
        kf.predict()
        kf.update(ZS[0])
        self.assertIs(kf.K, kf.K)
        kf.nis = 3.0  # Python-side value, replaced by the next call
        self.assertEqual(kf.nis, 3.0)
        kf.predict()
        kf.update(ZS[1])
        self.assertNotEqual(kf.nis, 3.0)

    def test_descriptors_are_reachable_on_the_class(self):
        # help(), inspect and doc tools read attributes off the class.
        for cls in CLASSES:
            names = dict(inspect.getmembers(cls))
            self.assertIn("K", names)
        self.assertIn("chol_P", dict(inspect.getmembers(TurboSRCKF)))


class SquareRootFactorTests(unittest.TestCase):
    def test_assigning_p_factors_once_and_counts_seed_jitter(self):
        kf = make(TurboSRCKF)
        kf.P = np.ones((4, 4))  # rank 1: needs jitter to factor
        self.assertEqual(kf.jitter_count, 1)
        self.assertGreater(kf.max_jitter, 0.0)
        self.assertGreater(kf.last_jitter, 0.0)
        for _ in range(3):
            kf.P
        kf.predict()
        self.assertEqual(kf.jitter_count, 1)
        self.assertEqual(kf.last_jitter, 0.0)  # the QR predict adds none

    def test_in_place_p_edit_is_factored_once_at_next_call(self):
        kf = make(TurboSRCKF)
        kf.P[:] = np.ones((4, 4))
        self.assertEqual(kf.jitter_count, 0)  # not factored until the next call
        kf.predict()
        self.assertEqual(kf.jitter_count, 1)
        kf.predict()
        self.assertEqual(kf.jitter_count, 1)

    def test_per_call_r_jitter_counts(self):
        kf = make(TurboSRCKF)
        kf.predict()
        kf.update(ZS[0], R=np.zeros((2, 2)))
        self.assertEqual(kf.jitter_count, 1)
        self.assertGreater(kf.last_jitter, 0.0)

    def test_factors_are_read_only(self):
        kf = make(TurboSRCKF)
        with self.assertRaisesRegex(AttributeError, "assign P"):
            kf.chol_P = np.eye(4)
        with self.assertRaises(ValueError):
            kf.chol_P[0, 0] = 2.0


class CopySerializationTests(unittest.TestCase):
    def test_copy_keeps_counters_through_a_step(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                kf = make_with_jitter(cls)
                twin = kf.copy()
                self.assertEqual(twin.jitter_count, kf.jitter_count)
                for f in (kf, twin):
                    steps(f, ZS[3:5])
                self.assertEqual(twin.jitter_count, kf.jitter_count)
                self.assertEqual(twin.max_jitter, kf.max_jitter)
                np.testing.assert_array_equal(twin.x, kf.x)

    def test_from_dict_keeps_counters_through_a_step(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                kf = make_with_jitter(cls)
                restored = cls.from_dict(kf.to_dict(), hx=hx, fx=fx)
                for f in (kf, restored):
                    steps(f, ZS[3:5])
                self.assertEqual(restored.jitter_count, kf.jitter_count)
                self.assertEqual(restored.max_jitter, kf.max_jitter)
                np.testing.assert_array_equal(restored.x, kf.x)
                np.testing.assert_array_equal(restored.P, kf.P)

    def test_pickle_round_trip_gives_identical_results(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                kf = make_with_jitter(cls)
                restored = pickle.loads(pickle.dumps(kf))
                self.assertIs(restored.fx, fx)
                for f in (kf, restored):
                    steps(f, ZS[3:10])
                np.testing.assert_array_equal(restored.x, kf.x)
                np.testing.assert_array_equal(restored.P, kf.P)
                self.assertEqual(restored.log_likelihood, kf.log_likelihood)
                self.assertEqual(restored.jitter_count, kf.jitter_count)

    def test_pickle_carries_adaptive_estimator(self):
        kf = make(TurboCKF)
        kf.enable_adaptive_noise(window=2, mode="R", alpha=0.2)
        steps(kf, ZS[:5])
        restored = pickle.loads(pickle.dumps(kf))
        for f in (kf, restored):
            steps(f, ZS[5:10])
        np.testing.assert_array_equal(restored.R, kf.R)
        np.testing.assert_array_equal(restored.Q, kf.Q)
        np.testing.assert_array_equal(restored.x, kf.x)

    def test_pickling_a_lambda_callback_fails_clearly(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                kf = cls(4, 2, 0.1, hx=lambda s: s[:, :2], fx=fx)
                with self.assertRaises((pickle.PicklingError, AttributeError)) as ctx:
                    pickle.dumps(kf)
                self.assertIn("lambda", str(ctx.exception))

    def test_srckf_copy_is_independent(self):
        kf = make(TurboSRCKF)
        steps(kf, ZS[:3])
        x, chol = kf.x.copy(), kf.chol_P.copy()
        twin = copy.deepcopy(kf)
        twin.x[0] = 99.0
        steps(twin, ZS[3:6])
        np.testing.assert_array_equal(kf.x, x)
        np.testing.assert_array_equal(kf.chol_P, chol)
        self.assertFalse(np.array_equal(twin.x, x))

    def test_srckf_restore_uses_the_stored_factor(self):
        kf = ill_conditioned_srckf()
        kf.reset_jitter_counters()
        state = kf.to_dict()
        self.assertIn("chol_P", state)
        restored = TurboSRCKF.from_dict(state, hx=hx, fx=fx)
        # Re-factoring this P would need jitter; the restore must not.
        self.assertEqual(restored.jitter_count, 0)
        self.assertEqual(restored.max_jitter, 0.0)
        np.testing.assert_array_equal(restored.chol_P, kf.chol_P)

    def test_from_dict_rejects_a_dict_for_the_other_class(self):
        with self.assertRaisesRegex(ValueError, "chol_P"):
            TurboSRCKF.from_dict(make(TurboCKF).to_dict(), hx=hx, fx=fx)

    def test_from_dict_loads_an_older_partial_dict(self):
        state = {
            "version": 1, "dim_x": 4, "dim_z": 2, "dt": 0.1,
            "x": [[1.0], [2.0], [3.0], [4.0]],
            "P": np.eye(4).tolist(), "Q": np.eye(4), "R": np.eye(2),
            "jitter_count": 3, "max_jitter": 1e-9,
        }
        kf = TurboCKF.from_dict(state, hx=hx, fx=fx)
        np.testing.assert_array_equal(kf.x, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(kf.jitter_count, 3)
        self.assertEqual(kf.max_jitter, 1e-9)

    def test_reset_zeroes_counters_and_keeps_q_r(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                kf = make_with_jitter(cls)
                q, r = kf.Q.copy(), kf.R.copy()
                kf.reset()
                self.assertEqual(kf.jitter_count, 0)
                self.assertEqual(kf.max_jitter, 0.0)
                np.testing.assert_array_equal(kf.Q, q)
                np.testing.assert_array_equal(kf.R, r)


class SkippedUpdateTests(unittest.TestCase):
    def test_skipped_posterior_persists_after_the_next_predict(self):
        for cls in CLASSES:
            with self.subTest(cls=cls.__name__):
                kf = make(cls)
                steps(kf, ZS[:2])
                kf.predict()
                kf.update(None)
                prior = kf.x_prior.copy()
                np.testing.assert_array_equal(kf.x_post, prior)
                kf.predict()
                np.testing.assert_array_equal(kf.x_post, prior)
                self.assertTrue(np.all(np.isnan(kf.z)))


class AdaptiveNoiseReachesBackendTests(unittest.TestCase):
    def test_adaptive_r_and_q_are_the_backend_values(self):
        kf = make(TurboCKF)
        kf.enable_adaptive_noise(window=2, mode="both", alpha=0.5)
        steps(kf, ZS[:5])
        self.assertFalse(np.allclose(kf.R, 0.1 * np.eye(2)))
        np.testing.assert_array_equal(kf._rust_backend.get("R"), kf.R)
        np.testing.assert_array_equal(kf._rust_backend.get("Q"), kf.Q)


if __name__ == "__main__":
    unittest.main()
