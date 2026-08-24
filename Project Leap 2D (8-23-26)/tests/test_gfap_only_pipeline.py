from __future__ import annotations

import inspect
import unittest
import weakref
from types import SimpleNamespace

import numpy as np

from project_leap_2d.analysis_modes.gfap_only.gfap_only_analysis import (
    GFAPOnlyResult,
)
from project_leap_2d.analysis_modes.gfap_only.gfap_only_pipeline import (
    GFAPZSelectionConfig,
    run_gfap_only_pipeline,
    select_gfap_active_z,
    validate_gfap_only_route,
)


def _fake_analysis(labels_zyx, gfap_zyx, pixel_size_um, z_spacing_um, **kwargs):
    shape = labels_zyx.shape[1:]
    whole = np.zeros(shape, dtype=np.int32)
    soma = np.zeros(shape, dtype=np.int32)
    processes = np.zeros(shape, dtype=np.int32)
    soma[2:4, 2:4] = 1
    processes[1, 2:4] = 1
    whole[soma > 0] = 1
    whole[processes > 0] = 1
    zeros = np.zeros(shape, dtype=np.float32)
    return GFAPOnlyResult(
        whole_labels=whole,
        soma_labels=soma,
        process_labels=processes,
        nucleus_labels_2d=soma.copy(),
        valid_nucleus_labels_2d=soma.copy(),
        corrected_gfap_projection=zeros,
        gfap_intensity=zeros,
        gfap_ridge_score=zeros,
        gfap_structural_score=zeros,
        gfap_structural_mask=whole > 0,
        diagnostics={"age_profile": "mature"},
    )


class GFAPOnlyPipelineTests(unittest.TestCase):
    def setUp(self):
        self.dapi = np.zeros((9, 8, 8), dtype=np.uint8)
        self.gfap = np.zeros_like(self.dapi)
        self.gfap[3:6, 2:6, 2:6] = 200
        self.detector_calls = []

        def detector(stack, pixel_height_um, pixel_width_um, *, z_indices):
            self.detector_calls.append(
                (stack.shape, pixel_height_um, pixel_width_um, tuple(z_indices))
            )
            labels = np.zeros((len(z_indices), 8, 8), dtype=np.int32)
            labels[:, 2:4, 2:4] = 1
            return SimpleNamespace(
                labels_zyx=labels,
                instance_counts=tuple(1 for _ in z_indices),
                model_sha256="pinned",
            )

        self.detector = detector

    def test_route_requires_dapi_gfap_and_rejects_valid_egfp(self):
        validate_gfap_only_route(
            {"DAPI", "GFAP", "KCNN2"},
            egfp_is_valid=False,
        )
        with self.assertRaisesRegex(ValueError, "requires DAPI and GFAP"):
            validate_gfap_only_route({"DAPI", "KCNN2"}, egfp_is_valid=False)
        with self.assertRaisesRegex(ValueError, "valid eGFP"):
            validate_gfap_only_route(
                {"DAPI", "eGFP", "GFAP"},
                egfp_is_valid=True,
            )

    def test_active_z_is_contiguous_and_uses_physical_padding(self):
        selection = select_gfap_active_z(
            self.gfap,
            0.5,
            config=GFAPZSelectionConfig(
                smoothing_sigma_um=0.0,
                padding_um=0.5,
                minimum_span_um=1.0,
            ),
        )
        self.assertEqual(selection.indices, tuple(range(2, 7)))
        self.assertEqual(selection.start_1based, 3)
        self.assertEqual(selection.end_1based_inclusive, 7)

    def test_flat_activity_keeps_complete_stack(self):
        selection = select_gfap_active_z(
            np.ones((5, 4, 4), dtype=np.uint8),
            1.0,
        )
        self.assertEqual(selection.indices, (0, 1, 2, 3, 4))

    def test_skip_fiji_calls_debug_only_and_preserves_partition(self):
        debug_calls = []
        fiji_calls = []
        stage_events = []
        result = run_gfap_only_pipeline(
            available_channels={"DAPI", "GFAP", "KCNN2"},
            egfp_is_valid=False,
            dapi_stack=self.dapi,
            gfap_stack=self.gfap,
            pixel_height_um=0.4,
            pixel_width_um=0.5,
            z_spacing_um=0.5,
            skip_fiji=True,
            debug=True,
            z_config=GFAPZSelectionConfig(
                smoothing_sigma_um=0.0,
                padding_um=0.0,
                minimum_span_um=1.0,
            ),
            nucleus_detector=self.detector,
            analyzer=_fake_analysis,
            measurement_projection_loader=lambda *_: np.full(
                (8, 8), 99, dtype=np.uint16
            ),
            debug_handler=lambda prepared: debug_calls.append(prepared) or "debug",
            fiji_handler=lambda prepared: fiji_calls.append(prepared),
            stage_reporter=lambda stage, status, elapsed: stage_events.append(
                (stage, status, elapsed)
            ),
        )
        self.assertTrue(result.skip_fiji)
        self.assertEqual(result.debug_result, "debug")
        self.assertEqual(len(debug_calls), 1)
        self.assertEqual(fiji_calls, [])
        prepared = result.prepared
        recombined = np.where(
            prepared.analysis.soma_labels > 0,
            prepared.analysis.soma_labels,
            prepared.analysis.process_labels,
        )
        np.testing.assert_array_equal(recombined, prepared.analysis.whole_labels)
        self.assertEqual(prepared.measurement_projection[0, 0], 99)
        self.assertEqual(prepared.best_row["analysis_mode"], "dapi_gfap_only")
        self.assertEqual(
            set(prepared.stage_timings_seconds),
            {
                "z_selection",
                "dapi_nucleus_model",
                "gfap_compartments",
                "measurement_preparation",
            },
        )
        self.assertTrue(
            all(value >= 0.0 for value in prepared.stage_timings_seconds.values())
        )
        self.assertEqual(
            [(stage, status) for stage, status, _elapsed in stage_events],
            [
                ("z_selection", "started"),
                ("z_selection", "completed"),
                ("dapi_nucleus_model", "started"),
                ("dapi_nucleus_model", "completed"),
                ("gfap_compartments", "started"),
                ("gfap_compartments", "completed"),
                ("measurement_preparation", "started"),
                ("measurement_preparation", "completed"),
            ],
        )

    def test_three_dimensional_inputs_are_released_before_handler(self):
        references = {}

        def make_stack(name, source):
            value = source.copy()
            references[name] = weakref.ref(value)
            return value

        def detector(stack, pixel_height_um, pixel_width_um, *, z_indices):
            class DetectorToken:
                pass

            labels = np.zeros((len(z_indices), 8, 8), dtype=np.int32)
            labels[:, 2:4, 2:4] = 1
            token = DetectorToken()
            references["labels"] = weakref.ref(labels)
            references["detector_token"] = weakref.ref(token)
            return SimpleNamespace(
                labels_zyx=labels,
                retained_only_by_detector_result=token,
                instance_counts=tuple(1 for _ in z_indices),
                model_sha256="pinned",
            )

        observed = []

        def debug_handler(prepared):
            observed.append(
                {
                    name: reference() is None
                    for name, reference in references.items()
                }
            )
            # CPython may retain temporary call arguments in the caller's value
            # stack until run_gfap_only_pipeline returns.  Intermediates created
            # inside the pipeline must already be gone before this handler.
            self.assertTrue(references["labels"]() is None)
            self.assertTrue(references["detector_token"]() is None)
            self.assertEqual(prepared.dapi_projection.ndim, 2)
            self.assertEqual(prepared.gfap_projection.ndim, 2)
            return "released"

        result = run_gfap_only_pipeline(
            available_channels={"DAPI", "GFAP"},
            egfp_is_valid=False,
            dapi_stack=make_stack("dapi", self.dapi),
            gfap_stack=make_stack("gfap", self.gfap),
            pixel_height_um=0.5,
            pixel_width_um=0.5,
            z_spacing_um=0.5,
            skip_fiji=True,
            nucleus_detector=detector,
            analyzer=_fake_analysis,
            debug_handler=debug_handler,
        )
        self.assertEqual(result.debug_result, "released")
        self.assertTrue(observed[0]["labels"])
        self.assertTrue(observed[0]["detector_token"])
        self.assertIsNone(references["dapi"]())
        self.assertIsNone(references["gfap"]())

    def test_measurement_is_loaded_after_roi_analysis_and_never_passed_to_it(self):
        order = []

        def analyzer(*args, **kwargs):
            order.append("analyze")
            self.assertEqual(len(args), 4)
            self.assertEqual(kwargs["config"].structure.projection_percentile, 95.0)
            self.assertEqual(kwargs["config"].structure.intensity_floor_percentile, 64.0)
            self.assertEqual(kwargs["config"].structure.structural_percentile, 84.0)
            self.assertEqual(kwargs["config"].structure.strong_ridge_percentile, 94.0)
            self.assertEqual(kwargs["config"].structure.connection_gap_um, 0.18)
            self.assertEqual(
                kwargs["config"].nucleus_ownership.min_shell_enrichment,
                4.5,
            )
            return _fake_analysis(*args, **kwargs)

        def measurement_loader(*_args):
            order.append("measurement")
            return np.zeros((8, 8), dtype=np.uint16)

        run_gfap_only_pipeline(
            available_channels={"DAPI", "GFAP", "KCNN1"},
            egfp_is_valid=False,
            dapi_stack=self.dapi,
            gfap_stack=self.gfap,
            pixel_height_um=0.5,
            pixel_width_um=0.5,
            z_spacing_um=0.5,
            skip_fiji=True,
            nucleus_detector=self.detector,
            analyzer=analyzer,
            measurement_projection_loader=measurement_loader,
        )
        self.assertEqual(order, ["analyze", "measurement"])

    def test_pipeline_exposes_only_frozen_mature_configuration(self):
        self.assertNotIn(
            "age_profile",
            inspect.signature(run_gfap_only_pipeline).parameters,
        )
        observed = {}

        def analyzer(*args, **kwargs):
            observed["config"] = kwargs["config"]
            return _fake_analysis(*args, **kwargs)

        run_gfap_only_pipeline(
            available_channels={"DAPI", "GFAP"},
            egfp_is_valid=False,
            dapi_stack=self.dapi,
            gfap_stack=self.gfap,
            pixel_height_um=0.5,
            pixel_width_um=0.5,
            z_spacing_um=0.5,
            skip_fiji=True,
            nucleus_detector=self.detector,
            analyzer=analyzer,
        )
        config = observed["config"]
        self.assertFalse(hasattr(config, "age_profile"))
        self.assertEqual(config.structure.structural_percentile, 84.0)
        self.assertEqual(config.structure.projection_percentile, 95.0)
        self.assertEqual(config.compartments.soma_base_margin_um, 0.80)
        self.assertEqual(config.compartments.soma_max_margin_um, 1.25)
        self.assertEqual(
            config.compartments.ownership_seed_max_margin_um,
            2.10,
        )
        self.assertEqual(
            config.nucleus_ownership.min_shell_enrichment,
            4.5,
        )

    def test_gross_whole_coverage_is_blocked_before_publication(self):
        def flooded_analyzer(*args, **kwargs):
            result = _fake_analysis(*args, **kwargs)
            result.whole_labels[:, :] = 1
            result.soma_labels[:, :] = 0
            result.soma_labels[2:4, 2:4] = 1
            result.process_labels[:, :] = 1
            result.process_labels[2:4, 2:4] = 0
            return result

        published = []
        with self.assertRaisesRegex(RuntimeError, "implausibly large"):
            run_gfap_only_pipeline(
                available_channels={"DAPI", "GFAP"},
                egfp_is_valid=False,
                dapi_stack=self.dapi,
                gfap_stack=self.gfap,
                pixel_height_um=0.5,
                pixel_width_um=0.5,
                z_spacing_um=0.5,
                skip_fiji=False,
                nucleus_detector=self.detector,
                analyzer=flooded_analyzer,
                fiji_handler=lambda prepared: published.append(prepared),
            )
        self.assertEqual(published, [])

    def test_normal_execution_requires_and_calls_shared_fiji_handler(self):
        seen = []
        stage_events = []
        result = run_gfap_only_pipeline(
            available_channels={"DAPI", "GFAP"},
            egfp_is_valid=False,
            dapi_stack=self.dapi,
            gfap_stack=self.gfap,
            pixel_height_um=0.5,
            pixel_width_um=0.5,
            z_spacing_um=0.5,
            skip_fiji=False,
            nucleus_detector=self.detector,
            analyzer=_fake_analysis,
            fiji_handler=lambda prepared: seen.append(prepared) or {"ok": True},
            stage_reporter=lambda stage, status, elapsed: stage_events.append(
                (stage, status, elapsed)
            ),
        )
        self.assertFalse(result.skip_fiji)
        self.assertEqual(result.fiji_result, {"ok": True})
        self.assertEqual(len(seen), 1)
        self.assertIn(
            "fiji_review_and_publication",
            result.prepared.stage_timings_seconds,
        )
        self.assertEqual(
            [(stage, status) for stage, status, _elapsed in stage_events[-2:]],
            [
                ("fiji_review_and_publication", "started"),
                ("fiji_review_and_publication", "completed"),
            ],
        )
        with self.assertRaisesRegex(ValueError, "fiji_handler is required"):
            run_gfap_only_pipeline(
                available_channels={"DAPI", "GFAP"},
                egfp_is_valid=False,
                dapi_stack=self.dapi,
                gfap_stack=self.gfap,
                pixel_height_um=0.5,
                pixel_width_um=0.5,
                z_spacing_um=0.5,
                skip_fiji=False,
                nucleus_detector=self.detector,
                analyzer=_fake_analysis,
            )

    def test_malformed_partition_is_rejected_before_measurement(self):
        def bad_analyzer(*args, **kwargs):
            result = _fake_analysis(*args, **kwargs)
            result.process_labels[2, 2] = 1
            return result

        called = []
        with self.assertRaisesRegex(RuntimeError, "exactly Soma union Processes"):
            run_gfap_only_pipeline(
                available_channels={"DAPI", "GFAP"},
                egfp_is_valid=False,
                dapi_stack=self.dapi,
                gfap_stack=self.gfap,
                pixel_height_um=0.5,
                pixel_width_um=0.5,
                z_spacing_um=0.5,
                skip_fiji=True,
                nucleus_detector=self.detector,
                analyzer=bad_analyzer,
                measurement_projection_loader=lambda *_: called.append(True),
            )
        self.assertEqual(called, [])

    def test_route_guard_runs_before_detector(self):
        with self.assertRaisesRegex(ValueError, "valid eGFP"):
            run_gfap_only_pipeline(
                available_channels={"DAPI", "GFAP", "eGFP"},
                egfp_is_valid=True,
                dapi_stack=self.dapi,
                gfap_stack=self.gfap,
                pixel_height_um=0.5,
                pixel_width_um=0.5,
                z_spacing_um=0.5,
                skip_fiji=True,
                nucleus_detector=self.detector,
                analyzer=_fake_analysis,
            )
        self.assertEqual(self.detector_calls, [])


if __name__ == "__main__":
    unittest.main()
