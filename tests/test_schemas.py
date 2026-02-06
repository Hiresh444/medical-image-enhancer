"""Tests for Pydantic schemas in schemas.py."""

from __future__ import annotations

import json

import pytest

from pipeline.schemas import (
    EnhancementParams,
    EnhancementPlan,
    ExplainabilityReport,
    IterationRecord,
    PARAM_BOUNDS,
)


class TestEnhancementParams:
    def test_defaults(self):
        p = EnhancementParams()
        assert 0.005 <= p.clahe_clip_limit <= 0.05
        assert 8 <= p.clahe_tile_size <= 32
        assert p.denoise_mode in ("soft", "hard")

    def test_custom_values(self):
        p = EnhancementParams(clahe_clip_limit=0.03, gamma=0.85)
        assert p.clahe_clip_limit == 0.03
        assert p.gamma == 0.85


class TestEnhancementPlan:
    def test_valid_plan(self, sample_enhancement_plan_dict: dict):
        plan = EnhancementPlan(**sample_enhancement_plan_dict)
        assert plan.recommended_ops == ["denoise", "clahe", "gamma", "unsharp"]
        assert plan.stop_reason is None
        assert plan.params.clahe_clip_limit == 0.02

    def test_json_roundtrip(self, sample_enhancement_plan_dict: dict):
        plan = EnhancementPlan(**sample_enhancement_plan_dict)
        json_str = plan.model_dump_json()
        restored = EnhancementPlan.model_validate_json(json_str)
        assert restored.recommended_ops == plan.recommended_ops
        assert restored.params.gamma == plan.params.gamma

    def test_empty_ops_with_stop_reason(self):
        plan = EnhancementPlan(
            recommended_ops=[],
            stop_reason="Image quality is already satisfactory.",
        )
        assert plan.stop_reason is not None
        assert len(plan.recommended_ops) == 0

    def test_missing_required_fields_raises(self):
        with pytest.raises(Exception):
            # recommended_ops is required
            EnhancementPlan.model_validate({"params": {}})


class TestIterationRecord:
    def test_basic_creation(self, sample_enhancement_plan_dict: dict):
        plan = EnhancementPlan(**sample_enhancement_plan_dict)
        rec = IterationRecord(
            iteration=1,
            plan=plan,
            metrics={"ssim": 0.85, "psnr": 28.0, "quality_improvement": 0.15},
            score=0.42,
            chosen=True,
        )
        assert rec.chosen is True
        assert rec.score == 0.42


class TestExplainabilityReport:
    def test_all_fields(self):
        report = ExplainabilityReport(
            detected_issues="Noise and blur detected.",
            corrective_measures="Wavelet denoise and unsharp mask recommended.",
            enhancement_applied="Denoise + CLAHE applied.",
            validation_outcome="SSIM=0.82, PSNR=25.1 — PASS.",
            limitations="Not for clinical diagnosis.",
        )
        assert "Noise" in report.detected_issues
        assert "clinical" in report.limitations.lower()

    def test_json_serialisable(self):
        report = ExplainabilityReport(
            detected_issues="None.",
            corrective_measures="None needed.",
            enhancement_applied="None.",
            validation_outcome="All green.",
            limitations="Research only.",
        )
        data = json.loads(report.model_dump_json())
        assert "detected_issues" in data


class TestParamBounds:
    def test_all_expected_keys(self):
        expected = {
            "clahe_clip_limit",
            "clahe_tile_size",
            "gamma",
            "unsharp_radius",
            "unsharp_amount",
            "post_denoise_strength",
            "bilateral_d",
            "bilateral_sigma_color",
            "bilateral_sigma_space",
            "tv_denoise_weight",
        }
        assert set(PARAM_BOUNDS.keys()) == expected

    def test_bounds_are_valid(self):
        for key, (lo, hi) in PARAM_BOUNDS.items():
            assert lo < hi, f"{key}: lower bound {lo} >= upper bound {hi}"
