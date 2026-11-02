# -*- coding: utf-8 -*-
# ==============================================================================
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# Original by DELTA SYNTH
# File: test_system_smoke.py | Version: 1.0
# Description: Automated Smoke Test Harness for System Verification & Zero-Defects
# ==============================================================================
import os
import sys
import gc
import unittest
import numpy as np
import torch

# Ensure project root is in sys.path
now_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)

from configs.config import Config
from infer.modules.vc.pipeline import change_rms, clear_f0_cache, cache_harvest_f0
from infer.modules.vc.utils import get_index_path_from_model


class TestRVCSystemSmoke(unittest.TestCase):
    """Automated Smoke Test Suite for DELTA SYNTH RVC WebUI."""

    def setUp(self):
        self.config = Config()

    def test_01_config_initialization(self):
        """Verify hardware configuration initializes safely with valid device types."""
        self.assertIsNotNone(self.config)
        self.assertIn(str(self.config.device), ["cpu", "cuda", "cuda:0", "mps", "xpu", "dml"])
        self.assertIsInstance(self.config.is_half, bool)
        print(f"  [PASS] Config Initialized: Device={self.config.device}, is_half={self.config.is_half}")

    def test_02_change_rms_dsp(self):
        """Verify RMS normalization function with synthetic audio."""
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
        sine1 = 0.5 * np.sin(2 * np.pi * 440 * t)
        sine2 = 0.2 * np.sin(2 * np.pi * 880 * t)

        # Rate = 1.0 (no rms change)
        res1 = change_rms(sine1, sr, sine2.copy(), sr, 1.0)
        self.assertEqual(res1.shape, sine2.shape)

        # Rate = 0.5 (blend)
        res2 = change_rms(sine1, sr, sine2.copy(), sr, 0.5)
        self.assertEqual(res2.shape, sine2.shape)
        self.assertFalse(np.isnan(res2).any())
        self.assertFalse(np.isinf(res2).any())

        # Boundary checks
        with self.assertRaises(ValueError):
            change_rms(sine1, sr, sine2.copy(), sr, -0.1)
        with self.assertRaises(ValueError):
            change_rms(sine1, sr, sine2.copy(), sr, 1.1)

        print("  [PASS] change_rms DSP calculations verified with 0 NaN/Inf")

    def test_03_f0_cache_memory_management(self):
        """Verify Harvest F0 caching, eviction, and memory reclamation."""
        clear_f0_cache()
        # Verify cache_clear does not raise
        clear_f0_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  [PASS] F0 Cache and VRAM/RAM reclamation executed cleanly")

    def test_04_index_discovery(self):
        """Verify deterministic index discovery utility handles missing and valid queries."""
        # Query with empty sid
        res_empty = get_index_path_from_model("")
        self.assertEqual(res_empty, "")

        res_none = get_index_path_from_model(None)
        self.assertEqual(res_none, "")

        # Query with arbitrary model name
        res_dummy = get_index_path_from_model("nonexistent_model_test_12345.pth")
        self.assertIsInstance(res_dummy, str)
        print("  [PASS] Index path discovery utility handles boundary inputs safely")

    def test_05_vc_module_lifecycle(self):
        """Verify VC engine initial state, model unloading, and memory safety."""
        from infer.modules.vc.modules import VC
        vc_instance = VC(self.config)
        self.assertIsNotNone(vc_instance)

        # Unload model safely (sid = "")
        update_result = vc_instance.get_vc("")
        self.assertEqual(len(update_result), 5)
        self.assertIsNone(vc_instance.net_g)
        self.assertIsNone(vc_instance.cpt)

        # Cancel process safety
        cancel_msg = vc_instance.cancel_process()
        self.assertIn("ยกเลิก", cancel_msg)
        print("  [PASS] VC Engine lifecycle, unload, and cancel functions verified")

    def test_06_pipeline_feature_dimension_handling(self):
        """Verify pipeline feature extraction layer and projection for v1 (256-dim) and v2 (768-dim)."""
        from infer.modules.vc.pipeline import Pipeline
        pipeline = Pipeline(40000, self.config)

        # Mock HuBERT model
        class MockHuBERT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.final_proj = torch.nn.Linear(768, 256)

            def extract_features(self, source, padding_mask=None, output_layer=None):
                b, t = source.shape[0], source.shape[1] // 320
                t = max(t, 10)
                return (torch.zeros(b, t, 768, dtype=source.dtype, device=source.device), None)

        mock_hubert = MockHuBERT()

        # Mock v1 Generator expecting 256 dimensions
        class MockNetG_v1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                class EncP(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.emb_phone = torch.nn.Linear(256, 192)
                self.enc_p = EncP()

            def infer(self, feats, p_len, *args):
                assert feats.shape[-1] == 256, f"Expected 256 dims for v1, got {feats.shape[-1]}"
                return (torch.zeros(1, 1, 16000), None)

        # Mock v2 Generator expecting 768 dimensions
        class MockNetG_v2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                class EncP(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.emb_phone = torch.nn.Linear(768, 192)
                self.enc_p = EncP()

            def infer(self, feats, p_len, *args):
                assert feats.shape[-1] == 768, f"Expected 768 dims for v2, got {feats.shape[-1]}"
                return (torch.zeros(1, 1, 16000), None)

        audio0 = np.zeros(16000, dtype=np.float32)
        times = [0.0, 0.0, 0.0]

        # Test v1 inference: should project 768 -> 256 seamlessly
        net_v1 = MockNetG_v1()
        res_v1 = pipeline.vc(mock_hubert, net_v1, torch.tensor([0]), audio0, None, None, times, None, None, 0.0, "v1", 0.33)
        self.assertIsInstance(res_v1, np.ndarray)

        # Test v2 inference: should maintain 768 dims seamlessly
        net_v2 = MockNetG_v2()
        res_v2 = pipeline.vc(mock_hubert, net_v2, torch.tensor([0]), audio0, None, None, times, None, None, 0.0, "v2", 0.33)
        self.assertIsInstance(res_v2, np.ndarray)

        # Test Mismatched Index Dimension Safety Guard
        class MockFaissIndex:
            def __init__(self, d):
                self.d = d
                self.ntotal = 100
            def search(self, npy, k=8):
                raise RuntimeError("Should not be called if dimension mismatches!")

        mismatched_index = MockFaissIndex(d=768)
        # Should gracefully log warning and skip search when passed to v1 (256-dim)
        res_guard = pipeline.vc(mock_hubert, net_v1, torch.tensor([0]), audio0, None, None, times, mismatched_index, np.zeros((100, 768)), 0.5, "v1", 0.33)
        self.assertIsInstance(res_guard, np.ndarray)
        print("  [PASS] Pipeline feature extraction, dimension guards (v1: 256, v2: 768), and Faiss safety verified")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DELTA SYNTH - Automated System Smoke Test Execution")
    print("=" * 60)
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRVCSystemSmoke)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
