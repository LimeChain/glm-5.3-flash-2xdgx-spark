"""patches/9004: the response's ``tensorfold`` stats object."""

import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

SERVER = Path(__file__).resolve().parents[1] / "tensorfold/patches/overlay/tensorfold/cuda/server.py"
spec = importlib.util.spec_from_file_location("tf_server_9004", SERVER)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)

ROUNDS = 5000
STATS = {
    "rounds": ROUNDS,
    "decode_s": 255.7,
    "tokens_per_round": 1.955,
    "prefill_s": 0.27,
    "cached": 0,
    "sha256": "c750872970636c60",
    "keeps": [2] * ROUNDS,
    "depths": [3] * ROUNDS,
    "drafters": "m" * ROUNDS,
    "round_kinds": {"verify_ms": 10.0},
    "tf_knobs": {"latent_kv": 1},
}


class StatsModeTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("GLM53_TF_STATS", None)

    def test_default_summary_drops_per_round_series(self):
        out = server.response_stats(STATS, server.stats_mode({}))
        self.assertNotIn("keeps", out)
        self.assertNotIn("depths", out)
        self.assertNotIn("drafters", out)
        # The canary and dashboards read these scalars.
        for key in ("rounds", "decode_s", "tokens_per_round", "prefill_s", "cached", "round_kinds", "tf_knobs"):
            self.assertEqual(out[key], STATS[key])
        self.assertLess(len(json.dumps(out)), 1024)

    def test_no_value_in_summary_exceeds_strict_client_bounds(self):
        out = server.response_stats(STATS, "summary")
        for value in out.values():
            if isinstance(value, (list, dict)):
                self.assertLessEqual(len(value), 64)

    def test_full_and_off(self):
        self.assertIs(server.response_stats(STATS, server.stats_mode({"tensorfold_stats": "full"})), STATS)
        self.assertIsNone(server.response_stats(STATS, server.stats_mode({"tensorfold_stats": "off"})))
        self.assertIsNone(server.response_stats(None, "summary"))

    def test_environment_default_and_request_override(self):
        os.environ["GLM53_TF_STATS"] = "full"
        self.assertEqual(server.stats_mode({}), "full")
        self.assertEqual(server.stats_mode({"tensorfold_stats": "summary"}), "summary")
        os.environ["GLM53_TF_STATS"] = "bogus"
        self.assertEqual(server.stats_mode({}), "summary")

    def test_summary_does_not_mutate_engine_stats(self):
        stats = dict(STATS)
        server.response_stats(stats, "summary")
        self.assertIn("keeps", stats)


if __name__ == "__main__":
    unittest.main()
