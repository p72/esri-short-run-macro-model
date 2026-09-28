"""忠実性の実験コードの回帰テスト（Issue #18）.

実行: python -m unittest discover -s tests -v
"""
import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import check_transcription as CT  # noqa: E402
import experiment_fidelity as F  # noqa: E402
import model as M  # noqa: E402

VALUES = dict(RTCI=0.10, PRTCP=0.52, PRTIF=0.18, PRTIH=0.80, PRTCG=0.48, PRTIG=0.45,
              CPV=300000.0, IFPV=90000.0, IHPV=20000.0, CGV=110000.0, IGV=30000.0, D972C=1.0)


def getter(**over):
    vals = {**VALUES, **over}
    return lambda n, k=0: vals[n]


class Eq134Printed(unittest.TestCase):
    def setUp(self):
        self.default = next(e for e in M.build_equations() if e.name == "TCIV")
        self.printed = F.eq134_printed()

    def test_same_as_default_when_dummy_is_zero(self):
        v = getter(DTCIC2=0.0)
        self.assertAlmostEqual(self.printed.rhs(v), self.default.rhs(v), places=12)

    def test_direct_formula_when_dummy_is_one(self):
        v = getter(DTCIC2=1.0)
        r = VALUES["RTCI"]
        main = (r / (1 + r * 0.52) * 300000 + r / (1 + r * 0.18) * 90000 + r / (1 + r * 0.80) * 20000
                + r / (1 + r * 0.48) * 110000 + r / (1 + r * 0.45) * 30000)
        dummy = main - r / (1 + r * 0.18) * 90000 + r / (1 + r + 0.18) * 90000  # 印刷: 設備投資だけ加算の分母
        expected = 0.915210 * np.log(main) - 0.011036 * np.log(dummy) + 0.191987
        self.assertAlmostEqual(self.printed.rhs(v), expected, places=12)
        self.assertNotAlmostEqual(self.printed.rhs(v), self.default.rhs(v), places=6)


class TranscriptionCompleteness(unittest.TestCase):
    def test_missing_identity_is_detected(self):
        # 係数のない恒等式（例: 式2 GDP の定義式）を model.py から消しても、式番号の欠落として失敗する
        orig = CT.model_blocks
        CT.model_blocks = lambda: {k: v for k, v in orig().items() if k != 2}
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
                CT.main()
            self.assertIn("欠落 [2]", out.getvalue())
        finally:
            CT.model_blocks = orig


if __name__ == "__main__":
    unittest.main()
