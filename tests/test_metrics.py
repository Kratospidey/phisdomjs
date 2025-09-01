import unittest
import sys
from pathlib import Path

# Allow running tests without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phisdom.metrics import roc_auc, pr_auc, fpr_at_tpr


class TestMetrics(unittest.TestCase):
    def test_roc_auc_trivial(self):
        y = [0, 0, 1, 1]
        s = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(roc_auc(y, s), 1.0, places=6)

    def test_pr_auc_balanced(self):
        y = [0, 1, 0, 1]
        s = [0.1, 0.9, 0.2, 0.8]
        auc = pr_auc(y, s)
        self.assertTrue(0.9 <= auc <= 1.0)

    def test_fpr_at_tpr(self):
        y = [0, 0, 1, 1, 1]
        s = [0.1, 0.3, 0.5, 0.7, 0.9]
        fpr, thr = fpr_at_tpr(y, s, 0.66)
        self.assertTrue(0.0 <= fpr <= 1.0)
        self.assertTrue(0.0 <= thr <= 1.0)


if __name__ == "__main__":
    unittest.main()
