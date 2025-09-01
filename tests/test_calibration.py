import unittest
import sys
from pathlib import Path

# Allow running tests without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phisdom.calibration import TemperatureScaler, _logloss


class TestCalibration(unittest.TestCase):
    def test_temperature_scaling_improves_logloss(self):
        # Synthetic miscalibrated logits: confident but wrong half the time
        y = [0, 1, 0, 1, 0, 1, 0, 1]
        logits = [3, 3, -3, -3, 3, 3, -3, -3]  # uncalibrated
        base_probs = [1 / (1 + pow(2.718281828, -z)) for z in logits]
        base_loss = _logloss(y, base_probs)

        ts = TemperatureScaler(is_logit=True)
        ts.fit(y, logits)
        cal_probs = ts.transform(logits)
        cal_loss = _logloss(y, cal_probs)
        self.assertLessEqual(cal_loss, base_loss)


if __name__ == "__main__":
    unittest.main()
