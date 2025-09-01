from typing import Sequence, Tuple
import math

_EPS = 1e-12


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logloss(y: Sequence[int], p: Sequence[float]) -> float:
    loss = 0.0
    for yi, pi in zip(y, p):
        pi = min(max(pi, _EPS), 1.0 - _EPS)
        loss += -yi * math.log(pi) - (1 - yi) * math.log(1 - pi)
    return loss / max(1, len(y))


class TemperatureScaler:
    """
    Binary temperature scaling on logits. If provided probabilities, we convert to logits first.
    Minimizes log loss on a validation set via golden-section search over T in [t_min, t_max].
    """

    def __init__(self, is_logit: bool = True, t_min: float = 0.05, t_max: float = 20.0):
        self.is_logit = is_logit
        self.t_min = t_min
        self.t_max = t_max
        self.T_: float | None = None

    def fit(self, y_true: Sequence[int], y_in: Sequence[float]) -> float:
        # Convert to logits if needed
        logits = [(_logit(p) if not self.is_logit else float(p)) for p in y_in]

        def loss_at(T: float) -> float:
            probs = [_sigmoid(z / T) for z in logits]
            return _logloss(y_true, probs)

        # Golden-section search
        gr = (1 + 5 ** 0.5) / 2
        a, b = self.t_min, self.t_max
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        fc = loss_at(c)
        fd = loss_at(d)
        for _ in range(80):  # sufficient for convergence
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - (b - a) / gr
                fc = loss_at(c)
            else:
                a, c, fc = c, d, fd
                d = a + (b - a) / gr
                fd = loss_at(d)
        self.T = (a + b) / 2
        self.T_ = self.T
        return self.T

    def transform(self, y_in: Sequence[float]) -> Tuple[float, ...]:
        assert self.T_ is not None, "Call fit() first"
        if self.is_logit:
            return tuple(_sigmoid(z / self.T_) for z in y_in)
        # y_in are probabilities
        logits = [_logit(p) for p in y_in]
        return tuple(_sigmoid(z / self.T_) for z in logits)
