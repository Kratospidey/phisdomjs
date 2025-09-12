from phisdom.ops import brier, ece, pick_operating_point

def test_brier_simple():
    y=[0,1,1,0]
    p=[0.1,0.9,0.8,0.2]
    # (0.1-0)^2 + (0.9-1)^2 + (0.8-1)^2 + (0.2-0)^2 = 0.01 + 0.01 + 0.04 + 0.04 = 0.10 -> /4 = 0.025
    assert abs(brier(y,p) - 0.025) < 1e-9

def test_ece_trivial_bins():
    y=[0,1]
    p=[0.0,1.0]
    assert ece(y,p, bins=2) == 0.0

def test_pick_operating_point_threshold():
    # Scores: positives high, negatives low
    y=[0,0,1,1]
    p=[0.1,0.2,0.8,0.9]
    op=pick_operating_point(y,p, target_recall=1.0)
    # To achieve recall=1.0, threshold must be <= lowest positive score (0.8)
    assert op['recall'] >= 1.0 - 1e-9
    assert op['threshold'] <= 0.8 + 1e-9
