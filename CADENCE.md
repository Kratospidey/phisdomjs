## Delivery Cadence & KPI Process (Phase 11)

This document outlines the recurring workflow for model improvements and monitoring.

### Weekly Cycle
1. Collect new data & curate hard negatives (Phase 2/3 refresh)
2. Retrain heads with augmentation & class weighting
3. Refit fusion + meta CV stacking (Phase 5)
4. Run cascade optimization (Phase 6) & refiner (Phase 8)
5. Execute enhanced evaluation aggregator (Phase 9)
6. Run CI gates (Phase 10) – block if degradation beyond thresholds
7. Append metrics to KPI dashboard (Phase 11)
8. Record notable changes in CHANGELOG (future enhancement)

### KPIs Tracked
- Fusion Precision/Recall/FPR @ operating point
- Cascade expected cost & recall
- Refiner precision uplift in confusion band
- Hard negative fraction in training set
- Calibration quality (Brier/ECE) trend

### Gating Thresholds (Initial)
- Max precision drop: 2% relative
- Max FPR increase: 2% relative
- Cascade cost should not regress >5% without precision gain justification

### Roles
- Data Curation: Maintain FP clusters & hard negative seeds
- Modeling: Implement stacking, loss experiments, refiner tuning
- Evaluation: Ensure dashboards and CI gates reflect latest metrics

### Automation Hooks
1. `evaluate_enhanced.py` generates consolidated metrics
2. `ci_gate.py` enforces thresholds
3. `build_kpi_dashboard.py` appends time-series

### Future Enhancements
- Add latency measurements per head from production telemetry
- Introduce bootstrap confidence intervals for precision/recall
- Expand refiner with lightweight transformer for band cases
