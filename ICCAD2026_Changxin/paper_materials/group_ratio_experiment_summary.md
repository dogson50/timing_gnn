# Group-Level Ratio Experiment Summary

- Generated at: 2026-04-20 13:36:28
- Source CSV: `ICCAD2026_Changxin/paper_materials/tables_ext_v5_r35_group_ratio/exp_summary_dual.csv`
- Statistics: 5-seed mean values (std is available in source CSV)

## 1) Protocol (this round)

- This round uses **group-level target/source training-data ratio** study.
- It is **not** row-level `target_label_ratio` subsampling.
- Target-domain split is group-disjoint by `group_id` (timing-table group).
- 50% target groups are fixed as validation groups.
- Remaining 50% groups form candidate pool.
- Ratio settings use complete groups: `g100/g050/g033/g020/g010`.

## 2) table_group (main protocol, full setting)

| Method | Val R2 | Val MAE | Val MAPE |
|---|---:|---:|---:|
| TCDP Joint | 0.93264 | 13.1247 | 25.4818 |
| Target-only | 0.89988 | 17.0658 | 33.7950 |
| Source-only | 0.44806 | 47.2136 | 121.6232 |
| Pretrain+Finetune | 0.91236 | 15.5790 | 29.8818 |

| TCDP Joint vs Target-only | Delta R2 | Delta MAE | Delta MAPE |
|---|---:|---:|---:|
| Improvement | +0.03276 | -3.9411 | -8.3132 |

## 3) table_group ablation (full setting)

| Ablation | Val R2 | Val MAE | Val MAPE |
|---|---:|---:|---:|
| w/o graph | 0.91196 | 15.3756 | 29.9564 |
| w/o domain_calib | 0.93018 | 13.2398 | 24.4948 |
| w/o topology_res | 0.86942 | 17.8330 | 29.1900 |

## 4) table_group group-level ratio (fixed val groups=50%)

| Setting | TCDP R2 | Target-only R2 | Delta R2 | TCDP MAE | Target-only MAE | Delta MAE | TCDP MAPE | Target-only MAPE | Delta MAPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| g100 | 0.98190 | 0.98176 | +0.00014 | 6.7350 | 6.7645 | -0.0295 | 14.6556 | 14.5178 | +0.1378 |
| g050 | 0.90822 | 0.88282 | +0.02540 | 14.3374 | 17.0160 | -2.6786 | 30.4756 | 34.0084 | -3.5328 |
| g033 | 0.86104 | 0.84794 | +0.01310 | 19.0597 | 19.8199 | -0.7602 | 39.6486 | 38.4990 | +1.1496 |
| g020 | 0.78680 | 0.70520 | +0.08160 | 24.1496 | 26.8846 | -2.7350 | 46.4712 | 49.2456 | -2.7744 |
| g010 | 0.63018 | 0.54120 | +0.08898 | 30.9250 | 33.2314 | -2.3064 | 61.8676 | 61.8782 | -0.0106 |

## 5) cell_type (replication line, full setting)

| Method | Val R2 | Val MAE | Val MAPE |
|---|---:|---:|---:|
| TCDP Joint | 0.76548 | 25.6135 | 44.4502 |
| Target-only | 0.69030 | 29.2688 | 52.0280 |
| Source-only | 0.39738 | 47.2469 | 115.5098 |
| Pretrain+Finetune | 0.80178 | 22.7069 | 34.4448 |

| TCDP Joint vs Target-only | Delta R2 | Delta MAE | Delta MAPE |
|---|---:|---:|---:|
| Improvement | +0.07518 | -3.6553 | -7.5778 |

## 6) cell_type ablation (full setting)

| Ablation | Val R2 | Val MAE | Val MAPE |
|---|---:|---:|---:|
| w/o graph | 0.72710 | 28.8057 | 51.4376 |
| w/o domain_calib | 0.75524 | 25.9219 | 44.8308 |
| w/o topology_res | 0.79462 | 22.8884 | 34.8338 |

## 7) Short takeaways

- Under table_group full setting, TCDP Joint remains the best mainline versus Target-only.
- Under group-level ratio settings, TCDP gains are larger in lower target-data budgets (`g020`, `g010`) on R2.
- Under cell_type full setting, Pretrain+Finetune has higher R2 than TCDP Joint.
