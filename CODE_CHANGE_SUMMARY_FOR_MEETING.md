# 代码修改汇总

日期：2026-03-23  
用途：组会汇报  
说明：最新代码仍在运行中，本文档只总结“已完成的代码修改”和“预期作用”，不包含最新实验结论。

## 1. 本轮修改的目标

当前课题任务是利用大量源域 `Nangate45` 标准单元库数据和少量目标域 `ASAP7` 数据，预测目标域不同标准单元在不同条件下的 delay。

本轮代码修改主要围绕以下几个问题展开：

- 目标域 `ASAP7` 的 `.sp` 网表虽然能匹配到 `subckt`，但原始图表示过于粗糙，没有利用 `nfin` 和寄生 `R/C`。
- 原始 `HGAT` 特征定义偏弱，只用了较少器件/网络属性。
- 原始训练脚本对 `timing arc` 的条件区分不够，`from_pin/to_pin`、`timing_sense`、`when/sdf_cond` 没有完整进入模型。
- 原始 `CLR/CMD` 定义不够合理，尤其是 `CMD` 对包含结构信息的整套数值特征做对齐，可能伤害预测。
- 原始训练脚本没有早停机制，不利于稳定比较不同实验设置。

## 2. 已完成的主要代码修改

## 2.1 数据集构建与切分方式

修改文件：

- `src/build_dataset.py`
- `src/options.py`

已完成修改：

- 新增 `table_group` 切分方式，避免将同一张 NLDM 表中的点随机拆到 train/test，减少“表内插值泄漏”。
- `group_id` 现在按完整 timing table 语义构造，而不是按单个样本点构造。
- 当前 `group_id` 包含以下信息：
  - `tech`
  - `cell_name`
  - `from_pin`
  - `to_pin`
  - `timing_sense`
  - `arc_cond`
  - `pol`
  - `voltage`
  - `temp`
- 数据集中新增并保留以下字段：
  - `when_cond`
  - `sdf_cond`
  - `timing_sense`
  - `arc_cond`
  - `group_id`
- 目标域 `ASAP7` 的 SPICE 特征提取改为先做层次展开，再提取晶体管统计，避免只看顶层 `subckt`。

当前意义：

- `table_group` 更适合作为主实验设置。
- `cell_type` 切分可以保留为 harder benchmark，用于评估“未见单元类型泛化”。

## 2.2 lib 解析增强

修改文件：

- `src/parse_lib.py`

已完成修改：

- 新增 `when`、`sdf_cond`、`timing_sense` 的解析与保存。
- 新增 `normalize_arc_condition()`，统一 `when/sdf_cond` 的条件表达形式。
- 对 timing arc 的解析结果中保留：
  - `from_pin`
  - `to_pin`
  - `when_cond`
  - `sdf_cond`
  - `timing_sense`
- 增加 `cell_type_to_topology_group()`，用于把不同 drive strength 的同拓扑单元归并到同一逻辑家族。

已确认解决的问题：

- `XOR2/XNOR2` 这种在相同 `related_pin` 下存在不同 `when` 条件和不同 `timing_sense` 的 arc，现在能被区分。
- 源域和目标域中的条件弧信息现在可以被保留到数据集与训练阶段。

## 2.3 SPICE 解析与 ASAP7 图表示增强

修改文件：

- `src/spi2graph.py`
- `src/hgat.py`

已完成修改：

- 重写 SPICE 解析逻辑，增强对数值单位、续行、器件参数的处理。
- 新增 `parse_transistors_spice()`，支持提取：
  - `W`
  - `L`
  - `nfin`
  - `m`
- 新增 `parse_passive_parasitics_spice()`，支持提取寄生：
  - `R`
  - `C`
- 新增层次化 `subckt` 展开：
  - `flatten_subckt_hierarchy()`
  - `extract_subckt_with_dependencies()`
- `extract_wl_features()` 现在额外返回：
  - `num_pmos`
  - `num_nmos`
  - `p_nfin_sum`
  - `n_nfin_sum`

HGAT 图表示增强如下：

- `PMOS/NMOS` 节点特征从原始的 `logW/logL` 扩展为：
  - `logW`
  - `logL`
  - `log(1+nfin)`
  - `log(m)`
- `NET` 节点特征扩展为 10 维，包括：
  - 电源/地/顶层 pin/internal 标志
  - 器件连接度
  - 寄生连接度
  - 接地电容总量
  - 耦合电容总量
  - 电阻总量
  - 等效导通强度
- 图中新增 `NET -> NET` 的寄生关系：
  - `res_to`
  - `cap_to`
- HGAT 最终输出的 `L2 normalize` 变为可选，默认关闭。

当前意义：

- `ASAP7` FinFET 工艺中的 `nfin` 终于被显式建模。
- 寄生 `R/C` 不再完全丢失。
- `HGAT` 的输入图更接近真实电路物理属性。

## 2.4 训练脚本接入新图特征

修改文件：

- `src/train_balanced_sampling_sep_mlp_shared_calib.py`
- `src/train_balanced_sampling_sep_mlp_disentangle_cell_type.py`
- `src/train_balanced_sampling.py`
- `src/train_base_pt_ft.py`
- `src/train_balanced_sampling_sep_mlp.py`
- `src/train_balanced_sampling_sep_mlp_bayesian.py`
- `src/train_balanced_sampling_sep_mlp_factorized_calib.py`
- `src/train_balanced_sampling_sep_mlp_disentangle_domain.py`
- `src/train.py`

已完成修改：

- 训练时改用新的 `build_graph_from_spice_text()` 构图。
- 各脚本同步适配新的 HGAT 输入维度映射。
- 支持 `hgat_l2_norm` 开关。
- 主要在用的 `shared_calib` 和 `disentangle_cell_type` 已经能够读取新图表示。

## 2.5 timing arc 条件进入模型

修改文件：

- `src/options.py`
- `src/train_balanced_sampling_sep_mlp_shared_calib.py`
- `src/train_balanced_sampling_sep_mlp_disentangle_cell_type.py`

已完成修改：

- 新增 arc 条件相关参数：
  - `--use_arc_cond`
  - `--pin_emb_dim`
  - `--pol_emb_dim`
  - `--sense_emb_dim`
  - `--cond_emb_dim`
  - `--arc_vocab_scope`
  - `--arc_sep_domain_emb`
  - `--arc_cond_mode`
  - `--disable_src_arc_cond`
- 训练头部现在可使用：
  - `from_pin`
  - `to_pin`
  - `pol`
  - `timing_sense`
  - `arc_cond`
- 支持两种 arc 词表构建方式：
  - `tgt`
  - `src_tgt`

当前意义：

- 模型现在能区分“同一 related_pin 但不同逻辑条件和不同翻转路径”的 timing arc。
- `XOR2` 等带条件控制的单元不再被简单混在一起。

## 2.6 CLR/CMD 重新定义

修改文件：

- `src/options.py`
- `src/train_balanced_sampling_sep_mlp_disentangle_cell_type.py`

已完成修改：

- `CLR` 不再直接作用在回归分支特征上，而是增加了独立的 `projection head`。
- 新增参数：
  - `--clr_proj_dim`
  - `--cmd_proj_dim`
  - `--clr_label_mode`
  - `--cmd_label_mode`
- 默认的 `clr_label_mode` 和 `cmd_label_mode` 改为 `semantic_arc`。

新的 `semantic_arc` 标签定义为：

- `topology group`
- `canonicalized from_pin role`
- `canonicalized to_pin role`
- `timing_sense`
- `canonicalized arc_cond`

其中 pin/arc 语义规范化已经考虑以下问题：

- 源域输出 pin 常为 `Z/ZN`，目标域常为 `Y`
- 源域输入 pin 常为 `A1/A2/A3...`，目标域常为 `A/B/C/D...`

为此已完成的处理是：

- 先在每个具体 `cell_name` 内部按输入 pin 顺序映射为 `IN1/IN2/...`
- 输出统一映射为 `OUT`
- 然后再提升到 `topology` 层面进行跨域对齐

同时，`CMD` 已从原始的“整批全局 moment matching”改为：

- 先通过独立 `cmd_projector`
- 再减去由 `log_slew/log_cap/pol_bit` 建模的上下文项
- 然后按共享语义标签做 `conditional CMD`

当前意义：

- `CLR` 目标更接近“同逻辑功能、同 arc 角色”的对齐，而不是记忆 exact `cell_type`。
- `CMD` 更接近“对残差域偏移做对齐”，而不是盲目对齐所有混合特征。

## 2.7 早停版本新脚本

新增文件：

- `src/train_balanced_sampling_sep_mlp_shared_calib_early_stop.py`
- `src/train_balanced_sampling_sep_mlp_disentangle_cell_type_early_stop.py`

新增原因：

- 用户当前已有训练在运行，不直接修改原始脚本，避免影响已有实验。

新增能力：

- 支持按 `val_r2` 进行早停。
- 支持参数：
  - `--early_stop_patience`
  - `--early_stop_min_delta`

早停规则：

- 若 `val_r2 > best_val_r2 + min_delta`，记为改进
- 连续 `patience` 个 epoch 无改进则提前终止训练

## 2.8 进一步修正：process 分支只保留真正接近工艺/条件的输入

修改文件：

- `src/train_balanced_sampling_sep_mlp_disentangle_cell_type_early_stop.py`

原因：

- 原始 `disentangle` 中 `process` 分支输入整套 `NUMERIC_COLS`，其中混入了大量结构相关特征：
  - `wp_sum`
  - `wn_sum`
  - `wp_over_wn`
  - `is_inv`
  - `req_*`
  - `rc_*`
  - `pn_balance`
- 若再对这类混合特征施加 `CMD`，容易把结构差异也强行对齐，反而伤害预测。

已完成修正：

早停版 `disentangle` 的 `process` 分支现在只使用以下 7 个特征：

- `voltage`
- `temp`
- `inv_v`
- `inv_temp`
- `log_slew`
- `log_cap`
- `pol_bit`

当前意义：

- `process` 分支更接近真正的“工艺/条件因素”。
- `CMD` 的作用对象更合理。
- 该问题在原始 `disentangle` 文件中仍然存在，但在早停版中已修正。

## 3. 用户已自行修复并纳入当前代码状态的问题

以下问题已确认在当前代码状态中被修复：

- `AND3X4` 曾错误映射到 `AND2_X4_lpe.spi`
- `XNOR2X2_ASAP7_6T_L` 曾被错误写成 `XOR2...`

## 4. 当前对实验设置的理解

## 4.1 关于 `shared_calib`

- `shared_calib` 不包含 `CLR/CMD`，是当前较强、较稳定的 baseline。
- 对于这条线，重点是验证：
  - 新的 HGAT 图表示是否有效
  - arc conditioning 是否有效
  - `table_group` 设定下的迁移效果是否稳定

## 4.2 关于 `disentangle`

- 若 `--weight_clr 0 --weight_cmd 0`，本质上只是“解耦结构的回归 baseline”，不是对齐实验。
- 若要验证新的 `CLR/CMD` 是否有效，必须显式设置非零权重。

建议起始值：

- `--weight_clr 0.05`
- `--weight_cmd 0.05`

如果不稳定，可降到：

- `--weight_clr 0.02`
- `--weight_cmd 0.02`

## 4.3 关于是否冻结 HGAT

当前建议：

- `shared_calib` 一般不建议使用 `--freeze_hgat`
- `disentangle` 也建议默认不冻结

原因：

- 不冻结时，HGAT 可随当前任务一起微调
- 当前 `shared_calib` 不具备完整的预训练 encoder 加载流程，直接冻结可能等价于“冻结随机初始化的 HGAT”

## 5. 推荐的当前实验入口

## 5.1 数据集构建

更推荐作为主实验的设置：

```powershell
python ./src/build_dataset.py --tgt_split_mode table_group --tgt_split_ratios 0.14 0.14 0.72
```

## 5.2 shared_calib 早停版

```powershell
python ./src/train_balanced_sampling_sep_mlp_shared_calib_early_stop.py `
  --model_saving_dir runs/shared_es_s9294 `
  --data_save_path output `
  --dataset_pkl_name dataset.pkl `
  --hgat_layers 2 `
  --hgat_dropout 0.1 `
  --hgat_use_net_readout `
  --use_arc_cond `
  --arc_vocab_scope src_tgt `
  --seed 9294 `
  --early_stop_patience 30 `
  --early_stop_min_delta 0.001
```

## 5.3 disentangle 早停版

```powershell
python ./src/train_balanced_sampling_sep_mlp_disentangle_cell_type_early_stop.py `
  --model_saving_dir runs/dis_sem_es_s9294 `
  --data_save_path output `
  --dataset_pkl_name dataset.pkl `
  --hgat_layers 2 `
  --hgat_dropout 0.1 `
  --hgat_use_net_readout `
  --use_arc_cond `
  --arc_vocab_scope src_tgt `
  --weight_clr 0.05 `
  --weight_cmd 0.05 `
  --clr_label_mode semantic_arc `
  --cmd_label_mode semantic_arc `
  --seed 9294 `
  --early_stop_patience 30 `
  --early_stop_min_delta 0.001
```

## 6. 当前状态与汇报建议

当前状态：

- 代码修改已完成。
- 关键脚本已通过语法检查。
- 最新代码仍在运行中，因此目前没有基于最新版代码的最终结果。
- 现阶段更适合在组会上汇报：
  - 本轮发现的问题
  - 已完成的代码级修正
  - 新的实验设计逻辑
  - 当前正在运行的实验配置

建议在组会上强调的重点：

- 本轮不是简单调参，而是修正了“数据语义、图表示、arc 条件建模、对齐定义”这几类基础问题。
- `table_group` 更符合课题原始目标，可作为主实验设置。
- `cell_type` 切分可作为 harder benchmark 保留。
- 最新结果尚未完成，当前重点是保证实验问题定义和代码实现逻辑正确。

## 7. 本轮新增/重点修改文件清单

- `src/build_dataset.py`
- `src/parse_lib.py`
- `src/spi2graph.py`
- `src/hgat.py`
- `src/options.py`
- `src/train_balanced_sampling_sep_mlp_shared_calib.py`
- `src/train_balanced_sampling_sep_mlp_disentangle_cell_type.py`
- `src/train_balanced_sampling_sep_mlp_shared_calib_early_stop.py`
- `src/train_balanced_sampling_sep_mlp_disentangle_cell_type_early_stop.py`

兼容性同步修改：

- `src/train_balanced_sampling.py`
- `src/train_base_pt_ft.py`
- `src/train_balanced_sampling_sep_mlp.py`
- `src/train_balanced_sampling_sep_mlp_bayesian.py`
- `src/train_balanced_sampling_sep_mlp_factorized_calib.py`
- `src/train_balanced_sampling_sep_mlp_disentangle_domain.py`
- `src/train.py`
