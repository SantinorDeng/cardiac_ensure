# Cardiac ENSURE Code Review Guide

这份文档不是逐行解释代码，而是帮你在精力有限的情况下，快速抓住 `cardiac_ensure` 里最值得看的部分。

建议阅读顺序：

1. `losses/ensure_loss.py`
2. `ops/cg_solver.py`
3. `ops/mri_ops.py`
4. `datasets/cardiac_cine_dataset.py`
5. `models/temporal_normunet.py`
6. `scripts/train_true_ensure.py`
7. `scripts/train_supervised_baseline.py`
8. `scripts/run_phase_d_smoke.py`
9. `scripts/eval_metrics.py`
10. `scripts/train_common.py`
11. `scripts/run_phase_c_sanity.py`
12. `datasets/preprocess_raw_cine.py`
13. `datasets/precompute_density_stats.py`

如果你只想先看 Phase C 主干，只看前 4 个文件就够了。

如果你想直接看 Phase D 训练主干，优先看第 5 到第 10 个文件。

---

## 1. 模块总览

### `cardiac_ensure/datasets`

职责：

- 从原始 `.h5` cardiac cine 数据构造训练样本
- 读取预处理 sidecar
- 在线生成 Bernoulli-Gaussian mask
- 返回 `kspace_us / mask / maps / zf / noise_sigma2 / density`

对应文件：

- [datasets/cardiac_cine_dataset.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/cardiac_cine_dataset.py)
- [datasets/preprocess_raw_cine.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/preprocess_raw_cine.py)
- [datasets/precompute_density_stats.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/precompute_density_stats.py)
- [datasets/__init__.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/__init__.py)

### `cardiac_ensure/ops`

职责：

- 定义动态 MRI 前向算子和伴随算子
- 处理密度加权
- 用 CG 求 `rho_ls` 和 `R_s e`

对应文件：

- [ops/mri_ops.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/ops/mri_ops.py)
- [ops/cg_solver.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/ops/cg_solver.py)

### `cardiac_ensure/losses`

职责：

- 组装 true ENSURE 的 data term
- 实现 Monte-Carlo divergence
- 把 `rho_ls + R_s e + divergence` 封成一个统一 loss 接口

对应文件：

- [losses/ensure_loss.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/losses/ensure_loss.py)

### `cardiac_ensure/models`

职责：

- 定义 Phase D 的动态重建 backbone
- 处理复数输入和时间窗折叠
- 提供全帧输出和中心帧输出两种模式

对应文件：

- [models/temporal_normunet.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/models/temporal_normunet.py)
- [models/__init__.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/models/__init__.py)

### `cardiac_ensure/scripts`

职责：

- 提供 supervised / true ENSURE 训练入口
- 提供通用评估与训练辅助工具
- 做 C1-C4 和 D 阶段 smoke test

对应文件：

- [scripts/train_supervised_baseline.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_supervised_baseline.py)
- [scripts/train_true_ensure.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py)
- [scripts/run_phase_d_smoke.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_d_smoke.py)
- [scripts/eval_metrics.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/eval_metrics.py)
- [scripts/train_common.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_common.py)
- [scripts/run_phase_c_sanity.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_c_sanity.py)

---

## 2. 哪些文件最值得看

### 第一优先级

#### `losses/ensure_loss.py`

你如果只想确认“true ENSURE 最终到底怎么算”，优先看这个文件。

最值得看：

- [ensure_data_term()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/losses/ensure_loss.py:22)
- [estimate_divergence_mc()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/losses/ensure_loss.py:51)
- [compute_true_ensure_loss()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/losses/ensure_loss.py:88)

为什么值得看：

- `ensure_data_term` 对应 Eq. (29) 之后的数据项
- `estimate_divergence_mc` 对应 C4 的 Monte-Carlo divergence
- `compute_true_ensure_loss` 是训练时最可能直接调用的总入口

#### `ops/cg_solver.py`

你如果关心 `rho_ls` 和 `R_s e` 是怎么解的，看这个文件。

最值得看：

- [complex_conjugate_gradient()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/ops/cg_solver.py:41)
- [solve_rho_ls()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/ops/cg_solver.py:86)
- [solve_weighted_projection()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/ops/cg_solver.py:113)

为什么值得看：

- `solve_rho_ls` 是 C2
- `solve_weighted_projection` 是 C3
- 这两个函数都依赖同一个复数 CG 内核，改收敛性时也主要改这里

#### `ops/mri_ops.py`

你如果关心动态 `A / A^H / A^H A` 和 shape 适配，看这个文件。

最值得看：

- [sense_forward / sense_adjoint / sense_normal](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/ops/mri_ops.py:138)
- [dynamic_a_forward / dynamic_a_adjoint / dynamic_a_normal](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/ops/mri_ops.py:390)
- `_prepare_dynamic_image_problem` 和 `_prepare_dynamic_kspace_problem`
- `_flatten_dynamic_weight`

为什么值得看：

- 单帧公式都落在 `sense_*`
- 动态时序和 batch reshape 都落在 `dynamic_*` 和 `prepare_*`
- 如果以后 shape 出 bug，90% 都会在这里

#### `models/temporal_normunet.py`

你如果关心 Phase D 的 backbone、时间维是怎么折到通道里、以及为什么网络现在默认输出全帧，看这个文件。

最值得看：

- [NormUnet()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/models/temporal_normunet.py:106)
- [extract_center_frame()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/models/temporal_normunet.py:183)
- [TemporalNormUnet()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/models/temporal_normunet.py:190)
- [TemporalNormUnet.forward()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/models/temporal_normunet.py:264)

为什么值得看：

- `NormUnet` 是从 MoDL 风格 U-Net 抽出来的复数 backbone
- `TemporalNormUnet` 把 `[B,T,H,W,2]` 折成 `2T` 个通道做联合建模
- `output_mode` 让你保留“全帧输出”和“中心帧输出”两种实验路径

### 第二优先级

#### `datasets/cardiac_cine_dataset.py`

你如果想知道训练时 sample 里到底返回了什么，看这个文件。

最值得看：

- [CardiacCineENSUREDataset.__init__()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/cardiac_cine_dataset.py:221)
- [_build_samples()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/cardiac_cine_dataset.py:268)
- [_read_preproc_values()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/cardiac_cine_dataset.py:380)
- [__getitem__()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/cardiac_cine_dataset.py:436)

为什么值得看：

- `__init__` 说明可配置项
- `_build_samples` 说明时间窗怎么建
- `_read_preproc_values` 说明 sidecar 读哪些字段
- `__getitem__` 是最终 sample 的真实出口

#### `scripts/run_phase_c_sanity.py`

你如果想快速验证 C1-C4 是怎么验收的，看这个文件。

最值得看：

- [run_c1()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_c_sanity.py:75)
- [run_c2()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_c_sanity.py:101)
- [run_c3()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_c_sanity.py:136)
- [run_c4()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_c_sanity.py:190)
- [evaluate_acceptance()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_c_sanity.py:225)

为什么值得看：

- 这是“代码怎么被证明能用”的地方
- 比起读底层实现，很多时候看验收脚本更快理解系统行为

#### `scripts/train_true_ensure.py`

你如果想知道 true ENSURE 在训练时是怎么真正接到 dataset / model / loss 上的，看这个文件。

最值得看：

- [_compute_train_loss()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py:116)
- [train_one_epoch()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py:137)
- [validate()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py:180)
- [run_experiment()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py:221)

为什么值得看：

- 这是 D4 的主入口
- 训练阶段故意让 `train_dataset.return_target = False`
- 验证阶段允许用 `target_rss` 计算 NMSE，同时也能额外记录 ENSURE risk

#### `scripts/train_supervised_baseline.py`

你如果想看 D2 的 supervised baseline 是怎么搭出来的，看这个文件。

最值得看：

- [_forward_supervised_loss()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_supervised_baseline.py:106)
- [train_one_epoch()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_supervised_baseline.py:120)
- [validate()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_supervised_baseline.py:155)
- [run_experiment()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_supervised_baseline.py:185)

为什么值得看：

- 这是 D2 的主入口
- 默认走全帧输出监督，但 `frame_mode` 支持切到只监督中心帧
- 训练骨架和 true ENSURE 尽量保持一致，便于对比

### 第三优先级

#### `datasets/preprocess_raw_cine.py`

作用：

- 为每个 volume 写 `.preproc.h5` sidecar
- 保存 maps、norm_scale、noise_sigma2

建议重点看：

- [process_file()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/preprocess_raw_cine.py:186)
- [compute_norm_scale()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/preprocess_raw_cine.py:95)
- [compute_noise_stats()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/preprocess_raw_cine.py:114)
- [estimate_maps_rss()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/preprocess_raw_cine.py:40)
- [estimate_maps_bart()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/preprocess_raw_cine.py:83)

#### `datasets/precompute_density_stats.py`

作用：

- 预先离线计算采样分布的经验密度和 `inv_sqrt_density`

建议重点看：

- [bernoulli_gaussian_line_prob()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/precompute_density_stats.py:20)
- [sample_density()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/precompute_density_stats.py:67)
- [save_density_stats()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/precompute_density_stats.py:100)

#### `datasets/__init__.py`

作用：

- 提供统一导出入口

建议看一眼就够：

- [datasets/__init__.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/__init__.py:1)

#### `scripts/run_phase_d_smoke.py`

作用：

- 跑一个真正的 end-to-end Phase D smoke test
- 同时调用 supervised 和 true ENSURE 两条训练链
- 输出统一的 `smoke_summary.json`

建议重点看：

- [main()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_d_smoke.py:36)
- `common_kwargs`
- `acceptance`

#### `scripts/eval_metrics.py`

作用：

- 统一计算 Phase D 里用到的 reconstruction 指标

建议重点看：

- [mean_nmse()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/eval_metrics.py:46)
- [frame_difference_nmse_per_sample()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/eval_metrics.py:62)
- [summarize_reconstruction_metrics()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/eval_metrics.py:90)

#### `scripts/train_common.py`

作用：

- 放训练脚本共用的设备迁移、中心裁窗、日志保存、checkpoint 保存逻辑

建议重点看：

- [move_to_device()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_common.py:32)
- [maybe_center_crop_batch()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_common.py:65)
- [save_checkpoint()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_common.py:129)
- [RunningStats](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_common.py:150)

---

## 3. 每个文件的精华总结

### A. `datasets/cardiac_cine_dataset.py`

核心思想：

- 把重计算放到预处理 sidecar
- 把 sample 级别的随机 mask、欠采样、zf 在线计算

最重要输出字段：

- `kspace_fs`: `[T, C, H, W]`
- `kspace_us`: `[T, C, H, W]`
- `mask`: `[T, 1, H, W]`
- `mask_prob`: `[T, 1, H, W]`
- `maps`: `[C, H, W]`
- `zf`: `[T, 1, H, W]`
- `noise_sigma2`: scalar
- `inv_sqrt_density`: `[W]`
- `meta`: dict

最值得注意的设计：

- 不返回 `rho_ls`
  - 这是故意的，因为 `rho_ls` 属于 loss / training step
- `maps` 默认在一个 slice 内固定
- 每帧 mask 是独立 realization，但来自同一概率分布

#### `CardiacCineENSUREDataset.__init__` 接口

参数说明：

- `root`
  - 数据根目录，可以是总目录，也可以直接是 `train/val/test`
- `split`
  - 可选，指定子目录
- `preproc_root`
  - `.preproc.h5` sidecar 根目录
- `density_root`
  - density `.npz` 根目录
- `acceleration`
  - 欠采样倍率，如 `4.0`
- `sigma_mask`
  - Bernoulli-Gaussian 概率分布的高斯宽度
- `window_size`
  - 时间窗长度
- `stride`
  - 时间窗滑动步长
- `window_mode`
  - `"centered"` 或 `"sliding"`
- `center_slice_fraction`
  - 只保留中心切片比例
- `deterministic_masks`
  - 是否为同一样本固定 mask
- `mask_seed`
  - mask 随机种子
- `max_prob`
  - 中心采样概率上限
- `normalize`
  - 是否按 `norm_scale` 归一化
- `return_target`
  - 是否额外返回 `target_rss`
- `require_preproc`
  - 是否强制要求 sidecar 存在
- `cache_maps`
  - 是否缓存 maps

#### `__getitem__` 的逻辑

流程：

1. 读取 sidecar 里的 `maps / norm_scale / noise_sigma2`
2. 从源 `.h5` 读取一个时间窗的 fully-sampled k-space
3. 根据 density stats 或在线概率生成每帧 mask
4. 得到 `kspace_us`
5. 计算 `zf = A^H y`
6. 打包成训练样本返回

---

### B. `datasets/precompute_density_stats.py`

核心思想：

- 先离线算出每种 `(H, W, R, sigma_mask)` 配置下的经验采样密度
- 训练时直接用 `inv_sqrt_density`

#### `bernoulli_gaussian_line_prob`

输入：

- `width`
- `acceleration`
- `sigma_mask`
- `max_prob`
- `tol`
- `max_iter`

输出：

- `line_prob: [W]`

作用：

- 通过二分搜索找到满足平均采样率约等于 `1/R` 的高斯型 Bernoulli 概率

#### `sample_density`

输入：

- `line_prob`
- `num_samples`
- `seed`
- `density_floor`

输出：

- `empirical_density: [W]`
- `inv_sqrt_density: [W]`

#### `save_density_stats`

输入：

- `output_dir`
- `shape=(H,W)`
- `acceleration`
- `sigma_mask`
- `num_samples`
- `seed`
- `max_prob`
- `density_floor`
- `overwrite`

输出：

- 返回写出的 `.npz` 路径

保存内容：

- `line_prob`
- `empirical_density`
- `inv_sqrt_density`
- 以及配置元数据

---

### C. `datasets/preprocess_raw_cine.py`

核心思想：

- 原始 `.h5` 不改
- 额外写一个 sidecar 文件，专门存预处理产物

#### `process_file`

这是预处理阶段最重要的函数。

输入：

- `source_path`
- `output_path`
- `map_method`
  - `"rss"` / `"bart"` / `"none"`
- `norm_source`
  - `"reconstruction_rss"` / `"rss_from_kspace"`
- `norm_percentile`
- `corner_fraction`
- `center_slice_fraction`
- `compression`
- `overwrite`
- `bart_fn`
- `bart_ecalib_crop`
- `bart_ecalib_cmd`

输出：

- 写出一个 `.preproc.h5`

sidecar 里最关键的字段：

- `maps`
- `norm_scale`
- `noise_sigma2`
- `noise_sigma2_raw`
- `noise_var_real`
- `noise_var_imag`
- `processed_slice_mask`

#### `compute_norm_scale`

作用：

- 给每个 slice 生成归一化尺度

默认：

- 优先用 `reconstruction_rss`
- 否则从 `kspace -> ifft -> RSS` 算

#### `compute_noise_stats`

作用：

- 用 k-space 四角估计噪声

输出：

- `sigma2_complex`
- `var_real`
- `var_imag`

#### `estimate_maps_rss` / `estimate_maps_bart`

作用：

- 估计静态 sensitivity maps

建议：

- 先用 `rss`
- 如果后面追求更高质量 map，再切 `bart`

---

### D. `ops/mri_ops.py`

核心思想：

- 把单帧算子和动态算子分开
- 把 shape 规范化逻辑集中处理

#### 单帧层

关键函数：

- `sense_forward(image, maps, mask)`
- `sense_adjoint(kspace, maps, mask)`
- `sense_normal(image, maps, mask)`

接口：

- `image`: `[N,1,H,W]` 或 `[N,H,W]`
- `kspace`: `[N,C,H,W]`
- `maps`: `[C,H,W]` 或 `[N,C,H,W]`
- `mask`: `[N,1,H,W]` 或可广播形式

#### 动态层

关键函数：

- `dynamic_a_forward(image, maps, mask)`
- `dynamic_a_adjoint(kspace, maps, mask)`
- `dynamic_a_normal(image, maps, mask)`

接口：

- 动态 image:
  - `[T,1,H,W]`
  - `[T,H,W]`
  - `[B,T,1,H,W]`
- 动态 kspace:
  - `[T,C,H,W]`
  - `[B,T,C,H,W]`
- maps:
  - `[C,H,W]` 或 `[B,C,H,W]`
- mask:
  - `[T,1,H,W]` 或 `[B,T,1,H,W]`

#### density 加权相关

关键函数：

- `apply_density_weight`
- `sense_weighted_forward`
- `sense_weighted_adjoint`
- `sense_weighted_normal`
- `dynamic_weighted_forward`
- `dynamic_weighted_adjoint`
- `dynamic_weighted_normal`

作用：

- 把 `inv_sqrt_density` 应到 k-space 上

#### `full_sense_pinv`

作用：

- 生成 fully-sampled 参考重建
- 主要用于 sanity check，不是训练主干

---

### E. `ops/cg_solver.py`

核心思想：

- 所有需要“解线性系统”的地方，都走统一的复数 CG

#### `complex_conjugate_gradient`

输入：

- `operator`
  - 一个函数，表示线性算子
- `rhs`
  - 右端项
- `x0`
  - 初值，可选
- `max_iter`
- `tol`

输出：

- `x`
- `CGInfo`

`CGInfo` 包含：

- `iterations`
- `converged`
- `residual_norms`

#### `solve_rho_ls`

对应 C2。

输入：

- `kspace`
- `maps`
- `mask`
- `l2lam`
- `max_iter`
- `tol`
- `x0`

输出：

- `rho_ls`
- `CGInfo`

数值含义：

- 解 `(A^H A + λI)x = A^H y`

#### `solve_weighted_projection`

对应 C3。

输入：

- `error`
- `maps`
- `mask`
- `density_weight`
- `l2lam`
- `max_iter`
- `tol`
- `x0`

输出：

- `projected_error`
- `CGInfo`

数值含义：

- 解 Eq. (29) 对应的加权投影问题

特别说明：

- 默认 `x0 = 0`
- 这是为了让 C3 的数值行为更稳定，也更容易看出加权效果

---

### F. `losses/ensure_loss.py`

这是训练真正会直接用到的文件。

#### `ensure_data_term`

输入：

- `prediction`
- `rho_ls`
- `maps`
- `mask`
- `density_weight`
- `l2lam`
- `max_iter`
- `tol`

输出：

- `projected_error`
- `frame_energy`
- `data_term`
- `projection_info`

数值流程：

1. `error = prediction - rho_ls`
2. 调 `solve_weighted_projection`
3. 对 `R_s e` 取能量

#### `estimate_divergence_mc`

输入：

- `model_fn`
- `inputs`
- `eps`
- `num_mc_samples`
- `post_fn`

输出：

- `divergence_per_sample`
- `divergence`
- `eps`

数值流程：

1. 采复高斯噪声方向
2. 算 `f(u + eps n) - f(u)`
3. 做 Hutchinson 型有限差分近似

关键设计：

- `eps=None` 时自动按输入 RMS 设定
- 默认一个 batch 内按 sample 平均

#### `compute_true_ensure_loss`

这是总入口。

输入：

- `model_fn`
  - 网络或任意可调用重建器
- `zf_input`
  - 网络输入，通常是 `A^H y`
- `kspace`
- `maps`
- `mask`
- `noise_sigma2`
- `density_weight`
- `cg_l2lam`
- `cg_max_iter`
- `cg_tol`
- `divergence_eps`
- `divergence_mc_samples`
- `divergence_post_fn`

输出：

- `loss`
- `data_term`
- `div_term`
- `risk_proxy`
- `prediction`
- `rho_ls`
- `projected_error`
- `rho_info`
- `projection_info`
- `frame_energy`
- `divergence_per_sample`
- `divergence_eps`

数值流程：

1. `prediction = model_fn(zf_input)`
2. `rho_ls = solve_rho_ls(...)`
3. `data_term = ensure_data_term(...)`
4. `div_term = estimate_divergence_mc(...)`
5. `loss = data_term + 2 sigma^2 div_term`

---

### G. `scripts/run_phase_c_sanity.py`

这是你最快理解“代码是否真的工作”的文件。

#### `run_c1`

看什么：

- 是否满足伴随性
- 单帧和多帧 shape 是否一致

#### `run_c2`

看什么：

- `rho_ls` 的 CG 残差是否下降
- fully-sampled 时是否接近参考重建

#### `run_c3`

看什么：

- `R_s e` 是否有限
- `e=0` 时是否为 0
- density weight 去掉后是否真的改变结果

#### `run_c4`

看什么：

- 小线性模型上的 divergence 是否接近解析 trace

#### `evaluate_acceptance`

作用：

- 定义 C1-C4 的通过阈值

这部分非常值得你看，因为它本质上就是“作者自己认为什么算通过”。

---

### H. `models/temporal_normunet.py`

这是 Phase D 里最核心的新模型文件。

#### `NormUnet`

作用：

- 保留 MoDL 风格的复数 U-Net 接口
- 负责 `complex <-> channel` 的重排、归一化、padding、unpadding

最值得注意的设计：

- 输入是 `[B, C, H, W, 2]`
- 实部和虚部分开做 group-wise normalization
- padding 对齐到 16 的倍数，保证 U-Net pooling 稳定

#### `TemporalNormUnet`

作用：

- 接收动态时间窗输入
- 把时间维折到通道维
- 用一个共享的 `NormUnet` 对整个窗口联合重建

输入支持：

- 复数：
  - `[B, T, 1, H, W]`
  - `[T, 1, H, W]`
  - `[B, T, H, W]`
  - `[T, H, W]`
- 实数尾维：
  - `[B, T, H, W, 2]`
  - `[T, H, W, 2]`

输出模式：

- `output_mode="all_frames"`
  - 返回整个时间窗
- `output_mode="center_frame"`
  - 只返回中间帧

当前训练脚本为什么默认用 `all_frames`：

- `compute_true_ensure_loss()` 现在是逐窗口、逐帧定义的
- `rho_ls / R_s e / divergence` 都按全帧 batch 来算更自洽
- 这样不会出现“网络只回中心帧，但 loss 还假定全窗口”的接口错位

如果你以后真想切中心帧实验，推荐优先改：

- `TemporalNormUnet(output_mode="center_frame")`
- `train_supervised_baseline.py` 里的 `frame_mode`
- `train_true_ensure.py` 里与 `rho_ls` 和 `mask` 对应的时间维约束

---

### I. `scripts/eval_metrics.py`

这个文件的作用是把“重建结果如何评估”从训练脚本里拆出来。

#### `to_magnitude`

作用：

- 不管输入是复数 tensor、最后一维是 `2` 的实数表示，还是普通实数 tensor，都统一转成幅值图

#### `ensure_bthw`

作用：

- 统一把张量整理成 `[B, T, H, W]`

这个函数很关键，因为：

- 训练脚本里的 supervised 和 true ENSURE 最终都要在这里对齐 shape
- 如果后面你把网络切成中心帧输出，最先需要检查的就是这里是否仍然满足预期

#### `summarize_reconstruction_metrics`

默认返回：

- `nmse`
- `nrmse`
- `psnr`
- `frame_diff_nmse`

可选：

- `ssim`

这里最值得注意的是 `frame_diff_nmse`：

- 它是当前文档里最接近 temporal fidelity 的轻量指标
- 比只看单帧 NMSE 更能暴露时间闪烁问题

---

### J. `scripts/train_common.py`

这个文件是训练脚本的胶水层。

#### `move_to_device`

作用：

- 递归把 batch 里的 tensor 全搬到 `cpu/cuda`

#### `maybe_center_crop_batch`

作用：

- 对真实 cardiac pilot 数据做中心裁窗
- 同时保证 `zf / target_rss / maps / mask / density` 这些字段一起裁，避免 shape 不一致

为什么 Phase D smoke test 里要有这个逻辑：

- 10.2 的 smoke test 目标是验证链路，而不是拼最大分辨率算力
- 先裁到较小窗口，可以更快验证 forward/loss/backward 是否闭环

#### `save_checkpoint` / `save_json`

作用：

- 把训练脚本里的持久化逻辑统一起来

#### `RunningStats`

作用：

- 累积训练和验证中的标量指标

这里有一个你值得注意的小点：

- `train_true_ensure.py` 的验证阶段现在把重建指标和 risk 指标分开累积
- 这是为了避免一次 val batch 既记 metrics 又记 risk 时把 `num_batches` 重复计数

---

### K. `scripts/train_supervised_baseline.py`

这是 Phase D2 的入口。

核心思想：

- 训练输入仍然是 dataset 返回的 `zf`
- 监督目标使用 `target_rss`
- loss 先用最直接的 `NMSE(pred, target)` 搭一个稳定上限 baseline

#### `_forward_supervised_loss`

流程：

1. `prediction = model(zf)`
2. 根据 `frame_mode` 选择全帧或中心帧
3. 和 `target_rss` 计算 mean NMSE

#### `train_one_epoch`

作用：

- 执行一轮 supervised 训练
- 记录 `observed_target_in_train`

这个字段为什么保留：

- 对 supervised 它应该是 `true`
- 和 true ENSURE 脚本放在一起看时，你能立刻确认哪条链路训练时真的用了 GT

#### `validate`

作用：

- 统一调用 `summarize_reconstruction_metrics`

#### `run_experiment`

这是最重要的入口。

它负责：

- 建 dataset / dataloader
- 建 `TemporalNormUnet`
- 建 optimizer
- 跑 epoch 循环
- 写 `config.json / history.json / best.pt / summary.json`

如果你以后要把 D2 改成“加 DC 的 supervised baseline”，大概率就是从这里开始改。

---

### L. `scripts/train_true_ensure.py`

这是 Phase D4 的入口，也是 Phase D 里最值得认真读的文件之一。

核心思想：

- 训练集 `return_target=False`
- 验证集 `return_target=True`
- 训练 loss 完全来自 `compute_true_ensure_loss()`
- 验证阶段既可以看 NMSE，也可以额外看 ENSURE risk

#### `_compute_train_loss`

作用：

- 把 `zf / kspace_us / maps / mask / noise_sigma2 / inv_sqrt_density` 喂给 `compute_true_ensure_loss`

你如果只想确认“训练时到底有没有 GT 参与 loss”，先看这里就够了。

#### `train_one_epoch`

流程：

1. 搬运 batch 到设备
2. 可选中心裁窗
3. 调 `_compute_train_loss`
4. 反传
5. 记录 `loss / data_term / div_term / risk_proxy`

最值得注意：

- 这里虽然会检查 `observed_target_in_train`，但 train dataset 默认不返回 `target_rss`
- 这是 smoke test 中判断 “GT-free train step” 的关键依据

#### `validate`

流程：

1. 用 `summarize_reconstruction_metrics` 算重建指标
2. 如果 `compute_val_risk=True`，再额外算一遍 ENSURE risk

为什么验证允许用 GT：

- 因为“无监督训练”只要求训练阶段不使用 label
- 用 GT 做验证是完全合理的，也是现在脚本的默认设计

#### `run_experiment`

它负责的事情和 supervised 类似，但有两个重要区别：

1. train/val dataset 的 `return_target` 不同
2. history 里会额外保存 `train_data_term / train_div_term / train_risk_proxy`

这个文件现在已经是你做真正 cardiac ENSURE 无监督训练的直接入口。

---

### M. `scripts/run_phase_d_smoke.py`

这是 10.2 集成测试在代码里的具体实现。

#### `common_kwargs`

这里定义了一套“只为 smoke test 服务”的小配置：

- `epochs = 1`
- `batch_size = 1`
- `max_train_steps = 1`
- `max_val_batches = 1`
- `crop_height = 64`
- `crop_width = 96`
- `chans = 4`
- `num_pools = 1`

为什么这样设计：

- 保证真链路都走一遍
- 又避免在 smoke test 上花太久算力

#### `acceptance`

当前检查：

- supervised forward/backward 是否成功
- supervised val NMSE 是否有限
- true ENSURE forward/backward 是否成功
- true ENSURE val NMSE 是否有限
- true ENSURE 训练步是否没有 GT

#### `smoke_summary.json`

最终会输出：

- supervised summary
- true ENSURE summary
- acceptance 布尔结果

这就是你现在最快复核 D2/D4 是否已经打通的文件。

---

## 4. 如果你时间只有 10 分钟，怎么读

### 只想抓训练主干

按这个顺序看：

1. [compute_true_ensure_loss()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/losses/ensure_loss.py:88)
2. [solve_rho_ls()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/ops/cg_solver.py:86)
3. [solve_weighted_projection()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/ops/cg_solver.py:113)
4. [TemporalNormUnet.forward()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/models/temporal_normunet.py:264)
5. [_compute_train_loss()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py:116)
6. [train_one_epoch()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py:137)
7. [validate()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py:180)
8. [CardiacCineENSUREDataset.__getitem__()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/cardiac_cine_dataset.py:436)

### 只想抓数据主干

按这个顺序看：

1. [process_file()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/preprocess_raw_cine.py:186)
2. [save_density_stats()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/precompute_density_stats.py:100)
3. [CardiacCineENSUREDataset.__init__()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/cardiac_cine_dataset.py:221)
4. [CardiacCineENSUREDataset.__getitem__()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/datasets/cardiac_cine_dataset.py:436)

### 只想抓验收逻辑

按这个顺序看：

1. [run_c1()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_c_sanity.py:75)
2. [run_c2()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_c_sanity.py:101)
3. [run_c3()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_c_sanity.py:136)
4. [run_c4()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_c_sanity.py:190)
5. [evaluate_acceptance()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_c_sanity.py:225)

### 只想抓 Phase D 主干

按这个顺序看：

1. [TemporalNormUnet()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/models/temporal_normunet.py:190)
2. [_forward_supervised_loss()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_supervised_baseline.py:106)
3. [_compute_train_loss()](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py:116)
4. [run_experiment() in supervised](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_supervised_baseline.py:185)
5. [run_experiment() in true ENSURE](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py:221)
6. [main() in smoke test](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/run_phase_d_smoke.py:36)

---

## 5. 目前代码中的“主入口”有哪些

训练/调用时你最可能直接用到的入口：

- `CardiacCineENSUREDataset`
- `TemporalNormUnet`
- `solve_rho_ls`
- `solve_weighted_projection`
- `estimate_divergence_mc`
- `compute_true_ensure_loss`
- `scripts/train_supervised_baseline.py`
- `scripts/train_true_ensure.py`

离线预处理时最可能直接用到的入口：

- `process_file`
- `save_density_stats`

验收时最可能直接用到的入口：

- `scripts/run_phase_c_sanity.py`
- `scripts/run_phase_d_smoke.py`

---

## 6. 我建议你实际 review 的最小集合

如果你想最高效 review，我建议只认真看这 12 个函数：

1. `compute_true_ensure_loss`
2. `ensure_data_term`
3. `estimate_divergence_mc`
4. `solve_rho_ls`
5. `solve_weighted_projection`
6. `TemporalNormUnet.forward`
7. `_compute_train_loss`
8. `_forward_supervised_loss`
9. `dynamic_a_forward`
10. `dynamic_a_adjoint`
11. `CardiacCineENSUREDataset.__getitem__`
12. `maybe_center_crop_batch`

这 12 个函数基本就覆盖了：

- 网络输入是什么
- `rho_ls` 怎么算
- `R_s e` 怎么算
- divergence 怎么算
- loss 怎么拼
- supervised baseline 怎么训练
- true ENSURE 怎么训练
- smoke test 为什么能快速闭环

如果你还有额外精力，再看：

- `process_file`
- `save_density_stats`
- `evaluate_acceptance`
- `summarize_reconstruction_metrics`
- `run_phase_d_smoke.main`

---

## 7. 一句话版总结

这套代码现在的主干是：

- `preprocess_raw_cine.py` 负责离线生成 `maps / norm_scale / noise_sigma2`
- `precompute_density_stats.py` 负责离线生成 `inv_sqrt_density`
- `cardiac_cine_dataset.py` 负责在线拼出训练 sample
- `mri_ops.py` 负责动态 `A / A^H / A^H A`
- `cg_solver.py` 负责 `rho_ls` 和 `R_s e`
- `ensure_loss.py` 负责 true ENSURE 的 loss 组装
- `temporal_normunet.py` 负责 Phase D 的动态 backbone
- `train_supervised_baseline.py` 负责 D2 supervised baseline
- `train_true_ensure.py` 负责 D4 true ENSURE 训练
- `eval_metrics.py` 负责验证指标
- `train_common.py` 负责训练脚本共用胶水逻辑
- `run_phase_c_sanity.py` 负责验收 C1-C4
- `run_phase_d_smoke.py` 负责 D2/D4 的 end-to-end smoke test

---

## 8. 增量更新：CNN + DC + Unroll

这一节是本次修改的增量说明，重点对应用户提出的 3 个要求：

1. 把原来的单次 `TemporalNormUnet` 改成更贴近论文 Fig.1 / Section D 的级联结构
2. 补上最基础可用的 data-consistency (DC) 项
3. 让训练脚本可以显式配置 unroll 步数

### 8.1 现在的 `TemporalNormUnet` 不再是“单次网络前向”

当前模型文件：

- [models/temporal_normunet.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/models/temporal_normunet.py)

现在的主干逻辑是：

1. 输入仍然是时间窗 `zf`
2. 时间维仍然折到通道维，不改原来数据链路
3. 每个 cascade 先走一个 paper-style CNN denoiser
4. 然后走一个 soft DC
5. 重复 `num_unrolls` 次
6. 最后再按 `output_mode` 决定输出全帧还是中心帧

也就是说，模型从原来的：

- `input -> network -> output`

变成了：

- `input -> CNN -> DC -> CNN -> DC -> ... -> output`

### 8.2 CNN 结构如何对应论文图

论文图里可以理解成单个 cascade 内部是一个浅层 CNN，通道形状接近：

- `input -> 2 -> 64 -> 64 -> 64 -> 64 -> 2 -> output`

但我们这里保留了原来的 temporal folding：

- 单帧 complex 的 `2` 个通道，会在时间窗里变成 `2 * T`

所以在本项目里，等价实现是：

- `2T -> 64 -> 64 -> 64 -> 64 -> 2T`

这样做的原因很简单：

- 保住当前能正常训练的数据接口
- 不把时间窗联合建模能力直接砍掉
- 同时把每个 cascade 的 CNN 宽度改成更接近论文里的 64-channel 设计

这里有一个需要特别记住的点：

- `--chans` 现在推荐直接用 `64`
- `--num-pools` 在当前实现里不再表示 U-Net pooling 深度，而是表示每个 cascade 里重复多少个 `64-channel` 卷积层
- 如果你想完全贴近论文图，推荐 `--num-pools 4`

### 8.3 DC 项是怎么实现的

当前 DC 用的是最基础、也最稳妥的 soft DC 形式：

```text
x_cnn = CNN(x_k)
x_{k+1} = x_cnn - lambda_k * (A^H A x_cnn - x_0)
```

这里：

- `x_0` 是输入的 zero-filled image，也就是 `batch["zf"]`
- `A^H A x_cnn` 通过已有的动态 MRI 算子计算
- `lambda_k` 是每个 cascade 一个可学习的 `dc_weight`

实现位置：

- [models/temporal_normunet.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/models/temporal_normunet.py)
- [ops/mri_ops.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/ops/mri_ops.py)

为什么这里用 `x_0` 而不是重新拼一个更复杂的 DC 版本：

- 它和 `promptmr_v2.py` 的 soft DC 思路是一致的
- 它只依赖现有数据集已经返回的 `zf / maps / mask`
- 对 true ENSURE 的 Hutchinson divergence 更友好，因为模型仍然主要是“以输入图像为自变量”

### 8.4 训练脚本现在怎么喂模型

当前训练脚本已经不是简单的：

- `prediction = model(batch["zf"])`

而是显式传入：

- `prediction = model(batch["zf"], maps=batch["maps"], mask=batch["mask"])`

对应文件：

- [scripts/train_supervised_baseline.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_supervised_baseline.py)
- [scripts/train_true_ensure.py](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py)

true ENSURE 训练里还额外做了一层 closure：

- `compute_true_ensure_loss()` 里使用的 `model_fn`
- 现在会把 `maps / mask` 一起 capture 进去
- 这样 Monte-Carlo divergence 估计时，模型 forward 仍然能走完整的 `CNN -> DC -> unroll`

### 8.5 新增和推荐的参数

这次最关键的新参数是：

- `--num-unrolls`

推荐值：

- `--num-unrolls 3`

这是为了和论文里提到的 `three unrolling steps` 对齐。

当前推荐的最小论文对齐配置是：

- `--chans 64`
- `--num-pools 4`
- `--num-unrolls 3`

如果你只是想关闭 DC 看纯 CNN 行为，当前代码层面仍然兼容只调用：

- `model(zf)`

只是这种调用不会启用 DC，因为没有传 `maps / mask`。

### 8.6 推荐训练命令

监督训练：

```bash
conda run -n cmr-blackwell python cardiac_ensure/scripts/train_supervised_baseline.py \
  --data-root /home/dengyipin/CMR2025/cmr001_pilot \
  --preproc-root /home/dengyipin/CMR2025/cmr001_pilot/preproc_c \
  --density-root /home/dengyipin/CMR2025/cmr001_pilot/density_stats \
  --train-split train \
  --val-split val \
  --output-dir /home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/outputs/supervised_baseline_r4_w5_unroll3 \
  --acceleration 4.0 \
  --sigma-mask 0.18 \
  --window-size 5 \
  --stride 1 \
  --window-mode centered \
  --frame-mode all \
  --center-slice-fraction 1.0 \
  --epochs 10 \
  --batch-size 1 \
  --num-workers 4 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --chans 64 \
  --num-pools 4 \
  --num-unrolls 3 \
  --drop-prob 0.0 \
  --device cuda:0 \
  --seed 7 \
  --save-every 1 \
  --val-deterministic-masks
```

true ENSURE 训练：

```bash
conda run -n cmr-blackwell python cardiac_ensure/scripts/train_true_ensure.py \
  --data-root /home/dengyipin/CMR2025/cmr001_pilot \
  --preproc-root /home/dengyipin/CMR2025/cmr001_pilot/preproc_c \
  --density-root /home/dengyipin/CMR2025/cmr001_pilot/density_stats \
  --train-split train \
  --val-split val \
  --output-dir /home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/outputs/true_ensure_r4_w5_unroll3 \
  --acceleration 4.0 \
  --sigma-mask 0.18 \
  --window-size 5 \
  --stride 1 \
  --window-mode centered \
  --center-slice-fraction 1.0 \
  --epochs 10 \
  --batch-size 1 \
  --num-workers 4 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --chans 64 \
  --num-pools 4 \
  --num-unrolls 3 \
  --drop-prob 0.0 \
  --device cuda:0 \
  --seed 7 \
  --save-every 1 \
  --val-deterministic-masks \
  --cg-l2lam 1e-6 \
  --cg-max-iter 25 \
  --cg-tol 1e-6 \
  --divergence-mc-samples 1
```

### 8.7 本次实际验证结果

本次验证使用环境：

- `conda` 环境：`cmr-blackwell`

真实数据路径：

- `/home/dengyipin/CMR2025/cmr001_pilot`

已经完成的验证：

- `run_phase_d_smoke.py` 在真实数据上跑通
- supervised 分支 forward/backward 正常
- true ENSURE 分支 forward/backward 正常
- true ENSURE 训练阶段没有偷看 `target_rss`
- smoke summary 已落盘到：
  [phase_d_smoke_unroll_dc](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/outputs/phase_d_smoke_unroll_dc)

smoke 关键结果：

- supervised smoke: `train_loss=0.1173`, `val_nmse=0.2652`
- true ENSURE smoke: `train_loss=10929.08`, `val_nmse=0.2673`

额外短跑结果：

- supervised 3-step 短跑输出目录：
  [supervised_short_unroll_dc](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/outputs/supervised_short_unroll_dc)
- true ENSURE 2-step 短跑输出目录：
  [true_ensure_short_unroll_dc](/home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/outputs/true_ensure_short_unroll_dc)

短跑里可以直接看到：

- supervised step loss: `0.0951 -> 0.2847 -> 0.1238`
- true ENSURE 前两步平均 loss 已降到 `7883.44`
- true ENSURE 第一 step 的即时 loss 约为 `10497.80`

需要注意：

- `edm` 环境里的旧版 PyTorch 对这台 Blackwell GPU 不兼容，所以这次验证统一切到了 `cmr-blackwell`
- 如果后面你继续正式训练，也建议直接使用 `cmr-blackwell`
