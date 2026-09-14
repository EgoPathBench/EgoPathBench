# Source Paths

## 发布目录

- publish root:
  - `/home/zhaoyang/Projects/navbench3d-code/published/navbench3d_release_20260417_rebuild5`

## 当前 canonical 数据根

- canonical rebuild root:
  - `/tmp/navbench3d_human_aligned_promptcount_20260417_surfacefix_rebuild5`
- canonical release manifest:
  - `/tmp/navbench3d_human_aligned_promptcount_20260417_surfacefix_rebuild5/release/release_manifest.json`
- canonical summary:
  - `/tmp/navbench3d_human_aligned_promptcount_20260417_surfacefix_rebuild5/summary.json`

## 上游 base release

- base release root:
  - `/mnt/data/zhaoyang/navbench3d-data/releases/scene_v3_full_8219_retry3_20260326_152454`

## 场景根目录

- scenes root:
  - `/mnt/data/zhaoyang/navbench3d-data/scenes_v3_4src_relaxed_refilter_post_objaverse_20260325_124913`

## 上游 c source

- c source dir:
  - `/tmp/navbench3d_human_aligned_promptcount_20260416_cuefix/release`

## 审计结果来源

- strict audit root:
  - `/tmp/navbench3d_human_aligned_promptcount_20260417_surfacefix_rebuild5/results/full_human_visual_audit_20260417_v5`

## 备注

- 当前发布包展示的是 rebuild5 的唯一正式版本。
- 当前 publish root 不是对 canonical rebuild5 的逐字节镜像。
- 发布包 benchmark 相对 canonical rebuild5 额外删除了 `1` 条在最终直线段协议下不合法的 `b2` 题。
- 发布包 `train / val` 也相对 canonical rebuild5 做了本地重切分：
  - canonical rebuild5 原始 split 约为 `train = 18753 / val = 14444 / benchmark = 1113`
  - 当前 publish root 为 `train = 31852 / val = 1345 / benchmark = 1111`
  - 其中 `val` 使用的是发布侧平衡开发集，而不是 canonical 的原始大验证集。
- 当前发布包的权威统计与 split 信息应以：
  - `data/release/release_manifest.json`
  为准。
- benchmark 评测补丁单独放在 `data/benchmark_evalfix/`，避免与 `data/release/benchmark/` 混淆。
