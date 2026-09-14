# Data Construction Order

下面记录的是这次最终发布包对应的实际数据构造顺序。为了避免协议混淆，分成三段写。

## A. Base Release 主链

输入：

- raw scenes:
  - `/mnt/data/zhaoyang/navbench3d-data/scenes_v3_4src_relaxed_refilter_post_objaverse_20260325_124913`

顺序：

1. `scripts/build_next_dataset_splits.py`
   - 从全量 scene 生成基础 split manifest。
2. `scripts/compose_scene.py`
   - 把场景与资产组合成可渲染场景。
3. `scripts/render_scenes.py`
   - 渲染每个视角，产出 `renders/<scene>/view_<id>/next_view_bundle_v3.json`。
4. `scripts/build_task_outputs.py`
   - 产出 `rgb_overlay_*` 与 `visible_waypoints_*.json`。
5. `scripts/build_navigation_gt_next.py`
   - 从 `task_outputs` 构造全量 GT。
   - 同时写入 fixed GT carrier、人类视觉唯一性元数据、cue 解析结果等。
6. `scripts/build_vqa_questions_next.py`
   - 从 GT 重建正式 VQA。
7. `scripts/build_next_release_datasets.py`
   - 从全量 GT/VQA 划分 `train / val / benchmark`。
   - 当前发布包使用的是发布侧重平衡后的 `val`：
     - benchmark 仍按 hard candidate scene mining 选；
     - val 改成小型 `balanced_navigation_dev_split` 开发集；
     - 如果候选池太小，脚本仍可回退到旧的低难度小场景策略。
8. `scripts/build_truth_aligned_release_manifest.py`
   - 由第 7 步内部消费，生成正式 release layout / manifest。

主链入口脚本：

- `scripts/run_next_full_release.sh`

## B. Human-Aligned Rebuild5 重建链

输入：

- base release:
  - `/mnt/data/zhaoyang/navbench3d-data/releases/scene_v3_full_8219_retry3_20260326_152454`
- legacy c source:
  - `/tmp/navbench3d_human_aligned_promptcount_20260416_cuefix/release`

顺序：

9. `scripts/rebuild_human_aligned_release_v2.py`
   - 以 base release 为输入，冻结既有 GT identity/path/route semantics。
   - 只重建 prompt-side human-visible uniqueness。
   - 产出当前 canonical `source_gt/`、`source_vqa/`、`release/`、`summary.json`。
10. `scripts/audit_human_visual_alignment.py`
   - 对 explicit `a2/b2` 与 implicit `c` 做整包 human-visual 对齐审计。
11. `scripts/report_next_dataset_stats.py`
   - 汇总发布包统计。
12. `scripts/visualize_next_audit.py`
   - 生成人工抽检所需的可视化核验图。

第 9 步内部实际依赖的上游模块：

- `scripts/build_hview_confusable_universe.py`
- `scripts/build_scene_relation_graph.py`
- `scripts/build_view_relation_projection.py`
- `scripts/cue_synthesis.py`
- `scripts/resolve_prompt_under_hview.py`
- `scripts/build_c_main_vnext_pilot_from_release.py`
- `scripts/build_navigation_gt_next.py`
- `scripts/build_vqa_questions_next.py`
- `scripts/build_next_release_datasets.py`
- `scripts/h_view_utils.py`
- `scripts/prompt_truth_contract.py`
- `scripts/tier_policy.py`

## C. Benchmark Evalfix 补包链

输入：

- final release benchmark:
  - `data/release/benchmark`

顺序：

13. `scripts/build_benchmark_evalfix_package.py`
   - 为 benchmark routing tasks 生成评测补丁。
   - 产出：
     - `direct_pairs_ref`
     - `acceptable_goal_ids`
     - `goal_ring_*`
     - `reference_path_display_ids`
14. 本次发布打包时，对第 13 步采用了 `16` 个 scene shard 的并行调度。
   - 目的只是缩短 occupancy grid 重建时间。
   - 核心 builder 没变，仍然是 `scripts/build_benchmark_evalfix_package.py`。

第 13 步内部依赖：

- `scripts/build_navigation_gt.py`

## 最终产物关系

- `data/release/`
  - 正式数据集主包。
  - 发布包版本相对 canonical rebuild5 额外剔除了 `1` 条在最终直线段协议下不合法的 `b2` benchmark 坏题。
  - 同时把 canonical rebuild5 的原始大 `val` 改成了发布侧平衡 `val`。
- `data/benchmark_evalfix/`
  - benchmark 评测补丁包。
- `data/sidecars/`
  - human-view / cue-resolution 辅助 sidecar。
