# Data Scripts Manifest

## Base Release 主链脚本

- `run_next_full_release.sh`
  - base release 端到端入口。
- `build_next_dataset_splits.py`
  - 全量场景切分。
- `compose_scene.py`
  - 组合场景资产。
- `render_scenes.py`
  - 渲染视角与 `next_view_bundle_v3.json`。
- `build_task_outputs.py`
  - 生成 overlay 与 `visible_waypoints_*`。
- `build_navigation_gt_next.py`
  - 生成 GT 与 promptability 元数据。
- `build_vqa_questions_next.py`
  - 从 GT 重建 VQA。
- `build_next_release_datasets.py`
  - 生成正式 `train / val / benchmark`。
  - benchmark 使用 hard-candidate scene mining。
  - val 在候选池足够时使用 `balanced_navigation_dev_split`，否则回退到旧的低难度小场景策略。
- `build_truth_aligned_release_manifest.py`
  - release manifest / package layout 支撑。

## Human-Aligned Rebuild 脚本

- `rebuild_human_aligned_release_v2.py`
  - 从 base release 重建 human-aligned source pool 与 final release。
- `build_hview_confusable_universe.py`
  - 构造 H(view) 混淆候选宇宙。
- `build_scene_relation_graph.py`
  - 从 scene/layout 生成关系图。
- `build_view_relation_projection.py`
  - 把 scene 关系投影到当前 view。
- `cue_synthesis.py`
  - 合成 cue package。
- `resolve_prompt_under_hview.py`
  - 检查 prompt 在 H(view) 下是否唯一。
- `build_c_main_vnext_pilot_from_release.py`
  - `c` family 辅助构造逻辑。

## 共享支撑模块

- `h_view_utils.py`
  - H(view) 构造与 sidecar 写出。
- `prompt_truth_contract.py`
  - fixed GT contract / prompt truth contract。
- `scene_nav_facts.py`
  - 导航事实字段定义。
- `tier_policy.py`
  - tier / reference axis 计算。
- `benchmark_policy.py`
  - benchmark 挑题与 manifest 生成策略。

## Benchmark Evalfix 脚本

- `build_benchmark_evalfix_package.py`
  - benchmark 专用评测补丁包。
- `build_navigation_gt.py`
  - evalfix builder 使用的 occupancy grid 构造依赖。

## 审计与可视化脚本

- `audit_human_visual_alignment.py`
  - 全量 human-visual 对齐审计。
- `report_next_dataset_stats.py`
  - 数据量统计。
- `visualize_next_audit.py`
  - 人工抽检可视化。

## 核心配置

- `internscene_local.yaml`
  - 渲染、导航、target、grid 等主配置。
- `truth_repair_protocol_v1.json`
  - truth repair 协议。
- `cue_repair_protocol_v1.json`
  - cue repair 协议。
- `hview_confusable_universe_v1.json`
  - H(view) 混淆宇宙协议。
- `c_intent_family_proxy_groups_v1.json`
  - `c` family proxy group 配置。
- `c_main_vnext_candidate_ontology.json`
  - `c` 目标/intent ontology。
- `c_prompt_catalog_v1.json`
  - `c` prompt catalog。
