# InteriorGS Benchmark 推理指南

本文档介绍如何在生成的 InteriorGS VQA Benchmark 上运行模型推理。

## 目录
- [数据格式说明](#数据格式说明)
- [推理流程](#推理流程)
- [评估方法](#评估方法)
- [常见问题](#常见问题)

---

## 数据格式说明

### 生成的 Benchmark 格式

生成的 `benchmark.jsonl` 格式如下：

```json
{
  "question": "What is the estimated distance of the mirror from the camera...",
  "answer": "1.7",
  "question_type": "object_distance_to_camera",
  "question_id": "object_distance_to_camera_95",
  "image": "images/mirror_0.png",
  "image_full_path": "around/0015_840888/0015_840888/images/mirror_0.png",
  "images": ["images/view1.png", "images/view2.png", ...],  // 多视图
  "scene_id": "0015_840888",
  "pattern": "around",
  "num_views": 1,
  ...
}
```

### Spatial-Consistency-Bench 格式

SC-Bench 的 `infer.py` 期望的格式：

```json
{
  "id": "unique_id",
  "question": "Given this view:\n<image_start>[image_1]<image_end>\n\nWhat is...",
  "images": {
    "image_1": "/full/path/to/image.png",
    "image_2": "/full/path/to/image2.png"
  },
  "gt_answer": "1.7",
  ...
}
```

### 格式差异

| 字段 | 生成格式 | SC-Bench 格式 |
|------|---------|--------------|
| 图片路径 | `"image": "images/xxx.png"` | `"images": {"image_1": "/full/path/xxx.png"}` |
| 答案 | `"answer": "1.7"` | `"gt_answer": "1.7"` |
| 问题文本 | 无图片占位符 | 包含 `<image_start>[image_1]<image_end>` 占位符 |
| ID | `question_id` | `id` (唯一标识) |

---

## 推理流程

### 前置条件

1. 已生成的 benchmark 数据集位于：
   ```
   /scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix/
   ```

2. Bagel-Zebra-CoT 模型和相关环境已配置

### Step 1: 转换数据格式

使用 `convert_to_scbench_format.py` 将 benchmark 转换为 SC-Bench 兼容格式：

```bash
cd /scratch/by2593/project/sceneshift/question_gen_InteriorGS/benchmark_gen
conda activate bagel

# 转换全部数据
python convert_to_scbench_format.py \
    --input /scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix/benchmark.jsonl \
    --output /scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix/scbench_format/all_questions.jsonl \
    --dataset_root /scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix

# 可选参数
#   --max_samples 100          # 只转换前100个样本（用于测试）
#   --question_types object_size object_distance_to_camera  # 只转换特定问题类型
#   --num_views 5              # 只转换有5个视图的样本
```

转换后的文件将保存在 `scbench_format/all_questions.jsonl`。

### Step 2: 运行模型推理

使用 Spatial-Consistency-Bench 的 `infer.py` 进行推理：

```bash
cd /scratch/by2593/Spatial-Consistency-Bench
conda activate bagel

python infer.py \
    --model_path /path/to/your/model/checkpoint \
    --base_model_path /scratch/by2593/Spatial-Consistency-Bench/Bagel-Zebra-CoT \
    --input_file /scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix/scbench_format/all_questions.jsonl \
    --image_base_dir . \
    --output_file /scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix/inference_results/predictions.jsonl \
    --output_dir /scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix/inference_results
```

#### 推理参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型 checkpoint 路径 | 必填 |
| `--base_model_path` | 基础模型路径（用于加载配置） | 同 model_path |
| `--input_file` | 输入 JSONL 文件 | 必填 |
| `--image_base_dir` | 图片基础目录 | `.` |
| `--output_file` | 输出预测结果 | 必填 |
| `--enable_cot` | 启用 Chain-of-Thought | False |
| `--text_temperature` | 文本生成温度 | 0.3 |
| `--num_gpus` | 使用的 GPU 数量 | 全部可用 |

#### 使用一键脚本

也可以使用预配置的脚本：

```bash
cd /scratch/by2593/Spatial-Consistency-Bench
bash exp_scripts/interiorgs/run_interiorgs_inference.sh \
    --model_path /path/to/your/model \
    --num_samples 100  # 可选：只处理前100个样本
```

### Step 3: 评估结果

使用评估脚本计算准确率和误差：

```bash
cd /scratch/by2593/Spatial-Consistency-Bench

python exp_scripts/interiorgs/evaluate_interiorgs.py \
    --predictions /scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix/inference_results/predictions.jsonl \
    --output_report /scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix/inference_results/evaluation_report.json
```

---

## 评估方法

### 评估指标

1. **准确率 (Accuracy)**：相对误差 < 20% 视为正确
2. **平均绝对误差 (Mean Absolute Error)**
3. **平均相对误差 (Mean Relative Error)**
4. **推理成功率 (Success Rate)**：模型成功生成输出的比例

### 按问题类型的评估

评估脚本会自动按问题类型分组统计：

```
Question Type                              Total    Acc    Mean Err   Med Err
--------------------------------------------------------------------------------
object_distance_to_camera                    100  45.00%     23.5%     18.2%
object_size                                  100  38.00%     31.2%     25.1%
object_pair_distance_center                  100  42.00%     28.3%     22.4%
...
```

### 答案解析

评估脚本支持多种答案格式：
- 数值类型: `1.7`, `"1.7"`, `{'answer': 1.7}`
- 向量类型: `[1.2, 0.5, 0.8]`, `"[1.2, 0.5, 0.8]"`

---

## 问题类型说明

| 问题类型 | 答案格式 | 说明 |
|----------|---------|------|
| `object_size` | `[L, W, H]` | 物体尺寸（长、宽、高） |
| `object_distance_to_camera` | `float` | 物体到相机的距离 |
| `object_size_comparison_relative` | `float` | 两物体尺寸比例 |
| `object_size_comparison_absolute` | `float` | 给定参照物尺寸的绝对尺寸 |
| `object_pair_distance_center` | `float` | 两物体中心距离 |
| `object_pair_distance_center_w_size` | `float` | 给定物体尺寸的距离 |
| `object_pair_distance_vector` | `[x, y, z]` | 物体间的向量 |
| `object_comparison_absolute_distance` | `float` | 多物体距离比较 |
| `object_comparison_relative_distance` | `float` | 多物体距离比例 |

---

## 相关文件

```
benchmark_gen/
├── convert_to_scbench_format.py   # 格式转换脚本
├── create_benchmark.py            # 创建基准测试集
├── convert_to_multiview.py        # 多视图转换
├── analyze_dataset_statistics.py  # 数据统计分析
└── INFERENCE_GUIDE.md             # 本文档

/scratch/by2593/Spatial-Consistency-Bench/
├── infer.py                       # 主推理脚本
├── config/interiorgs/             # InteriorGS 配置
│   └── tasks_interiorgs.yaml
└── exp_scripts/interiorgs/        # InteriorGS 推理脚本
    ├── run_interiorgs_inference.sh
    └── evaluate_interiorgs.py
```

---

## 常见问题

### Q: 图片路径找不到？

A: 确保转换时使用正确的 `--dataset_root` 参数，转换脚本会将相对路径转换为绝对路径。

### Q: 推理时出现 OOM？

A: 尝试减少 `--num_gpus` 或增加 `--max_mem_per_gpu`。

### Q: 如何只测试特定问题类型？

A: 使用转换脚本的 `--question_types` 参数：
```bash
python convert_to_scbench_format.py \
    --question_types object_distance_to_camera object_size \
    ...
```

### Q: 如何测试多视图性能？

A: 使用 `--num_views` 参数筛选特定视图数量的样本：
```bash
python convert_to_scbench_format.py --num_views 5 ...
```

---

## 更新日志

- **2026-01-28**: 初始版本，支持 Bagel-Zebra-CoT 模型推理
