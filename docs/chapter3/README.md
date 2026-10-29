# 第三章代码入口

从 `GenNBV/upstream` 目录运行命令。先激活独立 `.venv-gennbv` 并恢复已验证的 CUDA/Vulkan 环境；完整步骤见 [HAWP安装](HAWP_SETUP.md) 和 [实验指南](EXPERIMENTS.md)。目录整理不改变输入维度、网络层、训练预算和权重格式；服务器仍需短测确认。

## baseline 到底输入什么

本文的 baseline 指补齐图像后的三通道版本：`use_image=1, use_line=0`。

| 通道 | 原始网络输入 | 编码后的特征 | 作用 |
|---|---|---|---|
| 历史位姿 | 100×6，位置x/y/z及roll/pitch/yaw；不足历史由环境初始化/重置填充 | 256维 | 已经到过哪里、朝向哪里 |
| 占据栅格 | 20×20×20，空闲/未知/占据三态 | 256维 | 已观察空间及未确定空间 |
| 图像 | 最近两帧64×64灰度图，像素进编码器时除255 | 256维 | 当前和最近视图的外观线索 |

三项展平合计16792个数，经独立编码得到768维，再经融合层输出256维给策略头。HAWP、几何线证据不属于 baseline。

深度已经参与栅格构建，不是额外第四个独立网络分支。当前仿真实现还使用目标分割掩码筛选深度点，并依赖数据提供的场景范围/体素尺度；不能称为仅靠三张普通图片的纯视觉系统，也不能声称整个原基线完全不使用仿真先验。新增线证据模块使用当前RGB、原始深度和相机内参；背景线筛选还使用当前仿真的目标分割掩码，不读取GT覆盖栅格。

旧的官方预训练兼容模式 `use_image=0, use_line=0` 只编码位姿和栅格；它与这里的三通道 baseline 不同。

## 目录分工

```text
upstream/
├── chapter3/                   # 新增的第三章代码
│   ├── __main__.py             # python -m chapter3 统一入口
│   ├── paths.py                # 仓库根路径
│   ├── perception/
│   │   ├── hawp.py             # 冻结HAWP、token、相机接入
│   │   ├── geometry.py         # 单帧深度几何证据（纯NumPy）
│   │   └── legacy_lines.py     # 早期LSD/DeepLSD，仅可视化
│   ├── cli/
│   │   ├── train.py            # 训练启动、设备设置
│   │   ├── evaluate.py         # 模型评估、轨迹与图像保存
│   │   ├── ablate.py           # 四组同协议实验，默认短测试
│   │   ├── pair.py             # 早期HAWP与已有基线的两组对照
│   │   ├── inspect.py          # 已保存几何快照的重算检查
│   │   ├── verify_lines.py     # 线编码器梯度/HAWP验证
│   │   ├── verify_image.py     # 图像编码器验证
│   │   └── _process.py         # 两个实验入口共用的日志执行工具
│   └── tests/                  # CPU单元测试、入口路径检查
├── docs/chapter3/              # 本索引、安装、实验指南
├── gennbv/                     # 原环境与网络，接入点保留原类路径
│   ├── network/hybrid_encoder.py
│   ├── env/env_train_base.py
│   ├── wrapper/               # 拼装网络输入
│   └── train/train_gennbv.py   # PPO训练配置
├── stable_baselines3/          # 原策略与PPO实现
├── legged_gym/、rsl_rl/        # 原项目依赖
├── data_gennbv/、pretrained/   # 服务器数据与预训练模型
├── runs/                      # 训练输出，保持原路径
├── _artifacts/                # 评估输出，保持原路径
├── eval_pretrained.py          # 旧评估命令兼容入口
├── verify_image_encoder.py     # 旧图像验证命令兼容入口
└── line_detection.py           # 旧可视化命令兼容入口
```

论文根目录的 `src/state_representation/` 是旧稿对应的占位骨架，不是上述可运行代码；论文工作稿和比较脚本仍在论文仓库 `_artifacts/`。不将两套目录混合。

## 常用命令

```bash
python -m chapter3 --help
python -m chapter3 ablate --help
python -m chapter3 eval --help
python -m chapter3 verify-lines
python -m chapter3 verify-image
python -m unittest discover -s chapter3/tests -v
```

`train` 透传 GenNBV 参数，仍需安装 Isaac Gym 才能显示完整训练帮助。其余帮助无需初始化 GPU；实际网络验证和训练仍需服务器依赖。`pair`、`detect-legacy` 是保留工具，当前四组研究以 `ablate` 为主。

最小闭环（先确认0号卡可用、HAWP已安装）：

```bash
python -u -m chapter3 ablate \
  --stage smoke --seed 1 --gpu 0 --graphics 0 \
  --hawp-config ../third_party/hawp/configs/hawpv2.yaml \
  --hawp-weights ../third_party/hawp/checkpoints/hawpv2-edb9b23f.pth \
  --output _artifacts/ch3_lines_smoke_s1_01
```

## 代码同步（由用户执行）

本轮和此前HAWP接入尚未提交；先审查差异。不使用 `git add .`，避免将 `.DS_Store`、服务器备份等无关文件混入。

Mac：

```bash
cd /Users/zzc/Desktop/Graduation-thesis/GenNBV/upstream
git add chapter3 docs/chapter3 README.md \
  eval_pretrained.py verify_image_encoder.py line_detection.py \
  gennbv/env/env_train_base.py gennbv/network/hybrid_encoder.py \
  gennbv/train/train_gennbv.py \
  gennbv/wrapper/env_wrapper_gennbv_train.py gennbv/wrapper/env_wrapper_gennbv_eval.py
git diff --cached --stat
git commit -m "Organize Chapter 3 state modules, workflows and documentation"
git push origin codex/state-representation
cd ../..
git add .gitignore AGENTS.md GenNBV/upstream _artifacts/compare_gennbv_eval.py \
  _artifacts/第三章研究方案_GenNBV基线.md _artifacts/第三章_中期工作稿.md
git diff --cached --stat
git commit -m "Update Chapter 3 code index and submodule"
git push origin master
```

服务器：

```bash
cd /data/home/zzc/Graduation-thesis
git pull --ff-only origin master
cd GenNBV/upstream
git pull --ff-only origin codex/state-representation
source ../.venv-gennbv/bin/activate
python -m chapter3 --help
python -m unittest discover -s chapter3/tests -v
```

不需要重建虚拟环境；新增包由当前仓库目录直接导入。旧的根目录评估命令仍可用。未提交过的旧HAWP脚本若已手动拷贝到服务器，Git不会自动删除它们；今后使用上面的统一入口，确认没有独有改动后再人工归档。
