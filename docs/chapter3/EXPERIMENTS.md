# 第三章：几何证据线状态的实施与验收

更新：2026-10-29。本地代码与合成测试完成；服务器推理、训练和收益仍待验证。
当前工作不改变 PPO、奖励、动作空间或策略头；不使用完整网格、GT 覆盖栅格或未来图像；目标筛选使用当前仿真的分割标签。

2026-10-29 更新：线通道增加目标区域筛选，使用仿真已有、也用于基线栅格构建的目标分割掩码。每条线取16个内部点，至少4个点落在目标像素或其4像素邻域才保留。原始线、unknown、evidence 三组使用同一规则；建筑与天空交界线允许保留。此规则依赖仿真目标掩码，不能直接宣称可部署到真实无人机；真实场景须替换为可运行的目标定位方法并重做实验。该更新使既有短测和旧 HAWP 权重不能充当新规则的结果。

## 方法与边界

官方冻结 HAWPv2 在当前 RGB 中检测线。前 6 项保持原格式：
`x1/W, y1/H, x2/W, y2/H, detection_score, valid`。

新增证据版每条线为 9 项，最后 3 项为 `support, weak, unknown`。前 5 项和最后 3 项进入共享 `8→64→128` MLP，第 6 项只作为 padding mask；平均池化后与三个 256 维通道拼接，`896→256→原策略头`。

`chapter3/perception/geometry.py` 沿线内部取 5 个位置，在线两侧取小窗口，拟合 `1/z=a*x_ray+b*y_ray+c`。这是针孔模型下局部平面的表达；比较两侧在同一线位置外推的深度，避免把斜面自然深度差直接算作边界。法向由 `[a,b,c]` 归一化得到，比较两侧的法向夹角。单侧无深度、过短线、越界或拟合不稳定均退回未知，不删除检测线。

以相对深度差和法向角得到启发式支持度 r；以有效样本及逆深度拟合残差得到 q。每个局部检查输出 `(q*r, q*(1-r), 1-q)`，沿线平均，三个分量之和为 1。padding 全为 0。

默认设置为未调参的初始假设：5 个采样位置；窗口尺度为短边512像素时6像素、最小3像素；拟合相对残差容差2%；深度跳变尺度3%；法向角尺度15度；有效深度0.001～50米，并进一步受相机近远裁剪约束。全部写入训练 manifest。调整后必须重新训练并在验证集选参，不能拿测试集挑最高结果。

q 不是校准概率，r 不是建筑折边的真假标签。地面、家具、遮挡边界也可能得到支持；细小构件、噪声深度及 HAWP 定位偏差可能导致误判。当前只实现单帧局部证据，没有多视图关联、法向不确定性传播或已证明的鲁棒性。

## 四组对照

| 名称 | 状态 | 目的 |
|---|---|---|
| baseline | 位姿+栅格+两帧灰度图 | 补齐图像的三通道基线，不是创新 |
| raw | baseline + 原始二维 HAWP 线 | 测试增加线分支的整体收益 |
| unknown | 同证据版9项/条，但证据恒为(0,0,1) | 与 evidence 参数量、检测器及高分辨率 RGB 相同 |
| evidence | 同9项/条，使用实际深度证据 | 主要检验深度证据编码是否有额外收益 |

四组独立从头训练，同一训练种子、PPO、动作、奖励、交互预算、环境数和评估协议。unknown 与 evidence 神经网络结构相同。raw→evidence 多192个线编码参数（3×64），因此不能只做这两组就排除容量影响。基线及 raw 本身已通过栅格使用深度，evidence 增加的是更细粒度局部深度信息的编码；后续仍需直接深度边缘、未知信息删除等消融，才能区分几何输入与编码设计的贡献。

## 第一步：同步与恢复环境

先按目录入口的命令由用户提交、推送，再在服务器拉取。HAWP 尚未安装时，先执行 `HAWP_SETUP.md` 第1～2节的官方权重下载上传及 uv 安装步骤；不换现有 torch。

```bash
cd /data/home/zzc/Graduation-thesis
git pull --ff-only origin master
cd GenNBV/upstream
git pull --ff-only origin codex/state-representation
source ../.venv-gennbv/bin/activate
export CUDA_HOME="$HOME/.local/opt/cuda-11.7"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/opt/nvidia-550.135/vendor-libs:$CUDA_HOME/lib64"
export VK_ICD_FILENAMES="$HOME/.local/opt/nvidia-550.135/NVIDIA-Linux-x86_64-550.135/nvidia_egl_icd.json"
export TORCH_CUDA_ARCH_LIST="8.6"
unset CUDA_VISIBLE_DEVICES
python -m unittest discover -s chapter3/tests -v
python -m chapter3 verify-lines
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

目的：先验证坐标、平面反例、缺失掩码、网络输入及梯度。合成测试通过不表示真实传感器与 HAWP 已通过。0号卡不可用时先等，不盲目改成 CUDA5/Vulkan3；该组合此前相机互操作失败过。

## 第二步：四组最小闭环（现在先做这一步）

确认0号卡有足够空间后执行；输出目录必须不存在，程序不会覆盖旧结果。

```bash
python -u -m chapter3 ablate \
  --stage smoke --seed 1 --gpu 0 --graphics 0 \
  --hawp-config ../third_party/hawp/configs/hawpv2.yaml \
  --hawp-weights ../third_party/hawp/checkpoints/hawpv2-edb9b23f.pth \
  --output _artifacts/ch3_lines_smoke_s1_01
```

每组仅2训练环境×8步×1轮，随后1栋×2视点评估。目的：验证检测→证据→编码→PPO更新→保存→加载→评估全过程。不能以此判断哪种方法更好。

成功条件：退出码0，最后出现 `CHAPTER3_SMOKE_COMPLETE`；`evidence_audit.txt` 显示 `snapshot_recomputation_passed: true`。该审计使用保存的当前帧深度、内参和线坐标重算证据，与在线输出逐项核对。

查看 `_artifacts/ch3_lines_smoke_s1_01/evidence/house_01/hawp_001.png`、`hawp_002.png`：

- 绿：几何支持多；红：有证据但支持弱；蓝：证据缺失/拟合不可靠。是连续混色，不是三个人工类别。
- 对照原 RGB 和 `line_depth_m_*.npy`；线坐标须对齐。蓝线不应消失，斜面纹理不应普遍变绿。不能以颜色本身当检测精度真值。
- `mean_support_weak_unknown` 若几乎全未知，应先排查深度有效比例、线窗口尺度、内参及拟合容差，不直接长训练。
- `*_train.log` 与评估 `summary.json` 给出检测、几何耗时；当前几何版是 NumPy CPU原型。耗时过大时先优化或减少统一训练规模，不能宣称满足在线实时要求。

## 第三步：预实验，不等同充分训练

短测与可视化确认后，先按 `HAWP_SETUP.md` 容量检查方式，加 `--line_mode evidence` 做256环境短测。通过后在 tmux 中运行（重新进入会话要恢复上面的环境）。

```bash
python -u -m chapter3 ablate \
  --stage pilot --seed 1 --gpu 0 --graphics 0 \
  --hawp-config ../third_party/hawp/configs/hawpv2.yaml \
  --hawp-weights ../third_party/hawp/checkpoints/hawpv2-edb9b23f.pth \
  --output _artifacts/ch3_lines_pilot_s1_01
```

四组均256训练环境、n_steps128、8轮、262144交互步、batch128、epochs5、lr1e-4；50栋×30视点、评估seed100。脚本顺序执行四组，每组保存最终模型，任意一组失败即停止，输出 `experiment.json` 记录状态与实际模型路径。

先看 `raw_vs_evidence.txt`、`unknown_vs_evidence.txt`，再看 `baseline_vs_raw.txt`。主指标：30视点最终覆盖率、预算内平均覆盖率、达到90%的比例及达标场景视点数，辅助看碰撞场景比例、耗时。提前终止后覆盖保持；未达90%者不可当作30视点成功；这些是诊断指标，不冒充论文AUC。

至少重复seed1、2、3，分别更换 `--seed` 和输出目录。先审查训练曲线，再统一增加所有组训练预算；8轮预实验不能证明收敛或方法优劣。此前反复查看的50栋属于开发诊断集，正式结论需要明确保留未用于方法选择的测试集。

## 当前验收状态与后续缺口

- 本地：12项 token/几何测试和3项目录迁移/入口测试（共15项）通过；平面、倾斜面、深度跳变、连续折面、单侧缺失、远裁剪、端点互换、边界、短线、噪声均有检查。
- 本地：Python3.8语法检查、脚本帮助与合成评估结果检查；没有可运行的 torch/Isaac Gym/HAWP CUDA环境，因此未运行神经网络或仿真训练。
- 待用户服务器：HAWP官方依赖、网络验证、四组短测、真实相机可视化、容量与吞吐、同预算多种子实验。
- 研究仍缺：充分训练、独立测试集、深度边缘/硬筛选/未知消融、噪声缺失协议、跨论文新颖性逐项对照。不能写“已证实提高覆盖率”“首次提出”或“已实现多视图可信度”。

## Git 同步

统一提交和服务器同步命令见 [目录入口](README.md#代码同步由用户执行)。AI 不提交或推送。
