# HAWPv2 线状态接入与对照实验

本实现比较「位姿＋栅格＋图像」与「位姿＋栅格＋图像＋HAWP」。
补齐原论文图像通道不是创新；本文描述默认 `raw` 二维线模式。2026-10-29 新增的单帧深度证据原型、同结构对照和四组实验见 [EXPERIMENTS.md](EXPERIMENTS.md)，仍待服务器验证。

## 实际数据流

- 在 Isaac Gym 相机张量访问期间复制当前原分辨率 RGB，供 wrapper 在返回观测时检测；快照与当前返回的观测一致，不读取后续视点或评估真值，也不依赖 reset 后的图像缓存。
- 使用官方 HAWPv2 `configs/hawpv2.yaml`、`hawpv2-edb9b23f.pth`，官方图像缩放和归一化。原分辨率 RGB 经过官方 512×512 预处理。冻结全部 HAWP 参数，仅训练下游策略。
- 每帧按检测分数筛选并排序，阈值 0.5，最多 64 条；每条为 `x1/W,y1/H,x2/W,y2/H,score,valid`。端点规范排序，不足补零。score 是检测器分数，不是几何可信度。
- 不跨回合累计线历史；线通道表示当前帧。现有图像通道仍为两帧 64×64 灰度图。
- 共享 MLP 5→64→128，按有效掩码平均池化，得到 128 维线特征；与原三个 256 维特征拼接，经 896→256 融合后送入原 PPO。
- `--use_line 0` 保持原三通道模型结构和观测维度；线通道模型必须从头训练，不能直接加载旧三通道权重当作线模型。
- 官方 HAWPv2 推理只支持 batch=1，因此每个仿真环境逐张执行，未擅自修改官方算法。256 环境可能明显变慢，先运行短测试查看耗时。
- `training_manifest.json` 记录训练参数及 HAWP 配置、权重哈希、官方代码提交。评估强制核对这些检测器信息；复制策略权重时连同该 manifest 保留原目录关系。
- 输出线 token 和叠加图，记录 `hawp_timing.json`。它测量检测及预处理传输开销，不能代替整条流程的实时性指标。

## 1. Mac 获取官方依赖并上传

在可访问 GitHub 的 Mac 执行。若目录已存在，复用现有目录，不覆盖。

```bash
cd /Users/zzc/Desktop/Graduation-thesis/GenNBV
mkdir -p third_party
cd third_party
git clone https://github.com/cherubicXN/hawp.git
mkdir -p hawp/checkpoints
curl -fL --retry 3 https://github.com/cherubicXN/hawp-torchhub/releases/download/HAWPv2/hawpv2-edb9b23f.pth -o hawp/checkpoints/hawpv2-edb9b23f.pth
tar -czf hawp-offline.tar.gz hawp
```

通过你已有的服务器传文件方式，把压缩包放到服务器 `GenNBV/third_party/`。
源地址：https://github.com/cherubicXN/hawp （官方 docs/HAWPv2.md 和 downloads.sh）。

## 2. 服务器环境

```bash
cd /data/home/zzc/Graduation-thesis/GenNBV/third_party
tar -xzf hawp-offline.tar.gz
cd ../upstream
source ../.venv-gennbv/bin/activate
export CUDA_HOME="$HOME/.local/opt/cuda-11.7"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/opt/nvidia-550.135/vendor-libs:$CUDA_HOME/lib64"
export VK_ICD_FILENAMES="$HOME/.local/opt/nvidia-550.135/NVIDIA-Linux-x86_64-550.135/nvidia_egl_icd.json"
export TORCH_CUDA_ARCH_LIST="8.6"
unset CUDA_VISIBLE_DEVICES
```

使用已有约束文件保护 torch/numpy 等环境；不要直接运行官方 conda 安装说明或换 torch。

```bash
uv pip install --python ../.venv-gennbv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --constraint ../_artifacts/gennbv-constraints.txt \
  --no-build-isolation -e ../third_party/hawp
uv pip check
```

若依赖解析或官方模块导入失败，保留错误信息排查，不删除约束后强行升级已有环境。

## 3. 验证状态分支，再做短训练

这一步检验线输入确实改变特征、补零行不干扰特征、参数有梯度，并实际调用 HAWP。

```bash
python -m unittest chapter3.tests.test_hawp_tokens -v
python -m chapter3 verify-lines \
  --image _artifacts/pilot256_A_gpu0_smoke_01/house_01/rgb_001.png \
  --hawp-config ../third_party/hawp/configs/hawpv2.yaml \
  --hawp-weights ../third_party/hawp/checkpoints/hawpv2-edb9b23f.pth \
  --device cuda:0
```

确认 0 号卡可用后，跑 2 环境 × 8 步，检验采集→HAWP→策略更新→保存整个流程。

```bash
python -u -m chapter3 train \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --pipeline gpu --headless --stop_wandb \
  --num_envs 2 --seed 2 --n_steps 8 --batch_size 8 \
  --total_iters 1 --n_epochs 1 --max_episode_length 2 \
  --use_image 1 --use_line 1 --exp_name hawp_smoke_s2 \
  --hawp_config ../third_party/hawp/configs/hawpv2.yaml \
  --hawp_weights ../third_party/hawp/checkpoints/hawpv2-edb9b23f.pth
```

以 `Saved final trained model ...`、`TRAINING_COMPLETE` 为成功标志。这是功能检查，不是效果实验。
用实际输出的最终权重路径替换下文 `SMOKE_MODEL`，验证新架构评估也能加载与出图：

```bash
python -m chapter3 eval --weights SMOKE_MODEL \
  --device 0 --graphics-device 0 --num-envs 1 --views 2 \
  --use-image --use-line \
  --hawp-config ../third_party/hawp/configs/hawpv2.yaml \
  --hawp-weights ../third_party/hawp/checkpoints/hawpv2-edb9b23f.pth \
  --output _artifacts/hawp_smoke_eval_01
```

## 4. 同预算下游对比（短训练和评估通过后）

先用 256 环境 × 8 步做容量检查：复制短训练命令，仅将 `--num_envs 256 --batch_size 128 --exp_name hawp_capacity_s2`。
若显存不足，不只减少 HAWP 组环境数：两组都要采用相同的新采样设置并重新训练，否则比较包含额外变量。
查看该运行的 `hawp_timing.json`，估计 262144 帧的实际检测成本，再启动完整实验。

以下命令从头训练 HAWP 策略（seed 2，256 环境，128 步/轮，8 轮，batch 128，epochs 5，episode 100），随后重评已有 seed 2 三通道基线及新策略，并自动输出配对比较。不会拿旧的「位姿＋栅格」模型当基线。

先进入 tmux：`tmux new -s hawp_s2`。新终端中重新执行第 2 节的激活和 export（不用重复安装）。

```bash
python -u -m chapter3 pair \
  --baseline runs/pilot256_pose_grid_image_s2_1028_1919_04/models/rl_model_final.zip \
  --seed 2 --gpu 0 --graphics 0 \
  --hawp-config ../third_party/hawp/configs/hawpv2.yaml \
  --hawp-weights ../third_party/hawp/checkpoints/hawpv2-edb9b23f.pth \
  --output _artifacts/hawp_pair_s2_01
```

输出在该目录的 `comparison.txt`；A=三通道，B=三通道＋HAWP。只改变状态输入及其编码器，保留既有奖励、动作空间、训练步数。两组测试均 seed 100、50 栋、30 视点。官方 KL 提前停止可能使实际更新次数不同，保留训练日志。

如需 seed 1，对应参数为：
`--seed 1 --baseline runs/pilot256_pose_grid_image_s1_auto_1025_2030_23/models/rl_model_final.zip --output _artifacts/hawp_pair_s1_01`。

手动重新读结果：

```bash
python ../../_artifacts/compare_gennbv_eval.py \
  _artifacts/hawp_pair_s2_01/baseline _artifacts/hawp_pair_s2_01/hawp \
  --comparison hawp
```

## 结果解释及当前限制

- 比较固定预算覆盖率、各视点曲线、逐栋胜负、碰撞比例；报告多个训练种子，不能保证 HAWP 一定提升。
- 这首先是加入整个 HAWP 分支的收益实验。额外参数、外部预训练和更高分辨率 RGB 都可能影响结果，不能直接把收益全部归因于线几何；后续需要容量、分辨率或零线输入对照。
- 当前使用已有 50 栋诊断集，已被多次观察。若据此选择阈值或方法，最终结论应另外保留未参与选择的测试数据。
- 本地已运行 CPU token 测试与 Python 3.8 语法检查；Mac 无 torch/Isaac Gym/CUDA，官方 HAWP 推理、梯度测试、短训练和真实对比仍待服务器运行。

## Git 同步

统一提交和服务器同步命令见 [目录入口](README.md#代码同步由用户执行)。AI 不提交或推送。
