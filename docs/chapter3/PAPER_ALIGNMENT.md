# GenNBV 原论文对齐审计（2026-11-01）

状态：尚未完整复现；不要把现有线通道预实验或下述奖励修正称为98.26%结果复现。

## 依据与版本

- 用户提供 CVPR 正式版（10页）：2024-CVPR-GenNBV.pdf。
- 项目网站带补充材料版本：https://gennbv.tech/static/paper/GenNBV.pdf （13页）。与正式版本正文不同，不能无条件混用。
- 正式版补充材料的 CVF 链接本次访问返回403，尚未核验。

## 已确认

- 256并行训练环境、100步上限；训练子类的成功终止阈值实际为99%，不是基类95%。网站附录A.2也写99%。
- 现有训练预算256×128×8=262144交互步；仓库默认1000轮对应32768000交互步。论文“32 million iterations”含糊，后者只能作为有代码依据的预算候选。
- 当前图像编码器3层卷积；正文2层。卷积通道/核/步长没有充分证据，不应自行猜完后声称完全恢复。
- 正式正文及网站附录C说2帧，网站附录A.2说5帧，需确认正式补充材料。
- 网站附录描述栅格4通道（世界坐标3+占据1），现有网络只用占据1通道。需确认最终版本。
- 正文描述连续五维高斯策略，仓库MultiDiscrete六轴（roll固定）。当前net_arch=[]，动作头是线性层，不是单独三层MLP。
- 原始代码独立collision负奖励被注释；非超时结束统一正termination，成功与碰撞混淆。
- only_positive_rewards被CLI默认False覆盖。新训练manifest记录实际值及dt缩放后的权重，不能只看配置类。

## 本次安全修正

新增训练参数 `--collision_penalty 正数`：仅碰撞给负终止项，取消成功终止正奖励；同时碰撞且超时仍罚。惩罚在原有截零操作后加入，避免被截掉。传入的幅度仍乘环境dt。论文未明确系数，用户必须显式给值，不能把任意选值称为论文值。

不传参数保持既往实验可复现；使用该参数的新实验必须独立命名。现有ablate/并行脚本不会自动启用它。

这只是奖励方向对齐，不等于完成网络、动作、指标与超参数对齐，暂不启动长训练。

## 待完成顺序

1. 获取并核验正式补充材料/作者配置：网络维度、动作分布、奖励系数、帧数、占据通道。
2. 官方权重评估审计：固定checkpoint哈希、场景列表及顺序、初始观测、reset是否计入、30视点预算、deterministic选择、碰撞后统计；解释95.03%与98.26%的差距。
3. 核实AUC积分端点/归一化及Chamfer平方或非平方、双向归约、尺度。现有预算内平均覆盖率不能直接重命名为论文AUC。
4. 单独实现且验证有依据的论文网络与连续动作模式，严禁用离散旧checkpoint直接加载。
5. 服务器短训练、保存重载及负奖励检查通过后，从头充分训练并周期评估；最终补全跨数据集及消融。

在缺乏完整官方配置时只能提供有注明假设的独立重实现，不能保证复现原数值。

## 后续核验与结构选项（同日）

已通过检索读取CVF正式补充材料：
https://openaccess.thecvf.com/content/CVPR2024/supplemental/Chen_GenNBV_Generalizable_Next-Best-View_CVPR_2024_supplemental.pdf
第1.2节确认256环境、99%阈值、100步；仍描述5帧、4通道栅格，与正文2帧存在冲突。此前“未核验”的状态已更新，具体卷积宽度和奖励系数仍未明确。

新增训练选项 `--image_conv_layers 2 --policy_hidden_layers 2`：图像两层卷积，策略为两层隐藏线性层加动作输出线性层（合计三层）；价值分支也使用两个隐藏层。隐藏宽度256、卷积通道16/32、核3、步长2是显式实现选择，不能视为论文披露值。动作仍为离散，栅格仍一通道，因此这不是完整论文模型。

旧默认3层图像卷积/无策略隐藏层保持原权重结构。新模式评估必须传 `--image-conv-layers 2 --policy-hidden-layers 2 --use-image`，严格加载会拒绝不匹配权重。新实验必须从头训练。

服务器先运行 `python -m chapter3.cli.verify_paper_network` 验证新旧结构的维度、梯度和保存重载；本地只有语法检查，未运行Torch/Isaac Gym。暂不据此开始长训练。

评估审计新发现：公开代码的AUC以奖励增量加权求和，但结束后的填充逻辑需要核查；Chamfer计算出现 `(chamfer_distance(...) * 100)[0]`，PyTorch3D返回元组，该写法是元组重复，不能完成单位换算，且默认平方距离不能直接标注cm。应先明确论文距离定义，再单独修正并标记指标版本。

## 连续动作与坐标栅格接入（2026-11-01）

训练增加 `--continuous_actions --grid_channels 4`，评估对应 `--continuous-actions --grid-channels 4`。训练和评估共享 `chapter3/paper_inputs.py`：
- 五维归一化Box动作触发SB3对角高斯分布，顺序x/y/z/pitch/yaw；映射为原环境的浮点索引再经过原仿射位姿变换，完全不取整，roll固定0。
- 动作范围沿用公开仓库世界坐标边界，并未声称与正文约20×20×10m完全一致；归一化也是明确的工程选择。
- 四通道为x/y/z坐标和当前观测占据值。坐标来自同一栅格元数据；没有拼接真值占据。原系统已使用的真值边界先验仍存在。
- 训练、评估、网络输入维度同步调整；默认旧模式和旧权重形状保持兼容。
- 本地只通过编译和diff检查，尚缺服务器Torch和Isaac Gym的数值、相机及训练闭环验证。

### 服务器验证顺序

在已配置CUDA/Vulkan的shell中，从upstream运行：

```bash
source ../.venv-gennbv/bin/activate
python -m chapter3.cli.verify_paper_network
```

测试包括旧模式和新模式的卷积层数、策略隐藏层数、256维特征、图像梯度、权重保存重载、连续动作边界和不取整、四通道坐标与占据隔离。

通过后可运行1轮小规模闭环（GPU 0空闲时）：

```bash
python -u -m chapter3 train --sim_device cuda:0 --rl_device cuda:0 \
  --graphics_device_id 0 --pipeline gpu --headless --stop_wandb \
  --use_image 1 --image_conv_layers 2 --policy_hidden_layers 2 \
  --continuous_actions --grid_channels 4 --collision_penalty 10 \
  --num_envs 2 --n_steps 8 --total_iters 1 --n_epochs 1 --batch_size 8 \
  --max_episode_length 2 --seed 1 --exp_name paper_alignment_smoke
```

其中惩罚幅度10仅为短测选值，依据原代码被注释的collision权重幅度，不能当作作者实际实验系数。短测不是正式复现结果。训练日志会打印最终权重路径，用它执行：

```bash
python -m chapter3 eval --weights <上述实际权重路径> \
  --device 0 --graphics-device 0 --use-image --image-conv-layers 2 \
  --policy-hidden-layers 2 --continuous-actions --grid-channels 4 \
  --num-envs 1 --views 2 --output _artifacts/paper_alignment_eval_smoke
```

评估仍标记diagnostic；尚未完成论文AUC和Chamfer口径核验。暂不直接启动32768000步训练。
