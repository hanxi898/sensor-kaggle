# Notebook 结构（逐单元说明）
1) 导入与环境设置（新增）

导入 torch / numpy / DataLoader / nn 等。

pywt：小波变换库。

固定随机种子 seed_everything(42)、设置 device 与 cudnn.benchmark=True。

作用：保证可复现、让 GPU 上的卷积/Attention 更快。

2) 频域特征函数：FFT & 小波（新增）

fft_token_from_sequence(x_np, top_k=32)

对形状 [B, T, C] 的序列在时间维做 rfft，取幅度谱的低频前 k 个（通常包含大部分能量），log1p 压缩动态范围，最后展平为形如 [B, k*C] 的向量。

wavelet_token_from_sequence(x_np, wavelet="db4", level=3)

对每个通道做离散小波分解（wavedec），把每级系数做 均值 + 标准差 汇聚，得到 [B, 2(level+1)C] 的向量。

设计动机：

FFT 强于捕捉周期性/节律；

小波强于捕捉瞬态/多尺度局部变化；

两者互补，作为“全局摘要”喂给 Transformer。

3) TokenBuilder：把频域当“额外 token”拼接（新增）

对每个模态（IMU/TOF）分别做：

时间域每步特征：Linear(in_dim → d_model) 投影，形成长度 T 的 token 序列；

频域 token：把 FFT token 与 Wavelet token 先做均值池化为 1 个标量，再用小 Linear(1 → d_model) 投影，得到每个1 个 token；

再加上一个可学习的 [CLS] token；

最终序列是：[CLS] + time tokens (T) + FFT token + WAVE token，长度 T+3。

位置编码：正弦位置编码（可选，默认为 max_len 预先生成，也可以是可学习的位置编码）。

设计取舍：把高维频域信息“压成 1 个 token”能低成本引入频域线索，不拖慢训练；需要更强表达时，可把“均值池化”换成线性降维/MLP（详见下方“可调参数”）。

4) MultiModalTransformer 模型（新增）

两个独立的 TransformerEncoder：分别编码 IMU 序列 & TOF 序列；

各自取 CLS 向量（序列首位）作为模态表征；

融合：拼接两个 CLS → MLP（Linear → ReLU → Dropout → Linear）输出分类 logits。

关键超参：d_model=128, nhead=4, num_layers=2, dim_feedforward=4*d_model。这些在 Notebook 里都写成了参数，便于你改。

5) 数据自动检测 + DataLoader + collate 函数（新增）

自动检测：若全局变量里找得到 imu_train / tof_train / y_train，就用它们；否则生成演示数据：

IMU 形状 [N, T, C_imu]（示例中 C_imu=6）

TOF 形状 [N, T, C_tof]（示例中 C_tof=64）

标签 y_train：[N] 的整型类别，或 [N, num_classes] 的 one-hot

SimpleTensorDataset：把三者打包为 Dataset。

重头戏在 collate：collate_with_freq

对当前 batch 的 IMU/TOF 序列一并计算 fft_token 和 wavelet_token（批量计算更快）；

返回张量：imu_x, tof_x, y, imu_fft, imu_wav, tof_fft, tof_wav，其中前 3 个是 torch.Tensor，后 4 个是 numpy 数组（在模型里会自动转为 tensor 使用）。

这样做的好处：频域计算开销和 batch 同步，节省大量重复运算。

6) 训练循环（新增）

train_one_epoch：标准的监督训练（CrossEntropyLoss），统计 loss/acc。

主脚本：构建 DataLoader、实例化模型 & 优化器，跑 2 个 epoch 确保流程可执行。

接入真实数据后，把 epoch 数、优化器、学习率策略、验证评估等接上即可。

7) 注意事项：

形状必须是 [N, T, C]，而不是 [N, C, T] 或 [T, N, C]；

T（序列长度）在同一 batch 内需要一致。如果原始样本长度不同，先对齐（pad/crop）；

NaN 请先处理（前后向填充/常数填充），否则 FFT/小波会出现 nan；

多分类时，若 y_train 是 one-hot，代码会自动 argmax 转成整型类别；

模型的 num_classes 会从标签推断：max(y)+1 或 y.shape[1]（one-hot）。

# 关键可调参数（以及改它们会带来什么）

## FFT：top_k（默认 32；TOF 示例用 16）

越大 → 频域 token 更“细”，能捕更多高频；但计算/内存更高。一般先从 16/32 开始。

## Wavelet：wavelet="db4", level=3

常用 db2~db6；level 越大越“粗”，但会截断更多高频细节。2~4 较常用。

## TokenBuilder 的频域池化策略（当前是mean→Linear(1→d_model)）

如果想让频域表达更丰富，可以把“均值池化”改成**nn.Linear(F, d_model)（直接投影 FFT 向量），或两层 MLP**，代价是计算量上升。

## Transformer：d_model, nhead, num_layers

数据量大/模态复杂时可适度增大；小数据集不宜太大，容易过拟合。

## 优化器：Adam(lr=1e-3, weight_decay=1e-4)

对应分类任务比较稳健；若你接 BERT 式调度器，也可换成 AdamW + Cosine。

## 正则化：Dropout(0.3)、weight_decay、数据增强（时间遮蔽/抖动/时间扭曲）

类别不平衡时可考虑 class weights 或 focal loss。

## 和你现有折叠（fold）管线对接的思路

你现在的 CMIFoldDataset / DataLoader 若已经产出 [B, T, C] 的 IMU/TOF 张量和标签，只需要：

在 collate_fn 里像本 Notebook 一样，批量计算 fft/wavelet tokens 并一并返回；

训练循环里把返回的七元组直接喂给模型即可（模型 forward 已支持）。

如果你坚持沿用现有 collate，不想改它，也可以把频域计算放进 forward，但那样会重复计算（每步都做 FFT/小波），性能更差。

# 训练/运行时常见坑

## 长度不一致：确保同一 batch 内的 T 一致（通常在 Dataset 阶段做 pad/crop）。

## 维度对不齐：务必是 [B, T, C]；如果你是 [B, C, T]，要 transpose(1,2)。

## NaN/Inf：频域计算前务必清洗（前后填充/中位数填充等）。

## 类别起始不是 0：确保 y 是从 0 开始的连续整数；否则要映射一下。

## max_len 设置：模型构造时用 max_len=T+3（留给 [CLS]+FFT+WAV）；如果你后续换了 T，请确保 max_len≥T+3。

# 为什么用“额外 token”注入频域？

频域/小波是整段序列的全局摘要，不天然对应每个时间步；把它们作为独立 token 拼接到序列尾端，更符合语义；

Transformer 的自注意力会学习“何时引用这些全局摘要”，而不是被动把它们拼到每个时刻；

实践中，这种设计在泛化性/稳定性上更稳（尤其当 TOF 维度较高时）。

# 输出与扩展

Notebook 默认打印每个 epoch 的 训练 loss/acc，用于“可运行性”验证；

你可以把验证集接上（按你现有的 fold 划分），在训练循环里加 eval()/with torch.no_grad() 的验证段、早停（EarlyStopping）、模型保存等——这部分与你当前工程的逻辑一致，只需把模型与 collate替换成现在这套即可。

# FFT / rFFT / Wavelet token 是什么？

FFT (Fast Fourier Transform，快速傅里叶变换)

用来把一个时间序列信号（随时间变化的数据）变换到频率域。

它能告诉我们“这个信号里面有哪些周期/节奏成分，以及各自的强度”。

举例：心率信号在 1Hz 左右会有个尖峰，步态信号在 2Hz 左右可能有峰值。

rFFT (real FFT，实数快速傅里叶变换)

如果输入信号是实数（大多数传感器信号都是实数），FFT 的结果会对称。

rFFT 只保留正频率部分（非冗余），节省一半计算量与存储。

所以在深度学习里更常用 np.fft.rfft。

# Wavelet token（小波 token）

小波变换是一种多尺度分析方法：能同时捕捉时间局部特征和频率特征。

它不像 FFT 那样“全局平均”，而是可以在不同的分辨率下分析信号。

在我们 notebook 里：对每个序列做小波分解，得到不同频带的系数 → 对每个频带做统计量（均值、方差） → 拼成一个向量（token）。

token 只是把这个向量当成 Transformer 的“一个额外输入单元”，类似 NLP 的 [CLS]。

👉 总结：

FFT token：全局周期特征（比如整体节奏）。

Wavelet token：局部/多尺度特征（比如短暂异常）。

二者互补，一起给 Transformer 提供“频域摘要”。

# 如果原始样本长度不同，先对齐。如果不对齐会怎么样？

序列长度 T：就是每个样本的时间步数。

举例：一个 IMU 采样 2 秒，每 20ms 采样一次 → 一共有 100 个时间点 → 那么 T=100。

对齐的原因：

Transformer 要求一个 batch 内的输入张量维度一致（[B, T, C]）。

如果不同样本 T 不一样，比如一个是 80 步，一个是 120 步，堆叠成矩阵时形状就对不上，报错。

常见做法：

pad：把短序列在末尾补零（或补均值），直到最长 T。

crop：把长序列截断到固定 T。

有时会 pad/crop 到一个统一的长度（比如 128、256）。

👉 如果不对齐：

直接 torch.stack([...]) 会报错：RuntimeError: Sizes of tensors must match。

即便强行拼接（如 list of tensors），Transformer 也无法 batch 化训练，效率很低。

# 类别不平衡时的 Focal Loss 是什么？

问题：在类别不平衡时，普通的交叉熵（CrossEntropy Loss）会被多数类主导，模型学不到少数类。

Focal Loss（Lin et al., 2017，用在目标检测里很有名）解决这个问题：

直观解释：

如果样本很容易（预测概率高），则 (1 - p_t)^\gamma 很小 → loss 权重降低；

如果样本很难（预测概率低），则 (1 - p_t)^\gamma 很大 → loss 权重放大；

这样模型更关注“难例”和“少数类”。

👉 适合不平衡数据集，比如少数类只占 5% 的情况。

# T 是什么？如果 T 不一致会怎么样？

T = 序列长度（time steps）。

在输入 [B, T, C] 中：

B = batch size

T = 时间步数

C = 通道数（比如 IMU 的加速度 x/y/z → C=3；TOF 的光学信号 → C=64）。

T 不一致时：

一个 batch 内无法组成规则的 3D tensor，torch.stack 会失败；

Transformer、RNN 等时序模型需要 [B, T, C]，如果 T 不一致，注意力矩阵无法计算。

👉 所以必须 pad/crop → 保证 batch 内所有样本的 T 一样。

# “对每个频带做统计量（均值、方差） → 拼成一个向量（token）”是什么意思？

这是时序信号特征提取常见做法：

背景：假设你有一个原始时间序列（比如加速度、角速度信号），你想让 Transformer 或其他模型处理它。

步骤：

做频域分解：对信号做 FFT 或小波变换，把信号拆成不同的频带（比如 0–1 Hz, 1–2 Hz, …）。

统计量计算：对每个频带的信号计算一些统计量（常用的有均值、方差、最大值、最小值、能量等）。

例如频带 1 的均值 = mean(x1)

频带 1 的方差 = var(x1)

拼成向量：把每个频带的统计量串起来，形成一个向量（token）。

比如有 3 个频带，每个频带取均值和方差 → token = [mean1, var1, mean2, var2, mean3, var3]

作用：

这个 token 就是模型输入的一条“特征”，把原始时序信息浓缩成固定长度的向量，方便 Transformer 或其他模型处理。

💡小提示：这个 token 相当于给信号做了频域压缩表示，保留了主要的统计特性。

#  focal loss 中的 gamma 和 alpha 是什么？

Focal Loss 用于解决类别不平衡问题，gamma (γ)：控制难易样本的权重。γ 越大，模型越关注那些难以分类的样本。一般经验值：γ = 2。

alpha (α)：用来给不同类别设置权重，解决类别不平衡。如果正样本少，可以给正样本 α > 0.5。一般经验值：α ∈ [0.25, 0.75]。

所以 γ 和 α 都是超参数，通常通过交叉验证或实验调试确定。

#  不平衡数据集除了 focal loss 还有什么方法？

常见方法：

1. 重采样（Resampling）

   过采样：少数类样本重复/生成（如 SMOTE）

   欠采样：多数类样本减少

2. 类别权重（Class Weight）

   在 loss 里给少数类加权重
   
   PyTorch: CrossEntropyLoss(weight=class_weight_tensor)

4. 集成方法

   如 Balanced Random Forest、XGBoost 的 scale_pos_weight

5. 数据增强

   对少数类生成更多样本

6. 调整阈值

   预测概率 threshold 调低少数类的判定门限
