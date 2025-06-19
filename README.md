# FunASR-VAD 

## 概述

FunASR-VAD是基于FSMN（Feedforward Sequential Memory Network）的语音活动检测系统，能够实时检测音频中的语音段和静音段，相当于做了一个二分类输出，在语音领域简单但是也很重要。
## 算法整体流程


```
音频输入 → 特征提取 → 模型推理 → 概率输出 → 阈值判决 → 滑动窗口平滑 → 状态机处理 → 语音段输出
    ↓         ↓         ↓         ↓         ↓           ↓            ↓           ↓
  16kHz    Fbank     FSMN     248维度输出    二分类     时序平滑      起止检测     时间戳
  采样     80维      网络                     判决      减少噪声      状态转换     格式化
```

## 详细算法流程
### 0.算法框架
```
音频输入 → 特征提取 → FSMN模型推理 → 后处理 → 语音段输出
    ↓         ↓           ↓          ↓
  预处理   Fbank特征   VAD分数计算  状态机处理
```
### 1. 音频预处理阶段

#### 1.1 音频加载与标准化
```python
# 音频采样率标准化为16kHz
waveform = load_audio_file(audio_path, sample_rate=16000)
#load_audio_file函数的具体实现在下方
# 幅度标准化（可选）
# 幅度标准化：范围转移、提高精度
if upsacle_samples:
    waveform = waveform * (1 << 15)  # 放大到16位整数范围
```
    技术原理：
        1. 使用torchaudio加载原始音频数据和采样率
        2. 多声道转单声道：取各声道平均值，保留主要信息
        3. 重采样：使用线性插值将任意采样率转换为16kHz
        4. 奈奎斯特频率：16kHz对应8kHz奈奎斯特频率，覆盖语音频谱
```python
# 来源：vad_system/utils.py - load_audio_file函数
# 加载音频为张量
# 具体实现
import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional

def load_audio_file(file_path: str, fs: int = 16000) -> torch.Tensor:
    try:
        waveform, sample_rate = torchaudio.load(file_path
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # 重采样到指定采样率 - 统一采样率为16kHz
        if sample_rate != fs:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=fs,
                resampling_method='sinc_interpolation' 
            )
            waveform = resampler(waveform)

        # 返回一维张量 [samples]，便于后续处理
        return waveform.squeeze(0)

    except Exception as e:
        raise RuntimeError(f"音频加载失败: {file_path}, 错误: {str(e)}")
```
#### 1.2 分块处理
对于长音频，采用重叠分块策略：
- **分块大小**: 默认60秒
- **重叠比例**: 10%
- **时间偏移计算**: `time_offset = start_sample * 1000 / sample_rate`


**分块操作的数学描述**：
```
时间轴分割:
Chunk 1: [0s,      60s]     → samples [0,        959999]
Chunk 2: [54s,     114s]    → samples [864000,   1823999]
Chunk 3: [108s,    168s]    → samples [1728000,  2687999]
...

重叠区域: 每两个相邻块重叠6秒 (96,000 samples)
步进长度: 54秒 (864,000 samples)
```
### 2.特征提取和处理阶段
#### 2.1 Fbank特征提取
**什么是Fbank特征**：
Fbank（Filter Bank，滤波器组）是一种基于**梅尔频率刻度**的音频特征提取方法，专门设计用于语音信号处理。它模拟人类听觉系统的频率感知特性，将线性频率转换为更符合人耳感知的梅尔频率。

***附：梅尔频率倒谱系数(MFCC)与Fbank的关系***：
- **Fbank**: 梅尔滤波器组的对数输出，保留了频谱的详细信息
- **MFCC**: 在Fbank基础上进行离散余弦变换(DCT)，进一步压缩和去相关
- **VAD应用**: Fbank保留更多频谱细节，更适合语音活动检测任务
- **mel数学公式**: $mel(f) = 2595 \times \log_{10}(1 + \frac{f}{700})$

**Fbank特征计算公式详解**:

1. **短时傅里叶变换(STFT)**:
$$X(k) = \sum_{n=0}^{N-1} x(n) \cdot w(n) \cdot e^{-j2\pi kn/N}$$

参数说明：
- $x(n)$: 时域音频信号，长度N=400个采样点(25ms×16kHz)
- $w(n)$: 汉明窗函数，$w(n) = 0.54 - 0.46\cos(\frac{2\pi n}{N-1})$
- $X(k)$: 第k个频率bin的复数频域表示，k=0,1,...,N/2
- $N$: FFT点数，通常为512点

2. **功率谱计算**:
$$P(k) = |X(k)|^2 = \text{Real}(X(k))^2 + \text{Imag}(X(k))^2$$

3. **梅尔滤波器组输出**:
$$M(m) = \sum_{k=0}^{N/2} P(k) \cdot H_m(k)$$

其中 $H_m(k)$ 是第m个梅尔滤波器的频率响应：
- 梅尔刻度转换: $\text{mel}(f) = 2595 \log_{10}(1 + \frac{f}{700})$
- 80个滤波器均匀分布在梅尔刻度上，覆盖0-8000Hz
- 每个滤波器为三角形窗，相邻滤波器50%重叠

4. **对数变换**:
$$\text{Fbank}(m) = \log(M(m) + \epsilon)$$
其中 $\epsilon = 1e-8$ 防止对数运算中的数值不稳定



**详细技术流程**：
        1. 幅度标准化：提高数值精度
        2. 加窗分帧：25ms窗长，10ms帧移
        3. FFT变换：时域到频域转换
        4. 梅尔滤波：模拟人耳感知特性
        5. 对数变换：压缩动态范围
        6. LFR处理：降低帧率，减少计算量
        7. CMVN归一化：消除通道差异

```python
# 来源：vad_system/utils.py - compute_fbank_features函数
# 计算fbank特征相关流程代码
import kaldi
import torch
import torch.nn.functional as F
from typing import Optional

def compute_fbank_features(
    waveform: torch.Tensor,
    audio_config: 'AudioConfig',
    cmvn: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # 步骤1: 幅度标准化 - 将浮点数转换为16位整数范围
    # 目的：最大化利用数值精度，提高后续计算的稳定性
    if audio_config.upsacle_samples:
        waveform = waveform * (1 << 15)  # 乘以32768 = 2^15
        print(f"幅度标准化: 缩放因子={1 << 15}, 最大值={torch.max(torch.abs(waveform)):.1f}")

    # 确保输入为二维张量 [batch=1, samples]
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)

    # 步骤2-5: 提取Fbank特征
    # 使用Kaldi库进行高质量特征提取
    fbank = kaldi.fbank(
        waveform,
        num_mel_bins=audio_config.n_mels,        # 80个梅尔滤波器
        frame_length=audio_config.frame_length,   # 25.0ms窗长
        frame_shift=audio_config.frame_shift,     # 10.0ms帧移
        dither=audio_config.dither,               # 抖动系数，增加鲁棒性
        energy_floor=0.0,                         # 能量下限
        window_type=audio_config.window,          # hamming窗函数
        sample_frequency=audio_config.fs,         # 16000Hz采样率
        snip_edges=audio_config.snip_edges,       # 边界处理方式
    )

    print(f"Fbank特征提取: 输入{waveform.shape} -> 输出{fbank.shape}")
    print(f"帧数计算: {len(waveform[0])}样本 / {audio_config.frame_shift}ms = {fbank.shape[0]}帧")

    # 步骤6: LFR (Low Frame Rate) 处理
    # 通过帧堆叠和下采样减少计算量
    if audio_config.lfr_m != 1 or audio_config.lfr_n != 1:
        fbank = apply_lfr(fbank, audio_config.lfr_m, audio_config.lfr_n)
        print(f"LFR处理: 堆叠{audio_config.lfr_m}帧, 下采样{audio_config.lfr_n}, 输出{fbank.shape}")

    # 步骤7: CMVN归一化
    # 消除通道差异，提高模型鲁棒性
    if cmvn is not None:
        fbank = apply_cmvn(fbank, cmvn)
        print(f"CMVN归一化: 均值={torch.mean(fbank):.3f}, 标准差={torch.std(fbank):.3f}")

    return fbank.type(torch.float32)

```
#### 2.2 LFR (Low Frame Rate) 处理
**LFR处理的核心机制**：
LFR（Low Frame Rate）是一种通过**帧堆叠**和**下采样**来减少序列长度的技术，在保持关键信息的同时显著降低计算复杂度。

**具体操作步骤**：
1. **帧堆叠（Frame Stacking）**：
   ```
   原始特征: [frame_0, frame_1, frame_2, frame_3, frame_4, frame_5, ...]
   堆叠操作: [frame_0+frame_1+frame_2+frame_3+frame_4] → 新特征向量
   维度变化: 80维 → 400维 (80 × 5)
   ```

2. **下采样（Downsampling）**：
   ```
   原始时间轴: t=0, t=1, t=2, t=3, t=4, t=5, t=6, t=7, t=8, ...
   下采样后:   t=0,      t=3,      t=6,      t=9, ...
   采样间隔:   每3帧取1帧进行处理
   ```

**数学表示**：
```
输入: X ∈ R^(T×D), T=时间帧数, D=特征维度(80)
输出: Y ∈ R^(T'×D'), T'=⌈T/lfr_n⌉, D'=D×lfr_m

Y[i] = [X[i×lfr_n], X[i×lfr_n+1], ..., X[i×lfr_n+lfr_m-1]]
```
#### 2.3 CMVN (Cepstral Mean and Variance Normalization) 
对特征进行均值方差归一化以提高模型鲁棒性：

**归一化公式**:
$$\hat{x}_t = \frac{x_t - \mu}{\sigma}$$

**参数计算**:
- $\mu = \frac{1}{T}\sum_{t=1}^{T} x_t$ (全局均值，T为总帧数)
- $\sigma = \sqrt{\frac{1}{T}\sum_{t=1}^{T} (x_t - \mu)^2}$ (全局标准差)

**技术意义**:
- **消除通道差异**: 不同录音设备、环境的影响
- **数值稳定性**: 将特征值标准化到均值0、方差1的分布
- **加速收敛**: 帮助神经网络更快收敛
- **维度独立**: 对80维Fbank特征的每一维独立进行归一化

**实现方式**:
```python
# 来源：vad_system/utils.py - CMVN的计算
def compute_cmvn_stats(all_features: torch.Tensor) -> torch.Tensor:

    if all_features.dim() == 3:  # [N, T, D] -> [N*T, D]
        all_features = all_features.view(-1, all_features.shape[-1])

    # 计算全局统计量
    global_mean = torch.mean(all_features, dim=0)  # [D]
    global_std = torch.std(all_features, dim=0)    # [D]

    # 避免标准差为0的情况
    global_std = torch.clamp(global_std, min=1e-8)

    cmvn_stats = torch.stack([global_mean, global_std])

    print(f"CMVN统计量计算完成:")
    print(f"  总帧数: {all_features.shape[0]:,}")
    print(f"  特征维度: {all_features.shape[1]}")
    print(f"  均值范围: [{torch.min(global_mean):.3f}, {torch.max(global_mean):.3f}]")
    print(f"  标准差范围: [{torch.min(global_std):.3f}, {torch.max(global_std):.3f}]")

    return cmvn_stats


```

### 3. FSMN模型推理阶段

#### 3.1 FSMN网络结构
```python
# 来源：vad_system/detector.py - SimpleFSMNEncoder类和FSMNVADDetector类
class FSMNEncoder(nn.Module):
    """
    FSMN (Feedforward Sequential Memory Network) 特点：
    1. 前馈结构：无循环连接，支持并行计算
    2. 记忆机制：通过卷积层捕获时序依赖
    3. 高效计算：相比RNN/LSTM更适合实时处理
    """

    def __init__(self, input_dim: int = 400, output_dim: int = 2, hidden_dim: int = 256):
        super(SimpleFSMNEncoder, self).__init__()

        self.input_dim = input_dim    # LFR后的特征维度: 80*5=400
        self.output_dim = output_dim  # VAD二分类输出
        self.hidden_dim = hidden_dim  # 隐藏层维度

        # 1. 输入线性层 - 特征维度变换和非线性映射
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # 400 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),     # 256 -> 256
            nn.ReLU(inplace=True)
        )

        # 2. FSMN核心层 - 使用双向LSTM模拟FSMN的记忆机制
        # 实际FSMN使用卷积层，这里用LSTM简化实现
        self.fsmn_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0.1 if i < 2 else 0.0  # 最后一层不用dropout
            ) for i in range(3)  # 3层FSMN
        ])

        # 3. 输出层 - 分类头
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),     # 2分类：语音/静音
            nn.Softmax(dim=-1)                     # 概率输出
        )

        # 初始化权重
        self._initialize_weights()

        print(f"SimpleFSMNEncoder初始化:")
        print(f"  输入维度: {input_dim}")
        print(f"  隐藏维度: {hidden_dim}")
        print(f"  输出维度: {output_dim}")
        print(f"  FSMN层数: {len(self.fsmn_layers)}")
        print(f"  总参数量: {sum(p.numel() for p in self.parameters()):,}")

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        # 1. 输入线性层 - 特征变换
        x = self.input_projection(x)  # [batch, frames, hidden_dim]

        # 2. FSMN核心层 - 时序建模
        for i, lstm in enumerate(self.fsmn_layers):
            # LSTM前向传播
            x, _ = lstm(x)  # [batch, frames, hidden_dim]

            # 残差连接（除了第一层）
            if i > 0:
                x = x + residual
            residual = x

        # 3. 输出层 - 分类
        output = self.output_projection(x)  # [batch, frames, 2]

        return output

```
#### 3.2 FSMN前向传播公式
对于第t帧的输出：

$$h_t = \sigma(W_h \cdot x_t + \sum_{i=1}^{L} W_i \cdot h_{t-i} + \sum_{j=1}^{R} W_j \cdot h_{t+j})$$

其中：
- $x_t$: 第t帧输入特征
- $h_t$: 第t帧隐藏状态
- $W_h, W_i, W_j$: 权重矩阵
- $L, R$: 左右上下文长度
- $\sigma$: 激活函数

#### 3.3 VAD分数计算
模型输出2维概率分布：
```python
scores = model(features)  # [batch, frames, 2]
speech_probs = scores[:, :, 1]  # 语音概率
silence_probs = scores[:, :, 0]  # 静音概率
```

### 4. 后处理阶段

#### 4.1 帧级别判决
**VAD决策机制详解**：

1. **概率输出**: FSMN模型输出每帧的语音/静音概率分布 [P_silence, P_speech]
2. **阈值比较**: 将语音概率与预设阈值比较进行二分类判决
3. **状态转换**: 基于连续帧的判决结果进行状态机转换
4. **时序平滑**: 通过滑动窗口和时间约束减少误判

**阈值确定方法**：

1. **经验设定**: 默认0.5基于大量实验数据的统计结果
2. **ROC曲线优化**: 通过分析不同阈值下的FAR和FRR找到最优点
3. **应用场景调整**:
   - 低噪声环境: 可降低到0.3-0.4，提高敏感度
   - 高噪声环境: 可提高到0.6-0.7，减少误检
4. **自适应调整**: 根据环境噪声水平动态调整阈值

```python
# 来源：vad_system/detector.py 
def adaptive_threshold(noise_level_db):
    """根据噪声水平自适应调整阈值"""
    if noise_level_db < -40:  # 安静环境
        return 0.3
    elif noise_level_db < -20:  # 一般环境
        return 0.5
    else:  # 嘈杂环境
        return 0.7
def frame_decision(speech_prob: float, threshold: float = 0.5) -> FrameState:
 # 根据阈值将概率转换为帧状态：
    if speech_prob > threshold:
        return FrameState.kFrameStateSpeech
    else:
        return FrameState.kFrameStateSil

```

#### 4.2 滑动窗口平滑

使用滑动窗口减少噪声影响：

```python
# 来源：vad_system/detector.py - WindowDetector类
class WindowDetector:
    def __init__(self, window_size_ms, frame_size_ms):
        self.win_size_frame = window_size_ms // frame_size_ms
        self.win_state = [0] * self.win_size_frame
        self.win_sum = 0
        self.cur_win_pos = 0

    def detect_one_frame(self, frame_state):
        # 更新滑动窗口
        self.win_sum += frame_state.value - self.win_state[self.cur_win_pos]
        self.win_state[self.cur_win_pos] = frame_state.value
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame

        # 状态转换判决
        return self.state_transition()
        def apply_sliding_window_smoothing(
    speech_probs: torch.Tensor,
    window_size_ms: int = 200,
    frame_shift_ms: int = 10,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, List[AudioChangeState]]:
    """
    应用滑动窗口平滑的完整流程
    """
    # 1. 帧级判决
    frame_decisions = []
    for prob in speech_probs:
        state = frame_decision(prob.item(), threshold)
        frame_decisions.append(state)

    # 2. 滑动窗口平滑
    detector = WindowDetector(window_size_ms, frame_shift_ms)
    state_changes = detector.process_sequence(frame_decisions)

    # 3. 生成平滑后的帧状态
    smoothed_states = torch.zeros_like(speech_probs, dtype=torch.long)
    current_state = FrameState.kFrameStateSil

    for i, change in enumerate(state_changes):
        if change == AudioChangeState.kChangeStateSil2Speech:
            current_state = FrameState.kFrameStateSpeech
        elif change == AudioChangeState.kChangeStateSpeech2Sil:
            current_state = FrameState.kFrameStateSil

        smoothed_states[i] = current_state.value

    return smoothed_states, state_changes
```

**滑动窗口减少噪声影响的原理**：

1. **时间平滑**: 通过多帧平均减少单帧的随机噪声影响
2. **统计稳定**: 窗口内的统计量比单帧更稳定可靠
3. **突发抑制**: 短暂的噪声突发被窗口内的正常帧稀释
4. **决策延迟**: 用少量延迟换取更可靠的检测结果
5. **阈值平滑**: 避免在阈值附近的频繁跳变

#### 4.3 状态机处理
实现语音段的起始和结束检测：

**状态定义**:
- `kVadInStateStartPointNotDetected`: 未检测到起始点
- `kVadInStateInSpeechSegment`: 在语音段中
- `kVadInStateEndPointDetected`: 检测到结束点
 ```
    技术原理：
        1. 基于连续帧计数进行状态转换
        2. 防止短暂噪声触发误检
        3. 避免语音中的停顿导致错误分割
        4. 提供稳定的语音段边界检测
```
```python
# 来源：vad_system/detector.py - VAD状态机

    def process_frame(self, frame_state: FrameState, speech_prob: float = 0.0) -> bool:
   
        self.total_frame_count += 1
        segment_detected = False

        if frame_state == FrameState.kFrameStateSpeech:
            self.voice_frame_count += 1
            self.silence_frame_count = 0
            if speech_prob > 0:
                self.current_segment_frames.append(speech_prob)
        else:
            self.silence_frame_count += 1
            self.voice_frame_count = 0

        # 状态转换逻辑
        if self.current_state == self.State.kVadInStateStartPointNotDetected:
            # 寻找语音起始点
            if self.voice_frame_count >= self.sil_to_speech_threshold:
                self._start_speech_segment()

        elif self.current_state == self.State.kVadInStateInSpeechSegment:
            # 在语音段中，寻找结束点
            if self.silence_frame_count >= self.speech_to_sil_threshold:
                segment_detected = self._end_speech_segment()

        elif self.current_state == self.State.kVadInStateEndPointDetected:
            # 已检测到结束点，等待下一个起始点
            if self.voice_frame_count >= self.sil_to_speech_threshold:
                self._start_speech_segment()
            elif self.silence_frame_count >= self.max_end_silence_frames:
                self.current_state = self.State.kVadInStateStartPointNotDetected

        return segment_detected
```

#### 4.4 分贝计算详解

计算每帧的能量分贝值用于辅助VAD判决：

**RMS(均方根)计算**:
$$\text{RMS}_t = \sqrt{\frac{1}{N}\sum_{n=1}^{N} x_t^2(n)}$$

**分贝转换**:
$$dB_t = 20 \log_{10}(\text{RMS}_t + \epsilon)$$

```python
# 来源：vad_system/utils.py - compute_decibel函数
def compute_decibel(waveform, frame_length=160):
    """计算音频帧的分贝值"""
    # 计算RMS能量
    rms = torch.sqrt(torch.mean(waveform**2, dim=-1))
    # 转换为分贝
    db = 20 * torch.log10(rms + 1e-10)
    return db
```
**参数说明**:
- $x_t(n)$: 第t帧的第n个采样点，n=1,2,...,N (N=160，对应10ms)
- $\epsilon = 1e-10$: 防止对数运算中的数值错误
- 分贝值范围: 通常在-60dB到0dB之间
**在VAD中的应用**:
```python
# 来源：算法设计原理
def energy_based_vad(frame_db, threshold_db=-35):
    """基于能量的辅助VAD判决"""
    if frame_db > threshold_db:
        return True  # 可能是语音
    else:
        return False  # 可能是静音
```

**与神经网络VAD的结合**:
- 神经网络VAD提供主要判决
- 能量VAD作为辅助验证，防止低能量语音被误判
- 双重验证提高整体检测精度
 #### 4.5 后处理总结
 ```
    处理流程：
        1. 帧级判决：概率 -> 帧状态
        2. 滑动窗口平滑：减少噪声影响
        3. 状态机处理：检测语音段边界
        4. 后处理：合并和过滤
  ```
```python
def process_vad_sequence(
    speech_probs: torch.Tensor,
    vad_config: Dict[str, Any],
    frame_shift_ms: int = 10
) -> List[VadSpeechSegment]:

    # 1. 帧级判决
    threshold = vad_config.get('speech_noise_thres', 0.5)
    frame_states = batch_frame_decision(speech_probs, threshold)

    # 2. 滑动窗口平滑
    window_size_ms = vad_config.get('window_size_ms', 200)
    smoothed_states, _ = apply_sliding_window_smoothing(
        speech_probs, window_size_ms, frame_shift_ms, threshold
    )

    # 3. 状态机处理
    state_machine = VadStateMachine(
        sil_to_speech_time_ms=vad_config.get('sil_to_speech_time', 200),
        speech_to_sil_time_ms=vad_config.get('speech_to_sil_time', 700),
        max_end_silence_time_ms=vad_config.get('max_end_silence_time', 800),
        frame_shift_ms=frame_shift_ms
    )

    # 逐帧处理
    for i, (state_val, prob) in enumerate(zip(smoothed_states, speech_probs)):
        frame_state = FrameState.kFrameStateSpeech if state_val == 1 else FrameState.kFrameStateSil
        state_machine.process_frame(frame_state, prob.item())

    # 4. 获取最终结果
    speech_segments = state_machine.finalize()

    return speech_segments
```

### 5. 语音段输出格式化
#### 5.1 时间戳转换
将帧索引转换为毫秒时间戳：

```python
def frame_to_time(frame_idx, frame_shift_ms):
    return frame_idx * frame_shift_ms

start_time = frame_to_time(start_frame, 10)  # 10ms帧移
end_time = frame_to_time(end_frame, 10)
```

#### 5.2 语音段合并

合并相邻或重叠的语音段：
```
    合并策略：
        1. 按开始时间排序
        2. 检查相邻段是否需要合并
        3. 合并重叠或接近的段
        4. 过滤过短的段
        5. 重新计算置信度
```
```python
# 来源：vad_system/utils.py - 语音段合并
def merge_segments(
    segments: List[Union[VadSpeechSegment, List[float]]],
    tolerance_ms: float = 100,
    min_duration_ms: float = 200
) -> List[VadSpeechSegment]:

    if not segments:
        return []

    print(f"语音段合并前: {len(segments)}个段")

    # 转换为统一格式并排序
    normalized_segments = []
    for seg in segments:
        if isinstance(seg, VadSpeechSegment):
            normalized_segments.append(seg)
        elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
            # 从列表格式转换
            start_time, end_time = seg[0], seg[1]
            confidence = seg[2] if len(seg) > 2 else 0.5
            frame_count = int((end_time - start_time) / 10)  # 假设10ms帧移

            normalized_segments.append(VadSpeechSegment(
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
                frame_count=frame_count
            ))
        else:
            print(f"警告: 跳过无效语音段格式: {seg}")
            continue

    # 按开始时间排序
    normalized_segments.sort(key=lambda x: x.start_time)

    # 合并处理
    merged = []
    for current in normalized_segments:
        if not merged:
            # 第一个段
            merged.append(current)
            continue

        last = merged[-1]

        # 检查是否需要合并
        gap = current.start_time - last.end_time

        if gap <= tolerance_ms:
            # 合并段
            merged_start = last.start_time
            merged_end = max(last.end_time, current.end_time)
            merged_duration = merged_end - merged_start

            # 计算加权平均置信度
            last_duration = last.end_time - last.start_time
            current_duration = current.end_time - current.start_time
            total_duration = last_duration + current_duration

            if total_duration > 0:
                merged_confidence = (
                    last.confidence * last_duration +
                    current.confidence * current_duration
                ) / total_duration
            else:
                merged_confidence = (last.confidence + current.confidence) / 2

            # 更新最后一个段
            merged[-1] = VadSpeechSegment(
                start_time=merged_start,
                end_time=merged_end,
                confidence=merged_confidence,
                frame_count=int(merged_duration / 10)
            )

            print(f"  合并段: [{last.start_time:.0f}-{last.end_time:.0f}] + "
                  f"[{current.start_time:.0f}-{current.end_time:.0f}] -> "
                  f"[{merged_start:.0f}-{merged_end:.0f}] (间隔:{gap:.0f}ms)")
        else:
            # 不合并，添加新段
            merged.append(current)

    # 过滤过短的段
    filtered = []
    for seg in merged:
        duration = seg.end_time - seg.start_time
        if duration >= min_duration_ms:
            filtered.append(seg)
        else:
            print(f"  过滤短段: [{seg.start_time:.0f}-{seg.end_time:.0f}] "
                  f"(时长:{duration:.0f}ms < {min_duration_ms}ms)")

    print(f"语音段合并后: {len(filtered)}个段")

    # 输出合并统计
    if filtered:
        total_speech_time = sum(seg.end_time - seg.start_time for seg in filtered)
        avg_confidence = sum(seg.confidence for seg in filtered) / len(filtered)

        print(f"合并统计:")
        print(f"  总语音时长: {total_speech_time:.0f}ms")
        print(f"  平均置信度: {avg_confidence:.3f}")
        print(f"  平均段长: {total_speech_time/len(filtered):.0f}ms")

    return filtered

```
#### 5.3 格式化输出
 
 

 
    format_vad_result(格式化检测结果)
 
    Args:
        speech_segments: 语音段列表
        output_format: 输出格式 ("list", "dict", "json", "kaldi")
        include_confidence: 是否包含置信度信息
        time_unit: 时间单位 ("ms", "s", "frame")

    Returns:
        格式化后的结果

    支持的输出格式：
        - list: [[start, end], [start, end], ...]
        - dict: {"segments": [...], "stats": {...}}
        - json: JSON字符串
        - kaldi: Kaldi格式文本
