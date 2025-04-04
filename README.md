# LQGbasedANC

Originally Designed by [24s153046@stu.hit.edu.cn](mailto:24s153046@stu.hit.edu.cn)

---

bandlimitedNoise.m：根据采样频率、目标带限噪声的频带范围生成外系统模型（带通滤波器→状态空间模型）

systemidentification.m：对所设计的次级通路（带通滤波器）选取ARMAX模型辨识，并转换为状态空间模型

identificationDelaytest.m：由于使用MATLAB函数辨识ARMAX模型需要提前知道模型纯时延个数，可以使用程序测试最优的模型时延个数

LQGdemo.m：将上述两部分程序生成的状态空间模型合并为增广状态空间模型，并据此设计LQR与LQE（Kalman Filter），完成ANC仿真

---

## 理论基础

#### LQR环节

#### LQE/Kalman Filter环节

## 带限噪声生成

#### 实验内容

1. 设置实验采取的采样频率（Hz），带通滤波器的高低截止频率（Hz），带通滤波器类型以及滤波器阶次，使用MATLAB函数设计带通滤波器；程序中选用了常规的butterworth滤波器，输出结果为离散冲激响应函数/传递函数。
2. 使用tf2ss函数将上面得到的传递函数转换为相应的状态空间模型；
3. 输入信号选取全频带白噪声，状态空间模型仿真得到相应的带限噪声输出。

#### 实验结果

* 绘制所生成带限噪声的时域波形，使用fft函数分析其频谱（幅度谱），验证频率范围与所设范围一致；
* 绘制所涉及的带通滤波器的Bode幅频曲线，以及最终所生成噪声的pwelch功率谱密度函数曲线，验证频率范围与所设范围一致。

<figure align = center>
    <img src="assets\bndltd1.svg" width = 49% />
    <img src="assets\bndltd2.svg" width = 49% />
</figure>

+ 将所得到的外系统（exosystem）状态空间模型导出为bandlimitedNoise.mat，可导入到其它脚本文件中用于设计相应的控制器。

## ARMAX次级通路辨识

#### 实验内容

1. 设置实验采取的采样频率（Hz），带通滤波器的高低截止频率（Hz），带通滤波器类型以及滤波器阶次，使用MATLAB函数设计带通滤波器；参考MATLAB的ActiveNoiseCancellation示例程序，选用了Chebyshev II型滤波器，输出结果为传递函数。
2. 输入信号选取全频带白噪声，使用上述带通滤波器（次级通路）滤波，得到辨识所需的输入、输出信号序列。
3. 设定ARMAX模型的阶次以及延迟，使用 `iddata` 以及 `armax` 函数辨识，`polydata` 函数提取多项式系数。并绘制系列图像验证辨识效果。
4. 使用 `tf2ss` 函数，将传递函数模型转换为状态空间模型。并绘制系列图像验证辨识、状态空间模型的频率响应是否等价。

#### 实验结果

* 绘制所生成带限噪声的时域波形，使用fft函数分析其频谱（幅度谱），验证频率范围与所设范围一致；
* 绘制所涉及的带通滤波器的Bode幅频曲线，以及最终所生成噪声的pwelch功率谱密度函数曲线，验证频率范围与所设范围一致。

<figure align = center>
    <img src="assets\sysId1.svg" width = 49% />
    <img src="assets\sysId2.svg" width = 49% />
</figure>

+ 将所得到的次级通路/对象系统（plant）状态空间模型导出为systemIdentification.mat，可导入到其它脚本文件中用于设计相应的控制器。

## 基于LQR的主动控制



## 基于LQG的主动控制



<img src="assets\LQGdemo1.svg" width = 60% />

## 不足与改进建议

1. 所生成的传递函数究竟是连续复频域传递函数还是离散复频域传函？所生成的状态空间矩阵属于连续时间模型还是离散时间模型？仍有待理论说明。
2. 未对多种条件作重复试验，实验条件特殊，且设置比较随意，不确定推广到其它数据是否适用。
3. 未讨论LQR主动控制的局限性，未说明为什么要引入LQG；未探究LQR与LQG反馈控制的带宽限制。
4. 对于LQR，LQG设计的参数（权重矩阵选取）未提供理论参考，未探究超出建议值会发生什么

## 参考文献

[1] 钱梵梵. 基于Youla参数化的自适应输出调节及应用研究[D/OL]. 上海大学, 2022[2024-12-18]. [https://link.cnki.net/doi/10.27300/d.cnki.gshau.2022.000228](https://link.cnki.net/doi/10.27300/d.cnki.gshau.2022.000228). DOI:[10.27300/d.cnki.gshau.2022.000228](https://doi.org/10.27300/d.cnki.gshau.2022.000228).

[2] [Steve Brunton](https://www.youtube.com/@Eigensteve). Control Bootcamp: Linear Quadratic Gaussian (LQG)[视频/OL].  (2017-02-07). [2025-04-04]. [https://www.youtube.com/watch?v=H4_hFazBGxU.](https://www.youtube.com/watch?v=H4_hFazBGxU.)
