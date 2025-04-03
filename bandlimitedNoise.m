clear; close all; clc;
%% 设计带通滤波器
fs = 8e3;                   % 采样频率 (Hz)
f_low = 200;                  % 低截止频率 (Hz)
f_high = 500;                % 高截止频率 (Hz)
order = 4;                   % 滤波器阶数

% 将实际频率转换为归一化截止频率（相对于奈奎斯特频率fs/2）
% butter函数要求归一化频率范围为[0,1]
fc = [f_low f_high]/(fs/2);  

% 生成Butterworth带通滤波器
[b, a] = butter(order, fc, 'bandpass');

% 转换为状态空间模型
[A_w2, B_w2, C_w2, D_w2] = tf2ss(b, a);
state_dim = size(A_w2, 1);

% 生成白噪声输入
N = 10000;                   % 数据点数
n = randn(N, 1);             % 高斯白噪声

% 仿真状态空间模型
w2 = zeros(N, 1);
x = zeros(state_dim, 1);     % 初始化状态
for k = 1:N
    x = A_w2 * x + B_w2 * n(k);
    w2(k) = C_w2 * x + D_w2 * n(k);
end

% 绘制时域波形
figure; subplot(2,1,1);
t = (0:N-1)/fs;  % 时间向量（秒）
plot(t, w2);
xlabel('Time (s)'); ylabel('Amplitude');
title('Bandlimited Noise (Time Domain)');
% 计算信号的频谱
W2_fft = fft(w2); % 计算FFT
frequencies = (0:N/2-1)*(fs/N); % 频率轴 (Hz)
W2_magnitude = abs(W2_fft(1:N/2)); % 取前半部分幅值
% 绘制频谱
subplot(2,1,2);
plot(frequencies, 20*log10(W2_magnitude));
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Frequency Spectrum of Bandlimited Noise');
grid on;

% 计算并绘制频谱（使用实际频率Hz）
figure; subplot(2,1,1);
[h, f] = freqz(b, a, 1024, fs);  % 注意这里传入fs参数
mag = 20*log10(abs(h));
plot(f, mag);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Filter Frequency Response');
grid on;
% 标注截止频率
hold on;
max_mag = max(mag);
yline(max_mag-3, '--r', '-3dB', 'LabelHorizontalAlignment', 'left');
xline(f_low, '--g', [num2str(f_low) ' Hz'], 'LabelVerticalAlignment', 'bottom');
xline(f_high, '--g', [num2str(f_high) ' Hz'], 'LabelVerticalAlignment', 'bottom');
hold off;

% 计算信号的频谱
subplot(2,1,2);
[Pxx, f] = pwelch(w2, hanning(512), 256, 1024, fs);
plot(f, 10*log10(Pxx));
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('Power Spectral Density of Bandlimited Noise');
grid on;
% 标注截止频率
hold on;
xline(f_low, '--g', [num2str(f_low) ' Hz'], 'LabelVerticalAlignment', 'bottom');
xline(f_high, '--g', [num2str(f_high) ' Hz'], 'LabelVerticalAlignment', 'bottom');
hold off;

save('bandlimitedNoise.mat', 'A_w2', 'B_w2', 'C_w2', 'D_w2');
%% test:根据传函估算截止频率（还不通）
sys = ss(A_w2, B_w2, C_w2, D_w2, 1/fs);  % 创建离散系统模型
[H, f] = freqresp(sys, linspace(0, fs/2, 1000)); 
H_mag = squeeze(20*log10(abs(H)));
% 找到 -3dB 截止频率
max_mag = max(H_mag);
cutoff_idx = find(H_mag >= max_mag - 3);
f_low_actual = f(min(cutoff_idx))   % 实际低截止频率
f_high_actual = f(max(cutoff_idx))  % 实际高截止频率