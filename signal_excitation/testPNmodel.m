% testPNmodel.m - 分析Simulink模型中生成的PN序列
close all; clc;

%% 定义参数 - 修改这些参数即可改变整个仿真
sample_time = 0.0001;       % 采样时间(秒)
sim_time = 10^4*sample_time; % 仿真时间(秒)
fs = 1/sample_time;          % 采样频率(Hz)

% PN序列特定参数
pn_polynomial = [1 0 1 0 0 1];  % 默认生成多项式，可根据实际修改
pn_initial_states = [0 0 0 0 1]; % 初始状态，确保不是全0
pn_samples_per_bit = 1;         % 每比特的样本数

%% 打开模型和配置
model_name = 'simBandLtdWN';  % 修改为您的模型名称
open_system(model_name);

% 设置PN Sequence Generator参数
set_param([model_name '/PN Sequence Generator'], 'poly', mat2str(pn_polynomial));
set_param([model_name '/PN Sequence Generator'], 'ini_sta', mat2str(pn_initial_states));
set_param([model_name '/PN Sequence Generator'], 'sampPerFrame', num2str(1));
set_param([model_name '/PN Sequence Generator'], 'Ts', num2str(sample_time));
set_param([model_name '/PN Sequence Generator'], 'outDataType', 'double');

% 设置模型仿真时间
set_param(model_name, 'StopTime', num2str(sim_time));

%% 运行仿真
out = sim(model_name);

%% 提取数据和可视化
% 提取PN序列数据 (假设输出变量名为'prbs')
prbs_data = out.get('pn_sequence');  % 根据实际信号名修改
prbs_data = prbs_data(:);
t = 0:sample_time:sim_time;   % 时间轴

% 绘制时域波形
figure;
subplot(3,1,1);
stairs(t, prbs_data, 'LineWidth', 1.2);
grid on;
title(['PN序列波形 (采样率 = ' num2str(fs) ' Hz)']);
xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 min(0.001, sim_time)]); % 只显示部分时域以便观察细节

% 计算自相关函数
[acf, lags] = xcorr(prbs_data, 'coeff');
subplot(3,1,2);
plot(lags*sample_time, acf, 'LineWidth', 1.2);
grid on;
title('自相关函数 - PN序列特性');
xlabel('Lag (s)');
ylabel('Normalized Correlation');
xlim([-0.1, 0.1]);  % 只显示主要的自相关范围

% 计算功率谱密度（PSD）
signal_length = length(prbs_data);

% 自适应窗口长度
window_length = min(1024, floor(signal_length/4));
noverlap = floor(window_length/2);
nfft = max(1024, 2^nextpow2(window_length*4)); % 增加频率分辨率

[pxx, f] = pwelch(prbs_data, window_length, noverlap, nfft, fs);

% 绘制PSD
subplot(3,1,3);
semilogy(f, pxx, 'LineWidth', 1.2);
grid on;
title(['功率谱密度 (窗长 = ' num2str(window_length) ', NFFT = ' num2str(nfft) ')']);
xlabel('Frequency (Hz)'); ylabel('Power/Frequency');
xlim([0, fs/2]);  % 显示到奈奎斯特频率

%% PN序列特性分析
figure;

% 1. 信号幅值分布 - 应该是两个清晰的值
subplot(2,2,1);
[counts, edges] = histcounts(prbs_data, 'BinMethod', 'auto');
bar(edges(1:end-1), counts, 'BarWidth', 1);
title('幅值分布');
xlabel('幅值'); ylabel('计数');
grid on;

% 2. 比特转换特性
transitions = abs(diff(prbs_data));
subplot(2,2,2);
stem(t(1:100), prbs_data(1:100), 'filled', 'LineWidth', 1.2);
title('比特转换图');
xlabel('时间 (s)'); ylabel('幅值');
grid on;

% 3. 周期性分析 - 检测PN序列周期
correlation_peak_indices = find(acf > 0.9);  % 找到大于0.9的自相关峰值
correlation_peak_indices = correlation_peak_indices(correlation_peak_indices > length(acf)/2);
if ~isempty(correlation_peak_indices)
    period_samples = correlation_peak_indices(1) - length(acf)/2;
    fprintf('PN序列周期估计: %.0f 样本 (%.6f 秒)\n', period_samples, period_samples*sample_time);
    
    % 绘制周期性展示
    subplot(2,2,3);
    max_display = min(3*period_samples, length(prbs_data));
    plot(t(1:max_display), prbs_data(1:max_display));
    title(['周期性展示 (估计周期: ' num2str(period_samples) ' 样本)']);
    xlabel('时间 (s)'); ylabel('幅值');
    grid on;
else
    fprintf('未检测到明确的PN序列周期\n');
end

% 4. PN序列序列特性
% 计算理论PN序列长度
seq_length = 2^length(pn_initial_states)-1;
fprintf('理论PN序列长度: %d (基于 %d 位移位寄存器)\n', seq_length, length(pn_initial_states));
fprintf('理论PN序列周期: %.6f 秒\n', seq_length*sample_time);

% 累积能量分布
subplot(2,2,4);
cumulative_energy = cumsum(prbs_data.^2)/sum(prbs_data.^2);
plot(t(1:length(cumulative_energy)), cumulative_energy);
title('累积能量分布');
xlabel('时间 (s)'); ylabel('归一化累积能量');
grid on;

%% PRBS 统计特性
rms_value = rms(prbs_data);
peak_value = max(abs(prbs_data));
peak_factor = peak_value/rms_value;

fprintf('\n--- PN序列统计分析 ---\n');
fprintf('RMS值: %.4f\n', rms_value);
fprintf('峰值: %.4f\n', peak_value);
fprintf('峰值因数: %.4f\n', peak_factor);
fprintf('均值: %.4e\n', mean(prbs_data));
fprintf('方差: %.4f\n', var(prbs_data));

% 探测序列均衡性 (±1电平应基本相等)
level_ratio = sum(prbs_data > 0)/sum(prbs_data < 0);
fprintf('正/负电平比例: %.4f\n', level_ratio);