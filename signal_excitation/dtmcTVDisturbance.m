% 离散时间谐波信号生成与对比分析 - 支持时变频率
% 本程序生成多个离散时间简谐波信号，并对比合并前后的时域和频域特性
% 新增特性：支持时变频率分量

clear; close all; clc;
%% 采样参数设置
fs = 100;            % 采样频率 (Hz)
dt = 1/fs;           % 采样周期
N = 1000;            % 仿真步数
t = (0:N-1)*dt;      % 时间向量

%% 定义基础频率分量
% 基本频率定义（初始值）
base_frequencies = [5, 10, 15, 20];  % 起始频率分量 (Hz)
amplitudes = [1, 0.8, 0.6, 0.4];     % 对应振幅
num_components = length(base_frequencies);

%% 定义频率变化模式
% 频率变化模式选择: 'constant', 'linear', 'sinusoidal', 'custom'
freq_variation_mode = 'custom';  

% 根据选择的模式定义时变频率
freq_variation = cell(num_components, 1);  % 存储每个分量的频率变化

switch freq_variation_mode
    case 'constant'
        % 常数频率（不变）
        for i = 1:num_components
            freq_variation{i} = base_frequencies(i) * ones(1, N);
        end
        
    case 'linear'
        % 线性变化频率 (类似啁啾信号)
        slope = [0.01, 0.02, 0.015, -0.01];  % 每个分量的频率变化斜率
        for i = 1:num_components
            freq_variation{i} = base_frequencies(i) + slope(i) * t * fs;
        end
        
    case 'sinusoidal'
        % 正弦变化频率
        mod_depth = [0.2, 0.3, 0.25, 0.15];  % 频率调制深度（相对于基频）
        mod_freq = [0.5, 0.3, 0.4, 0.6];     % 调制频率 (Hz)
        for i = 1:num_components
            freq_variation{i} = base_frequencies(i) * (1 + mod_depth(i) * sin(2*pi*mod_freq(i)*t));
        end
        
    case 'custom'
        % 自定义频率变化（示例：阶跃变化）
        for i = 1:num_components
            freq_variation{i} = base_frequencies(i) * ones(1, N);
            step_point = round(N/2);  % 在中间位置发生阶跃
            freq_variation{i}(step_point:end) = base_frequencies(i) * 1.5;  % 增加50%频率
        end
        
    otherwise
        error('未知的频率变化模式');
end

%% 预分配存储空间
x_states = cell(num_components, 1);
Y_signals = zeros(num_components, N);
A_matrices = cell(num_components, N);  % 存储每个时刻每个分量的状态矩阵

% 初始化每个分量的状态
for i = 1:num_components
    x_states{i} = [amplitudes(i); 0];  % 调整初始幅值
end

%% 单独生成各分量时变频率信号
for k = 1:N
    for i = 1:num_components
        % 获取当前时刻的频率
        current_freq = freq_variation{i}(k);
        omega_i = 2*pi*current_freq;
        
        % 为此时刻计算离散状态转移矩阵
        A_matrices{i, k} = [cos(omega_i*dt), -sin(omega_i*dt);
                           sin(omega_i*dt),  cos(omega_i*dt)];
        
        % 使用当前的转移矩阵更新状态
        x_states{i} = A_matrices{i, k} * x_states{i};          
        Y_signals(i, k) = x_states{i}(1);  % 提取第一个分量作为输出
    end
end

%% 合并信号生成 - 使用时变块对角系统
x_combined = zeros(2 * num_components, 1);
for i = 1:num_components
    x_combined(2*i-1:2*i) = [amplitudes(i); 0];
end

% 构造输出矩阵，将所有信号的第一个分量相加
C = zeros(1, 2 * num_components);
for i = 1:num_components
    C(1, 2*i-1) = 1;  % 选择每个子系统的第一个状态变量
end

% 初始化输出信号
Y_combined = zeros(1, N);

% 使用合并后的系统进行迭代
for k = 1:N
    % 构建当前时刻的块对角系统矩阵
    current_A_matrices = cell(1, num_components);
    for i = 1:num_components
        current_A_matrices{i} = A_matrices{i, k};
    end
    Ad_k = blkdiag(current_A_matrices{:});
    
    % 使用当前的块对角矩阵更新状态
    x_combined = Ad_k * x_combined;
    Y_combined(k) = C * x_combined;
end

% 计算直接相加的信号
Y_direct_sum = sum(Y_signals, 1);

%% 时频分析 - 添加短时傅里叶变换分析
window = 128;  % 窗口大小
noverlap = 100;  % 重叠点数
nfft = 256;    % FFT点数

% 生成时频谱图数据
[S_direct, F_direct, T_direct] = spectrogram(Y_direct_sum, window, noverlap, nfft, fs);
[S_combined, F_combined, T_combined] = spectrogram(Y_combined, window, noverlap, nfft, fs);

%% 可视化结果
figure('Position', [100, 100, 1200, 800]);

% 时域信号比较 - 各分量
subplot(3, 3, 1);
hold on;
for i = 1:num_components
    plot(t(1:N/5), Y_signals(i, 1:N/5), 'LineWidth', 1.5);
end
grid on;
title('各频率分量的时域信号');
xlabel('时间 (s)'); ylabel('幅值');
legend_labels = cell(num_components, 1);
for i = 1:num_components
    legend_labels{i} = sprintf('分量 %d', i);
end
legend(legend_labels);

% 时域信号比较 - 合并与直接相加
subplot(3, 3, 2);
plot(t(1:N/5), Y_direct_sum(1:N/5), 'g-', t(1:N/5), Y_combined(1:N/5), 'k--', 'LineWidth', 1.5);
grid on;
title('直接相加与状态空间合并对比');
xlabel('时间 (s)'); ylabel('幅值');
legend('直接相加', '状态空间合并');

% 时变频率轨迹
subplot(3, 3, 3);
hold on;
for i = 1:num_components
    plot(t(1:N/5), freq_variation{i}(1:N/5), 'LineWidth', 1.5);
end
grid on;
title('时变频率轨迹');
xlabel('时间 (s)'); ylabel('频率 (Hz)');
legend(legend_labels);

% 时频谱图 - 直接相加信号
subplot(3, 3, 4);
imagesc(T_direct, F_direct, 10*log10(abs(S_direct)));
axis xy;
title('直接相加信号的时频谱图');
xlabel('时间 (s)'); ylabel('频率 (Hz)');
colorbar;
caxis([-60, 0] + max(10*log10(abs(S_direct(:)))));

% 时频谱图 - 状态空间合并信号
subplot(3, 3, 5);
imagesc(T_combined, F_combined, 10*log10(abs(S_combined)));
axis xy;
title('状态空间合并信号的时频谱图');
xlabel('时间 (s)'); ylabel('频率 (Hz)');
colorbar;
caxis([-60, 0] + max(10*log10(abs(S_combined(:)))));

% 时频谱图 - 差异
subplot(3, 3, 6);
imagesc(T_direct, F_direct, 10*log10(abs(S_direct - S_combined)));
axis xy;
title('时频谱图差异');
xlabel('时间 (s)'); ylabel('频率 (Hz)');
colorbar;

% 误差分析
error = Y_combined - Y_direct_sum;
subplot(3, 3, [7, 8, 9]);
plot(t, error);
grid on;
title('两种方法的误差');
xlabel('时间 (s)'); ylabel('误差幅值');

% 计算误差统计
max_error = max(abs(error));
rms_error = sqrt(mean(error.^2));
annotation('textbox', [0.5, 0.05, 0.4, 0.05], 'String', ...
    sprintf('最大误差: %.4e, RMS误差: %.4e', max_error, rms_error), ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white');

% 在命令窗口中打印误差统计信息和频率变化模式
fprintf('频率变化模式: %s\n', freq_variation_mode);
fprintf('最大误差: %.4e\n', max_error);
fprintf('均方根误差: %.4e\n', rms_error);

%% 保存一个额外的图形，用于频率变化动画
figure('Position', [100, 100, 800, 600]);
subplot(2, 1, 1);
plot(t, Y_direct_sum, 'g-', 'LineWidth', 1.5);
hold on;
h_marker = plot(0, Y_direct_sum(1), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
title('信号随时间的变化');
xlabel('时间 (s)'); ylabel('幅值');
grid on;
ylim([min(Y_direct_sum)*1.1, max(Y_direct_sum)*1.1]);

subplot(2, 1, 2);
hold on;
for i = 1:num_components
    plot(t, freq_variation{i}, 'LineWidth', 1.5);
    h_freq(i) = plot(0, freq_variation{i}(1), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'auto');
end
title('频率随时间的变化');
xlabel('时间 (s)'); ylabel('频率 (Hz)');
grid on;
legend(legend_labels);
ylim([0, 1.1*max(cellfun(@max, freq_variation))]);

% 创建动画效果（可选）
animate_freq_variation = true;  % 设为true以启用动画
if animate_freq_variation
    for k = 1:10:N
        set(h_marker, 'XData', t(k), 'YData', Y_direct_sum(k));
        for i = 1:num_components
            set(h_freq(i), 'XData', t(k), 'YData', freq_variation{i}(k));
        end
        drawnow;
        pause(0.05);
    end
end