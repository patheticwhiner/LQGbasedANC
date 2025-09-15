% 离散时间谐波信号生成与对比分析
% 本程序生成多个离散时间简谐波信号，并对比合并前后的时域和频域特性
clear; close all; clc;
%% 采样参数设置
fs = 200;            % 采样频率 (Hz)
dt = 1/fs;           % 采样周期
N = 1000;            % 仿真步数
t = (0:N-1)*dt;      % 时间向量

%% 定义多个频率分量
% 可以方便地添加或减少频率分量
frequencies = [5, 10, 15, 20];  % 频率分量 (Hz)
amplitudes = [1, 0.8, 0.6, 0.4]; % 对应振幅
num_components = length(frequencies);

%% 构造每个频率的离散转移矩阵
A_matrices = cell(num_components, 1);
for i = 1:num_components
    omega_i = 2*pi*frequencies(i);
    A_matrices{i} = [cos(omega_i*dt), -sin(omega_i*dt);
                     sin(omega_i*dt),  cos(omega_i*dt)];
end

%% 单独生成各分量信号
x_states = cell(num_components, 1);
Y_signals = zeros(num_components, N);

% 初始化每个分量的状态
for i = 1:num_components
    x_states{i} = [amplitudes(i); 0];  % 调整初始幅值
end

% 迭代生成各个分量信号
for k = 1:N
    for i = 1:num_components
        x_states{i} = A_matrices{i} * x_states{i};          
        Y_signals(i, k) = x_states{i}(1);  % 提取第一个分量作为输出
    end
end

%% 合并成块对角矩阵方式生成信号
% 构造合并的块对角矩阵
Ad = blkdiag(A_matrices{:});

% 构造输出矩阵，将所有信号的第一个分量相加
C = zeros(1, 2 * num_components);
for i = 1:num_components
    C(1, 2*i-1) = 1;  % 选择每个子系统的第一个状态变量
end

% 初始化合并状态向量
x_combined = zeros(2 * num_components, 1);
for i = 1:num_components
    x_combined(2*i-1:2*i) = [amplitudes(i); 0];
end

% 初始化输出信号
Y_combined = zeros(1, N);

% 使用合并后的系统进行迭代
for k = 1:N
    x_combined = Ad * x_combined;          % 状态更新
    Y_combined(k) = C * x_combined;        % 输出计算
end

% 计算直接相加的信号
Y_direct_sum = sum(Y_signals, 1);

%% 频域分析
% 计算频率轴
freq = (0:N/2) * fs / N;  % 单边频谱的频率轴

% 计算直接相加和状态空间合并的功率谱密度
[psd_combined, f_combined] = pwelch(Y_combined, hamming(256), 128, 1024, fs);
[psd_direct_sum, f_direct] = pwelch(Y_direct_sum, hamming(256), 128, 1024, fs);

% 计算各分量的功率谱密度
psd_components = zeros(num_components, length(f_combined));
f_components = cell(num_components, 1);
for i = 1:num_components
    [psd_components(i, :), f_components{i}] = pwelch(Y_signals(i, :), hamming(256), 128, 1024, fs);
end

%% 可视化结果
figure('Position', [100, 100, 1000, 800]);

% 时域信号比较 - 各分量
subplot(3, 2, 1);
hold on;
for i = 1:num_components
    plot(t(1:N/5), Y_signals(i, 1:N/5), 'LineWidth', 1.5);
end
grid on;
title('各频率分量的时域信号');
xlabel('时间 (s)'); ylabel('幅值');
legend_labels = cell(num_components, 1);
for i = 1:num_components
    legend_labels{i} = sprintf('f = %dHz', frequencies(i));
end
legend(legend_labels);

% 时域信号比较 - 合并与直接相加
subplot(3, 2, 2);
plot(t(1:N/5), Y_direct_sum(1:N/5), 'g-', t(1:N/5), Y_combined(1:N/5), 'k--', 'LineWidth', 1.5);
grid on;
title('直接相加与状态空间合并对比');
xlabel('时间 (s)'); ylabel('幅值');
legend('直接相加', '状态空间合并');

% 频域分析 - 各分量
subplot(3, 2, 3);
hold on;
for i = 1:num_components
    semilogy(f_components{i}, psd_components(i, :), 'LineWidth', 1.5);
end
grid on;
title('各频率分量的功率谱密度');
xlabel('频率 (Hz)'); ylabel('功率/频率 (dB/Hz)');
legend(legend_labels);
xlim([0, fs/2]);

% 频域分析 - 合并与直接相加
subplot(3, 2, 4);
semilogy(f_direct, psd_direct_sum, 'g-', f_combined, psd_combined, 'k--', 'LineWidth', 1.5);
grid on;
title('直接相加与状态空间合并功率谱密度对比');
xlabel('频率 (Hz)'); ylabel('功率/频率 (dB/Hz)');
legend('直接相加', '状态空间合并');
xlim([0, fs/2]);

% 误差分析
error = Y_combined - Y_direct_sum;
subplot(3, 2, [5,6]);
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

% 在命令窗口中打印误差统计信息
fprintf('最大误差: %.4e\n', max_error);
fprintf('均方根误差: %.4e\n', rms_error);