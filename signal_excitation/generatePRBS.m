%% 生成长度为10s，采样频率10kHz的PRBS白噪声信号 
clear; close all; clc;

%% 参数设置
fs = 10000;                     % 采样频率(Hz)
duration = 10;                  % 信号持续时间(秒)
total_samples = fs * duration;  % 总样本数
n = 15;                         % 移位寄存器位数(2^15-1=32767)
taps = [15 14];                 % 抽头位置，使用15阶PRBS的标准抽头位置

%% 1. 生成基础PRBS序列
fprintf('正在生成PRBS序列...\n');
prbs_base = generate_prbs(n, taps);
prbs_length = length(prbs_base);
fprintf('基础PRBS序列长度: %d\n', prbs_length);

%% 2. 调整序列长度至所需样本数
% 计算需要重复的次数，并调整到所需长度
repeats = ceil(total_samples / prbs_length);
extended_prbs = repmat(prbs_base, 1, repeats);
prbs_signal = extended_prbs(1:total_samples);  % 截取所需长度

% 将±1序列调整为适当的幅度
amplitude = 1.0;  % 可调整信号幅度
prbs_signal = amplitude * prbs_signal;

%% 3. 信号分析与可视化
t = (0:total_samples-1)/fs;  % 时间轴(秒)

% 绘制信号时域波形(部分)
figure('Name', 'PRBS白噪声信号分析', 'Position', [100 100 900 700]);
subplot(3,1,1);
plot_samples = min(5000, length(prbs_signal));  % 只显示前0.5秒
plot(t(1:plot_samples), prbs_signal(1:plot_samples));
grid on;
title('PRBS白噪声信号(前0.5秒)');
xlabel('时间 (s)');
ylabel('幅度');

% 计算并绘制自相关函数
[acf, lags] = xcorr(prbs_signal, 'coeff');
subplot(3,1,2);
plot(lags/fs, acf);  % 转换为时间单位
grid on;
title('PRBS信号自相关函数');
xlabel('滞后时间 (s)');
ylabel('归一化相关');
xlim([-0.01 0.01]);  % 只显示近零区域

% 计算并绘制功率谱密度
window_length = 1024;
noverlap = window_length/2;
nfft = max(2048, 2^nextpow2(window_length));
[pxx, f] = pwelch(prbs_signal, hamming(window_length), noverlap, nfft, fs);

subplot(3,1,3);
plot(f, 10*log10(pxx));  % 转为dB
grid on;
title('PRBS信号功率谱密度');
xlabel('频率 (Hz)');
ylabel('功率/频率 (dB/Hz)');
xlim([0 fs/2]);

%% 4. 信号特性统计
fprintf('信号统计特性:\n');
fprintf('均值: %.6f\n', mean(prbs_signal));
fprintf('标准差: %.6f\n', std(prbs_signal));
fprintf('RMS值: %.6f\n', rms(prbs_signal));

%% 5. 保存信号数据
mydata = [(1:length(prbs_signal))'./fs, prbs_signal'];
save('prbs_signal_10s_10kHz.mat', 'mydata', 'fs', 'duration');
fprintf('信号已保存到文件: prbs_signal_10s_10kHz.mat\n');

%% PRBS生成函数
function seq = generate_prbs(n, taps)
    % 生成标准PRBS序列(长度2^n-1)
    % n: 寄存器位数
    % taps: 反馈抽头位置
    
    register = ones(1, n);    % 初始寄存器状态全1
    seq_length = 2^n - 1;     % 最大长度序列
    seq = zeros(1, seq_length);
    
    for i = 1:seq_length
        % 输出当前最右位
        seq(i) = register(end);
        
        % 计算抽头位的异或结果(mod 2 sum)
        feedback = mod(sum(register(taps)), 2);
        
        % 移位寄存器
        register = [feedback, register(1:end-1)];
    end
    
    % 转换为±1电平序列
    seq = 2*seq - 1;
end