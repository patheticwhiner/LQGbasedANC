%% 修复版本的PRBS生成与频谱分析
clear; close all; clc;

%% 参数设置
n = 8;                      % 移位寄存器位数
taps = [8 6 5 4];           % 抽头位置
p_values = [1, 2, 3];       % 分频系数
fs = 1;                     % 归一化采样频率

%% 1. 生成基础PRBS序列
prbs_base = generate_prbs(n, taps);
N_base = length(prbs_base); % 应为2^n-1=255

% 验证生成的序列
figure('Name', 'PRBS序列验证');
subplot(2,1,1);
stairs(1:50, prbs_base(1:50));
title('生成的PRBS序列(前50个样本)');
grid on;
ylim([-1.5 1.5]);

% 验证自相关特性
[acf, lags] = xcorr(prbs_base, 'coeff');
subplot(2,1,2);
plot(lags, acf);
title('PRBS序列自相关函数');
xlabel('滞后');
ylabel('自相关');
grid on;
xlim([-50 50]);

%% 2. 生成分频序列并计算PSD
figure('Name', 'PRBS分频及功率谱密度', 'Position', [100 100 800 600]);

% 子图1: 显示不同分频的时域信号
subplot(2,1,1);
hold on;
for i = 1:length(p_values)
    p = p_values(i);
    signal_p = repelem(prbs_base, p);
    % 只显示前100个样本
    if i == 1
        stairs(1:100, signal_p(1:100), 'b-', 'LineWidth', 1.5);
        stairs(1:100, signal_p(1:100) - 3*i, 'Color', [0 0.5 0], 'LineWidth', 1.5);
    else
        stairs(1:100, signal_p(1:100) - 3*i, 'Color', [0 0.5 0], 'LineWidth', 1.5);
    end
end
hold off;
yticks([-9 -6 -3 0]);
yticklabels({'p=3', 'p=2', 'p=1', '幅值'});
grid on;
title('不同分频系数的PRBS序列(前100个样本)');
xlabel('样本');

% 子图2: 显示PSD及凹陷位置
subplot(2,1,2);
hold on;

colors = {'b', 'g', 'r'};
legends = cell(1, length(p_values));
LenN = 128;

for i = 1:length(p_values)
    p = p_values(i);
    signal_p = repelem(prbs_base, p);
    
    % 提高FFT点数和修改窗口参数
    signal_length = length(signal_p);
    win_length = min(LenN, signal_length);
    overlap = 0.75;  % 75%重叠率（增加平滑度）
    noverlap = floor(win_length * overlap);
    nfft = max(4096, 2^nextpow2(signal_length));  % 大幅增加FFT点数
    
    % 使用改进的参数计算PSD
    [pxx_raw, f] = pwelch(signal_p, blackmanharris(win_length), noverlap, nfft, fs);
    
    % 对PSD进行额外的平滑处理（可选）
    if signal_length > LenN
        window_size = 5;  % 平滑窗口大小
        pxx = movmean(pxx_raw, window_size);
    else
        pxx = pxx_raw;
    end
    
    % 归一化PSD
    pxx_norm = 10*log10(pxx/max(pxx)) + 30;  % 转换为dB并平移
    
    % 绘制PSD曲线
    plot(f, pxx_norm, 'Color', colors{i}, 'LineWidth', 1.5);
    
%     % 标记理论凹陷位置
%     if p > 1
%         notch_freq = fs/p;
%         xline(notch_freq, '--', sprintf('f_s/%d', p), 'Color', colors{i}, 'Alpha', 0.7);
%     end
    
    legends{i} = sprintf('p=%d', p);
end

hold off;
grid on;
xlim([0 0.5]);
ylim([-20 30]);
xlabel('归一化频率 (× f_s)');
ylabel('功率谱密度 (dB)');
title('PRBS分频序列的功率谱密度');
legend(legends, 'Location', 'southwest');

%% 辅助函数：改进的PRBS生成器
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
    
    % 转换为±1电平序列(更适合信号处理)
    seq = 2*seq - 1;
end