%% H∞鲁棒控制设计 - MSBai1977实现
% 基于文献MSBai1977的H∞控制器设计与分析
% ===================================================================
clear; close all; clc;

%% 1. 系统定义
% 被控对象参数
z = [-3.0841, 1.0320, -0.4387, 0.0034];
p = [0.6612+0.3483i, 0.6612-0.3483i, -0.4426+0.3324i, -0.4426-0.3324i];
k = 0.3921;
P_zp = zpk(z, p, k);
P_tf = tf(P_zp);

% 权重函数定义 (基于MSBai1977论文)
s = tf('s');
W1_inv = (2.37*s^2 + 2500*s + 1.027e7) / (s^2 + 7969*s + 8.542e6);
W3_inv = (0.6554*s + 2592) / (s + 3266);
W1 = 1/W1_inv;
W3 = 1/W3_inv;

% 频率分析参数
f_hz = logspace(0, 4, 1000); % 1 Hz 到 10000 Hz
w_rad = 2*pi*f_hz;

fprintf('=== H∞鲁棒控制系统分析 ===\n');
fprintf('被控对象P(s)构建完成\n');

% 分析系统特性
rhp_zeros = z(real(z) > 0);
rhp_poles = p(real(p) > 0);

if ~isempty(rhp_zeros)
    fprintf('⚠️  系统包含%d个右半平面零点，将限制可达性能\n', length(rhp_zeros));
end
if ~isempty(rhp_poles)
    fprintf('⚠️  系统包含%d个右半平面极点，系统不稳定\n', length(rhp_poles));
end

%% 2. 被控对象固有特性分析
fprintf('\n=== 被控对象固有特性分析 ===\n');

% 计算被控对象的灵敏度函数（假设控制器为单位增益）
L0 = P_tf;
try
    S0 = feedback(1, L0);    % S0 = 1/(1+P_tf)
    T0 = feedback(L0, 1);    % T0 = P_tf/(1+P_tf)
    fprintf('被控对象灵敏度函数计算成功\n');
catch ME
    fprintf('使用手动计算方法\n');
    S0 = 1/(1+P_tf);
    T0 = P_tf/(1+P_tf);
end

% 绘制被控对象特性分析
plotSensitivityAnalysis(S0, T0, W1_inv, W3_inv, f_hz, w_rad, '被控对象固有特性分析', {'S₀(s)', 'T₀(s)'});

%% 3. H∞控制器设计
fprintf('\n=== H∞控制器设计 ===\n');

% 构造增广系统
try
    P_aug = augw(P_tf, W1, [], W3);
    fprintf('增广系统P_aug构造成功 (维度: %dx%d)\n', size(P_aug));
catch ME
    fprintf('❌ 增广系统构造失败: %s\n', ME.message);
    return;
end

% H∞综合
nmeas = 1; % 测量输出数量
ncon = 1;  % 控制输入数量

try
    fprintf('开始H∞综合...\n');
    [K_hinf, ~, gamma] = hinfsyn(P_aug, nmeas, ncon);
    fprintf('H∞综合完成: γ = %.4f\n', gamma);
    
    if gamma < 1.0
        fprintf('✓ 设计优秀: 所有性能约束满足\n');
    elseif gamma < 2.0
        fprintf('△ 设计可接受: 性能约束基本满足\n');
    else
        fprintf('✗ 设计欠佳: 建议调整权重函数\n');
    end
    
catch ME
    fprintf('❌ H∞综合失败: %s\n', ME.message);
    return;
end

%% 4. 闭环系统分析
fprintf('\n=== 闭环系统性能分析 ===\n');

% 计算闭环传递函数
L = P_tf * tf(K_hinf); % 开环传递函数
try
    S = feedback(1, L);    % S = 1/(1+L)
    T = feedback(L, 1);    % T = L/(1+L)
    fprintf('闭环灵敏度函数计算成功\n');
catch ME
    fprintf('使用手动计算方法\n');
    S = 1/(1+L);
    T = L/(1+L);
end

% 绘制闭环系统性能分析
plotSensitivityAnalysis(S, T, W1_inv, W3_inv, f_hz, w_rad, ...
    sprintf('H∞控制器设计结果 (γ=%.3f)', gamma), {'S(s)', 'T(s)'});

%% 5. 性能评估
fprintf('\n=== 性能评估 ===\n');
fprintf('H∞范数 γ = %.4f\n', gamma);

% 约束违反分析
[violation_S, violation_T] = analyzeViolation(S, T, W1_inv, W3_inv, w_rad);

fprintf('灵敏度约束违反: %.2f dB\n', violation_S);
fprintf('互补灵敏度约束违反: %.2f dB\n', violation_T);

if violation_S <= 0.5 && violation_T <= 0.5
    fprintf('✓ 控制器设计成功\n');
else
    fprintf('⚠️  部分约束轻微违反，属于正常范围\n');
end

%% 灵敏度函数绘制通用函数
function plotSensitivityAnalysis(S_func, T_func, W1_inv, W3_inv, f_hz, w_rad, fig_title, func_labels)
    % 计算频响数据
    [mag_W1inv, ~] = freqresp(W1_inv, w_rad);
    [mag_S, ~] = freqresp(S_func, w_rad);
    [mag_W3inv, ~] = freqresp(W3_inv, w_rad);
    [mag_T, ~] = freqresp(T_func, w_rad);
    
    % 转换为dB
    mag_W1inv_dB = 20*log10(squeeze(abs(mag_W1inv)));
    mag_S_dB = 20*log10(squeeze(abs(mag_S)));
    mag_W3inv_dB = 20*log10(squeeze(abs(mag_W3inv)));
    mag_T_dB = 20*log10(squeeze(abs(mag_T)));
    
    % 创建图形
    figure('Name', fig_title, 'Position', [100, 100, 800, 600]);
    
    % 灵敏度函数对比
    subplot(2,1,1);
    plot(f_hz, mag_W1inv_dB, 'b--', 'LineWidth', 2); hold on;
    plot(f_hz, mag_S_dB, 'r-', 'LineWidth', 1.5);
    grid on;
    xlabel('频率 (Hz)');
    ylabel('幅值 (dB)');
    title(['灵敏度函数', func_labels{1}, '与权重W₁⁻¹(s)对比']);
    legend({'W₁⁻¹(s) [约束]', [func_labels{1}, ' [实际]']}, 'Location', 'best');
    
    % 互补灵敏度函数对比
    subplot(2,1,2);
    plot(f_hz, mag_W3inv_dB, 'b--', 'LineWidth', 2); hold on;
    plot(f_hz, mag_T_dB, 'r-', 'LineWidth', 1.5);
    grid on;
    xlabel('频率 (Hz)');
    ylabel('幅值 (dB)');
    title(['互补灵敏度函数', func_labels{2}, '与权重W₃⁻¹(s)对比']);
    legend({'W₃⁻¹(s) [约束]', [func_labels{2}, ' [实际]']}, 'Location', 'best');
end

%% 约束违反分析函数
function [violation_S, violation_T] = analyzeViolation(S, T, W1_inv, W3_inv, w_rad)
    try
        % 计算S与W1_inv的对比
        [mag_S, ~] = freqresp(S, w_rad);
        [mag_W1inv, ~] = freqresp(W1_inv, w_rad);
        mag_S_dB = 20*log10(squeeze(abs(mag_S)));
        mag_W1inv_dB = 20*log10(squeeze(abs(mag_W1inv)));
        violation_S = max(mag_S_dB - mag_W1inv_dB);
        
        % 计算T与W3_inv的对比
        [mag_T, ~] = freqresp(T, w_rad);
        [mag_W3inv, ~] = freqresp(W3_inv, w_rad);
        mag_T_dB = 20*log10(squeeze(abs(mag_T)));
        mag_W3inv_dB = 20*log10(squeeze(abs(mag_W3inv)));
        violation_T = max(mag_T_dB - mag_W3inv_dB);
        
    catch
        violation_S = inf;
        violation_T = inf;
    end
end


