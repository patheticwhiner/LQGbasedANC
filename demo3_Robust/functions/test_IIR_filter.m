%% IIR_filter函数测试用例
% 验证自定义IIR_filter函数与MATLAB内置filter函数的一致性

clear; close all; clc;
addpath('functions');

%% 测试用例1: 简单一阶低通滤波器
fprintf('=== 测试用例1: 一阶低通滤波器 ===\n');

% 定义一阶低通滤波器 H(z) = 0.5/(z-0.3) = 0.5z^(-1)/(1-0.3z^(-1))
% 标准形式: H(z) = b0/(1 + a1*z^(-1))
% 差分方程: y[n] = b0*x[n] - a1*y[n-1]

num1 = [0, 0.5];        % [0, b0] -> 0*z^0 + 0.5*z^(-1)
den1 = [1, -0.3];       % [1, a1] -> 1 - 0.3*z^(-1)

% 测试输入: 单位脉冲
N = 10;
x = [1, zeros(1, N-1)];  % 单位脉冲

% MATLAB内置filter函数结果
y_matlab = filter(num1, den1, x);

% 自定义IIR_filter函数结果
y_custom = zeros(1, N);
u_hist = zeros(1, length(num1));
y_hist = zeros(1, length(den1));

for k = 1:N
    [y_custom(k), u_hist, y_hist] = IIR_filter(num1, den1, x(k), u_hist, y_hist);
end

% 比较结果
error1 = max(abs(y_matlab - y_custom));
fprintf('一阶滤波器最大误差: %.2e\n', error1);

if error1 < 1e-12
    fprintf('✓ 测试通过!\n');
else
    fprintf('✗ 测试失败!\n');
end

%% 测试用例2: 二阶带通滤波器
fprintf('\n=== 测试用例2: 二阶带通滤波器 ===\n');

% 设计二阶带通滤波器
fs = 1000;  % 采样频率
fc = 100;   % 中心频率
Q = 5;      % 品质因数

s = tf('s');
H_analog = s / (s^2 + 2*pi*fc/Q*s + (2*pi*fc)^2);
H_digital = c2d(H_analog, 1/fs, 'tustin');

[num2, den2] = tfdata(H_digital, 'v');

% 测试输入: 随机信号
N = 100;
x = randn(1, N);

% MATLAB内置filter函数结果
y_matlab = filter(num2, den2, x);

% 自定义IIR_filter函数结果
y_custom = zeros(1, N);
u_hist = zeros(1, length(num2));
y_hist = zeros(1, length(den2));

for k = 1:N
    [y_custom(k), u_hist, y_hist] = IIR_filter(num2, den2, x(k), u_hist, y_hist);
end

% 比较结果
error2 = max(abs(y_matlab - y_custom));
fprintf('二阶滤波器最大误差: %.2e\n', error2);

if error2 < 1e-12
    fprintf('✓ 测试通过!\n');
else
    fprintf('✗ 测试失败!\n');
end

%% 测试用例3: 复杂高阶滤波器
fprintf('\n=== 测试用例3: 高阶椭圆滤波器 ===\n');

% 设计5阶椭圆低通滤波器
[b, a] = ellip(5, 1, 40, 0.2);  % 5阶，1dB通带波纹，40dB阻带衰减，归一化截止频率0.2

% 测试输入: 正弦波 + 噪声
N = 200;
t = (0:N-1);
x = sin(0.1*pi*t) + 0.5*sin(0.8*pi*t) + 0.1*randn(1, N);

% MATLAB内置filter函数结果
y_matlab = filter(b, a, x);

% 自定义IIR_filter函数结果
y_custom = zeros(1, N);
u_hist = zeros(1, length(b));
y_hist = zeros(1, length(a));

for k = 1:N
    [y_custom(k), u_hist, y_hist] = IIR_filter(b, a, x(k), u_hist, y_hist);
end

% 比较结果
error3 = max(abs(y_matlab - y_custom));
fprintf('高阶滤波器最大误差: %.2e\n', error3);

if error3 < 1e-10  % 高阶滤波器可能有稍大的数值误差
    fprintf('✓ 测试通过!\n');
else
    fprintf('✗ 测试失败!\n');
end

%% 测试用例4: 边界条件测试
fprintf('\n=== 测试用例4: 边界条件测试 ===\n');

% 测试零输入
num4 = [1, 0.5];
den4 = [1, -0.8, 0.2];

% 零输入测试
x_zero = zeros(1, 10);
y_matlab_zero = filter(num4, den4, x_zero);

y_custom_zero = zeros(1, 10);
u_hist = zeros(1, length(num4));
y_hist = zeros(1, length(den4));

for k = 1:10
    [y_custom_zero(k), u_hist, y_hist] = IIR_filter(num4, den4, x_zero(k), u_hist, y_hist);
end

error4 = max(abs(y_matlab_zero - y_custom_zero));
fprintf('零输入测试最大误差: %.2e\n', error4);

if error4 < 1e-15
    fprintf('✓ 零输入测试通过!\n');
else
    fprintf('✗ 零输入测试失败!\n');
end

%% 绘制对比结果
figure('Name', 'IIR_filter测试结果对比', 'Position', [100, 100, 1200, 800]);

% 子图1: 一阶滤波器脉冲响应
subplot(2,3,1);
plot(0:N-1, y_matlab, 'b-', 'LineWidth', 2); hold on;
plot(0:N-1, y_custom, 'r--', 'LineWidth', 1.5);
title('一阶滤波器脉冲响应');
xlabel('样本'); ylabel('幅值');
legend('MATLAB filter', '自定义 IIR\_filter', 'Location', 'best');
grid on;

% 子图2: 一阶滤波器误差
subplot(2,3,2);
plot(0:N-1, abs(y_matlab - y_custom), 'g-', 'LineWidth', 2);
title('一阶滤波器误差');
xlabel('样本'); ylabel('绝对误差');
grid on;

% 子图3: 二阶滤波器随机信号响应
subplot(2,3,3);
plot(0:N-1, y_matlab, 'b-', 'LineWidth', 1); hold on;
plot(0:N-1, y_custom, 'r--', 'LineWidth', 1);
title('二阶滤波器随机信号响应');
xlabel('样本'); ylabel('幅值');
legend('MATLAB filter', '自定义 IIR\_filter', 'Location', 'best');
grid on;

% 子图4: 二阶滤波器误差
subplot(2,3,4);
plot(0:N-1, abs(y_matlab - y_custom), 'g-', 'LineWidth', 2);
title('二阶滤波器误差');
xlabel('样本'); ylabel('绝对误差');
grid on;

% 子图5: 高阶滤波器复合信号响应
subplot(2,3,5);
plot(0:199, y_matlab, 'b-', 'LineWidth', 1); hold on;
plot(0:199, y_custom, 'r--', 'LineWidth', 1);
title('高阶滤波器复合信号响应');
xlabel('样本'); ylabel('幅值');
legend('MATLAB filter', '自定义 IIR\_filter', 'Location', 'best');
grid on;

% 子图6: 高阶滤波器误差
subplot(2,3,6);
plot(0:199, abs(y_matlab - y_custom), 'g-', 'LineWidth', 2);
title('高阶滤波器误差');
xlabel('样本'); ylabel('绝对误差');
grid on;

%% 性能测试
fprintf('\n=== 性能测试 ===\n');

% 测试大量数据的处理速度
N_perf = 10000;
x_perf = randn(1, N_perf);

% MATLAB内置filter性能
tic;
y_matlab_perf = filter(b, a, x_perf);
time_matlab = toc;

% 自定义IIR_filter性能
tic;
y_custom_perf = zeros(1, N_perf);
u_hist = zeros(1, length(b));
y_hist = zeros(1, length(a));

for k = 1:N_perf
    [y_custom_perf(k), u_hist, y_hist] = IIR_filter(b, a, x_perf(k), u_hist, y_hist);
end
time_custom = toc;

fprintf('MATLAB filter处理%d样本耗时: %.4f秒\n', N_perf, time_matlab);
fprintf('自定义IIR_filter处理%d样本耗时: %.4f秒\n', N_perf, time_custom);
fprintf('速度比值: %.1fx\n', time_custom/time_matlab);

% 最终准确性检查
final_error = max(abs(y_matlab_perf - y_custom_perf));
fprintf('大数据量处理最大误差: %.2e\n', final_error);

%% 总结
fprintf('\n=== 测试总结 ===\n');
if all([error1, error2] < 1e-12) && error3 < 1e-10 && error4 < 1e-15
    fprintf('✓ 所有测试均通过! IIR_filter函数实现正确。\n');
else
    fprintf('✗ 部分测试失败，需要检查IIR_filter函数实现。\n');
end

%% IIR_filter函数定义 (复制到这里用于测试)
function [y_out, u_hist_new, y_hist_new] = IIR_filter(num, den, u_in, u_hist, y_hist)
    % 实现IIR滤波器: H(z) = num(z)/den(z)
    % 差分方程: den(z)Y(z) = num(z)U(z)
    % 即: a0*y[n] + a1*y[n-1] + ... = b0*u[n] + b1*u[n-1] + ...
    
    % 更新输入历史（新样本加入队首）
    u_hist_new = [u_in, u_hist(1:end-1)];
    
    % 计算输出：y[n] = (num部分 - den[1:end]部分) / den[0]
    num_part = FIR(num, u_hist_new);
    den_part = FIR(den(2:end), y_hist);  % 不包括den(1)，因为那是当前输出项
    
    y_out = (num_part - den_part) / den(1);
    
    % 更新输出历史
    y_hist_new = [y_out, y_hist(1:end-1)];
end