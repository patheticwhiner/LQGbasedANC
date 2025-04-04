% 需要检查状态空间表示中是否含有延迟
clear; close all; clc;
%%
% % 创建白噪声，并且生成外系统模型
% run bandlimitedNoise.m
% % 设定并且辨识次级通路
% run systemIdentification.m
load('bandlimitedNoise.mat');
load('systemIdentification.mat');

%% 耦合系统
% 增广状态矩阵
n = size(Af, 1);    % 原系统状态维度
p = size(A_w2, 1);  % 干扰模型状态维度
A = blkdiag(Af, A_w2);  % 块对角矩阵 [Af 0; 0 A_w2]

% 增广输入矩阵
B = [Bf; zeros(p, size(Bf, 2))];  % 原系统控制输入通道
G = [zeros(n, size(B_w2, 2)); B_w2];  % 干扰输入通道

% 增广输出矩阵
C = [Cf , C_w2];

%% 设计LQR控制器
% 状态权重矩阵 Q（半正定）
Q = C' * C;
% 输入权重矩阵 R（正定）
R = 10e-4;
% 离散 LQR 求解（状态反馈增益）
[K, ~, ~] = dlqr(A, B, Q, R);

% 计算闭环系统的状态矩阵
A_cl = A - B * K;
% 计算闭环系统的特征值
eig_cl = eig(A_cl);
% 检查闭环系统的稳定性
if all(abs(eig_cl) < 1)
    disp('闭环系统是稳定的（所有极点都在单位圆内）。');
else
    disp('闭环系统是不稳定的（存在极点在单位圆外）。');
end

%% 设计卡尔曼滤波器（状态估计器）
% 假设过程噪声协方差矩阵和测量噪声协方差矩阵
Qn = 2;  % 过程噪声协方差
Rn = 1e-4 * eye(size(C, 1));  % 测量噪声协方差

% 使用离散卡尔曼滤波器求解最优增益矩阵
[L, ~, ~] = dlqe(A, G, C, Qn, Rn);

% 显示卡尔曼滤波器增益
disp('卡尔曼滤波器增益矩阵 L:');
disp(L);

% 计算引入卡尔曼滤波器后的闭环系统状态矩阵
A_cl = A - B * K - L * C;
% 计算闭环系统的特征值
eig_cl = eig(A_cl);
% 显示闭环系统的极点
disp('引入卡尔曼滤波器后的闭环系统极点:');
disp(eig_cl);
% 检查闭环系统的稳定性
if all(abs(eig_cl) < 1)
    disp('引入卡尔曼滤波器后的闭环系统是稳定的（所有极点都在单位圆内）。');
else
    disp('引入卡尔曼滤波器后的闭环系统是不稳定的（存在极点在单位圆外）。');
end

%% 初始化参数
N = 2000;                    % 仿真步数
x = zeros(n + p, 1);    % 增广状态初始化
x_hat = zeros(n + p, 1);    % 状态估计初始化
x_history = zeros(n + p, N);  % 状态轨迹记录
x_hat_history = zeros(n + p, N);  % 状态估计轨迹记录
y_history = zeros(size(Cf, 1), N);  % 输出轨迹记录
u_history = zeros(1, N);  % 控制输入记录

%% 闭环控制仿真
for k = 1:N
    % 生成过程噪声和测量噪声
    e = sqrt(Qn) * randn(size(G, 2), 1);  % 过程噪声
    v = chol(Rn)' * randn(size(C, 1), 1);  % 测量噪声

    % 状态反馈控制律
    u = -K * x_hat;  % 使用估计状态进行反馈控制
    u_history(k) = u;  % 记录控制输入

    % 增广系统动态更新
    x = A * x + B * u + G * e;  % 系统状态更新
    y = C * x + v;  % 输出测量

    % 卡尔曼滤波器更新（状态估计）
    x_hat = A * x_hat + B * u;  % 预测
    x_hat = x_hat + L * (y - C * x_hat);  % 校正

    % 记录状态和输出
    x_history(:, k) = x;
    x_hat_history(:, k) = x_hat;
    y_history(:, k) = y;
end

% 计算干扰信号和反噪声信号
d_history = C_w2 * x_history(n+1:end, :);  % 干扰信号
anti_history = Cf * x_history(1:end-p, :);  % 反噪声信号

figure;
subplot(2, 1, 1);
plot(1:N, u_history, 'DisplayName', '控制信号'); hold on;
plot(1:N, anti_history, 'DisplayName', '反噪声信号');
xlabel('样本n/samples'); ylabel('幅度');
legend;
grid on; 

subplot(2, 1, 2);
plot(1:N, d_history, 'DisplayName', '干扰信号d'); hold on;
plot(1:N, -anti_history, 'DisplayName', '反噪声信号');
plot(1:N, y_history, 'DisplayName', '残余信号 y');
xlabel('样本n/samples'); ylabel('幅度');
legend;
grid on;