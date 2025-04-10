% filepath: e:\Code\MATLAB_ANC\LQGbasedANC\LQGdemo.m
% 使用 LQR 实现控制（不引入 Kalman Filter）
clear; close all; clc;

%% 加载数据
load('bandlimitedNoise.mat');
load('systemIdentification.mat');

%% 耦合系统
% 增广状态矩阵
n = size(Af, 1);    % 原系统状态维度
p = size(A_w2, 1);  % 干扰模型状态维度
A = blkdiag(Af, A_w2);  % 块对角矩阵 [Af 0; 0 A_w2]

% 增广输入矩阵
B = [Bf; zeros(p, size(Bf, 2))];  % 原系统控制输入通道

% 增广输出矩阵
C = [Cf , C_w2];

%% 设计 LQR 控制器
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

%% 初始化参数
N = 2000;                    % 仿真步数
x = zeros(n + p, 1);    % 增广状态初始化
x_history = zeros(n + p, N);  % 状态轨迹记录
y_history = zeros(size(Cf, 1), N);  % 输出轨迹记录
u_history = zeros(1, N);  % 控制输入记录

%% 闭环控制仿真
for k = 1:N
    % 生成过程噪声
    e = sqrt(4) * randn(size(B, 2), 1);  % 过程噪声

    % 状态反馈控制律
    u = -K * x;  % 使用真实状态进行反馈控制
    u_history(k) = u;  % 记录控制输入

    % 增广系统动态更新
    x = A * x + B * u + e;  % 系统状态更新
    y = C * x;  % 输出测量

    % 记录状态和输出
    x_history(:, k) = x;
    y_history(:, k) = y;
end

% 计算干扰信号和反噪声信号
d_history = C_w2 * x_history(n+1:end, :);  % 干扰信号
anti_history = Cf * x_history(1:end-p, :);  % 反噪声信号

%% 绘图
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