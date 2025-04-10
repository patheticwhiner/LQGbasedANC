%%
run systemIddemo.m;
idinput = [1/fs*(1:length(randomSignal))',randomSignal];
uiopen('E:\Code\MATLAB_ANC\LQGbasedANC\systemId.slx',1)
%%
simin = out.simin;
simout = out.simout;
%% 使用ARMAX程序和使用系统辨识工具箱为什么会存在差异？
% 2.2 定义 ARMAX 模型阶数
na = 8;  % 自回归部分的阶数
nb = 8;  % 输入延迟部分的阶数
nc = 8;  % 移动平均部分的阶数
nk = 7;  % 输入延迟（通常为 1）

% 2.3 使用 MATLAB 的 armax 函数进行辨识
data = iddata(simout, simin, 1/fs); % 创建辨识数据对象
armaxModel = armax(data, [na nb nc nk]); % 辨识 ARMAX 模型

% 2.4 提取 ARMAX 模型的系数
[A_armax, B_armax, C_armax] = polydata(armaxModel); % 提取多项式系数

% 2.5 绘制辨识结果
figure;
subplot(3, 1, 1);
plot(simout, 'DisplayName', '期望信号');
hold on;
simulatedSignal = sim(armaxModel, simin); % 使用 ARMAX 模型仿真输出
plot(simulatedSignal, 'DisplayName', 'ARMAX 模型输出信号');
xlabel('样本点');
ylabel('信号幅值');
title('期望信号与 ARMAX 模型输出信号对比');
legend;

subplot(3, 1, 2);
plot(simout - simulatedSignal, 'DisplayName', '误差信号');
xlabel('样本点');
ylabel('误差幅值');
title('误差信号');
legend;

subplot(3, 1, 3);
impulseResponse = filter(numS, denS, [1; zeros(N - 1, 1)]); % 实际冲激响应
estimatedImpulseResponse = filter(B_armax, A_armax, [1; zeros(N - 1, 1)]); % 估计冲激响应
plot(impulseResponse, 'DisplayName', '实际冲激响应');
hold on;
plot(estimatedImpulseResponse, 'DisplayName', '估计冲激响应');
xlabel('样本点');
ylabel('幅值');
title('实际冲激响应与估计冲激响应对比');
legend;
grid on;
