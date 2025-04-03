Fs     = 8e3;  % 8 kHz 采样频率              用于整个程序 
%% 1 使用滤波器系数建模具有带通特性的次级通路S(z)
% 设计带通滤波器所用参数
Nfilt  = 8;    % 滤波器阶数
Flow   = 160;  % 频带下限：160 Hz
Fhigh  = 2000; % 频带上限：2000 Hz
Ast    = 20;   % 20 dB 阻带衰减
delayS = 7;

% 使用设计函数生成滤波器系数
funcS = fdesign.bandpass('N,Fst1,Fst2,Ast',Nfilt,Flow,Fhigh,Ast,Fs);
SFilterCoeffs = design(funcS, 'cheby2', 'FilterStructure', 'df2tsos');

% 转换SOS矩阵为传递函数
[numS, denS] = sos2tf(SFilterCoeffs.sosMatrix);       % 自动处理级联关系
numS = [zeros(1, delayS),numS];
S_total = tf(numS, denS, 1/Fs);           % 将延迟与滤波器串联
S_total = minreal(S_total);                         % 保持简化操作

% 频谱验证（直接使用SOS结构更稳定）
fvtool(SFilterCoeffs, 'Fs', Fs);                     % 改用原滤波器对象分析
% fvtool(num, den, 'Fs', Fs);                        % 备用方案

%% 2 次级通路辨识与辨识效果验证（ARMAX 模型）
N = 800;
% 2.1 生成随机信号作为输入
ntrS = 30000; % 训练样本数
randomSignal = randn(ntrS, 1); % 输入信号（白噪声）
% 使用次级通路滤波器生成期望信号
measureSignal = filter(numS, denS, randomSignal) + 0.01 * randn(ntrS, 1); % 添加测量噪声

% 2.2 定义 ARMAX 模型阶数
na = 8;  % 自回归部分的阶数
nb = 8;  % 输入延迟部分的阶数
nc = 8;  % 移动平均部分的阶数
nk = 7;  % 输入延迟（通常为 1）

% 2.3 使用 MATLAB 的 armax 函数进行辨识
data = iddata(measureSignal, randomSignal, 1/Fs); % 创建辨识数据对象
armaxModel = armax(data, [na nb nc nk]); % 辨识 ARMAX 模型

% 2.4 提取 ARMAX 模型的系数
[A_armax, B_armax, C_armax] = polydata(armaxModel); % 提取多项式系数

% 2.5 绘制辨识结果
figure;
subplot(3, 1, 1);
plot(measureSignal, 'DisplayName', '期望信号');
hold on;
simulatedSignal = sim(armaxModel, randomSignal); % 使用 ARMAX 模型仿真输出
plot(simulatedSignal, 'DisplayName', 'ARMAX 模型输出信号');
xlabel('样本点');
ylabel('信号幅值');
title('期望信号与 ARMAX 模型输出信号对比');
legend;

subplot(3, 1, 2);
plot(measureSignal - simulatedSignal, 'DisplayName', '误差信号');
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

%% 3 将辨识出的次级通路 ARMAX 模型转换为状态空间模型
% 使用 ARMAX 模型的多项式系数转换为状态空间模型
[Af, Bf, Cf, Df] = tf2ss(B_armax, A_armax);
save('systemIdentification.mat', 'Af', 'Bf', 'Cf', 'Df');

% 检查状态空间模型是否正确生成
if isempty(Af) || isempty(Bf) || isempty(Cf)
    error('状态空间模型生成失败，请检查输入的多项式系数。');
end

% 显示状态空间模型
disp('辨识出的次级通路状态空间模型：');
disp('A 矩阵:');
disp(Af);
disp('B 矩阵:');
disp(Bf);
disp('C 矩阵:');
disp(Cf);
disp('D 矩阵:');
disp(Df);

% 验证实际次级通路与辨识结果的频率响应对比
figure;

% 实际次级通路频率响应
[H_actual, f_actual] = freqz(numS, denS, 1024, Fs);
plot(f_actual, 20*log10(abs(H_actual)), 'k', 'DisplayName', '实际次级通路频率响应');
hold on;

% ARMAX 模型频率响应
[H_armax, f_armax] = freqz(B_armax, A_armax, 1024, Fs);
plot(f_armax, 20*log10(abs(H_armax)), 'b', 'DisplayName', 'ARMAX 模型频率响应');

% 状态空间模型频率响应
f_ss = linspace(0, Fs/2, 1024); % 频率范围 (Hz)
omega_ss = 2 * pi * f_ss;      % 转换为弧度/秒
H_ss = freqresp(ss(Af, Bf, Cf, Df, 1/Fs), omega_ss); % 计算频率响应
H_ss = squeeze(H_ss);          % 去掉多余维度
plot(f_ss, 20*log10(abs(H_ss)), '--r', 'DisplayName', '状态空间模型频率响应');

% 图形设置
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('实际次级通路与辨识结果频率响应对比');
legend('Location', 'Best');
grid on;
hold off;