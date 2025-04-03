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
data = iddata(measureSignal, randomSignal, 1/Fs); % 创建辨识数据对象

V = arxstruc(data, data, struc(1:10, 1:10, 1:10));
bestOrder = selstruc(V, 0); % 自动选择最佳阶数
na = bestOrder(1);
nb = bestOrder(2);
nc = bestOrder(3);

% 延迟确定
delayRange = 1:20;
bestNk = 0;
minError = inf;
for nk = delayRange
    armaxModel = armax(data, [na nb nc nk]);
    simulatedSignal = sim(armaxModel, randomSignal);
    error = norm(measureSignal - simulatedSignal); % 计算误差
    if error < minError
        minError = error;
        bestNk = nk;
    end
end
disp(['最佳延迟 nk: ', num2str(bestNk)]);