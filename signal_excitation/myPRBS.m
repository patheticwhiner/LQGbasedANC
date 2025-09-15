%% 使用自己编写的脚本函数生成PRBS
function p = myPRBS(order, length)
% myPRBS - 生成伪随机二进制序列
%   p = myPRBS(order, length) 生成阶数为order，长度为length的PRBS序列
%   
%   参数:
%       order - PRBS的阶数，决定了反馈移位寄存器的长度
%       length - 需要生成的序列长度
%
%   返回值:
%       p - 生成的PRBS序列，为0和1组成的向量
%
%   例子:
%       p = myPRBS(10, 1023);  % 生成长度为1023的10阶PRBS序列

    % 确保参数有效
    if order < 2 || order > 20
        error('PRBS阶数应在2到20之间。');
    end
    
    % 为不同阶数的PRBS定义抽头位置（基于最大长度序列的特性）
    switch order
        case 2
            taps = [2, 1];
        case 3
            taps = [3, 2];
        case 4
            taps = [4, 3];
        case 5
            taps = [5, 3];
        case 6
            taps = [6, 5];
        case 7
            taps = [7, 6];
        case 8
            taps = [8, 6, 5, 4];
        case 9
            taps = [9, 5];
        case 10
            taps = [10, 7];
        case 11
            taps = [11, 9];
        case 12
            taps = [12, 11, 10, 4];
        case 13
            taps = [13, 12, 11, 8];
        case 14
            taps = [14, 13, 12, 2];
        case 15
            taps = [15, 14];
        case 16
            taps = [16, 15, 13, 4];
        case 17
            taps = [17, 14];
        case 18
            taps = [18, 11];
        case 19
            taps = [19, 18, 17, 14];
        case 20
            taps = [20, 17];
    end
    
    % 初始化移位寄存器（避免全0状态）
    register = ones(1, order);
    
    % 分配空间给输出序列
    p = zeros(1, length);
    
    % 生成PRBS序列
    for i = 1:length
        % 取第一个寄存器位作为输出
        p(i) = register(1);
        
        % 计算反馈位
        feedback = 0;
        for j = 1:numel(taps)
            feedback = xor(feedback, register(taps(j)));
        end
        
        % 移位操作
        register = [register(2:end), feedback];
    end
end
