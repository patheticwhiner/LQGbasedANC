% DataManager.m - 仿真数据加载与导出工具
function [x, y, t, fs] = DataManager(dataSource, exportPath)
    % 数据加载与导出工具
    %
    % 输入:
    %   dataSource - 字符串或数字，指定要加载的数据源
    %   exportPath - [可选] 字符串，数据导出路径，不提供则不导出
    %
    % 输出:
    %   x - 输入信号
    %   y - 输出信号
    %   t - 时间向量
    %   fs - 采样频率
    %
    % 用法示例:
    %   [x, y, t, fs] = DataManager('pipe_sim');
    %   [x, y, t, fs] = DataManager(1); % 数字索引
    %   [x, y, t, fs] = DataManager('PRBS_001', 'E:\Code\MATLAB_ANC\data');
    
    if nargin < 2
        exportPath = '';  % 默认不导出
    end
    
    % 数据加载
    fprintf('正在加载数据源: %s...\n', num2str(dataSource));
    
    % 判断输入类型并确定文件名和参数
    if isnumeric(dataSource)
        % 数字输入：直接映射到rec1_0xx格式
        if dataSource < 10
            fileName = sprintf('rec1_00%d', dataSource);
        else
            fileName = sprintf('rec1_0%d', dataSource);
        end
        filePath = [fileName '.mat'];
        dataName = fileName;
        
    else
        % 文本输入：特殊情况处理
        switch dataSource
            case 'primpath'
                fileName = 'primpath';
                filePath = 'primpath.mat';
                dataName = 'primpath';
                
            case 'secpath'
                fileName = 'secpath';
                filePath = 'secpath.mat';
                dataName = 'secpath';
                
            otherwise
                error('未知数据源: %s', dataSource);
        end
    end
    
    % 统一加载和处理数据
    try
        % 加载数据文件
        data = load(filePath);
        
        % 获取数据结构（假设数据结构名与文件名相同）
        structName = fieldnames(data);
        dataStruct = data.(structName{1});
        
        % 提取数据
        t = dataStruct.X.Data';
        dt = t(2)-t(1);
        
        % 根据dt计算采样频率，找到最接近的整1000倍数
        fs_actual = 1/dt;
        fs = round(fs_actual/1000) * 1000;  % 四舍五入到最近的1000倍数
        
        x = dataStruct.Y(2).Data';
        y = dataStruct.Y(1).Data';
        
        % 设置裁剪参数
        trimTop = 2*fs;  % 数字输入默认裁剪末尾2秒
        % 裁剪数据
        topIdx = 1 + trimTop;
        t = t(topIdx:end);
        x = x(topIdx:end);
        y = y(topIdx:end);
        
    catch ME
        error('加载数据文件失败: %s\n错误信息: %s', filePath, ME.message);
    end
    
    % 显示数据信息
    fprintf('数据加载完成!\n');
    fprintf('数据源名称: %s\n', dataName);
    fprintf('采样率: %.0f Hz\n', fs);
    fprintf('数据长度: %d 点 (%.2f 秒)\n', length(x), length(x)/fs);
    fprintf('输入信号RMS: %.4f\n', rms(x));
    fprintf('输出信号RMS: %.4f\n', rms(y));
    
    % 导出数据（如果提供了导出路径）
    if ~isempty(exportPath)
        % 确保导出目录存在
        if ~exist(exportPath, 'dir')
            mkdir(exportPath);
            fprintf('创建导出目录: %s\n', exportPath);
        end
        
        % 创建导出文件名
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        exportFile = fullfile(exportPath, sprintf('%s_%s.mat', dataName, timestamp));
        
        % 保存数据
        save(exportFile, 'x', 'y', 't', 'fs', 'dataName');
        fprintf('数据已导出至: %s\n', exportFile);
    end
    
    % 可视化数据概览（可选）
    if nargout == 0 || nargin > 1
        figure('Name', ['数据概览: ', dataName]);
        
        % 时域绘图
        subplot(2,2,1);
        plot(t(1:min(1000, length(t))), x(1:min(1000, length(x)))); 
        grid on;
        title('输入信号(前1000点)');
        xlabel('时间(秒)');
        
        subplot(2,2,3);
        plot(t(1:min(1000, length(t))), y(1:min(1000, length(y)))); 
        grid on;
        title('输出信号(前1000点)');
        xlabel('时间(秒)');
        
        % 频域绘图
        subplot(2,2,2);
        window_length = min(1024, floor(length(x)/3));
        window = hamming(window_length);
        noverlap = window_length/2;
        nfft = max(2048, 2^nextpow2(window_length));
        [px, f] = pwelch(x, window, noverlap, nfft, fs);
        plot(f, 10*log10(px));
        grid on;
        title('输入信号功率谱');
        xlabel('频率(Hz)'); 
        ylabel('功率/频率(dB/Hz)');
        xlim([0 fs/2]);
        
        subplot(2,2,4);
        [py, f] = pwelch(y, window, noverlap, nfft, fs);
        plot(f, 10*log10(py));
        grid on;
        title('输出信号功率谱');
        xlabel('频率(Hz)'); 
        ylabel('功率/频率(dB/Hz)');
        xlim([0 fs/2]);
    end
end

% 在主脚本中调用示例:
% [x, y, t, fs] = DataManager('pipe_sim', 'E:\Code\MATLAB_ANC\SysId_SecondaryPath\exported_data');