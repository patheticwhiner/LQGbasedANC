% DataManager.m - �������ݼ����뵼������
function [x, y, t, fs] = DataManager(dataSource, exportPath)
    % ���ݼ����뵼������
    %
    % ����:
    %   dataSource - �ַ��������֣�ָ��Ҫ���ص�����Դ
    %   exportPath - [��ѡ] �ַ��������ݵ���·�������ṩ�򲻵���
    %
    % ���:
    %   x - �����ź�
    %   y - ����ź�
    %   t - ʱ������
    %   fs - ����Ƶ��
    %
    % �÷�ʾ��:
    %   [x, y, t, fs] = DataManager('pipe_sim');
    %   [x, y, t, fs] = DataManager(1); % ��������
    %   [x, y, t, fs] = DataManager('PRBS_001', 'E:\Code\MATLAB_ANC\data');
    
    if nargin < 2
        exportPath = '';  % Ĭ�ϲ�����
    end
    
    % ���ݼ���
    fprintf('���ڼ�������Դ: %s...\n', num2str(dataSource));
    
    % �ж��������Ͳ�ȷ���ļ����Ͳ���
    if isnumeric(dataSource)
        % �������룺ֱ��ӳ�䵽rec1_0xx��ʽ
        if dataSource < 10
            fileName = sprintf('rec1_00%d', dataSource);
        else
            fileName = sprintf('rec1_0%d', dataSource);
        end
        filePath = [fileName '.mat'];
        dataName = fileName;
        
    else
        % �ı����룺�����������
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
                error('δ֪����Դ: %s', dataSource);
        end
    end
    
    % ͳһ���غʹ�������
    try
        % ���������ļ�
        data = load(filePath);
        
        % ��ȡ���ݽṹ���������ݽṹ�����ļ�����ͬ��
        structName = fieldnames(data);
        dataStruct = data.(structName{1});
        
        % ��ȡ����
        t = dataStruct.X.Data';
        dt = t(2)-t(1);
        
        % ����dt�������Ƶ�ʣ��ҵ���ӽ�����1000����
        fs_actual = 1/dt;
        fs = round(fs_actual/1000) * 1000;  % �������뵽�����1000����
        
        x = dataStruct.Y(2).Data';
        y = dataStruct.Y(1).Data';
        
        % ���òü�����
        trimTop = 2*fs;  % ��������Ĭ�ϲü�ĩβ2��
        % �ü�����
        topIdx = 1 + trimTop;
        t = t(topIdx:end);
        x = x(topIdx:end);
        y = y(topIdx:end);
        
    catch ME
        error('���������ļ�ʧ��: %s\n������Ϣ: %s', filePath, ME.message);
    end
    
    % ��ʾ������Ϣ
    fprintf('���ݼ������!\n');
    fprintf('����Դ����: %s\n', dataName);
    fprintf('������: %.0f Hz\n', fs);
    fprintf('���ݳ���: %d �� (%.2f ��)\n', length(x), length(x)/fs);
    fprintf('�����ź�RMS: %.4f\n', rms(x));
    fprintf('����ź�RMS: %.4f\n', rms(y));
    
    % �������ݣ�����ṩ�˵���·����
    if ~isempty(exportPath)
        % ȷ������Ŀ¼����
        if ~exist(exportPath, 'dir')
            mkdir(exportPath);
            fprintf('��������Ŀ¼: %s\n', exportPath);
        end
        
        % ���������ļ���
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        exportFile = fullfile(exportPath, sprintf('%s_%s.mat', dataName, timestamp));
        
        % ��������
        save(exportFile, 'x', 'y', 't', 'fs', 'dataName');
        fprintf('�����ѵ�����: %s\n', exportFile);
    end
    
    % ���ӻ����ݸ�������ѡ��
    if nargout == 0 || nargin > 1
        figure('Name', ['���ݸ���: ', dataName]);
        
        % ʱ���ͼ
        subplot(2,2,1);
        plot(t(1:min(1000, length(t))), x(1:min(1000, length(x)))); 
        grid on;
        title('�����ź�(ǰ1000��)');
        xlabel('ʱ��(��)');
        
        subplot(2,2,3);
        plot(t(1:min(1000, length(t))), y(1:min(1000, length(y)))); 
        grid on;
        title('����ź�(ǰ1000��)');
        xlabel('ʱ��(��)');
        
        % Ƶ���ͼ
        subplot(2,2,2);
        window_length = min(1024, floor(length(x)/3));
        window = hamming(window_length);
        noverlap = window_length/2;
        nfft = max(2048, 2^nextpow2(window_length));
        [px, f] = pwelch(x, window, noverlap, nfft, fs);
        plot(f, 10*log10(px));
        grid on;
        title('�����źŹ�����');
        xlabel('Ƶ��(Hz)'); 
        ylabel('����/Ƶ��(dB/Hz)');
        xlim([0 fs/2]);
        
        subplot(2,2,4);
        [py, f] = pwelch(y, window, noverlap, nfft, fs);
        plot(f, 10*log10(py));
        grid on;
        title('����źŹ�����');
        xlabel('Ƶ��(Hz)'); 
        ylabel('����/Ƶ��(dB/Hz)');
        xlim([0 fs/2]);
    end
end

% �����ű��е���ʾ��:
% [x, y, t, fs] = DataManager('pipe_sim', 'E:\Code\MATLAB_ANC\SysId_SecondaryPath\exported_data');