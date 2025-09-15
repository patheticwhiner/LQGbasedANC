function plot_simulation_results(results)
% PLOT_SIMULATION_RESULTS 绘制自适应控制仿真结果
%
%   输入:
%       results - 包含仿真结果的结构体，应包含以下字段:
%           .t                - 时间向量
%           .y                - 系统输出
%           .w                - 外部扰动
%           .u                - 控制输入
%           .theta_history    - 频率估计历史
%           .M_val            - 持续激励判定量
%           .m0               - 激励阈值
%           .lambda0          - 激励衰减参数
%           .x_adaptive_history - 自适应观测器状态历史
%           .x_system_history   - 系统真实状态历史

figure;

% 系统输出
subplot(2,3,1);
plot(results.t, results.y, 'b-', 'LineWidth', 1.5);
xlabel('时间 (s)'); ylabel('y(t)');
title('系统输出 y(t)'); grid on;
xlim([0 results.t(end)]); ylim([-20 10]);

% 扰动
subplot(2,3,2);
plot(results.t, results.w, 'r-', 'LineWidth', 1.5);
xlabel('时间 (s)'); ylabel('w(t)');
title('外部扰动 w(t)'); grid on;
xlim([0 results.t(end)]); ylim([-2 2]);

% 控制输入
subplot(2,3,3);
plot(results.t, results.u, 'g-', 'LineWidth', 1.5);
xlabel('时间 (s)'); ylabel('u(t)');
title('控制输入 u(t)'); grid on;
xlim([0 results.t(end)]); ylim([-10 10]);

% 频率估计
subplot(2,3,4);
plot(results.t, results.theta_history, 'm-', 'LineWidth', 1.5);
xlabel('时间 (s)'); ylabel('\theta(t)');
title('频率估计 \theta(t)'); grid on;
xlim([0 results.t(end)]); ylim([-0.5 1.5]);

% M(t)判定量
subplot(2,3,5);
semilogy(results.t, results.M_val, 'c-', 'LineWidth', 1.5); hold on;
semilogy(results.t, results.m0*exp(-results.lambda0*results.t), 'k--', 'LineWidth', 1.5);
xlabel('时间 (s)'); ylabel('M(t)');
title('持续激励判定 M(t)'); grid on;
legend('M(t)', 'm_0 e^{-\lambda_0 t}', 'Location', 'best');
xlim([0 results.t(end)]);

% 自适应观测器状态
subplot(2,3,6);
plot(results.t, results.x_adaptive_history(1,:), 'LineWidth', 1.5); hold on;
plot(results.t, results.x_system_history(1,:), '--', 'LineWidth', 1.5);
xlabel('时间 (s)'); ylabel('状态');
title('自适应观测器 vs 真实状态'); grid on;
legend('z_1(t)', 'x_1(t)', 'Location', 'best');
xlim([0 results.t(end)]);

sgtitle('向量化自适应控制仿真结果 - Marino & Tomei 2013', 'FontSize', 15, 'FontWeight', 'bold');

end
