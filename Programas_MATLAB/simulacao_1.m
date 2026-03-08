%Este programa plota a resposta da planta (tartaruga) para algumas entradas
%padrão, utilizadas em projetos de sistemas de controle.
%Função de transferência da planta: 1/s
%Entradas analisadas: impulso, degrau unitário e rampa.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INICIALIZAÇÃO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Limpando as variáveis e a tela
clear;
close all;
clc;

%Definindo as cores para plotagem
cor1 = [43, 47, 138]*1/255;     % Azul escuro
cor2 = [178, 89, 94]*1/255;     % Vermelho rosado
cor3 = [68, 178, 134]*1/255;    % Verde azulado
cor4 = [79, 13, 57]*1/255;      % Roxo escuro
cor5 = [16, 202, 115]*1/255;    % Verde claro
cor6 = [182, 0, 29]*1/255;      % Vermelho
cor7 = [49, 217, 76]*1/255;     % Verde fluorescente
cor8 = [0, 127, 195]*1/255;     % Azul
cor9 = [0, 0, 0]*1/255;         % Preto
cor10 = [0, 138, 0]*1/255;      % Verde escuro
cor11 = [179, 57, 81]*1/255;    % Vermelho vinho

%Tempo de simulação
t = 0:0.01:3; 

%Criando a função de transferência da planta
num = 1;
den = [1 0];
G = tf(num,den);

% Variável s para transformada de Laplace
s = tf('s');

fprintf('Transfer Function:\n');
display(G);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CÁLCULO DAS RESPOSTAS (UMA ÚNICA VEZ)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calcular todas as respostas uma única vez
[y_impulse, t_impulse] = impulse(G, t);
[y_step, t_step] = step(G, t);
ramp_response = step(G/s, t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%IMPULSE RESPONSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
plot(t_impulse, y_impulse, 'LineWidth', 3.5, 'Color', cor1);
title('Impulse Response', 'FontSize', 16);
xlabel('Time [s]', 'FontSize', 16);
ylabel('Position [m]', 'FontSize', 16);
set(gca, 'FontSize', 16);
ylim([0 5]);
xlim([0 t(end)]);
grid on;
print('Figure_1_impulse', '-dpng', '-r300');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%STEP RESPONSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2)
plot(t_step, y_step, 'LineWidth', 3.5, 'Color', cor1);
title('Step Response', 'FontSize', 16);
xlabel('Time [s]', 'FontSize', 16);
ylabel('Position [m]', 'FontSize', 16);
set(gca, 'FontSize', 16);
ylim([0 5]);
xlim([0 t(end)]);
grid on;
print('Figure_2_step', '-dpng', '-r300');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%RAMP RESPONSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(3)
plot(t, ramp_response, 'LineWidth', 3.5, 'Color', cor1);
title('Ramp Response', 'FontSize', 16);
xlabel('Time [s]', 'FontSize', 16);
ylabel('Position [m]', 'FontSize', 16);
set(gca, 'FontSize', 16);
ylim([0 5]);
xlim([0 t(end)]);
grid on;
print('Figure_3_ramp', '-dpng', '-r300');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%COMBINED RESPONSE PLOT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(5);
hold on;

% Impulse response
plot(t_impulse, y_impulse, 'LineWidth', 3.5, 'Color', cor1, 'DisplayName', 'Impulse Response');

% Step response
plot(t_step, y_step, 'LineWidth', 3.5, 'Color', cor2, 'DisplayName', 'Step Response', 'LineStyle', '--');

% Ramp response
plot(t, ramp_response, 'LineWidth', 3.5, 'Color', cor3, 'DisplayName', 'Ramp Response', 'LineStyle', '-.');

% Graph settings
title('System Response Comparison', 'FontSize', 16);
xlabel('Time [s]', 'FontSize', 16);
ylabel('Position [m]', 'FontSize', 16);
legend('Location', 'northwest', 'FontSize', 16);
grid on;
set(gca, 'FontSize', 16);
xlim([0 t(end)]);
ylim([0 5]);

% Reference line at y=0
plot([0 t(end)], [0 0], 'k:', 'LineWidth', 0.5, 'HandleVisibility', 'off');

hold off;
print('Figure_5_combined_responses', '-dpng', '-r300');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%POLES AND ZEROS MAP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get poles and zeros
[p, z] = pzmap(G);

figure(4);
hold on;

% Plot zeros (if any)
if ~isempty(z)
    plot(real(z), imag(z), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Zeros', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b');
end

% Plot poles with larger marker
if ~isempty(p)
    plot(real(p), imag(p), 'x', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', 'Poles', 'Color', 'r');
end

% Add grid lines and axes
grid on;
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';

% Secondary grid for better visualization
grid minor;
ax.GridAlpha = 0.3;
ax.MinorGridAlpha = 0.1;

% Set axis limits
xlim([-2, 2]);  % Adjust as needed
ylim([-1.5, 1.5]);  % Adjust as needed

% Add labels
legend('Location', 'southeast', 'FontSize', 16);
xlabel('Real Axis (σ)', 'FontSize', 16);
ylabel('Imaginary Axis (jω)', 'FontSize', 16);
title('Pole-Zero Map', 'FontSize', 16);
set(gca, 'FontSize', 16);
hold off;
print('Figure_4_poles', '-dpng', '-r300');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FINALIZAÇÃO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Operação realizada com sucesso
disp('Plotting completed successfully!')

%Deletando os arquivos .asv
delete *.asv