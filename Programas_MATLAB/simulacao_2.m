%Este programa plota a resposta do sistema controlado à entrada degrau, bem
%como o lugar das raízes.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INICIALIZAÇÃO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Limpando as variáveis e a tela
clear;
close all;
clc;

%Definindo as cores para plotagem
cor1 = [43, 47, 138]*1/255;
cor2 = [178, 89, 94]*1/255;
cor3 = [68, 178, 134]*1/255;
cor4 = [79, 13, 57]*1/255;
cor5 = [16, 202, 115]*1/255;
cor6 = [182, 0, 29]*1/255;
cor7 = [49, 217, 76]*1/255;
cor8 = [0, 127, 195]*1/255;
cor9 = [0, 0, 0]*1/255;
cor10 = [0, 138, 0]*1/255;
cor11 = [179, 57, 81]*1/255;

%Tempo de simulação
t = 0:0.01:3; 

%Vetor com os ganhos
K = [1 2 4 10];

%Criando as matrizes para receber os resultados das simulações
Y = zeros(length(t),length(K));
T = zeros(length(t),length(K));

%Entrada degrau
degrau = ones(1, length(t));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SIMULAÇÃO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Simulação a respota ao degrau para os valores de ganho contidos no vetor
%K.

for i=1:length(K)

%Determinando a função de transferência
num = K(i);
den = [1 K(i)];
G = tf(num, den);

[y,tOut] = step(G, t);

Y(:,i)=y;
T(:,i)=tOut;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PLOTANDO O RESULTADO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
plot(t, degrau, 'r-*', 'LineWidth', 3.5, 'HandleVisibility','off');
hold on
plot(T(:,1), Y(:,1), 'LineWidth', 3.5, 'Color', cor1);
plot(T(:,2), Y(:,2), 'LineWidth', 3.5, 'Color', cor2);
plot(T(:,3), Y(:,3), 'LineWidth', 3.5, 'Color', cor3);
plot(T(:,4), Y(:,4), 'LineWidth', 3.5, 'Color', cor4);
title('Step Response','FontSize', 16);
L = legend('K=1','K=2','K=4', 'K=10','Location','southeast', 'Orientation','vertical');
L.Box = 'off';
xlabel('Time [s]', 'FontSize', 16);
ylabel('Position [m]', 'FontSize', 16);
set(gca, 'FontSize', 16);
ylim([0 1.2]);
xlim([0 t(end)]);
grid on;

%Salva a figura
f = gcf;
figureName = strcat('Figure_6_step.png');
exportgraphics(f,figureName,'Resolution',300);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PLOTANDO O ROOT LOCUS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Determinando a função de transferência da planta:
num = 1;
den = [1 0];
G = tf(num, den);
figure(2);
rlocus(G);
r = findobj(gcf, 'Type', 'Line');
set(r, 'LineWidth', 3.5, 'Color', cor6);
grid on;
title('Root Locus for the System K/(K+s)', 'FontSize', 16);
xlabel('Real Axis', 'FontSize', 16);
ylabel('Imaginary Axis', 'FontSize', 16);

%Salva a figura
g = gcf;
figureName = strcat('Figure_7_rlocus.png');
%figureName2 = strcat('Figure_8_rlocus.emf');
exportgraphics(g,figureName,'Resolution',300);
%exportgraphics(g,figureName2,'Resolution',300);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FINALIZANDO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Operação realizada com sucesso
disp('Plotagem realizada com sucesso!')

%Deletando os arquivos .asv
delete *.asv