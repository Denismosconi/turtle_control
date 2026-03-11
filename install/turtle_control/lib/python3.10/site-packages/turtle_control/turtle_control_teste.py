#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
import numpy as np
import time
import math
import json
import csv
import os
from datetime import datetime

###########################################################################################
#CRIANDO O NÓ ROS2
###########################################################################################
class ControlTurtle(Node):
    def __init__(self, target_x, target_y, Kp_angular, Kp_linear, experiment_name=None):
        super().__init__('turtle_control')

        #Definição de variáveis
        self.target_x = target_x #Posição x final desejada
        self.target_y = target_y #Posição y final desejada
        self.Kp_angular = Kp_angular #Ganho Kp para controlador proporcional de rotação
        self.Kp_linear = Kp_linear #Ganho Kp para controlador proporcional de translação
        self.estado = "INICIAR"
        self.angle_tolerance = 0.001
        self.linear_tolerance = 0.001
        self.vetor_velocidade = Twist() # Criando o vetor com os dados de velocidade
        self.max_linear_speed = 1.0 #Limite de velocidade linear
        self.max_angular_speed = 1.0 #Limite de velocidade angular
        self.shutdown_timer = None #Timer para encerrar o programa
        self.should_shutdown = False  # Flag para controle de shutdown
        self.executor = None # Executor para gerenciar o shutdown
        self.already_printed_final = False #Flag para controle de impressão
        self.already_printed_shutdown = False #Flag para controle de impressão
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}" #Nome do experimento
        self.save_dir = "turtle_experiments" #Pasta para salvar dados
        
        # Criar diretório para salvar dados, se não existir
        os.makedirs(self.save_dir, exist_ok=True)

        # VETORES PARA ARMAZENAMENTO DE DADOS
        self.timestamps = []  # Timestamps para cada amostra
        self.pos_x = []       # posicao.x
        self.pos_y = []       # posicao.y
        self.error_x_list = []  # error_x
        self.error_y_list = []  # error_y
        self.vel_x_global = []  # velocidade_x (global)
        self.vel_y_global = []  # velocidade_y (global)
        self.vel_linear_x = []  # vetor_velocidade.linear.x
        self.angle_rad_list = []  # self.angle em radianos
        self.angle_deg_list = []  # self.angle em graus
        self.error_angle_list = []  # erro_angle
        self.vel_angular_z = []  # vetor_velocidade.angular.z
        self.estados_list = []  # Estado atual em cada amostra
        self.time_elapsed = []  # Tempo decorrido desde o início

        # Variável para tempo inicial
        self.start_time = None

        #Criação do subscriber
        self.posicao_sensor = self.create_subscription(Pose, "turtle1/pose", self.callback_controle, 10)

        #Criação do publisher
        self.velocidade_atuador = self.create_publisher(Twist, "turtle1/cmd_vel", 10)

###########################################################################################
#CRIANDO O MÉTODO DE CONTROLE
###########################################################################################
    def callback_controle(self, posicao: Pose):
        # Registrar tempo inicial na primeira chamada
        if self.start_time is None:
            self.start_time = time.time()
        
        # Calcular tempo atual e tempo decorrido
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Salvar timestamp
        self.timestamps.append(current_time)
        self.time_elapsed.append(elapsed_time)

        # Salvar posição atual
        self.pos_x.append(posicao.x)
        self.pos_y.append(posicao.y)

        # Salvar estado atual
        self.estados_list.append(self.estado)

        if self.estado == "INICIAR":
            """Definição da oritenção para posição desejada - laço executado apenas na primeira vez"""
            # Calcular o vetor diferença
            dx = self.target_x - posicao.x
            dy = self.target_y - posicao.y

            # Calcular o ângulo usando arco tangente (atan2 retorna ângulo em [-pi, pi])
            self.angle = math.atan2(dy, dx)

            #Calcula o seno e o cosseno do ângulo
            self.seno_angle = math.sin(self.angle)
            self.cosseno_angle = math.cos(self.angle)

            # Salvar dados de ângulo
            self.angle_rad_list.append(self.angle)
            self.angle_deg_list.append(math.degrees(self.angle))
            
            # Inicializar outras listas com valores iniciais
            self.error_x_list.append(dx)
            self.error_y_list.append(dy)
            self.vel_x_global.append(0.0)
            self.vel_y_global.append(0.0)
            self.vel_linear_x.append(0.0)
            self.error_angle_list.append(0.0)
            self.vel_angular_z.append(0.0)

            #Atualiza estado
            self.estado = "ROTACIONAR"

        elif self.estado == "ROTACIONAR":
            #Calcula o erro de posição angular
            self.erro_angle = self.angle - posicao.theta
            
            # Salvar dados de ângulo e erro
            self.angle_rad_list.append(self.angle)
            self.angle_deg_list.append(math.degrees(self.angle))
            self.error_angle_list.append(self.erro_angle)

            #Determina sinal de controle
            if abs(self.erro_angle) > self.angle_tolerance:
                self.vetor_velocidade.angular.z = self.Kp_angular * self.erro_angle
                self.vetor_velocidade.angular.z = np.clip(self.vetor_velocidade.angular.z, -self.max_angular_speed, self.max_angular_speed)
            else: #Chegou na orientação desejada
                self.vetor_velocidade.angular.z = 0.0
                time.sleep(0.5)  # Pequena pausa para estabilização
                self.estado = "TRANSLADAR"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                self.get_logger().info("Ângulo corrigido. Iniciando movimento de translação")

            # Salvar velocidade angular
            self.vel_angular_z.append(self.vetor_velocidade.angular.z)
            
            # Para fase ROTACIONAR, velocidade linear é zero
            self.vel_linear_x.append(0.0)
            
            # Calcular erros lineares (apenas para registro)
            self.error_x = self.target_x - posicao.x
            self.error_y = self.target_y - posicao.y
            self.error_x_list.append(self.error_x )
            self.error_y_list.append(self.error_y)
            
            # Velocidades globais zero durante rotação
            self.vel_x_global.append(0.0)
            self.vel_y_global.append(0.0)

        elif self.estado == "TRANSLADAR":
            #Calculando o erro de posição
            self.error_x = self.target_x - posicao.x
            self.error_y = self.target_y - posicao.y
            
            # Salvar erros
            self.error_x_list.append(self.error_x)
            self.error_y_list.append(self.error_y)
            
            # Salvar ângulo atual
            self.angle_rad_list.append(self.angle)
            self.angle_deg_list.append(math.degrees(self.angle))
            
            # Durante translação, erro angular é zero (já alinhado)
            self.error_angle_list.append(0.0)
            self.vel_angular_z.append(0.0)

            #Determina sinal de controle
            if abs(self.error_x) > self.linear_tolerance or abs(self.error_y) > self.linear_tolerance:

                #Calculando velocidade no sistema de coordenadas global
                velocidade_x = self.error_x * self.Kp_linear
                velocidade_y = self.error_y * self.Kp_linear
                
                # Salvar velocidades globais
                self.vel_x_global.append(velocidade_x)
                self.vel_y_global.append(velocidade_y)

                #Calculando a velocidade no sistema de coordenadas móvel
                velocidade = velocidade_x * self.cosseno_angle + velocidade_y * self.seno_angle

                #Atribuindo a velocidade calculada
                self.vetor_velocidade.linear.x = velocidade
                self.vetor_velocidade.linear.x = np.clip(self.vetor_velocidade.linear.x , -self.max_linear_speed, self.max_linear_speed)

            else: #Chegou na posição desejada
                # Velocidade zero quando chegou ao destino
                self.vel_x_global.append(0.0)
                self.vel_y_global.append(0.0)
                self.vetor_velocidade.linear.x = 0.0
                
                #Atualizando estado
                self.estado = "FINALIZADO"
            
            # Salvar velocidade linear
            self.vel_linear_x.append(self.vetor_velocidade.linear.x)
            
        elif self.estado == "FINALIZADO":    
            # Parar completamente
            self.vetor_velocidade.linear.x = 0.0
            self.vetor_velocidade.angular.z = 0.0
            
            # Salvar dados finais (tudo zero)
            self.error_x_list.append(0.0)
            self.error_y_list.append(0.0)
            self.vel_x_global.append(0.0)
            self.vel_y_global.append(0.0)
            self.vel_linear_x.append(0.0)
            self.angle_rad_list.append(self.angle)
            self.angle_deg_list.append(math.degrees(self.angle))
            self.error_angle_list.append(0.0)
            self.vel_angular_z.append(0.0)
            
            if not self.already_printed_final:
                self.get_logger().info("Posição final alcançada!")
                self.already_printed_final = True
                self.iniciar_shutdown_timer()

        # Publicando a velocidade
        self.velocidade_atuador.publish(self.vetor_velocidade)
        
        # Log para debug (opcional - pode remover depois)
        if len(self.timestamps) % 50 == 0:  # Log a cada 50 amostras
            self.get_logger().debug(f"Amostra {len(self.timestamps)}: "
                                   f"x={posicao.x:.3f}, y={posicao.y:.3f}, "
                                   f"estado={self.estado}")

###########################################################################################
# ENCERRAMENTO DO NÓ
###########################################################################################
    def iniciar_shutdown_timer(self):
        """Inicia o timer de 2 segundos para encerrar o programa"""
        if not self.already_printed_shutdown:
            self.get_logger().info("Destino alcançado! O programa será encerrado em 2 segundos...")
            
            # Imprimir resumo dos dados coletados
            self.get_logger().info(f"Dados coletados: {len(self.timestamps)} amostras")
            self.get_logger().info(f"Tempo total: {self.time_elapsed[-1]:.2f} segundos")
            
            self.already_printed_shutdown = True
            
        # Criar timer para encerrar após 2 segundos
        self.shutdown_timer = self.create_timer(2.0, self.encerrar_programa)
    
    def encerrar_programa(self):
        """Encerra o programa graciosamente"""
        self.get_logger().info("Encerrando programa...")
        
        # Mostrar informações sobre os vetores coletados
        print("\n" + "="*60)
        print("RESUMO DOS DADOS COLETADOS")
        print("="*60)
        print(f"Total de amostras: {len(self.timestamps)}")
        print(f"Tamanho dos vetores:")
        print(f"  timestamps: {len(self.timestamps)}")
        print(f"  pos_x: {len(self.pos_x)}")
        print(f"  pos_y: {len(self.pos_y)}")
        print(f"  error_x_list: {len(self.error_x_list)}")
        print(f"  error_y_list: {len(self.error_y_list)}")
        print(f"  vel_x_global: {len(self.vel_x_global)}")
        print(f"  vel_y_global: {len(self.vel_y_global)}")
        print(f"  vel_linear_x: {len(self.vel_linear_x)}")
        print(f"  angle_rad_list: {len(self.angle_rad_list)}")
        print(f"  angle_deg_list: {len(self.angle_deg_list)}")
        print(f"  error_angle_list: {len(self.error_angle_list)}")
        print(f"  vel_angular_z: {len(self.vel_angular_z)}")
        print(f"  estados_list: {len(self.estados_list)}")
        print(f"  time_elapsed: {len(self.time_elapsed)}")
        print("\nPrimeiros 5 valores de cada vetor:")
        print(f"  pos_x: {self.pos_x[:5]}")
        print(f"  pos_y: {self.pos_y[:5]}")
        print(f"  error_x: {self.error_x_list[:5]}")
        print("="*60)

        # Marcar que o programa deve encerrar
        self.should_shutdown = True
        
        # Destruir o timer de shutdown para evitar chamadas repetidas
        if self.shutdown_timer:
            self.shutdown_timer.destroy()
        
        self.get_logger().info("Node pronto para encerramento...")

# ... (O restante do código permanece igual - funções get_user_input() e main())

###########################################################################################
# FUNÇÃO PARA OBTER ENTRADAS DO USUÁRIO
###########################################################################################
def get_user_input():
    """Obtém a posição alvo, ganhos e nome do experimento"""
    print("\n" + "="*60)
    print("CONTROLE DA TARTARUGA - ROS2 TURTLESIM")
    print("="*60)
    print("Digite a posição desejada e os ganhos do controlador")
    print("Formato: x y Kp_angular Kp_linear [nome_experimento]")
    print("\nExemplos:")
    print("  5.0 8.0 1.5 1.0                (nome automático)")
    print("  5.0 8.0 1.5 1.0 teste_estavel  (nome personalizado)")
    print("\nPosição: valores entre 0 e 11")
    print("Ganhos: valores positivos (sugestão: 0.5 a 3.0)")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nDigite os valores: ").strip()
            
            if not user_input:
                print("Por favor, digite pelo menos quatro valores.")
                continue
                
            values = user_input.split()
            
            if len(values) < 4:
                print("Erro: Digite pelo menos quatro valores (x y Kp_angular Kp_linear)")
                continue
            
            # Primeiros 4 valores obrigatórios
            x = float(values[0])
            y = float(values[1])
            Kp_angular = float(values[2])
            Kp_linear = float(values[3])
            
            # Nome do experimento (opcional)
            experiment_name = None
            if len(values) >= 5:
                experiment_name = values[4]
            
            # Validar valores
            if x < 0 or x > 11 or y < 0 or y > 11:
                print("Aviso: Valores de posição devem estar entre 0 e 11")
                print("Deseja continuar mesmo assim? (s/n)")
                confirm = input("> ").lower()
                if confirm != 's':
                    continue
            
            if Kp_angular <= 0 or Kp_linear <= 0:
                print("Aviso: Ganhos devem ser valores positivos")
                print("Deseja continuar mesmo assim? (s/n)")
                confirm = input("> ").lower()
                if confirm != 's':
                    continue
            
            print(f"\nConfiguração definida:")
            print(f"  Posição alvo: x={x:.2f}, y={y:.2f}")
            print(f"  Ganho angular (Kp_angular): {Kp_angular:.2f}")
            print(f"  Ganho linear (Kp_linear): {Kp_linear:.2f}")
            if experiment_name:
                print(f"  Nome do experimento: {experiment_name}")
            print("\nIniciando controle...")
            return x, y, Kp_angular, Kp_linear, experiment_name
            
        except ValueError:
            print("Erro: Digite números válidos.")
        except KeyboardInterrupt:
            print("\n\nPrograma cancelado pelo usuário.")
            exit(0)

###########################################################################################
# FUNÇÃO MAIN PARA EXECUÇÃO DO NÓ
###########################################################################################
def main(args=None):
    # Obtendo a posição, ganhos e nome do experimento
    target_x, target_y, Kp_angular, Kp_linear, experiment_name = get_user_input()

    rclpy.init(args=args)
    node = ControlTurtle(target_x, target_y, Kp_angular, Kp_linear, experiment_name)
    
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    
    print("\nControlador iniciado. Aguardando conclusão...")
    print(f"Dados serão salvos em: turtle_experiments/{experiment_name or node.experiment_name}*")
    
    try:
        while rclpy.ok() and not node.should_shutdown:
            executor.spin_once(timeout_sec=0.1)
            
        if node.should_shutdown:
            print("\n" + "="*60)
            print("PROGRAMA CONCLUÍDO COM SUCESSO!")
            print("="*60)
            print(f"Dados coletados em vetores, prontos para exportação.")
            print(f"Total de amostras: {len(node.timestamps)}")
            print("="*60)
            
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário.")
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()
        print("Node encerrado com sucesso!")

if __name__ == '__main__':
    main()