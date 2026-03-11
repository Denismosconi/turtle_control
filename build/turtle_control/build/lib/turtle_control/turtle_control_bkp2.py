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
#CRIANDO O NÓ ROS2 E INICIALIZANDO VARIÁVEIS
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
        #self.timestamps = []  # Timestamps para cada amostra
        self.pos_x = []       # posicao.x
        self.pos_y = []       # posicao.y
        self.error_x_list = []  # error_x
        self.error_y_list = []  # error_y
        self.vel_x_global = []  # velocidade_x (global)
        self.vel_y_global = []  # velocidade_y (global)
        self.vel_linear_x = []  # vetor_velocidade.linear.x
        self.angle_rad_list = []  # self.angle desejado em radianos
        self.pos_theta = [] #Posição angular em radianos
        self.error_angle_list = []  # erro_angle
        self.vel_angular_z = []  # vetor_velocidade.angular.z
        self.estados_list = []  # Estado atual em cada amostra
        self.time_elapsed = []  # Tempo decorrido desde o início
        self.error_distance = [] #Erro em distância

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

        if self.estado == "INICIAR":
            """Definição da oritenção para posição desejada - laço executado apenas na primeira vez"""
            # Calcular o vetor diferença (erro) de posição
            self.error_x = self.target_x - posicao.x
            self.error_y = self.target_y - posicao.y

            # Calcular o ângulo usando arco tangente (atan2 retorna ângulo em [-pi, pi])
            self.angle = math.atan2(self.error_y, self.error_x)

            #Calcula o seno e o cosseno do ângulo
            self.seno_angle = math.sin(self.angle)
            self.cosseno_angle = math.cos(self.angle)

            #Atualiza estado
            self.estado = "ROTACIONAR"

        elif self.estado == "ROTACIONAR":
            #Calcula o erro de posição angular
            self.erro_angle = self.angle-posicao.theta

            #Determina sinal de controle
            if abs(self.erro_angle) > self.angle_tolerance:
                self.vetor_velocidade.angular.z = self.Kp_angular*self.erro_angle
                self.vetor_velocidade.angular.z = np.clip(self.vetor_velocidade.angular.z, -self.max_angular_speed, self.max_angular_speed)
            
            else: #Chegou na orientação desejada
                self.vetor_velocidade.angular.z = 0.0
                time.sleep(0.5)  # Pequena pausa para estabilização
                self.estado = "TRANSLADAR"
                self.get_logger().info("Ângulo corrigido. Iniciando movimento de translação")

        elif self.estado == "TRANSLADAR":
            #Calculando o erro de posição
            self.error_x = self.target_x-posicao.x
            self.error_y = self.target_y-posicao.y

            #Determina sinal de controle
            if abs(self.error_x) > self.linear_tolerance or abs(self.error_y) > self.linear_tolerance:

                #Calculando velocidade no sistema de coordenadas global
                self.velocidade_x = self.error_x*self.Kp_linear
                self.velocidade_y = self.error_y*self.Kp_linear

                #Calculando a velocidade no sistema de coordenadas móvel
                self.velocidade = self.velocidade_x*self.cosseno_angle + self.velocidade_y*self.seno_angle

                #Atribuindo a velocidade calculada
                self.vetor_velocidade.linear.x = self.velocidade
                self.vetor_velocidade.linear.x = np.clip(self.vetor_velocidade.linear.x , -self.max_linear_speed, self.max_linear_speed)

            else: #Chegou na posição desejada
                #Para completamente            
                self.vetor_velocidade.linear.x = 0.0
                self.vetor_velocidade.angular.z = 0.0
                #Atualizando estado
                self.estado = "FINALIZADO"
            
        elif self.estado == "FINALIZADO":    
            if not self.already_printed_final:
                self.get_logger().info("Posição final alcançada!")
                self.save_all_data()
                self.already_printed_final = True
                self.iniciar_shutdown_timer()
     
        # Salvar timestamp
        #self.timestamps.append(current_time)
        self.time_elapsed.append(elapsed_time)

        # Salvar estado atual
        self.estados_list.append(self.estado)

        # Salvar dados de ângulo
        self.angle_rad_list.append(self.angle) #Ângulo desejado em radianos
        self.pos_theta.append(posicao.theta) #Posição angular
        self.error_angle_list.append(self.erro_angle) #Erro de posição angular
        self.vel_angular_z.append(self.vetor_velocidade.angular.z) #Velocidade angular
        
        # Salvar posição linear
        self.pos_x.append(posicao.x)
        self.pos_y.append(posicao.y)
        self.error_x_list.append(self.error_x )
        self.error_y_list.append(self.error_y)
        self.vel_x_global.append(self.velocidade_x)
        self.vel_y_global.append(self.velocidade_y)
        self.vel_linear_x.append(self.vetor_velocidade.linear.x)
        self.error_dist = math.sqrt(self.error_x**2 + self.error_y**2) #Calcula erro em distância
        self.error_distance.append(self.error_dist)

        # Publicando a velocidade
        self.velocidade_atuador.publish(self.vetor_velocidade)


###########################################################################################
#ENCERRAMENTO DO NÓ
###########################################################################################
    def iniciar_shutdown_timer(self):
        """Inicia o timer de 2 segundos para encerrar o programa"""
        if not self.already_printed_shutdown:
            self.get_logger().info("Destino alcançado! O programa será encerrado em 2 segundos...")
            self.already_printed_shutdown = True
            
        # Criar timer para encerrar após 2 segundos
        self.shutdown_timer = self.create_timer(2.0, self.encerrar_programa)
    
    def encerrar_programa(self):
        """Encerra o programa graciosamente"""
        self.get_logger().info("Encerrando programa...")

        # Marcar que o programa deve encerrar
        self.should_shutdown = True
        
        # Destruir o timer de shutdown para evitar chamadas repetidas
        if self.shutdown_timer:
            self.shutdown_timer.destroy()
        
        self.get_logger().info("Node pronto para encerramento...")

###########################################################################################
# FUNÇÕES DE SALVAMENTO DE DADOS
###########################################################################################
def save_to_json(self):
    """Salva todos os dados em formato JSON"""
    # Calcular o tempo total
    total_time = self.time_elapsed[-1] if self.time_elapsed else 0
    
    data_dict = {
        'metadata': {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'target_position': {'x': self.target_x, 'y': self.target_y},
            'controller_gains': {
                'Kp_angular': self.Kp_angular, 
                'Kp_linear': self.Kp_linear
            },
            'tolerances': {
                'angle': self.angle_tolerance, 
                'linear': self.linear_tolerance
            },
            'speed_limits': {
                'max_linear_speed': self.max_linear_speed,
                'max_angular_speed': self.max_angular_speed
            },
            'total_samples': len(self.time_elapsed),
            'total_time': total_time
        },
        'data': {
            'time': self.time_elapsed,
            'pos_x': self.pos_x,
            'pos_y': self.pos_y,
            'pos_theta': self.pos_theta if hasattr(self, 'pos_theta') else [],
            'target_angle_rad': self.angle_rad_list,
            'target_angle_deg': self.angle_deg_list,
            'error_angle': self.error_angle_list,
            'error_x': self.error_x_list,
            'error_y': self.error_y_list,
            'error_distance': self.error_distance if hasattr(self, 'error_distance') else [],
            'vel_linear_x': self.vel_linear_x,
            'vel_angular_z': self.vel_angular_z,
            'vel_x_global': self.vel_x_global,
            'vel_y_global': self.vel_y_global,
            'state': self.estados_list
        }
    }
    
    filename = os.path.join(self.save_dir, f"{self.experiment_name}.json")
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=2, default=str)
    
    self.get_logger().info(f"Dados JSON salvos em: {filename}")

def save_to_csv(self):
    """Salva dados principais em formato CSV (MATLAB-friendly)"""
    filename = os.path.join(self.save_dir, f"{self.experiment_name}_main.csv")
    
    # Garantir que todas as listas tenham o mesmo tamanho
    n_samples = len(self.time_elapsed)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Cabeçalho
        writer.writerow([
            'time', 'pos_x', 'pos_y', 'target_angle_deg', 'error_angle_rad',
            'error_x', 'error_y', 'vel_linear_x', 'vel_angular_z',
            'vel_x_global', 'vel_y_global', 'state'
        ])
        
        # Dados
        for i in range(n_samples):
            writer.writerow([
                self.time_elapsed[i] if i < len(self.time_elapsed) else 0,
                self.pos_x[i] if i < len(self.pos_x) else 0,
                self.pos_y[i] if i < len(self.pos_y) else 0,
                self.angle_deg_list[i] if i < len(self.angle_deg_list) else 0,
                self.error_angle_list[i] if i < len(self.error_angle_list) else 0,
                self.error_x_list[i] if i < len(self.error_x_list) else 0,
                self.error_y_list[i] if i < len(self.error_y_list) else 0,
                self.vel_linear_x[i] if i < len(self.vel_linear_x) else 0,
                self.vel_angular_z[i] if i < len(self.vel_angular_z) else 0,
                self.vel_x_global[i] if i < len(self.vel_x_global) else 0,
                self.vel_y_global[i] if i < len(self.vel_y_global) else 0,
                self.estados_list[i] if i < len(self.estados_list) else ""
            ])
    
    self.get_logger().info(f"Dados CSV salvos em: {filename}")

def update_experiments_summary(self):
    """Atualiza o arquivo CSV de resumo de todos os experimentos"""
    summary_file = os.path.join(self.save_dir, "experiments_summary.csv")
    
    current_summary = {
        'experiment_name': self.experiment_name,
        'timestamp': datetime.now().isoformat(),
        'target_x': self.target_x,
        'target_y': self.target_y,
        'Kp_angular': self.Kp_angular,
        'Kp_linear': self.Kp_linear,
        'initial_x': self.pos_x[0] if self.pos_x else None,
        'initial_y': self.pos_y[0] if self.pos_y else None,
        'final_x': self.pos_x[-1] if self.pos_x else None,
        'final_y': self.pos_y[-1] if self.pos_y else None,
        'total_samples': len(self.time_elapsed),
        'total_time': self.time_elapsed[-1] if self.time_elapsed else 0,
        'final_error_x': self.error_x_list[-1] if self.error_x_list else None,
        'final_error_y': self.error_y_list[-1] if self.error_y_list else None
    }
    
    # Verificar se o arquivo de resumo já existe
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)
            
            # Adicionar novo experimento
            existing_data.append(current_summary)
            
            # Escrever de volta
            with open(summary_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=current_summary.keys())
                writer.writeheader()
                writer.writerows(existing_data)
        except Exception as e:
            self.get_logger().error(f"Erro ao ler resumo existente: {str(e)}")
            # Criar novo arquivo
            with open(summary_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=current_summary.keys())
                writer.writeheader()
                writer.writerow(current_summary)
    else:
        # Criar novo arquivo
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=current_summary.keys())
            writer.writeheader()
            writer.writerow(current_summary)
    
    self.get_logger().info(f"Resumo atualizado em: {summary_file}")

def save_all_data(self):
    """Salva todos os dados (JSON, CSV e atualiza resumo)"""
    self.save_to_json()
    self.save_to_csv()
    self.update_experiments_summary()
    self.get_logger().info("Todos os dados foram salvos com sucesso!")

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
            print(f"Dados salvos em: turtle_experiments/")
            print(f"Arquivos gerados:")
            print(f"  • {node.experiment_name}.json (dados completos)")
            print(f"  • {node.experiment_name}_main.csv (tabela principal)")
            print(f"  • {node.experiment_name}_*.csv (dados por fase)")
            print(f"  • experiments_summary.csv (resumo de todos experimentos)")
            print("="*60)
            
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário.")
        # Salvar dados mesmo se interrompido
        node.save_all_data()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()
        print("Node encerrado com sucesso!")

if __name__ == '__main__':
    main()
