import math
import numpy as np
import ac3
import matplotlib.pyplot as plt
import pandas as pd


class Problem:
    """
    Representa un problema de optimización con una función objetivo y restricciones dinámicas.
    """
    def __init__(self, dimension, lower_bound, upper_bound, objective_function, constraints_function, goal="maximize"):
        """
        Inicializa el problema con parámetros definidos por el usuario.

        Args:
            dimension (int): El número de variables en el problema.
            lower_bound (float): El límite inferior para todas las variables.
            upper_bound (float): El límite superior para todas las variables.
            objective_function (callable): Una función que toma un vector de solución (array de numpy)
                                          y devuelve su valor objetivo.
            constraints_function (callable): Una función que toma un vector de solución (array de numpy)
                                            y devuelve True si es factible, False en caso contrario.
            goal (str): "maximize" o "minimize" la función objetivo.
        """
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._objective_function = objective_function
        self._constraints_function = constraints_function
        self.goal = goal.lower() # Asegura que esté en minúsculas para una comparación consistente

        if self.goal not in ["maximize", "minimize"]:
            raise ValueError("El objetivo debe ser 'maximize' o 'minimize'.")

    def check(self, x_raw):
        """
        Verifica si una solución dada satisface las restricciones del problema.
        Las soluciones se redondean a enteros antes de verificar las restricciones para problemas enteros.
        """
        # Llama a la función de restricciones definida por el usuario
        return self._constraints_function(x_raw)

    def fit(self, x_raw):
        """
        Calcula el valor de la función objetivo para una solución dada.
        Para soluciones inviables, devuelve -inf para maximización o +inf para minimización.
        """
        # Las soluciones se redondean a enteros para el cálculo del fitness según la definición del problema.
        # Es importante que la función constraints_function también maneje el redondeo si es necesario.
        if not self.check(x_raw):
            return -np.inf if self.goal == "maximize" else np.inf

        # Llama a la función objetivo definida por el usuario
        return self._objective_function(x_raw)

    def keep_domain(self, x_val, min_val, max_val):
        """
        Recorta los valores de posición para que permanezcan dentro de los límites definidos.
        """
        return np.clip(x_val, min_val, max_val)


class SecretaryBird:
    """
    Representa un solo ave secretario (candidato a solución) en el enjambre SBOA.
    """
    def __init__(self, problem: Problem):
        self.p = problem
        self.dimension = self.p.dimension
        
        # Inicializa la posición aleatoriamente dentro de los límites definidos
        self.position = np.random.uniform(self.p.lower_bound, self.p.upper_bound, self.dimension)
        
        self.fitness = self.p.fit(self.position)

    def update_fitness(self):
        """Recalcula y actualiza el fitness del ave."""
        self.fitness = self.p.fit(self.position)

    def is_better_than(self, other_bird):
        """
        Compara el fitness de esta ave con el de otra ave según el objetivo de optimización.
        """
        if self.p.goal == "maximize":
            return self.fitness > other_bird.fitness
        else: # minimizar
            return self.fitness < other_bird.fitness

    def copy_from(self, other_bird):
        """Copia la posición y el fitness de otra ave."""
        if isinstance(other_bird, SecretaryBird):
            self.position = other_bird.position.copy()
            self.fitness = other_bird.fitness


    def __str__(self):
        # Redondea la posición para mostrarla en formato entero
        x_display = np.round(self.position).astype(int)
        return f"Posición (redondeada): {x_display}, Fitness: {self.fitness:.4f}"


class SBOA:
    """
    Implementación del Algoritmo de Optimización de Aves Secretarias (SBOA).
    """
    def __init__(self, problem: Problem, n_birds=50, max_iterations=200, 
                 f_max=2, f_min=0.5, # Parámetros para F_t (ajustar según sea necesario según el artículo o pruebas)
                 initial_c=2,        # Parámetro para C_t (ajustar según sea necesario)
                 perturbation_factor=0.5): # Parámetro para la etapa de Ataque a la Presa (Ec. 9)

        self.n_birds = n_birds
        self.max_iterations = max_iterations
        
        self.problem = problem # Acepta una instancia de Problem
        
        self.swarm = []
        self.P_best = None # Mejor ave global (objeto SecretaryBird)
        # Inicializa P_best_fitness según el objetivo
        self.P_best_fitness = -np.inf if self.problem.goal == "maximize" else np.inf

        # Parámetros específicos de SBOA
        self.f_max = f_max
        self.f_min = f_min
        self.initial_c = initial_c
        self.perturbation_factor = perturbation_factor
        self.s_levy = 0.01 # De la descripción del artículo para el vuelo de Levy
        self.eta_levy = 1.5 # De la descripción del artículo para el vuelo de Levy
        self.p_best_history = []
        self.avg_fitness_history = []
        self.min_fitness_history = []
        self.max_fitness_history = []


    def _initialize_population(self):
        """
        Paso 1: Inicializa la población de aves secretarias.
        """
        for _ in range(self.n_birds):
            bird = SecretaryBird(self.problem)
            self.swarm.append(bird)

            # Actualiza el mejor global inicial
            if self.P_best is None or bird.is_better_than(self.P_best):
                self.P_best_fitness = bird.fitness
                self.P_best = bird # Almacena el objeto ave

    def _update_global_best(self):
        """
        Actualiza P_best basándose en el fitness actual del enjambre.
        """
        for bird in self.swarm:
            bird.update_fitness() # Asegura que el fitness del ave esté actualizado
            if bird.is_better_than(self.P_best):
                self.P_best_fitness = bird.fitness
                self.P_best = bird # Almacena el objeto ave
        self.p_best_history.append(self.P_best_fitness)
        current_swarm_fitnesses = [bird.fitness for bird in self.swarm]
        self.avg_fitness_history.append(np.mean(current_swarm_fitnesses))
        self.min_fitness_history.append(np.min(current_swarm_fitnesses))
        self.max_fitness_history.append(np.max(current_swarm_fitnesses))



    def _calculate_F_t(self, t):
        """
        Ec. (3): Inicializa el vector de la función objetivo F_t.
        F_t = (F_max - F_min) * (Best_fitness / Worst_fitness) + F_min si Best_fitness <= Worst_fitness, de lo contrario F_min
        Requiere el peor fitness actual del enjambre.
        """
        # Determina el peor fitness según el objetivo
        if self.problem.goal == "maximize":
            current_worst_fitness = min(bird.fitness for bird in self.swarm)
        else: # minimizar
            current_worst_fitness = max(bird.fitness for bird in self.swarm)
        
        # La condición de verificación debe estar alineada con el objetivo también
        if self.problem.goal == "maximize":
            condition = self.P_best_fitness <= current_worst_fitness
        else: # minimizar
            condition = self.P_best_fitness >= current_worst_fitness # El mejor fitness debe ser menor o igual que el peor

        if condition:
            # Maneja casos extremos para el cálculo de la razón (ej. división por cero, o -inf/inf)
            if current_worst_fitness == 0 and self.problem.goal == "maximize":
                 ratio = 1 
            elif current_worst_fitness == np.inf and self.problem.goal == "minimize":
                 ratio = 1 
            elif current_worst_fitness == -np.inf or current_worst_fitness == np.inf: # Si el peor ya es infinito/-infinito
                 ratio = 1 # Establece un valor neutral para evitar problemas
            elif self.P_best_fitness == current_worst_fitness: # Para evitar división por cero en la razón si todos son iguales
                ratio = 1
            else:
                 ratio = self.P_best_fitness / current_worst_fitness

            F_t = (self.f_max - self.f_min) * ratio + self.f_min
        else:
            F_t = self.f_min
        return F_t

    def _generate_levy_flight_vector(self, dimension):
        """
        Genera un vector de paso de vuelo de Levy basado en las fórmulas del artículo (Ec. 12 y 13 implícitas).
        """
        sigma_num = math.gamma(1 + self.eta_levy) * math.sin(math.pi * self.eta_levy / 2)
        sigma_den = math.gamma((1 + self.eta_levy) / 2) * self.eta_levy * (2**((self.eta_levy - 1) / 2))
        sigma = (sigma_num / sigma_den)**(1 / self.eta_levy)
        
        u = np.random.normal(0, sigma**2, dimension) # N(0, sigma^2)
        v = np.random.normal(0, 1, dimension)       # N(0, 1)
        
        # Evita la división por cero si v está demasiado cerca de 0
        v[np.abs(v) < 1e-10] = 1e-10 
        
        step = u / (np.abs(v)**(1 / self.eta_levy))
        
        return self.s_levy * step 

    def _exploration_phase(self, t):
        """
        Fase de exploración (Caza de Serpientes) - Etapa 2 (Ec. 4 y 5) y Etapa 3 (Ec. 9 y 10).
        """
        F_t = self._calculate_F_t(t) 

        for bird in self.swarm:
            current_pos = bird.position.copy()
            r_vector = np.random.rand(self.problem.dimension) # Vector aleatorio para actualizaciones

            # Etapa 2: Observación y ataque
            if abs(F_t) <= 0.5:
                # Ec. (4): P_i^{t+1} = Best_position - r_vector * (Best_position - P_i^t) / F_t
                if abs(F_t) < 1e-10: F_t = 1e-10 # Evita la división por cero
                new_pos_stage2 = self.P_best.position - r_vector * (self.P_best.position - current_pos) / F_t
            else:
                # Ec. (5): P_i^{t+1} = P_i^t + r_vector * F_t
                new_pos_stage2 = current_pos + r_vector * F_t

            new_pos_stage2 = self.problem.keep_domain(new_pos_stage2, self.problem.lower_bound, self.problem.upper_bound)
            
            # Etapa 3: Ataque a la Presa
            levy_flight_vec = self._generate_levy_flight_vector(self.problem.dimension)

            # Ec. (9): P_i^{t+1} = Best_position + Perturbation_factor * Levy_flight(D) * (P_i^t - Best_position)
            new_pos_stage3_eq9 = self.P_best.position + self.perturbation_factor * levy_flight_vec * (new_pos_stage2 - self.P_best.position)

            # Ec. (10): P_i^{t+1} = P_i^t + random_vector * (Best_position - P_i^t)
            new_pos_stage3_eq10 = new_pos_stage3_eq9 + np.random.rand(self.problem.dimension) * (self.P_best.position - new_pos_stage3_eq9)

            bird.position = self.problem.keep_domain(new_pos_stage3_eq10, self.problem.lower_bound, self.problem.upper_bound)


    def _calculate_C_t(self, t):
        """
        Ec. (7): Calcula el factor de peligro C_t.
        C_t = Initial_C - t * (Initial_C / Max_iteration)
        """
        return self.initial_c - t * (self.initial_c / self.max_iterations)


    def _exploitation_phase(self, t):
        """
        Fase de explotación (Escape de Depredadores) - Etapa 2 (C1 o C2).
        """
        C_t = self._calculate_C_t(t) 

        for bird in self.swarm:
            current_pos = bird.position.copy()
            rand_vector = np.random.rand(self.problem.dimension) # Vector aleatorio para actualizaciones

            # Elige la estrategia de escape (C1 o C2) basándose en C_t
            if C_t < 0.5: 
                # Estrategia C1 (Camuflaje): P_i^{t+1} = P_i^t + rand_vector * (Best_position - P_i^t)
                new_pos_escape = current_pos + rand_vector * (self.P_best.position - current_pos)
            else:
                # Estrategia C2 (Volar o huir): P_i^{t+1} = P_i^t + Levy_flight(D) * Best_position
                levy_flight_vec = self._generate_levy_flight_vector(self.problem.dimension)
                new_pos_escape = current_pos + levy_flight_vec * self.P_best.position 

            bird.position = self.problem.keep_domain(new_pos_escape, self.problem.lower_bound, self.problem.upper_bound)


    def optimizer(self):
        """
        Bucle principal de ejecución del algoritmo SBOA.
        """
        self._initialize_population()
        self._update_global_best() 

        print(f"Fitness inicial del P_best: {self.P_best_fitness:.4f}")

        for t in range(self.max_iterations): 
            # Fase de Exploración
            self._exploration_phase(t)

            # Fase de Explotación
            self._exploitation_phase(t)

            # Actualiza el mejor global
            self._update_global_best()

            print(f"Iteración {t+1}/{self.max_iterations}, Fitness del P_best: {self.P_best_fitness:.4f}, soluciones: {self.P_best}")

        return self.P_best, self.P_best_fitness


# --- Bloque de ejecución principal para ejemplos ---
if __name__ == "__main__":
    # Parte 1 de la solución al problema, verificar arco consistencia y nodo consistencia con ac-3
    ac3.domains = {
        'x1': list(range(0, 16)), # 0 a 15
        'x2': list(range(0, 11)), # 0 a 10
        'x3': list(range(0, 26)), # 0 a 25
        'x4': list(range(0, 5)),  # 0 a 4
        'x5': list(range(0, 31)), # 0 a 30
    }
    ac3.constraints = {
        ('x1', 'x2'): lambda x1, x2: x1 * 174 <= 3800 - 320 * x2,
        ('x2', 'x1'): lambda x2, x1: 3800 - 320 * x2 >= x1 * 174,
        ('x3', 'x4'): lambda x3, x4: 50 * x3 <= 2800 - 105 * x4,
        ('x4', 'x3'): lambda x4, x3: 2800 - 105 * x4 >= 50 * x3,
        ('x3', 'x5'): lambda x3, x5: 50 * x3 <= 3500 - 16 * x5,
        ('x5', 'x3'): lambda x5, x3: 3500 - 16 * x5 >= 50 * x3,
    }
    ac3.arcs = [
        ('x1', 'x2'), ('x2', 'x1'),
        ('x3', 'x4'), ('x4', 'x3'),
        ('x3', 'x5'), ('x5', 'x3'),
    ]

    print("-- PRIMERA PARTE DEL PROBLEMA: Verificación de consistencia de arcos y nodos con AC-3 --")
    print("Dominio inicial, antes de ac-3:", ac3.domains)
    ac3.ac3(ac3.arcs)
    print("Dominio final, después de ac-3:", ac3.domains)
    print("AC3 ejecutado sin problemas")
    # Puedes calcular los upper bounds a partir de los dominios finales de ac3:
    upper_bounds = [max(ac3.domains[var]) for var in ['x1', 'x2', 'x3', 'x4', 'x5']]
    overall_upper_bound = max(upper_bounds)
    for var, ub in zip(['x1', 'x2', 'x3', 'x4', 'x5'], upper_bounds):
        print(f"Límite superior para {var}: {ub}")
    # 1. Define la función objetivo para el Problema III
    def problema_iii_objective(x_raw):
        # Redondea a entero para los cálculos según la especificación del problema
        x = np.round(x_raw).astype(int) 
        values = [72, 92, 45, 65, 26] # Corresponde a x1, x2, x3, x4, x5
        Z1 = (values[0] * x[0] + values[1] * x[1] +
              values[2] * x[2] + values[3] * x[3] +
              values[4] * x[4])
        return Z1
    
    def multi_objective_combined(x_raw, w_ganancia=0.7, w_costo=0.3): # Igual ganancia que costo, modificar pesos de cada objetivo si es necesario
        x = np.round(x_raw).astype(int)

        # Calcular Ganancia (Z_ganancia)
        Z_ganancia = (72 * x[0] + 92 * x[1] + 45 * x[2] + 65 * x[3] + 26 * x[4])

        # Calcular Costo (Z_costo) - Asegúrate de que esta suma sea la correcta de todos los costos
        cost_tv = (174 * x[0] + 320 * x[1])
        cost_dm = (50 * x[2] + 105 * x[3])
        cost_dr = (50 * x[2] + 16 * x[4]) # x3 (Diario) puede estar aquí y en cost_dm, cuidado con doble conteo si no es intencional
        Z_costo = cost_tv + cost_dm + cost_dr

        # Función objetivo combinada (queremos maximizar las ganancias y minimizar los costos)
        # Convertimos la minimización de costo en maximización de -costo
        return (w_ganancia * Z_ganancia) - (w_costo * Z_costo)

    # 2. Define la función de restricciones para el Problema III
    def problema_iii_constraints(x_raw):
        x = np.round(x_raw).astype(int) # Redondea a entero para la verificación

        # Restricciones de cantidad individual
        x1_max, x2_max, x3_max, x4_max, x5_max = 15, 10, 25, 4, 30
        if not (x[0] <= x1_max and x[1] <= x2_max and
                x[2] <= x3_max and x[3] <= x4_max and
                x[4] <= x5_max):
            return False

        # Restricciones presupuestarias
        costs_tv = [174, 320] # Para x1, x2 (Televisión)
        costs_dm = [50, 105]  # Para x3, x4 (Diario + Revista)
        costs_dr = [50, 16]   # Para x3, x5 (Diario + Radio)

        budget_tv_max = 3800
        budget_dm_max = 2800
        budget_dr_max = 3500

        if (costs_tv[0] * x[0] + costs_tv[1] * x[1]) > budget_tv_max:
            return False
        
        if (costs_dm[0] * x[2] + costs_dm[1] * x[3]) > budget_dm_max:
            return False
            
        if (costs_dr[0] * x[2] + costs_dr[1] * x[4]) > budget_dr_max:
            return False

        return True # Todas las restricciones satisfechas

    # 3. Crea una instancia de Problem dinámica para el Problema III
    # Determina el límite superior general a partir de las cantidades máximas individuales
   

    problema_instance = Problem(
        dimension=5,
        lower_bound=0,
        upper_bound=overall_upper_bound,
        objective_function=multi_objective_combined, # Usa la función combinada para múltiples objetivos, la otra es problema_iii_objective
        constraints_function=problema_iii_constraints, #problema_iii_constraints,
        goal="maximize"
    )
    print("\n--- SEGUNDA PARTE DEL PROBLEMA, OPTIMIZACIÓN MEDIANTE ESCALARIZING ---")
    # 4. Instancia y ejecuta SBOA con el problema dinámico
    sboa_optimizer = SBOA(problem=problema_instance, n_birds=100, max_iterations=500) 
    
    best_bird_found, best_fitness_found = sboa_optimizer.optimizer()

    print("\n--- Optimización Completada ---")
    
    best_solution_rounded = np.round(best_bird_found.position).astype(int)
    
    print(f"Solución óptima (redondeada): x1={best_solution_rounded[0]}, x2={best_solution_rounded[1]}, "
          f"x3={best_solution_rounded[2]}, x4={best_solution_rounded[3]}, x5={best_solution_rounded[4]}")
    print(f"Fitness óptimo (Z1): {best_fitness_found:.4f}")
    
    if not problema_instance.check(best_solution_rounded):
        print("Advertencia: La solución 'óptima' final es en realidad inviable al ser redondeada y verificada.")
    else:
        actual_rounded_fitness = problema_instance.fit(best_solution_rounded)
        print(f"Fitness real de la solución redondeada: {actual_rounded_fitness:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(sboa_optimizer.p_best_history, label='Best Fitness (P_best)')
    plt.plot(sboa_optimizer.avg_fitness_history, label='Average Swarm Fitness')
    plt.plot(sboa_optimizer.min_fitness_history, label='Min Swarm Fitness')
    plt.plot(sboa_optimizer.max_fitness_history, label='Max Swarm Fitness')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('SBOA Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()
    data = {
    'Iteration': list(range(1, sboa_optimizer.max_iterations + 1)),
    'Best Fitness': sboa_optimizer.p_best_history,
    'Average Fitness': sboa_optimizer.avg_fitness_history,
    'Min Fitness': sboa_optimizer.min_fitness_history,
    'Max Fitness': sboa_optimizer.max_fitness_history
    }
    df_fitness_summary = pd.DataFrame(data)
    print("\nDescriptive Fitness Summary:")
    print(df_fitness_summary.to_string()) # .to_string() for full display in console

