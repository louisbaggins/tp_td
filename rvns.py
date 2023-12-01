"""
@author: Lucas S Batista
"""


'''
Importa os módulos usados
'''
import numpy as np
import copy
import random 
import pandas as pd
from collections import Counter
import math

DISTANCE = 'distance'

def shake(current_solution, current_pas):
 
    new_solution = current_solution.copy()
    
    # Escolha aleatoriamente um ponto de entrega (PA) e um cliente
    i = random.randint(0, len(new_solution) - 1)
    j = random.randint(0, len(new_solution) - 1)
    
    pa_1 = random.choice(current_pas[current_pas['isActive'] == True]['id'].values) 
    pa_2 = random.choice(current_pas[current_pas['isActive'] == True]['id'].values) 
    
    # Troca o PA do cliente
    new_solution['access_point'][i] = pa_1
    new_solution['access_point'][j] = pa_2
    return new_solution
    

def distance(x1, y1, x2, y2) -> float:
    return math.sqrt((x2 - x1)**2 + (y2-y1)**2)

def calculate_band(b_c: float, b_pa: float):
    return b_c + b_pa < 54

def ValidSol(x_ii_c: pd.DataFrame):
    na_clients = len([value for value in x_ii_c['distance'] if value < 70])
    return (na_clients*100 / len(x_ii_c)) <= 5

def MinSol(x_ii_c: pd.DataFrame, x_ii: pd.DataFrame) -> bool:
     l_c = list(x_ii_c['access_point'].unique())
     l_ii = list(x_ii['access_point'].unique())
     
     return len(l_c) < len(l_ii)

def NeighborhoodChange(x: pd.DataFrame, x_ii: pd.DataFrame, pa_t: pd.DataFrame, obj_function: function, approachinfo, k: int):
    x_ii_copy = x_ii.copy()
    pa_t_copy = pa_t.copy()

    # Deactivate less connected PA's
    if k == 1:
        shortest_distance = 1000
        temp_clients = x_ii.copy()
        access_points_least_used = temp_clients['access_point'].tolist()
        least_common = int((Counter(access_points_least_used).most_common()[-1])[0])
        x_ii_copy.loc[x_ii_copy['access_point'] == least_common, 'access_point'] = -1
        pa_t_copy.loc[least_common, 'isActive'] = False
        for index, row in x_ii_copy.iterrows():
            if x_ii_copy['access_point'][index] == -1:
                for idx, r in pa_t_copy.iterrows():
                    dist = distance(
                        x_ii_copy['x'][index],
                        x_ii_copy['y'][index],
                        pa_t_copy['x'][idx],
                        pa_t_copy['y'][idx]
                    )
                    if dist < shortest_distance and calculate_band(x_ii_copy['consumo'][index], pa_t_copy['band'][idx]):
                        shortest_distance = dist
                        index_t, idx_t = index, idx

                if shortest_distance < 70:
                    x_ii_copy.loc[index_t, 'access_point'] = pa_t_copy.loc[idx_t, 'id']
                    x_ii_copy.loc[index_t, DISTANCE] = shortest_distance
                    accumulated_band = pa_t_copy.loc[idx_t, 'band'] + x_ii_copy.loc[index_t, 'consumo']
                    pa_t_copy.loc[idx_t, 'band'] = accumulated_band
                     

    if k == 2:         
        # Escolha aleatoriamente um ponto de entrega (PA) e um cliente
        i = random.randint(0, len(x_ii_copy) - 1)
        j = random.randint(0, len(x_ii_copy) - 1)
        
        pa_1 = random.choice(pa_t[pa_t['isActive'] == True]['id'].values) 
        pa_2 = random.choice(pa_t[pa_t['isActive'] == True]['id'].values) 
        
        # Troca o PA do cliente
        x_ii_copy.loc[i, 'access_point'] = pa_1
        x_ii_copy.loc[j, 'access_point'] = pa_2

    # Swap Two Clients
    if k == 3:        
        # Escolha aleatoriamente um ponto de entrega (PA) e um cliente
        i = random.randint(0, len(x_ii_copy) - 1)
        j = random.randint(0, len(x_ii_copy) - 1)
        
        

        t_hold = x_ii_copy['access_point'][i]
        x_ii_copy.loc[i, 'access_point'] = x_ii_copy['access_point'][j]
        x_ii_copy.loc[j, 'access_point'] = t_hold
    
    if obj_function(x_ii, approachinfo) < obj_function(x, approachinfo):
        x = x_ii
        k += 1
        pa_t = pa_t_copy

    
    return x_ii, k, pa_t

'''
Implementa a função shake
'''
# def shake(x,k,probdata):
    
#     y = copy.deepcopy(x)
#     r = np.random.permutation(probdata.n)       
    
#     if k == 1:             # apply not operator in one random position
#         y.solution[r[0]] = not(y.solution[r[0]])
        
#     elif k == 2:           # apply not operator in two random positions        
#         y.solution[r[0]] = not(y.solution[r[0]])
#         y.solution[r[1]] = not(y.solution[r[1]])
        
#     elif k == 3:           # apply not operator in three random positions
#         y.solution[r[0]] = not(y.solution[r[0]])
#         y.solution[r[1]] = not(y.solution[r[1]])
#         y.solution[r[2]] = not(y.solution[r[2]])        
    
#     return y

'''
Implementa a função neighborhoodChange
'''
def neighborhoodChange(x, y, k):
    
    if y.single_objective_value < x.single_objective_value:
        x = copy.deepcopy(y)
        k = 1
    else:
        k += 1
        
    return x, k


'''
Implementa uma metaheurística RVNS
'''
def rvns_approach(fobj,x,probdata,approachinfo,maxeval=1000):
    
    # Contador do número de soluções candidatas avaliadas
    num_sol_avaliadas = 0

    # Máximo número de soluções candidatas avaliadas
    max_num_sol_avaliadas = maxeval

    # Número de estruturas de vizinhanças definidas
    kmax = 3
       
    # Avalia solução inicial
    x = fobj(x,approachinfo,probdata)
    num_sol_avaliadas += 1
    
    
    # Ciclo iterativo do método
    while num_sol_avaliadas < max_num_sol_avaliadas:

        k = 1
        while k <= kmax:

            # Gera uma solução candidata na k-ésima vizinhança de x          
            y = shake(x,k,probdata)
            y = fobj(y,approachinfo,probdata)
            num_sol_avaliadas += 1

            # Atualiza solução corrente e estrutura de vizinhança (se necessário)
            x, k = neighborhoodChange(x, y, k)
    
    return x


def vns_approach(fobj, x, pas, approachinfo, maxeval=1000):
    # Contador do número de soluções candidatas avaliadas
    num_sol_avaliadas = 0

    # Máximo número de soluções candidatas avaliadas
    max_num_sol_avaliadas = maxeval

    # Sequência fixa de vizinhanças
    neighborhoods = [1, 2, 3]  # Adapte de acordo com suas necessidades

    # Avalia solução inicial
    x = fobj(x, approachinfo)
    num_sol_avaliadas += 1
    c = 0
    pas_copy = pas.copy()
    x
    # Ciclo iterativo do método
    while num_sol_avaliadas < max_num_sol_avaliadas:
        for k in range(1,3):
            # Gera uma solução candidata na k-ésima vizinhança de x
            y = shake(x, pas_copy)
            y = fobj(y, approachinfo)
            num_sol_avaliadas += 1

            # Atualiza solução corrente e estrutura de vizinhança (se necessário)
            x, k, _ = NeighborhoodChange(x, y, pas_copy, fobj, approachinfo, k)  # 0 indica que não houve mudança na vizinhança

    return x
