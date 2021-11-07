import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt


# read data
distance_mt = []
time_windows = []
coor_matrix = []
time_mt = []
depot = 0
demand = []
capacity  = 0
service_time = []
f_read_name = open("name_data.txt")
name_data = f_read_name.readline().split()[0]
f_name = "In/" + name_data
file = open(f_name, "r")
line_count = 0
for line in file:
    if line != "\n":
        line_count += 1
file = open(f_name, "r")
for i in range(line_count+2):
    string_seq = (file.readline().split())
    if i == 4:
        capacity = int(string_seq[1])
    if i >= 9:
        x_cord = int(string_seq[1])
        y_cord = int(string_seq[2])
        cord = [x_cord, y_cord]
        coor_matrix.append(cord)
        demand.append(int(string_seq[3]))
        time_window = [int(string_seq[4]), int(string_seq[5])]
        time_windows.append(time_window)
        service_time.append(int(string_seq[6]))
num_of_point = len(demand)
num_of_cus = num_of_point - 1
for i in range(num_of_point):
    distance_tmp = []
    for j in range(num_of_point):
        distance_tmp.append(math.sqrt(pow(coor_matrix[i][0] - coor_matrix[j][0], 2) + pow(coor_matrix[i][1] - coor_matrix[j][1], 2)))
    distance_mt.append(distance_tmp)
time_mt = copy.deepcopy(distance_mt)
# parameter of GA
generation_span = 100   
population_size = 100
p_c = 0.8
p_m = 0.1


# generate initial population

random_pop_size = int(population_size*90/100)
greedy_pop_size = population_size - random_pop_size
init_population = []
# generate 90% random
set_point = []
for i in range(1, num_of_point):
    set_point.append(i)
for i in range(random_pop_size):
    random_sequence = np.random.permutation(set_point)
    init_population.append(random_sequence)

# generate 10% greedy
# sort nearest point is side by side
for i in range(greedy_pop_size):
    set_point = []
    for i in range(1, num_of_point):
        set_point.append(i)
    greedy_seq = []
    exist_of_nearest_point = False
    while True:
        if exist_of_nearest_point == False:
            remove_index = random.randrange(len(set_point))
            remove_point = set_point[remove_index]
        greedy_seq.append(remove_point)
        set_point.remove(remove_point)
        min_distance = 10000000000000
        nearest_point = remove_point
        for j in range(1, num_of_point):
            if distance_mt[remove_point][j] < min_distance and j != remove_point:
                min_distance == distance_mt[remove_point][j] 
                nearest_point = j
        if nearest_point in set_point:
            remove_point = nearest_point
            exist_of_nearest_point = True
        else:
            exist_of_nearest_point = False
        if len(greedy_seq) == num_of_cus:
            break
    init_population.append(greedy_seq)

# loop
current_population = copy.deepcopy(init_population)
for generation in range(generation_span):
    new_population = copy.deepcopy(current_population)
    size_of_current_pop = len(current_population)
    ranking_of_pop = np.zeros(size_of_current_pop, int)
    ranked_list = np.zeros(size_of_current_pop, int)
    pareto_vectors = []
    non_dominated_vector = [100000000, 10000000]
    dominated_list = np.ones(size_of_current_pop)
    all_current_routes = []
    for chro_index in range(size_of_current_pop):
        #Routing Scheme and evaluate rank
        choro_presentation = current_population[chro_index]
        routes = []
        time_of_routes = []
        current_route = []
        time_arrive_points = []
        current_route.append(depot)
        time_of_current_route = 0
        demand_of_current_route = 0
        distance_of_current_route = 0
        number_of_vehicles = 1
        pareto_vector = [0, 0]
        # phase 1 
        for gene_index in range(num_of_cus):
            gene = choro_presentation[gene_index]
            previous_gene = current_route[-1]
            if demand_of_current_route + demand[gene] <= capacity and time_of_current_route + time_mt[previous_gene][gene] <= time_windows[gene][1]:
                current_route.append(gene)
                demand_of_current_route += demand[gene]
                time_arrive = time_of_current_route + time_mt[previous_gene][gene]
                time_of_current_route = max(time_arrive, time_windows[gene][depot]) + service_time[gene]
                distance_of_current_route += distance_mt[previous_gene][gene]
            else:
                distance_of_current_route += distance_mt[previous_gene][depot] + distance_mt[depot][gene]
                number_of_vehicles += 1
                routes.append(current_route)
                time_arrive_points.append([0])
                current_route = [depot]
                current_route.append(gene)
                demand_of_current_route = demand[gene]
                time_arrive = time_mt[previous_gene][gene]
                time_of_current_route = max(time_arrive, time_windows[gene][0]) + service_time[gene]
        routes.append(current_route)
        distance_of_current_route += distance_mt[gene][depot]
        #
        pareto_vector[0] = number_of_vehicles
        pareto_vector[1] = distance_of_current_route
        # find 1 vector not dominated
        if pareto_vector[0] <= non_dominated_vector[0] and pareto_vector[1] <= non_dominated_vector[1]:
            if pareto_vector[0] < non_dominated_vector[0] or pareto_vector[1] < non_dominated_vector[1]:
                non_dominated_vector[0] = pareto_vector[0]
                non_dominated_vector[1] = pareto_vector[1]
        pareto_vectors.append(pareto_vector)
        all_current_routes.append(routes)
    # ranking
    cur_rank = 1
    number_of_non_ranking_vectors = size_of_current_pop
    while number_of_non_ranking_vectors != 0:
    
        # find non dominated vector
        non_dominated_vector = [100000000, 10000000]
        for i in range(size_of_current_pop):
            if ranked_list[i] == 0: 
                if pareto_vectors[i][0] <= non_dominated_vector[0] and pareto_vectors[i][1] <= non_dominated_vector[1]:
                    if pareto_vectors[i][0] < non_dominated_vector[0] or pareto_vectors[i][1] < non_dominated_vector[1]:
                        non_dominated_vector[0] = pareto_vectors[i][0]
                        non_dominated_vector[1] = pareto_vectors[i][1]
        for i in range(size_of_current_pop):
            if ranked_list[i] == 0:
                if pareto_vectors[i][0] < non_dominated_vector[0] or pareto_vectors[i][1] < non_dominated_vector[1]:
                    dominated_list[i] = 0
                elif pareto_vectors[i][0] == non_dominated_vector[0] and pareto_vectors[i][1] == non_dominated_vector[1]:
                    dominated_list[i] = 0
        #
        for i in range(size_of_current_pop):
            if dominated_list[i] == 0 and ranked_list[i] == 0:
                ranking_of_pop[i] = cur_rank
                ranked_list[i] = 1
                number_of_non_ranking_vectors -= 1
        #
        cur_rank += 1
    current_population = []
    all_new_routes = copy.deepcopy(all_current_routes)
    all_current_routes = []
    new_ranking_pop = copy.deepcopy(ranking_of_pop)
    ranking_of_pop = []
    size_of_tournament_set = 4
    for i in range(population_size):
        r = random.random()
        tournament_set = []
        best_rank_in_tournament = 1000000000000
        best_index_in_tournament = 1
        for j in range(size_of_tournament_set):
            random_index = random.randrange(0, size_of_current_pop)
            tournament_set.append(random_index)
            if new_ranking_pop[random_index] < best_rank_in_tournament:
                best_rank_in_tournament = new_ranking_pop[random_index]
                best_index_in_tournament = random_index
        if r < 0.8:
            survey_index = best_index_in_tournament
        else:
            random_index = random.randrange(0, 4)
            while tournament_set[random_index] == best_index_in_tournament:
                random_index = random.randrange(0, 4)
            survey_index = tournament_set[random_index]
        current_population.append(new_population[survey_index])
        ranking_of_pop.append(new_ranking_pop[survey_index])
        all_current_routes.append(all_new_routes[survey_index])
    size_of_current_pop = len(current_population)
    # GA operator
    # parent selection
    for i in range(population_size):
        r = random.random()
        if r < p_c:
            #find father
            tournament_set = []
            best_rank_in_tournament = 1000000000000
            best_index_in_tournament = 1
            for j in range(size_of_tournament_set):
                random_index = random.randrange(0, size_of_current_pop)
                tournament_set.append(random_index)
                if ranking_of_pop[random_index] < best_rank_in_tournament:
                    best_rank_in_tournament = ranking_of_pop[random_index]
                    best_index_in_tournament = random_index
            if r < 0.8:
                father_index = best_index_in_tournament
            else:
                random_index = random.randrange(0, 4)
                while tournament_set[random_index] == best_index_in_tournament:
                    random_index = random.randrange(0, 4)
                father_index = tournament_set[random_index]
            # find mother
            tournament_set = []
            best_rank_in_tournament = 1000000000000
            best_index_in_tournament = 1
            for j in range(size_of_tournament_set):
                random_index = random.randrange(0, size_of_current_pop)
                tournament_set.append(random_index)
                if ranking_of_pop[random_index] < best_rank_in_tournament:
                    best_rank_in_tournament = ranking_of_pop[random_index]
                    best_index_in_tournament = random_index
            if r < 0.8:
                mother_index = best_index_in_tournament
            else:
                random_index = random.randrange(0, 4)
                while tournament_set[random_index] == best_index_in_tournament:
                    random_index = random.randrange(0, 4)
                mother_index = tournament_set[random_index]
            # crossover
            offspring1 = []
            offspring2 = []
            route_of_father = copy.deepcopy(all_current_routes[father_index])
            size_of_father = len(route_of_father)
            route_of_mother = copy.deepcopy(all_current_routes[mother_index])
            size_of_mother = len(route_of_mother)
            remove_route_in_father = copy.deepcopy(route_of_father[random.randrange(size_of_father)])
            remove_route_in_mother = copy.deepcopy(route_of_mother[random.randrange(size_of_mother)])

            if route_of_father == route_of_mother:
                continue
            # remove point
            for remove_index in range(1, len(remove_route_in_mother)):
                remove_point = remove_route_in_mother[remove_index]
                for j in range(len(route_of_father)):
                    k = 1
                    while k != len(route_of_father[j]):
                        if remove_point == route_of_father[j][k]:
                            route_of_father[j].remove(remove_point)
                        else:
                            k += 1
            for remove_index in range(1, len(remove_route_in_father)):
                remove_point = remove_route_in_father[remove_index]
                for j in range(len(route_of_mother)):
                    k = 1
                    while k != len(route_of_mother[j]):
                        if remove_point == route_of_mother[j][k]:
                            route_of_mother[j].remove(remove_point)
                        else:
                            k += 1
            # caculate cost of route father and mother
            cost_of_father = 0
            for j in range(len(route_of_father)):
                tmp_size = len(route_of_father[j])
                for k in range(1, tmp_size):
                    this_point = route_of_father[j][k]
                    pre_point = route_of_father[j][k - 1]
                    cost_of_father += distance_mt[pre_point][this_point]
                cost_of_father += distance_mt[route_of_father[j][-1]][depot]
            cost_of_mother = 0
            for j in range(len(route_of_mother)):
                tmp_size = len(route_of_mother[j])
                for k in range(1, tmp_size):
                    this_point = route_of_mother[j][k]
                    pre_point = route_of_mother[j][k - 1]
                    cost_of_mother += distance_mt[pre_point][this_point]
                cost_of_mother += distance_mt[route_of_mother[j][-1]][depot]    
            # try to insert
            while len(remove_route_in_mother) != 1:
                random_index = random.randrange(1, len(remove_route_in_mother))
                remove_point = remove_route_in_mother[random_index]
                remove_route_in_mother.remove(remove_point)
                best_tmp_cost = 1000000000
                non_feasible_insert = True
                for j in range(len(route_of_father)):
                    for k in range(1, len(route_of_father[j]) + 1):
                        tmp_route = copy.deepcopy(route_of_father[j])
                        tmp_route.insert(k, remove_point)
                        feasible = True
                        tmp_demand = 0
                        tmp_time = 0
                        tmp_cost = 0
                        for t in range(1, len(tmp_route)):
                            this_point = tmp_route[t]
                            pre_point = tmp_route[t - 1]
                            tmp_arrive_time = tmp_time + time_mt[this_point][pre_point]
                            if tmp_demand + demand[this_point] > capacity or tmp_arrive_time > time_windows[this_point][1]:
                                feasible = False
                                break
                            else:
                                feasible = True
                                tmp_demand += demand[this_point]
                                tmp_time = max(time_windows[this_point][0], tmp_arrive_time) + service_time[this_point]
                        if feasible == True:
                            non_feasible_insert = False
                            if k == len(tmp_route) - 1 :
                                pre_of_remove_point = tmp_route[k - 1]
                                tmp_cost += distance_mt[remove_point][depot] + distance_mt[pre_of_remove_point][remove_point]
                            else:
                                pre_of_remove_point = tmp_route[k - 1]
                                succ_of_remove_point = tmp_route[k + 1]
                                tmp_cost += distance_mt[remove_point][succ_of_remove_point] + distance_mt[pre_of_remove_point][remove_point]
                            if tmp_cost < best_tmp_cost:
                                best_tmp_cost = tmp_cost
                                coor_insert = [j, k]
                if non_feasible_insert == False:
                    route_of_father[coor_insert[0]].insert(coor_insert[1], remove_point)
                    cost_of_father += best_tmp_cost
                else:
                    route_of_father.append([depot, remove_point])
                    cost_of_father += distance_mt[depot][remove_point] + distance_mt[remove_point][depot]
            #
            while len(remove_route_in_father) != 1:
                random_index = random.randrange(1, len(remove_route_in_father))
                remove_point = remove_route_in_father[random_index]
                remove_route_in_father.remove(remove_point)
                best_tmp_cost = 1000000000
                non_feasible_insert = True
                for j in range(len(route_of_mother)):
                    for k in range(1, len(route_of_mother[j]) + 1):
                        tmp_route = copy.deepcopy(route_of_mother[j])
                        tmp_route.insert(k, remove_point)
                        feasible = True
                        tmp_demand = 0
                        tmp_time = 0
                        tmp_cost = 0
                        for t in range(1, len(tmp_route)):
                            this_point = tmp_route[t]
                            pre_point = tmp_route[t - 1]
                            tmp_arrive_time = tmp_time + time_mt[this_point][pre_point]
                            if tmp_demand + demand[this_point] > capacity or tmp_arrive_time > time_windows[this_point][1]:
                                feasible = False
                                break
                            else:
                                feasible = True
                                tmp_demand += demand[this_point]
                                tmp_time = max(time_windows[this_point][0], tmp_arrive_time) + service_time[this_point]
                        if feasible == True:
                            non_feasible_insert = False
                            if k == len(tmp_route) - 1:
                                pre_of_remove_point = tmp_route[k - 1]
                                tmp_cost += distance_mt[remove_point][depot] + distance_mt[pre_of_remove_point][remove_point]
                            else:
                                pre_of_remove_point = tmp_route[k - 1]
                                succ_of_remove_point = tmp_route[k + 1]
                                tmp_cost += distance_mt[remove_point][succ_of_remove_point] + distance_mt[pre_of_remove_point][remove_point]
                            if tmp_cost < best_tmp_cost:
                                best_tmp_cost = tmp_cost
                                coor_insert = [j, k]
                if non_feasible_insert == False:
                    route_of_mother[coor_insert[0]].insert(coor_insert[1], remove_point)
                    cost_of_mother += best_tmp_cost
                else:
                    route_of_mother.append([depot, remove_point])
                    cost_of_mother += distance_mt[depot][remove_point] + distance_mt[remove_point][depot]
            #
            offspring1_routes = copy.deepcopy(route_of_father)
            offspring2_routes = copy.deepcopy(route_of_mother)
            for t in range(len(offspring1_routes)):
                for v in range(1, len(offspring1_routes[t])):
                    if offspring1_routes[t][v] != 0:
                        offspring1.append(offspring1_routes[t][v])
            for t in range(len(offspring2_routes)):
                for v in range(1, len(offspring2_routes[t])):
                    if offspring2_routes[t][v] != 0:
                        offspring2.append(offspring2_routes[t][v])
            current_population.append(offspring1)
            current_population.append(offspring2)

    # mutation
    for chro_index in range(population_size):
        r = random.random()
        if r < p_m:
            chro_route = all_current_routes[chro_index]
            mutation_route = copy.deepcopy(chro_route)
            random_route_index = random.randrange(len(mutation_route))
            random_route = mutation_route[random_route_index]
            if len(random_route) > 3:
                len_of_segment = random.randrange(2,4)
                random_index = random.randrange(1, len(random_route) - len_of_segment + 1)
                for i in range(int(len_of_segment/2)):
                    tmp = random_route[i + random_index]
                    random_route[i + random_index] = random_route[random_index + len_of_segment - 1 -i]
                    random_route[random_index + len_of_segment - 1 -i] = tmp
                feasible = True
                tmp_demand = 0
                tmp_time = 0
                tmp_cost = 0
                for t in range(1, len(random_route)):
                    this_point = random_route[t]
                    pre_point = random_route[t - 1]
                    tmp_arrive_time = tmp_time + time_mt[this_point][pre_point]
                    if tmp_demand + demand[this_point] > capacity or tmp_arrive_time > time_windows[this_point][1]:
                        feasible = False
                        break
                    else:
                        feasible = True
                        tmp_demand += demand[this_point]
                        tmp_time = max(time_windows[this_point][0], tmp_arrive_time) + service_time[this_point]
                if feasible == True:
                    mutation_route[random_route_index] = copy.deepcopy(random_route)
                mutation_offspring = []
                for t in range(len(mutation_route)):
                    for v in range(len(mutation_route[t])):
                        if mutation_route[t][v] != 0:
                            mutation_offspring.append(mutation_route[t][v] )
                current_population.append(mutation_offspring)
size_of_current_pop = len(current_population)
ranking_of_pop = np.zeros(size_of_current_pop, int)
ranked_list = np.zeros(size_of_current_pop, int)
pareto_vectors = []
non_dominated_vector = [100000000, 10000000]
dominated_list = np.ones(size_of_current_pop)
all_current_routes = []
for chro_index in range(size_of_current_pop):
    #Routing Scheme and evaluate rank
    choro_presentation = current_population[chro_index]
    routes = []
    time_of_routes = []
    current_route = []
    time_arrive_points = []
    current_route.append(depot)
    time_of_current_route = 0
    demand_of_current_route = 0
    distance_of_current_route = 0
    number_of_vehicles = 1
    pareto_vector = [0, 0]
    # phase 1 
    for gene_index in range(num_of_cus):
        gene = choro_presentation[gene_index]
        previous_gene = current_route[-1]
        if demand_of_current_route + demand[gene] <= capacity and time_of_current_route + time_mt[previous_gene][gene] <= time_windows[gene][1]:
            current_route.append(gene)
            demand_of_current_route += demand[gene]
            time_arrive = time_of_current_route + time_mt[previous_gene][gene]
            time_of_current_route = max(time_arrive, time_windows[gene][depot]) + service_time[gene]
            distance_of_current_route += distance_mt[previous_gene][gene]
        else:
            distance_of_current_route += distance_mt[previous_gene][depot] + distance_mt[depot][gene]
            number_of_vehicles += 1
            routes.append(current_route)
            time_arrive_points.append([0])
            current_route = [depot]
            current_route.append(gene)
            demand_of_current_route = demand[gene]
            time_arrive = time_mt[previous_gene][gene]
            time_of_current_route = max(time_arrive, time_windows[gene][0]) + service_time[gene]
    routes.append(current_route)
    distance_of_current_route += distance_mt[gene][depot]
    # phase 2 
    #
    pareto_vector[0] = number_of_vehicles
    pareto_vector[1] = distance_of_current_route
    # find 1 vector not dominated
    if pareto_vector[0] <= non_dominated_vector[0] and pareto_vector[1] <= non_dominated_vector[1]:
        if pareto_vector[0] < non_dominated_vector[0] or pareto_vector[1] < non_dominated_vector[1]:
            non_dominated_vector[0] = pareto_vector[0]
            non_dominated_vector[1] = pareto_vector[1]
    pareto_vectors.append(pareto_vector)
    all_current_routes.append(routes)
# ranking
cur_rank = 1
number_of_non_ranking_vectors = size_of_current_pop
while number_of_non_ranking_vectors != 0:

    # find non dominated vector
    non_dominated_vector = [100000000, 10000000]
    for i in range(size_of_current_pop):
        if ranked_list[i] == 0: 
            if pareto_vectors[i][0] <= non_dominated_vector[0] and pareto_vectors[i][1] <= non_dominated_vector[1]:
                if pareto_vectors[i][0] < non_dominated_vector[0] or pareto_vectors[i][1] < non_dominated_vector[1]:
                    non_dominated_vector[0] = pareto_vectors[i][0]
                    non_dominated_vector[1] = pareto_vectors[i][1]
    for i in range(size_of_current_pop):
        if ranked_list[i] == 0:
            if pareto_vectors[i][0] < non_dominated_vector[0] or pareto_vectors[i][1] < non_dominated_vector[1]:
                dominated_list[i] = 0
            elif pareto_vectors[i][0] == non_dominated_vector[0] and pareto_vectors[i][1] == non_dominated_vector[1]:
                dominated_list[i] = 0
    #
    for i in range(size_of_current_pop):
        if dominated_list[i] == 0 and ranked_list[i] == 0:
            ranking_of_pop[i] = cur_rank
            ranked_list[i] = 1
            number_of_non_ranking_vectors -= 1
    #
    cur_rank += 1
for i in range(len(ranking_of_pop)):

    if ranking_of_pop[i] == 1:
        for j in range(len(all_current_routes[i])):
            x = []
            y = []
            for k in range(len(all_current_routes[i][j])):
                found_point = all_current_routes[i][j][k]
                x.append(coor_matrix[found_point][0])
                y.append(coor_matrix[found_point][1])
            x.append(coor_matrix[0][0])
            y.append(coor_matrix[0][1])
            plt.plot(x,y)
        break
plt.show()

   


