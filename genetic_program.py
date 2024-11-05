import multiprocessing
from pprint import pprint
import pandas as pd
import math
import operator
import math
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from random import choice
from expirement import run_eval

tournament_size = 5
seed = 9
sample_size = 100
test_size = 0.3
crossover_pb = 0.5
mutation_pb = 0.5
generations = 25
pop_size = 150
mu = 100
lambda_ = 100

depth = 3  # Root to nth defined as a global area of the search space
end_depth = 8  # depth to end_depth as specific part of the local search

glob_search_iter_max = 5
local_search_iter_max = 3

# when to explore or exploit


best_individual = None

def random_prop(x):
    return x*random.uniform(0.1, 1.0)

def protected_div(left, right):
    """Division function"""
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def protected_sqrt(x):
    """Square root function"""
    try:
        return math.sqrt(x)
    except ValueError:
        return 1

def sigmoid(x):
    """Sigmoid function"""
    return 1/(1 + numpy.exp(-x))

def expfunc(x):
    return numpy.exp(x)

pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(protected_sqrt, 1)
pset.addPrimitive(sigmoid, 1)
pset.addPrimitive(random_prop, 1)
pset.addPrimitive(expfunc, 1)

# Rename terminal set arguments
pset.renameArguments(ARG0="Violation")
pset.renameArguments(ARG1="TotalViolation")

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


class Node:
    def __init__(self, element, children=None):
        self.element = element
        self.children = children if children else []

    def add_child(self, element):
        self.children.append(element)


def build_tree(tree):
    new_tree = []
    for node in tree:
        new_tree.append(node)

    return build_tree_rec(new_tree)


def build_tree_rec(tree):
    element = tree.pop(0)
    if isinstance(element, gp.Terminal):
        return Node(element)

    else:
        curr = Node(element)
        for i in range(element.arity):
            curr.add_child(build_tree_rec(tree))

        return curr


def height(node):
    if not node.children:
        # node is a leaf node, so its height is 0
        return 0
    else:
        # node has children, so its height is 1 plus the maximum height of its children
        return 1 + max(height(child) for child in node.children)


def to_list(tree_obj):
    list_tree_form = []
    to_list_rec(tree_obj, list_tree_form)
    return list_tree_form


def to_list_rec(tree_obj, list_tree_form):
    if tree_obj is not None:
        list_tree_form.append(tree_obj.element)
        if tree_obj.children:
            for child in tree_obj.children:
                to_list_rec(child, list_tree_form)


def get_nodes_at_depth(node, depth):
    if depth == 0:
        return [node.element]
    elif depth > 0:
        result = []
        for child in node.children:
            result += get_nodes_at_depth(child, depth - 1)
        return result
    else:
        return []


def get_cumaletive_nodes_at_depth(node, depth):
    res = []
    for i in range(depth + 1):
        res += get_nodes_at_depth(node, i)

    return res


def get_global_sim_index(tree1, tree2, depth):
    sub_node = get_cumaletive_nodes_at_depth(tree1, depth)
    sub_node2 = get_cumaletive_nodes_at_depth(tree2, depth)
    return get_sim_index(sub_node, sub_node2)


def get_sim_index(sub_tree1, sub_tree2):
    sim_index = 0
    for i in range(len(sub_tree1)):
        try:
            if sub_tree1[i] is sub_tree2[i]:
                sim_index += 1

        except IndexError as ie:
            break

    return sim_index


def get_local_sim_index(tree1, tree2, f_depth, t_depth):
    sub_node1 = get_cumaletive_nodes_at_depth(tree1, f_depth)
    sub_node_1 = get_cumaletive_nodes_at_depth(tree1, t_depth)
    local_nodes1 = sub_node_1[len(sub_node1) - 1:]

    sub_node2 = get_cumaletive_nodes_at_depth(tree2, f_depth)
    sub_node_2 = get_cumaletive_nodes_at_depth(tree2, t_depth)
    local_nodes2 = sub_node_2[len(sub_node2) - 1:]

    return get_sim_index(local_nodes1, local_nodes2)


def evaluate_fitness(individual, evaluate, hof=None, depth=0, end_depth=0, is_global=True):
    sim = 0
    func = toolbox.compile(expr=individual)  # type: ignore
    print(individual)
    total_violations = evaluate(func, 0) + evaluate(func, 1) + evaluate(func, 2)
    print("Total violation: ", total_violations)
    
    if hof is not None:
        sim = evaluate_similarity(individual, hof, depth, end_depth, is_global)

    return total_violations, sim

# generate the similarity index of an individual comparerd to the previouse hall of fame
def evaluate_similarity(individual, local_optima, depth, end_depth, is_global):
    cust_tree = build_tree(local_optima)
    cust_tree1 = build_tree(individual)
    if is_global:
        return get_global_sim_index(cust_tree, cust_tree1, depth)
    else:
        return get_local_sim_index(cust_tree, cust_tree1, depth, end_depth)


def mutate(indi, invalid_indecies, toolbox=toolbox, pset=pset):
    index = choice(list(set(range(0, len(indi))) - set(invalid_indecies)))
    individual = toolbox.clone(indi)
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = toolbox.expr(pset=pset, type_=type_)
    return individual


def create_local(indi, depth, pop_size):
    if depth > indi.height:
        depth = 0

    cust_tree = build_tree(indi)
    nodes_at_depth = get_cumaletive_nodes_at_depth(cust_tree, depth)
    invalid_indecies = [indi.index(ind) for ind in nodes_at_depth]
    new_pop = [mutate(indi, invalid_indecies) for i in range(pop_size)]
    return new_pop


toolbox.register("evaluate", evaluate_fitness, evaluate=run_eval)
toolbox.register("select", tools.selTournament, tournsize=tournament_size)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=1, max_=15)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)  # type: ignore

# cut of tree on height 20
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))


def evaluate_best_individual(best_individual=best_individual):
    """Evaluate the best individual on the test data"""
    rmse = best_individual.fitness  # type: ignore
    return rmse


def gen_metrics():
    """Generate metrics for the best individual"""
    results = evaluate_best_individual()

    metrics = {
        "total_violation": [results[0]],
    }
    # turn metrics into a dataframe
    metrics = pd.DataFrame(metrics)
    return metrics


def evolve_program(toolbox, pop, hof, mstatsm, results_train, results_test):
    """Runs the evolution algorithm """
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=mu, lambda_=lambda_, cxpb=crossover_pb, mutpb=mutation_pb,
                                         ngen=generations, stats=mstatsm, halloffame=hof, verbose=True)
    best_individual = hof[0]

    return best_individual


def final_metrics_report(last_ten):
    """Generate the final metrics report"""
    df_train = pd.DataFrame(columns=["Total Violations"])
    
    for indi in last_ten:
        df_train.loc[len(df_train)] = list(evaluate_best_individual(indi))

    print(df_train)

    return df_train


if __name__ == "__main__":
    # pool = multiprocessing.Pool(4)
    # toolbox.register("map", pool.map)

    pop = toolbox.population(n=pop_size)  # type: ignore
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])  # type: ignore
    stats_height = tools.Statistics(lambda ind: ind.height)  # type: ignore
    mstats = tools.MultiStatistics(fitness=stats_fit, height=stats_height)
    mstats.register("avg", numpy.mean)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    results_train = pd.DataFrame()
    results_test = pd.DataFrame()

    last_ten = []
    best_global = None
    for i in range(10):
        best_individual = []
        for i in range(glob_search_iter_max):  # number of iterations of the global search
            if i == 0:
                try:
                    del creator.FitnessMax
                    del creator.FitnessMin
                except Exception as e:
                    pass

                del creator.Individual
                toolbox.unregister("individual")
                toolbox.unregister("population")

                # change fitness to include minimizing simalrity
                creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
                creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
                toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
                toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", evaluate_fitness, evaluate=run_eval, hof=best_global,
                             depth=depth, end_depth=end_depth, is_global=True)

            best_global = evolve_program(toolbox, pop, hof, mstats, results_train, results_test)

            print()
            print("_______________________________________________________________________________________")
            print(f'Fitness of the current best individual before exploiting: {best_global.fitness}')
            print("_______________________________________________________________________________________")
            print()

            pop = create_local(best_global, depth, pop_size)
            local = []
            best_local = best_global
            # local search
            for j in range(local_search_iter_max):  # number of iterations of the local search
                if j == 0:
                    # change fitness to include maximizing simalrity
                    try:
                        del creator.FitnessMax
                        del creator.FitnessMin
                    except Exception as e:
                        pass

                    del creator.Individual
                    toolbox.unregister("individual")
                    toolbox.unregister("population")
                    creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1.0))
                    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
                    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
                    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

                toolbox.register("evaluate", evaluate_fitness, evaluate=run_eval, hof=best_local,
                                 depth=depth, end_depth=end_depth, is_global=False)
                best_local = evolve_program(toolbox, pop, hof, mstats, results_train, results_test)

                print()
                print("_______________________________________________________________________________________")
                print(f'Fitness of the current best individual after exploiting: {best_local.fitness}')
                print("_______________________________________________________________________________________")
                print()
                local.append(best_local)

            # get the best local individual
            best_local = min(local, key=lambda x: x.fitness)
            best_individual.append(best_local)

        best_globally = min(best_individual, key=lambda x: x.fitness)
        last_ten.append(best_globally)

    # get the best individual from the last ten
    best_individual_last_ten = min(last_ten, key=lambda x: x.fitness)

    results_train = final_metrics_report(last_ten)

    # get the metrics for the last ten best individuals

    # print last_ten results
    # print("_______________________________________________________________________________________")
    # print("Last ten results")
    # print(last_ten)
    # print("_______________________________________________________________________________________")

    print(f'Average Total Violations: {results_train["Total Violation"].mean()}')


    # print the best Train and Test results
    print("_______________________________________________________________________________________")
    print("Best Results")
    print(results_train.loc[results_train["Total Violation"].idxmin()])


    print(best_individual_last_ten)
    # pool.close()
