import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from itertools import combinations

def create_paths_dict(route_list):
    #Assuming list of lists
    extra_paths = {}
    for route in route_list:
        if len(route) >= 2:
                  #Origin, destination
            key = (route[0], route[-1])
            if key not in extra_paths:
                extra_paths[key] = []
            extra_paths[key].append(route)
    return extra_paths

def p_matrix_from_undirected_shortest_paths(G, shortest_paths_dict):
    #Roads and locations
    roads = [tuple(sorted(edge)) for edge in G.edges()] #Sorting for consistency
    #locations = list(G.nodes())

    P = np.zeros((len(roads), len(shortest_paths_dict))) #IxJ matrix: I roads, J shortest paths
    #Shortest paths for each location pair
    for j, ((source, target), paths) in enumerate(shortest_paths_dict.items()):
        #If the source and target are connected by an edge, set the corresponding entry in P to 1
        edge = tuple(sorted((source, target)))
        if edge in roads: #Not necessary, but typically faster
            i = roads.index(edge)
            P[i, j] = 1
        else:
            p_value = 1 / len(paths)
            for path in paths:
                path_edges = [tuple(sorted(edge)) for edge in zip(path[:-1], path[1:])] #Sorting for consistency
                for edge in path_edges:
                    i = roads.index(edge)
                    P[i, j] += p_value

    return P

def v_P_odmbp_shortest_paths(G, removed_nodes=None, hidden_locations=None, extra_paths_dict=None, round_P = False):
    """
    Given a graph + extra parameters, return v, P, a blueprint for the O-D matrix, and corresponding names/info.
    The computation of P is based on the number of shortest paths between two locations, plus included extra paths. 
    Removed nodes are not present during computation. Hidden locations are included in computing, but not in the output.
    The function returns: road traffic (v) of size I x 1, an origin-destination matrix (ODM) footprint of size J x 1,
    a matrix P of size I x J, all ordered in correspondance to each other and an extra info tuple containing road names,
    locations, location pairs, and the shortest paths dictionary.
    For further use, v = P * odm is assumed, from which one can estimate the ODM matrix. This is why order matters.

    Parameters:
    G (networkx.Graph): The graph on which shortest paths are to be calculated.
    removed_nodes (list, optional): Nodes to be removed from the graph. Defaults to None.
    hidden_locations (list, optional): Locations to be hidden in the final output. Defaults to None.
    extra_paths_dict (dict, optional): Extra paths to be included in the shortest paths. Defaults to None.

    Returns:
    v (numpy.ndarray): Vector of road traffics.
    P (numpy.ndarray): Matrix representing the shortest paths.
    odm_blueprint (numpy.ndarray): Blueprint vector for an origin-destination matrix.
    extra_info (dict): A dict containing road names, locations, location pairs, and the shortest paths dictionary.
    """


    #Vector of locations (if needed)
    locations = list(G.nodes())
    if removed_nodes:
        for node in removed_nodes:
            locations.remove(node)

    #Vector of location pairs
    location_pairs = list(combinations(locations, 2))
    #Create a "blueprint" for O-D matrix
    odm_blueprint = np.full(len(location_pairs), 1) #0.1
    
    #Vector of road traffics
    v = np.array([G.get_edge_data(*edge)['weight'] for edge in G.edges()])
    road_names = [edge for edge in G.edges()]

    #Shortest + extra paths between all pairs of locations
    shortest_paths_dict = {}
    for i in range(len(locations)):
        source = locations[i]
        for j in range(i+1,len(locations)):
            target = locations[j]
            if source != target:
                paths = nx.all_shortest_paths(G, source=source, target=target)
                shortest_paths_dict[(source, target)] = list(paths)  # Convert generator to list
    if (extra_paths_dict is not None): #A possible checking of the type might be needed, e.g. if list, try create_paths_dict()
        for key in extra_paths_dict:
            if key not in shortest_paths_dict:
                shortest_paths_dict[key] = []
            for key_extra_path in extra_paths_dict[key]:
                shortest_paths_dict[key].append(key_extra_path)

    #P matrix
    P = p_matrix_from_undirected_shortest_paths(G, shortest_paths_dict)
    if round_P:
        P = np.around(P, 5)

    #Post-compute removing hidden locations from location_pairs, P and odm_blueprint
    if hidden_locations is not None:
        for i in reversed(range(len(location_pairs))):  # Iterate in reverse order to avoid index errors
            if any(loc in location_pairs[i] for loc in hidden_locations):
                odm_blueprint = np.delete(odm_blueprint, i) #Delete the i-th row from the vector
                del location_pairs[i]
                P = np.delete(P, i, axis=1) #Delete the i-th column from the matrix
    
    extra_info = {
        "road_names": road_names,
        "locations": locations,
        "location_pairs": location_pairs,
        "shortest_paths_dict": shortest_paths_dict
    }
    return v, P, odm_blueprint, extra_info #v, P, odm, extra_info

def get_all_zero_rows(matrix):
    #Rows are selected based on the condition: not containing any non-zero elements
    return np.where(~matrix.any(axis=1)) [0]

def remove_full_zero_rows(P, v):
    non_zero_row_condition = ~np.all(P == 0, axis=1)
    P_reduced = P[non_zero_row_condition]; v_reduced = v[non_zero_row_condition]
    zero_rows = np.where(non_zero_row_condition == False)
    if zero_rows[0].size > 0:
        print(f"Removed full-zero rows at indexes: {zero_rows}")
    return P_reduced, v_reduced

def find_dependent_rows(P, verbose=True, return_independent=False):
    #Note: The case when after one group is found, the next groups will include it, is not fixed
    dependencies = []
    independent_rows = []
    #rank = np.linalg.matrix_rank(P)
    new_rank = 0
    for i in range(P.shape[0]):
        submatrix = P[:i+1]
        previous_rank = new_rank
        new_rank = np.linalg.matrix_rank(submatrix)
        if new_rank == previous_rank:
            #The new row is linearly dependent with some of the previous rows
            #We first assume all of them, then remove those which are independent
            dependent_rows = list(range(i+1))
            for j in range(i):
                #Check if deleting row j reduces the rank 
                reduced_submatrix = np.delete(submatrix, j, axis=0)
                if np.linalg.matrix_rank(reduced_submatrix) < new_rank:
                    #Row j is independent from all the previous rows
                    dependent_rows.remove(j)
            if dependent_rows:
                dependencies.append(dependent_rows)
            else:
                if verbose:
                    print("Found empty group")

    independent_rows = [i for i in range(P.shape[0]) if i not in [idx for group in dependencies for idx in group]]
    if verbose:
        print(f"Dependent rows: {dependencies}")        
        print(f"Independent rows count: {len(independent_rows)}, out of {P.shape[0]} rows")
        length_1_dependencies = [d for d in dependencies if len(d) == 1]
        print(f"Amount of 1-length dependent groups: {len(length_1_dependencies)}")
        if length_1_dependencies:
            print(f"1-length dependent groups: {length_1_dependencies}")
    if return_independent:
        return dependencies, independent_rows
    return dependencies #return [d for d in dependencies if len(d) > 1]

def hard_remove_dependent_rows(P_reduced):
    #Cases for when the rank is not maximal according to numpy
    #Remove rows that do not reduce the rank, until it is maximal (equal to the number of rows in this case)
    dependent_rows = find_dependent_rows(P_reduced)
    potential_rows = [i for group in dependent_rows for i in group]
    rank = np.linalg.matrix_rank(P_reduced)
    deleted_rows_indexes = []
    P_current = P_reduced.copy()

    while np.linalg.matrix_rank(P_current) < np.min(P_current.shape):
        for i in (potential_rows):
            if i in deleted_rows_indexes:
                #Row is already removed
                continue
            included_indexes = [j for j in range(P_reduced.shape[0]) if j not in deleted_rows_indexes]
            P_current = P_reduced[included_indexes, :]
            P_temp = P_reduced[[idx for idx in included_indexes if idx != i], :]

            if np.linalg.matrix_rank(P_temp) < rank:
                #Rank decreased, row i is independent: should not be removed
                continue
            else:
                #Rank didn't decrease: we can remove this row
                deleted_rows_indexes.append(i)
                P_current = P_temp

    #Matrix with dependent rows removed
    P_independent = np.delete(P_reduced, deleted_rows_indexes, axis=0)
    return P_independent, deleted_rows_indexes

def v_P_odmbp_reduced_matrix(G, f = v_P_odmbp_shortest_paths, **model_parameters):
    try:
        v, P, odm_blueprint, extra_info = f(G, **model_parameters)
    except TypeError:
        raise TypeError(f"TypeError: function {f} does not accept some of the given parameters: {model_parameters}.")
    
    import sympy

    P_reduced, v_reduced = remove_full_zero_rows(P, v)
    
    _, independent_rows_indexes = sympy.Matrix(P_reduced).T.rref() 
    independent_rows_indexes = list(independent_rows_indexes)

    P_reduced = P_reduced[independent_rows_indexes, :]
    v_reduced = v_reduced[independent_rows_indexes]
    extra_info["road_names"] = [extra_info["road_names"][i] for i in independent_rows_indexes]

    if np.linalg.matrix_rank(P_reduced) < np.min(P_reduced.shape):
        print("Sympy measured higher rank than numpy, extra steps are needed to take")
        #Try removing rows until the rank is maximal
        _, deleted_rows_indexes = hard_remove_dependent_rows(P_reduced)
        
        print(f"Deleted rows list (index): {deleted_rows_indexes}")
        P_reduced = np.delete(P_reduced, deleted_rows_indexes, axis=0)
        v_reduced = np.delete(v_reduced, deleted_rows_indexes)
        extra_info["road_names"] = [extra_info["road_names"][i] for i in range(len(extra_info["road_names"])) if i not in deleted_rows_indexes]

    return v_reduced, P_reduced, odm_blueprint, extra_info

def v_P_odmbp_reduced_matrix_complete(G, f = v_P_odmbp_shortest_paths, **model_parameters):
    """
    Old version of the function, before dividing into smaller functions.
    """
    try:
        v, P, odm_blueprint, extra_info = f(G, **model_parameters)
    except TypeError:
        raise TypeError(f"TypeError: function {f} does not accept some of the given parameters: {model_parameters}.")
    
    import sympy

    P_reduced, v_reduced = remove_full_zero_rows(P, v)
    
    #Reduced row echelon form: great approach for finding dependent and independent rows. In practice, might give different results (floats)
    _, independent_rows_indexes = sympy.Matrix(P_reduced).T.rref() #rref finds independent (pivot) columns, so we transpose 
    independent_rows_indexes = list(independent_rows_indexes)
    #removable_rows_indexes = set(range(P.shape[0])) - set(independent_rows_indexes)

    P_reduced = P_reduced[independent_rows_indexes, :]
    v_reduced = v_reduced[independent_rows_indexes]
    extra_info["road_names"] = [extra_info["road_names"][i] for i in independent_rows_indexes]

    #Sympy isn't always computing the same rank as numpy does.
    #See this: https://stackoverflow.com/a/53793829/19626271
    if np.linalg.matrix_rank(P_reduced) < np.min(P_reduced.shape):
        print("Sympy measured higher rank than numpy, extra steps are needed to take")
        #Try removing rows until the rank is maximal
        dependent_rows = find_dependent_rows(P_reduced)
        potential_rows = [i for group in dependent_rows for i in group]
        rank = np.linalg.matrix_rank(P_reduced)
        deleted_rows_indexes = []
        P_current = P_reduced.copy()

        while np.linalg.matrix_rank(P_current) < np.min(P_current.shape):
            for i in (potential_rows):
                if i in deleted_rows_indexes:
                    #Row is already removed
                    continue
                included_indexes = [j for j in range(P_reduced.shape[0]) if j not in deleted_rows_indexes]
                P_current = P_reduced[included_indexes, :]
                P_temp = P_reduced[[idx for idx in included_indexes if idx != i], :]

                if np.linalg.matrix_rank(P_temp) < rank:
                    #Rank decreased, row i is independent: should not be removed
                    continue
                else:
                    #Rank didn't decrease: we can remove this row
                    deleted_rows_indexes.append(i)
                    P_current = P_temp
        
        print(f"Deleted rows list (index): {deleted_rows_indexes}")
        P_reduced = np.delete(P_reduced, deleted_rows_indexes, axis=0)
        v_reduced = np.delete(v_reduced, deleted_rows_indexes)
        extra_info["road_names"] = [extra_info["road_names"][i] for i in range(len(extra_info["road_names"])) if i not in deleted_rows_indexes]

    return v_reduced, P_reduced, odm_blueprint, extra_info

def get_odm_2d_symmetric(odm, location_pairs):
    locations = sorted(set([loc for pair in location_pairs for loc in pair]))
    odm_2d = np.zeros((len(locations), len(locations)))

    #For any origin-destination pair A-B and B-A, we store the same value
    for pair, value in zip(location_pairs, odm):
        i, j = locations.index(pair[0]), locations.index(pair[1])
        odm_2d[i][j] = value
        odm_2d[j][i] = value #Symmetric

    return odm_2d, locations

def odm_ids_df_to_odm_2d(odm_df, id_place_dict, places_sorted=None):
    if places_sorted is None:
        places_sorted = [id_place_dict[int(id)] for id in odm_df['destination'].unique()]
    odm_2d = np.zeros((len(places_sorted), len(places_sorted)))

    for i, row in odm_df.iterrows():
        origin = row['origin']; destination = row['destination']
        odm_2d[places_sorted.index(id_place_dict[origin]), places_sorted.index(id_place_dict[destination])] = row['flow']
    return odm_2d

def odm_ids_df_to_odm_2d_symmetric(odm_df, id_place_dict, places_sorted=None):
    if places_sorted is None:
        places_sorted = [id_place_dict[int(id)] for id in odm_df['destination'].unique()]
    odm_2d = np.zeros((len(places_sorted), len(places_sorted)))

    for i, row in odm_df.iterrows():
        origin = row['origin']; destination = row['destination']
        odm_2d[places_sorted.index(id_place_dict[origin]), places_sorted.index(id_place_dict[destination])] = row['flow']
        odm_2d[places_sorted.index(id_place_dict[destination]), places_sorted.index(id_place_dict[origin])] = row['flow']
    return odm_2d

def odm_location_names_df_to_odm_2d(odm_df, places_sorted=None):
    if places_sorted is None:
        places_sorted = list(np.sort([name for name in odm_df['destination'].unique()]))
    odm_2d = np.zeros((len(places_sorted), len(places_sorted)))

    for i, row in odm_df.iterrows():
        origin = row['origin']; destination = row['destination']
        odm_2d[places_sorted.index(origin), places_sorted.index(destination)] = row['flow']
    return odm_2d

def odm_location_names_df_to_odm_2d_symmetric(odm_df, places_sorted=None):
    if places_sorted is None:
        places_sorted = list(np.sort([name for name in odm_df['destination'].unique()]))
    odm_2d = np.zeros((len(places_sorted), len(places_sorted)))

    for i, row in odm_df.iterrows():
        origin = row['origin']; destination = row['destination']
        odm_2d[places_sorted.index(origin), places_sorted.index(destination)] = row['flow']
        odm_2d[places_sorted.index(destination), places_sorted.index(origin)] = row['flow']
    return odm_2d

def plot_odm(odm_2d, locations, plot_type='heatmap', order = None, log_scale=False):
    if plot_type == 'heatmap':
        if order is not None:
            original_order = locations
            current_order = [original_order.index(i) for i in order]
            odm_2d = odm_2d[current_order][:, current_order]
            locations = order
        if not log_scale:
            sns.heatmap(odm_2d, xticklabels=locations, yticklabels=locations)
        else:
            sns.heatmap(odm_2d, xticklabels=locations, yticklabels=locations, norm=colors.LogNorm())
        plt.show()
    elif plot_type == 'scatterplot':
        if log_scale:
            print("Log scale is not supported for scatterplot")
        odm_long = odm_2d.reshape(-1)
        x = np.repeat(locations, len(locations))
        y = np.tile(locations, len(locations))
        if order is not None:
            odm_long = odm_long[order]
            x = x[order]
            y = y[order]
        sns.scatterplot(x=x, y=y, size=odm_long)
        plt.show()
    else:
        print(f"Unknown plot type: {plot_type}")#potential change to raise ValueError

def plot_odm_axis(odm_2d, locations, plot_type='heatmap', order=None, ax=None, log_scale=False, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    if plot_type == 'heatmap':
        if not log_scale:
            sns.heatmap(odm_2d, xticklabels=locations, yticklabels=locations, ax=ax)
        else:
            sns.heatmap(odm_2d, xticklabels=locations, yticklabels=locations, norm=colors.LogNorm(), ax=ax)
    elif plot_type == 'scatterplot':
        if log_scale:
            print("Log scale is not supported for scatterplot")
        odm_long = odm_2d.reshape(-1)
        x = np.repeat(locations, len(locations))
        y = np.tile(locations, len(locations))
        if order is not None:
            odm_long = odm_long[order]
            x = x[order]
            y = y[order]
        sns.scatterplot(x=x, y=y, size=odm_long, ax=ax)
    else:
        print(f"Unknown plot type: {plot_type}")  # potential change to raise ValueError
    if title is not None:
        ax.set_title(title)

def evaluate_model(predicted, real, model_name="model", verbose=True):
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(real.flatten(), predicted.flatten())
    corr, _ = pearsonr(real.flatten(), predicted.flatten())

    #Skip cases where real value is 0 (division by 0)
    mask = (real != 0)
    real_masked = real[mask]
    predicted_masked = predicted[mask]

    ssrd = np.sum(((real_masked - predicted_masked) / real_masked) ** 2)

    if verbose:
        print(f'MSE for {model_name}: {mse}')
        print(f'Correlation for {model_name}: {corr}')
        print(f'SoS of relative differences for {model_name}: {ssrd}\n')
    return mse, corr, ssrd