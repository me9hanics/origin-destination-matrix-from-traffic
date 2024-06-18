import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from itertools import combinations
import warnings
from itertools import islice

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

def remove_nodes_from_graph(G, nodes_to_remove, verbose=False):
    G_ = G.copy()
    if nodes_to_remove:
        if verbose:
            print(f"Removing {len(nodes_to_remove)} nodes from the graph.")
        for node in nodes_to_remove:
            G_.remove_node(node)
    return G_

def get_ODM_locations(G, hidden_locations, verbose=False):
    if verbose:
        print(f"Only focusing on non-hidden locations, removing hidden locations from computations (not graph)")
    locations = list(set(G.nodes()) - set(hidden_locations))
    if verbose:
        print(f"Locations after removal: {len(locations)}")
    return locations

def create_odm_blueprint_and_road_info(G, locations, weight):
    location_pairs = list(combinations(locations, 2))
    odm_blueprint = np.full(len(location_pairs), 1)
    v = np.array([G.get_edge_data(*edge)[weight] for edge in G.edges()])
    road_names = [edge for edge in G.edges()]
    return odm_blueprint, v, road_names, location_pairs

def add_extra_paths_to_dict(shortest_paths_dict, extra_paths_dict, verbose=False):
    if (extra_paths_dict is not None): 
        if verbose:
            print(f"Adding {len(extra_paths_dict)} extra paths to the shortest paths matrix.")
        for key in extra_paths_dict:
            if key not in shortest_paths_dict:
                shortest_paths_dict[key] = []
            for key_extra_path in extra_paths_dict[key]:
                shortest_paths_dict[key].append(key_extra_path)
    return shortest_paths_dict

def p_matrix_from_undirected_shortest_paths(G, shortest_paths_dict):
    """
    Given a graph and a dictionary of shortest paths, return a matrix P of size I x J,
    where I is the number of roads and J is the number of shortest paths.
    The matrix is constructed based on the number of shortest paths between two locations.
    For every two nodes, check all shortest paths. For every path, add 1/n to each road it uses.
    If the path uses a road multiple times, the value is added multiple times.
    """
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

def v_P_odmbp_shortest_paths(G, removed_nodes=None, hidden_locations=None, extra_paths_dict=None, round_P = False, verbose=False):
    """
    Compressive (memory efficient) version of the v_P_odmbp_shortest_paths_depreciated function, with the same functionality.

    Given a graph + extra parameters, return v, P, a blueprint for the O-D matrix, and corresponding names/info.
    The computation of P is based on the number of shortest paths between two locations, plus included extra paths. 
    Removed nodes are not present during computation. Hidden locations are included in computing, but not in the output.
    The function returns: road traffic (v) of size I x 1, an origin-destination matrix (ODM) blueprint of size J x 1,
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

    if verbose:
        print("Starting P-matrix computation.")
    G_ = G.copy()
    #Vector of locations (if needed)
    if removed_nodes:
        G_ = remove_nodes_from_graph(G_, removed_nodes, verbose)
    #Only include non-hidden locations in the O-D matrix
    locations = get_ODM_locations(G_, hidden_locations, verbose)

    #Create a "blueprint" for O-D matrix, vector of road traffics, and road names
    location_pairs = list(combinations(locations, 2))
    odm_blueprint = np.full(len(location_pairs), 1)
    #Vector of road traffics
    v = np.array([G_.get_edge_data(*edge)['weight'] for edge in G_.edges()])
    road_names = [edge for edge in G_.edges()]

    #Shortest + extra paths between all pairs of locations
    if verbose:
        process = 0
        max_process = len(locations) * (len(locations) - 1) / 2
        limits = [int(max_process * i / 10) for i in range(1, 11)]
        if verbose:
            print(f"Starting computation of shortest paths between {len(locations)} locations.")
    shortest_paths_dict = {}
    for i in range(len(locations)):
        source = locations[i]
        for j in range(i+1,len(locations)):
            target = locations[j]
            if source != target: #Should theoretically not happen
                paths = nx.all_shortest_paths(G_, source=source, target=target)
                shortest_paths_dict[(source, target)] = list(paths)  #Convert generator to list
            if verbose:
                process += 1
                if process in limits:
                    if verbose:
                        print(f"{int(process*100/max_process)}% completed.")
    if verbose:
        print("Shortest paths computation completed.")
    if (extra_paths_dict is not None): #A possible checking of the type might be needed
        shortest_paths_dict = add_extra_paths_to_dict(shortest_paths_dict, extra_paths_dict, verbose)

    #P matrix
    if verbose:
        print("Constructing the P matrix from the found shortest paths.")
    P = p_matrix_from_undirected_shortest_paths(G_, shortest_paths_dict)
    if round_P:
        P = np.around(P, 5)

    extra_info = {
        "road_names": road_names,
        "locations": locations,
        "location_pairs": location_pairs,
        "shortest_paths_dict": shortest_paths_dict
    }
    return v, P, odm_blueprint, extra_info #v, P, odm, extra_info

def v_P_odmbp_shortest_times(G, max_selected_paths=3, time_threshold=None, removed_nodes=None, hidden_locations=None, extra_paths_dict=None, round_P = False, verbose=False):
    """
    Given a graph + parameters, compute the shortest time paths between locations, returning v, P, and other info.

    The computation of P is based on the shortest time paths between locations. The shortest time path is always included.
    If the time threshold is set, only those paths are included that are shorter than the shortest time * (1+threshold).
    If it is not set, the max_selected_paths are considered.
    Removed nodes are not present during computation. Hidden locations are included in computing, but not in the output.
    The function returns: road traffic (v) of size I x 1, an origin-destination matrix (ODM) blueprint of size J x 1,
    a matrix P of size I x J, all ordered in correspondance to each other and an extra info tuple containing road names,
    locations, location pairs, and the shortest paths dictionary.
    For further use, v = P * odm is assumed, from which one can estimate the ODM matrix. This is why order matters.

    Parameters:
    G (networkx.Graph): The graph on which shortest paths are to be calculated.
    max_selected_paths (int, optional): The maximum number of selected shortest paths between two locations. Defaults to 3.
    time_threshold (float, optional): The threshold for the time diff between the shortest path and other selected paths. Defaults to None.
    removed_nodes (list, optional): Nodes to be removed from the graph before computation. Defaults to None.
    hidden_locations (list, optional): Locations to be hidden in the final output. Defaults to None.
    extra_paths_dict (dict, optional): Extra paths to be included in the shortest paths. Defaults to None.

    Returns:
    v (numpy.ndarray): Vector of road traffics.
    P (numpy.ndarray): Matrix representing the shortest paths.
    odm_blueprint (numpy.ndarray): Blueprint vector for an origin-destination matrix.
    extra_info (dict): A dict containing road names, locations, location pairs, and the shortest paths (by time) dictionary.
    """

    if time_threshold is not None:
        warnings.warn("The time_threshold parameter is not yet implemented, using the max_selected_paths instead.")
        #TODO

    if verbose:
        print("Starting P-matrix creation - using time as a shortest path indicator.")
    G_ = G.copy()
    #Vector of locations (if needed)
    if removed_nodes:
        G_ = remove_nodes_from_graph(G_, removed_nodes, verbose)
    #Only include non-hidden locations in the O-D matrix
    locations = get_ODM_locations(G_, hidden_locations, verbose)

    #Create a "blueprint" for O-D matrix
    location_pairs = list(combinations(locations, 2))
    odm_blueprint = np.full(len(location_pairs), 1)
    
    #Vector of road traffics
    v = np.array([G_.get_edge_data(*edge)['time'] for edge in G_.edges()])
    road_names = [edge for edge in G_.edges()]

    #Shortest + extra paths between all pairs of locations
    if verbose:
        process = 0
        max_process = len(locations) * (len(locations) - 1) / 2
        limits = [int(max_process * i / 10) for i in range(1, 11)]
        if verbose:
            print(f"Starting computation of shortest paths between {len(locations)} locations.")
    #num_selected_paths = max_selected_paths
    shortest_paths_dict = {}
    for i in range(len(locations)):
        source = locations[i]
        for j in range(i+1,len(locations)):
            target = locations[j]
            #TODO: Account for the time_threshold
            if source != target: #Should theoretically not happen
                paths = list(islice(nx.shortest_simple_paths(G_, source=source, target=target, weight='time'), max_selected_paths))
                shortest_paths_dict[(source, target)] = paths
            if verbose:
                process += 1
                if process in limits:
                    if verbose:
                        print(f"{int(process*100/max_process)}% completed.")
    if verbose:
        print("Shortest paths computation completed.")
    if (extra_paths_dict is not None): #A possible checking of the type might be needed
        shortest_paths_dict = add_extra_paths_to_dict(shortest_paths_dict, extra_paths_dict, verbose)
    #P matrix
    if verbose:
        print("Constructing the P matrix from the found shortest paths.")
    P = p_matrix_from_undirected_shortest_paths(G_, shortest_paths_dict)
    if round_P:
        P = np.around(P, 5)

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

def find_dependent_rows_simplified(P, verbose=True, return_independent=False):
    """Find those rows which are linearly dependent/independent from others - simplified version for speed."""
    dependent_rows = []
    independent_rows = []
    if P.shape[0]>P.shape[1]:
        if verbose:
            print("More rows than columns - automatically flagging last M-N rows as dependent")
        dependent_rows += list(range(P.shape[1], P.shape[0]))
        P = P[:P.shape[1]] #Keep only the first N rows

    rank = np.linalg.matrix_rank(P)
    previous_rank = 0
    for i in range(P.shape[0]):
        #Add the i-th row to the submatrix
        #Check if the rank doesn't change by removing row i
        submatrix = P[:i+1]
        new_rank = np.linalg.matrix_rank(submatrix)
        if new_rank == previous_rank:
            #The new row is linearly dependent with some of the previous rows
            dependent_rows.append(i)
        else:
            independent_rows.append(i)
        previous_rank = new_rank
        
    independent_rows = [i for i in range(P.shape[0]) if i not in dependent_rows]
    if verbose:
        print(f"Dependent rows: {dependent_rows}")  
        print(f"Independent rows count: {len(independent_rows)}, out of {P.shape[0]} rows")
    if return_independent:
        return dependent_rows, independent_rows
    return dependent_rows #return [d for d in dependencies if len(d) > 1]

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

def sort_odm_loc_names_df(odm_df, ordered_o_d_tuple_list, return_ordering=False):
    #Column with the index of each (origin, destination) pair in the ordered list
    #origins = np.array(odm_df['origin'])
    #destinations = np.array(odm_df['destination'])
    #sort_keys = np.array([None]*len(origins))
    odm_df['sort_key'] = None
    for i, (o, d) in enumerate(ordered_o_d_tuple_list):
        odm_df.loc[(odm_df['origin'] == o) & (odm_df['destination'] == d), 'sort_key'] = i
    #odm_df['sort_key'] = [ordered_o_d_tuple_list.index((o, d)) for o, d in zip(odm_df['origin'], odm_df['destination'])]
    if odm_df['sort_key'].isnull().sum() > 0:
        print("Missing values in the sort_key column (ordering of original dataframe).\n\
              ASSUMING symmetric ODM - filling the missing values with the reverse (destination, origin).\n\
              All other missing values will be removed.")
        for i, (o, d) in enumerate(ordered_o_d_tuple_list):
            odm_df.loc[(odm_df['origin'] == d) & (odm_df['destination'] == o) & (odm_df['sort_key'].isnull()), 'sort_key'] = i
        odm_df = odm_df[odm_df['sort_key'].notnull()]
    
    ordering = np.array(odm_df['sort_key'].values).astype(int)
    sorted_df = odm_df.sort_values(by='sort_key')
    
    sorted_df = sorted_df.drop(columns='sort_key')
    
    if return_ordering:
        return sorted_df, ordering
    else:
        return sorted_df


###################### Plotting, evaluation and other functions ######################

#Moved to plotting_functions.py (helper_functions folder)