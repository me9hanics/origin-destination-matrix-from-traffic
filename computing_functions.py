import numpy as np
import networkx as nx
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

def v_P_odmbp_shortest_paths(G, removed_nodes=None, hidden_locations=None, extra_paths_dict=None):
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
    #P = np.around(P, 5)  # Round P to 5 decimal places

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

def remove_zero_rows(P, v):
    non_zero_row_condition = ~np.all(P == 0, axis=1)
    P_reduced = P[non_zero_row_condition]; v_reduced = v[non_zero_row_condition]
    zero_rows = np.where(non_zero_row_condition == False)
    if zero_rows[0].size > 0:
        print(f"Removed full-zero rows at indexes: {zero_rows}")
    return P_reduced, v_reduced

def v_P_odmbp_reduced_matrix(G, f = v_P_odmbp_shortest_paths, **model_parameters):
    try:
        v, P, odm_blueprint, extra_info = f(G, **model_parameters)
    except TypeError:
        raise TypeError(f"TypeError: function {f} does not accept some of the given parameters: {model_parameters}.")
    
    import sympy

    P_reduced, v_reduced = remove_zero_rows(P, v)
    
    #Reduced row echelon form: great for finding dependent and independent rows
    _, independent_rows_indexes = sympy.Matrix(P_reduced).T.rref() #rref finds independent (pivot) columns, so we transpose 
    independent_rows_indexes = list(independent_rows_indexes)
    #removable_rows_indexes = set(range(P.shape[0])) - set(independent_rows_indexes)

    P_reduced = P_reduced[independent_rows_indexes, :]
    v_reduced = v_reduced[independent_rows_indexes]
    extra_info["road_names"] = [extra_info["road_names"][i] for i in independent_rows_indexes]

    return v_reduced, P_reduced, odm_blueprint, extra_info
