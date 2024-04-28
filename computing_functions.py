import numpy as np
import networkx as nx
from itertools import combinations

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
    P = computing_functions.p_matrix_from_undirected_shortest_paths(G, shortest_paths_dict)

    #Post-compute removing hidden locations from location_pairs, P and odm_blueprint
    if hidden_locations is not None:
        for i in reversed(range(len(location_pairs))):  # Iterate in reverse order to avoid index errors
            if any(loc in location_pairs[i] for loc in hidden_locations):
                odm_blueprint = np.delete(odm_blueprint, i) #Delete the i-th row from the vector
                del location_pairs[i]
                P = np.delete(P, i, axis=1) #Delete the i-th column from the matrix
    
    return v, P, odm_blueprint, (road_names,locations, location_pairs, shortest_paths_dict) #v, P, odm, extra_info