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

def v_P_odm0_shortest_paths(G, removed_nodes=None):
    #Vector of locations (if needed)
    locations = list(G.nodes())
    for node in removed_nodes:
        locations.remove(node)

    #Vector of location pairs
    location_pairs = list(combinations(locations, 2))
    #Create a "blueprint" for O-D matrix
    odm0 = np.full(len(location_pairs), 0.1)
    
    #Vector of road traffics
    v = np.array([G.get_edge_data(*edge)['weight'] for edge in G.edges()])
    
    #Shortest paths between all pairs of locations
    shortest_paths_dict = {}
    for i in range(len(locations)):
        source = locations[i]
        for j in range(i+1,len(locations)):
            target = locations[j]
            if source != target:
                paths = nx.all_shortest_paths(G, source=source, target=target)
                shortest_paths_dict[(source, target)] = list(paths)  # Convert generator to list

    #P matrix
    P = p_matrix_from_undirected_shortest_paths(G, shortest_paths_dict)
    
    return v, P, odm0, locations, location_pairs