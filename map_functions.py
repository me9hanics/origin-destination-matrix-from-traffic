import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

######## Network creation functions ########

def road_data_dict_from_OD(origin_destination_tuple_list, roads_ids, dataframe): #Note: nothing dealt about order, e.g. Szolnok_Budapest vs Budapest_Szolnok
    road_data = {}
    for origin,destination in origin_destination_tuple_list:
        road_data[(origin, destination)] = []
        lookup_name_start = str(origin) + "_" + str(destination) + "_"
        all_matching_roads = [key for key in roads_ids.keys() if key.startswith(lookup_name_start)]#Not sure if .startswith() is faster than e.g. regex
        if len(all_matching_roads) == 0:
            print("No road found for the given origin-destination pair: " + str(origin) + "-" + str(destination))
        for road in all_matching_roads:
            road_name = road.split(lookup_name_start)[-1]
            road_ids = roads_ids[road]
            min_traffic = np.min(dataframe[dataframe['id'].isin(road_ids)]['anf'])
            max_traffic = np.max(dataframe[dataframe['id'].isin(road_ids)]['anf'])
            avg_traffic = np.mean(dataframe[dataframe['id'].isin(road_ids)]['anf'])
            road_dict = {
                "road_name": road_name,
                "min_traffic": min_traffic,
                "max_traffic": max_traffic,
                "avg_traffic": avg_traffic,
                "road_ids": road_ids
            }
            road_data[(origin, destination)].append(road_dict)
    return road_data

######## Plotting functions ########
def plot_map_simple(G, pos, node_names=None):
    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', label='Cities')

    #Taken from https://stackoverflow.com/a/60638452/19626271
    ax = plt.gca()
    for u, v, key, data in G.edges(keys=True, data=True):
        weight =  int(data['weight']) #"{:.2f}".format(data['weight'])
        edge_text = f"{data['name']}: {weight}"
        start_pos = pos[u]; end_pos = pos[v]
        text_pos = ((start_pos[0] + end_pos[0]) / 2 + 0.1*key-0.05, (start_pos[1] + end_pos[1]) / 2+ 0.1*key-0.05)
        ax.annotate("",
                    xy=end_pos, xycoords='data',
                    xytext=start_pos, textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*key)
                                    ),
                                    ),
                    )
        ax.text(text_pos[0], text_pos[1], edge_text, fontsize=10, ha='center')

    if node_names: #Probably redundant, just in case
        nx.draw_networkx_labels(G, pos, labels=node_names)
    else:
        nx.draw_networkx_labels(G, pos)

    plt.axis('off')
    plt.show()