import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
#import json
#import geojson
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

######## Road combining/splitting/reordering functions ########

def combine_roads_total_simple(gdf):
    gdf_new = gpd.GeoDataFrame(columns=['kszam', 'min_traffic', 'min_5_traffic', 'max_traffic', 'max_5_traffic', 'avg_traffic','geometry', 'data_df', 'data_json'])
    kszam_values = gdf['kszam'].unique()
    for road_name in kszam_values:
        instances = gdf[gdf['kszam'] == road_name]
        gdf_new.loc[len(gdf_new)] = {
            'kszam': road_name,
            'min_traffic': instances['anf'].min(),
            'min_5_traffic': np.sort(instances['anf'])[5] if len(instances) > 5 else None,
            'max_traffic': instances['anf'].max(),
            'max_5_traffic': np.sort(instances['anf'])[-5] if len(instances) > 5 else None,
            'avg_traffic': instances['anf'].mean(),
            'geometry': instances.unary_union,
            'data_df': instances,
            'data_json': instances.to_json()
        }
    return gdf_new

def instance_list_to_df(instance_list):
    df = pd.DataFrame(instance_list)
    return df

def instance_list_to_gdf(instance_list):
    gdf = gpd.GeoDataFrame(instance_list)
    return gdf

def route_road_chain_reordering(route_gdf): #maybe add a silent=True parameter
    #Basically, instances are not in order, but we can order them as they are connected. 
    #One could create a "chain graph", or just a simple "linked" list. I choose the latter, even though I adore the first more.
    #Another idea is sorting.

    #Assuming that no input route is a "cycle"
    chain_groups = [] #List of groups
    for t, road in route_gdf.iterrows():
        road_geom = (road['geometry'])
        touches = []
        for g in (range(len(chain_groups))): #We pass through all existing groups
            group = chain_groups[g]
            for i in range(len(group)): #Check if the road touches any of the roads in the group
                chain_road_geo = group[i]['geometry']
                if road_geom.touches(chain_road_geo):
                    touches.append((g, i))

        if len(touches) == 0:
            #Add new group
            chain_groups.append([road])
        elif len(touches) == 1:
            group = chain_groups[touches[0][0]]
            i = touches[0][1] #This should be either the beginning, or the end.
            if i == 0:
                group.insert(0, road)
            else: #Assuming the new road is connected to the end road, theoretically it cannot be connected to a road in the middle of the group
                if i != len(group)-1:
                    print(f"Instance {t}: Road touches a road in the middle of the group. This is not expected.")
                group.append(road)

        elif len(touches) == 2: #We assume two groups are both touched: we merge them
            index1 = touches[0][0]
            index2 = touches[1][0]
            group1 = chain_groups[index1]
            group2 = chain_groups[index2]
            i1 = touches[0][1]
            i2 = touches[1][1]
            if i1 == 0 and i2 == 0:
                #Reverse group1, put the element at the end of group1 and append group2 to group1
                group1.reverse()
                group1.append(road)
                group1.extend(group2)
                chain_groups.pop(index2) #Delete group2 (it's now connected with group1)
            elif i1 == 0 and i2 != 0: #Assuming i2 is the end
                group2.append(road)
                group2.extend(group1)
                chain_groups.pop(index1)
            elif i1 != 0 and i2 == 0: #Assuming i1 is the end
                group1.append(road)
                group1.extend(group2)
                chain_groups.pop(index2)
            elif i1 != 0 and i2 != 0: #Assuming both are the end
                group2.reverse()
                group1.append(road)
                group1.extend(group2)
                chain_groups.pop(index2)
        else:
            print("Road touches more than 2 groups. This is not expected.")
            print(touches), print(road)

    #Post-algorithm analysis
    if np.sum([len(group) for group in chain_groups]) != len(route_gdf):
        print("Not all roads are in the groups. Missing roads:")
        print(set(route_gdf['id']) - set([road['id'] for group in chain_groups for road in group]))

    return chain_groups

def reorder_full_gdf(gdf):
    #Warning: potential issue: when ran for the whole dataframe, previously, I got 3 non-obvious empty copying warnings (+1 obvious one, the first)
    #FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
    gdf_reordered = gpd.GeoDataFrame(columns=gdf.columns)
    for name in gdf['kszam'].unique():
        road_segments = gdf[(gdf['kszam'] == name) & (gdf['pkod']!='2')]
        connected_ordered_groups = route_road_chain_reordering(road_segments)
        for group in connected_ordered_groups:
            gdf_reordered = pd.concat([gdf_reordered, instance_list_to_gdf(group)], ignore_index=True)
    return gdf_reordered

def reorder_full_gdf_groupindexed(gdf):
    gdf_reordered = gpd.GeoDataFrame(columns=list(gdf.columns) + ['group'])
    for name in gdf['kszam'].unique():
        road_segments = gdf[(gdf['kszam'] == name) & (gdf['pkod']!='2')]
        connected_ordered_groups = route_road_chain_reordering(road_segments)
        for i in range(len(connected_ordered_groups)):
            group = connected_ordered_groups[i]
            gdf_group = instance_list_to_gdf(group)
            gdf_group['group'] = i
            gdf_reordered = pd.concat([gdf_reordered, gdf_group], ignore_index=True)
    return gdf_reordered

def create_ordered_roads_json(gdf):
    roads_ordered = {}
    for name in gdf['kszam'].unique():
        road_segments = gdf[(gdf['kszam'] == name) & (gdf['pkod']!='2')]
        roads_ordered[name] = []
        connected_ordered_groups = route_road_chain_reordering(road_segments)
        for component in connected_ordered_groups:
            component_ = []
            for segment in component:
                dict_segment = segment.to_dict()
                component_.append(dict_segment)
            roads_ordered[name].append(component_)
    return roads_ordered

def split_route_by_locations(route_instances, location_long_lat_pairs, split__range=0.2):
    #TODO
    pass