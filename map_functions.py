import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString
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

def process_roads(direct_routes, roads_ids, gdf):
    import json
    from copy import deepcopy

    roads_dict = road_data_dict_from_OD(direct_routes, roads_ids, gdf)
    roads_dict_string = {str(key): value for key, value in roads_dict.items()}
    roads_simple_dict_string = deepcopy(roads_dict_string)
    for key, value in roads_simple_dict_string.items():
        for road in value:
            del road['road_ids']
    
    return roads_dict, roads_dict_string, roads_simple_dict_string

def combined_paralel_roads_dict(roads_dict_new):
    roads_dict_new_simple = {}
    for (origin, destination), roads in roads_dict_new.items():
        if (origin, destination) in roads_dict_new_simple:
            roads_dict_new_simple[(origin, destination)][0]['min_traffic'] += sum(road['min_traffic'] for road in roads)
            roads_dict_new_simple[(origin, destination)][0]['max_traffic'] += sum(road['max_traffic'] for road in roads)
                                                                            #Not weighted sum, fine for now
            roads_dict_new_simple[(origin, destination)][0]['avg_traffic'] += sum(road['avg_traffic'] for road in roads)
            roads_dict_new_simple[(origin, destination)][0]['road_name'] += "&" + "&".join(road['road_name'] for road in roads)
        else:
            roads_dict_new_simple[(origin, destination)] = [{
                'road_name': "&".join(road['road_name'] for road in roads),
                'min_traffic': sum(road['min_traffic'] for road in roads),
                'max_traffic': sum(road['max_traffic'] for road in roads),
                'avg_traffic': sum(road['avg_traffic'] for road in roads)
            }]
    return roads_dict_new_simple

def combine_edges(G):
    simple_G = nx.Graph()
    for u, v, data in G.edges(data=True):
        if simple_G.has_edge(u, v):
            simple_G[u][v]['weight'] += data['weight']
            simple_G[u][v]['name'] += "&" + data['name']
        else:
            simple_G.add_edge(u, v, name=data['name'], weight=data['weight'])
    return simple_G

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

def LUT_geopositions(location_list, geoposition_dict):
    #Subset of location geopositions
    return {location: geoposition_dict[location] for location in location_list}

def create_location_gdf_with_crs(location_lat_long_pairs, crs):
     #Turn the location dict into a GeoDataFrame: this way it is easy to transform the CRS
    df = pd.DataFrame({
        'Location': list(location_lat_long_pairs.keys()),
        'Latitude': [coords[0] for coords in location_lat_long_pairs.values()],
        'Longitude': [coords[1] for coords in location_lat_long_pairs.values()]
    })
    gdf_loc = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
    #Set the CRS to WGS84 (epsg:4326), WorldGeodeticSystem84 is the most common coordinate reference system for lat/lon data
    gdf_loc.crs = "EPSG:4326" 
    #Conversion: from WGS84 to the CRS of gdf2
    gdf_loc = gdf_loc.to_crs(crs) #Now we have the location points in the same CRS as the original gdf
    return gdf_loc

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

def reorder_full_gdf(gdf, exclude_direction = True):
    #Warning: potential issue: when ran for the whole dataframe, previously, I got 3 non-obvious empty copying warnings (+1 obvious one, the first)
    #FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
    gdf_reordered = gpd.GeoDataFrame(columns=gdf.columns)
    gdf_reordered.set_crs(gdf.crs, inplace=True) #Set CRS
    for name in gdf['kszam'].unique():
        road_segments = gdf[(gdf['kszam'] == name)]
        if exclude_direction:
            road_segments = road_segments[road_segments['pkod']!='2']
        connected_ordered_groups = route_road_chain_reordering(road_segments)
        for group in connected_ordered_groups:
            gdf_group = instance_list_to_gdf(group)
            gdf_group.set_crs(gdf.crs, inplace=True) #Set CRS
            gdf_reordered = pd.concat([gdf_reordered, gdf_group], ignore_index=True)
    return gdf_reordered

def reorder_full_gdf_groupindexed(gdf, exclude_direction = True):
    gdf_reordered = gpd.GeoDataFrame(columns=list(gdf.columns) + ['group'])
    gdf_reordered.set_crs(gdf.crs, inplace=True) #Set CRS
    for name in gdf['kszam'].unique():
        road_segments = gdf[(gdf['kszam'] == name)]
        if exclude_direction:
            road_segments = road_segments[road_segments['pkod']!='2']
        connected_ordered_groups = route_road_chain_reordering(road_segments)
        for i in range(len(connected_ordered_groups)):
            group = connected_ordered_groups[i]
            gdf_group = instance_list_to_gdf(group)
            gdf_group['group'] = i
            gdf_group.set_crs(gdf.crs, inplace=True) #Set CRS
            gdf_reordered = pd.concat([gdf_reordered, gdf_group], ignore_index=True)
    return gdf_reordered

def create_ordered_gdf_list_from_road_names(gdf, road_name_list, exclude_direction = True):
    gdf_list = []
    for road_name in road_name_list:
        road_segments = gdf[(gdf['kszam'] == road_name)]
        if exclude_direction:
            road_segments = road_segments[road_segments['pkod']!='2']
        connected_ordered_groups = route_road_chain_reordering(road_segments)
        gdf_road = gpd.GeoDataFrame(columns=list(gdf.columns)+['group'])
        for i in range(len(connected_ordered_groups)):
            group = connected_ordered_groups[i]
            gdf_group = instance_list_to_gdf(group)
            gdf_group['group'] = i
            gdf_group.set_crs(gdf.crs, inplace=True) #Set CRS
            gdf_road = pd.concat([gdf_road, gdf_group], ignore_index=True)
        gdf_list.append(gdf_road)
    return gdf_list

def create_ordered_roads_json(gdf, exclude_direction = True):
    roads_ordered = {}
    for name in gdf['kszam'].unique():
        road_segments = gdf[(gdf['kszam'] == name)]
        if exclude_direction:
            road_segments = road_segments[road_segments['pkod']!='2']
        roads_ordered[name] = []
        connected_ordered_groups = route_road_chain_reordering(road_segments)
        for component in connected_ordered_groups:
            component_ = []
            for segment in component:
                dict_segment = segment.to_dict()
                component_.append(dict_segment)
            roads_ordered[name].append(component_)
    return roads_ordered

def find_intersections_cross(gdfs):
    #Find the intersection points
    intersection_points = []
    intersection_segments = []
    for i in range(len(gdfs)):
        for j in range(i+1, len(gdfs)):
            for k in range(len(gdfs[i])):
                for l in range(len(gdfs[j])):
                    if gdfs[i].geometry.iloc[k].intersects(gdfs[j].geometry.iloc[l]):
                        intersection_point = gdfs[i].geometry.iloc[k].intersection(gdfs[j].geometry.iloc[l])
                        if "Point" == intersection_point.geom_type:
                            intersection_points.append(intersection_point)
                            intersection_segments.append(((i, k, gdfs[i]['kszam'].iloc[k]), (j, l, gdfs[j]['kszam'].iloc[l])))
    
    return intersection_points, intersection_segments

def get_separate_groups(gdf):
    if 'group' not in gdf.columns:
        print('No "group" column')
        return None
    group_ids = gdf['group'].unique()
    groups = []
    for group_id in group_ids:
        groups.append(gdf[gdf['group'] == group_id])
        #If given an ordered geodataframe, it should keep order
    return groups

def split_route_by_locations(route_instances, location_lat_long_pairs, split__range=0.2):
    #TODO
    pass

######## Plotting functions ########

def plot_roads_and_locations(gdf, geoposition_dict, road_name_list, location_list, with_text = True, fontsize=15):
    if type(gdf)==list:
        gdf = pd.concat(gdf)
    roads_gdf = gdf[gdf['kszam'].isin(road_name_list)]
    ax=roads_gdf.plot(color='black', linewidth=1)
    
    if with_text: #Route's name
        for _, route_gdf in roads_gdf.groupby('kszam'):
            plt.text(route_gdf['geometry'].iloc[0].centroid.x, route_gdf['geometry'].iloc[0].centroid.y, route_gdf['kszam'].iloc[0], fontsize=fontsize, color='red')

    loc_gdf = create_location_gdf_with_crs(LUT_geopositions(location_list,geoposition_dict), gdf.crs)
    loc_gdf.plot(ax=ax, color='red', markersize=20)
    
    if with_text: #Location's name
        for _, location in loc_gdf.iterrows():
            plt.text(location['geometry'].x, location['geometry'].y, location['Location'], fontsize=fontsize, color='blue')
    plt.axis('off')

def plot_map_multigraph(G, pos, node_names=None):
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

def plot_map_simple_graph(G, pos, node_names=None):
    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', label='Cities')

    ax = plt.gca()
    for u, v, data in G.edges(data=True):
        weight =  int(data['weight']) #"{:.2f}".format(data['weight'])
        edge_text = f"{data['name']}: {weight}"
        start_pos = pos[u]; end_pos = pos[v]
        text_pos = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
        ax.annotate("",
                    xy=end_pos, xycoords='data',
                    xytext=start_pos, textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=0",
                                    ),
                    )
        ax.text(text_pos[0], text_pos[1], edge_text, fontsize=10, ha='center')

    if node_names: #Probably redundant, just in case
        nx.draw_networkx_labels(G, pos, labels=node_names)
    else:
        nx.draw_networkx_labels(G, pos)

    plt.axis('off')
    plt.show()

def plot_roads_and_locations_intersections(gdfs, geoposition_dict, road_name_list, location_list, with_text = True, fontsize=15, randomize = False, randomize_factor = 0.1):
    if type(gdfs)==list:
        gdf = pd.concat(gdfs)
    else:
        gdf = gdfs
        gdfs = [gdfs]
    roads_gdf = gdf[gdf['kszam'].isin(road_name_list)]
    ax=roads_gdf.plot(color='black', linewidth=1)
    
    if with_text: #Route's name
        for _, route_gdf in roads_gdf.groupby('kszam'):
            plt.text(route_gdf['geometry'].iloc[0].centroid.x, route_gdf['geometry'].iloc[0].centroid.y, route_gdf['kszam'].iloc[0], fontsize=fontsize, color='red')

    loc_gdf = create_location_gdf_with_crs(LUT_geopositions(location_list,geoposition_dict), gdf.crs)
    loc_gdf.plot(ax=ax, color='red', markersize=20)
    
    if with_text: #Location's name
        for _, location in loc_gdf.iterrows():
            plt.text(location['geometry'].x, location['geometry'].y, location['Location'], fontsize=fontsize, color='blue')
    
    #Intersection points across gdframes
    intersection_points, intersection_segments = find_intersections_cross(gdfs)
    for point, segment in zip(intersection_points, intersection_segments):
        plt.plot(*point.xy, 'go')
        if randomize:
            d = (np.random.rand(2)-np.array([0.5,0.5]))*randomize_factor
        else:
            d = [0,0]
        plt.text(point.x*(1+d[0]), point.y*(1+d[1]), f'A: {segment[0]}, B: {segment[1]}', fontsize=fontsize, color='green')

    plt.axis('off')

def plot_road_traffic_with_given_locations(road_gdf,location_lat_long_pairs, location_radius=0.1, title = None):
    #Important assumption: the road must be ordered (chain-like, see route_road_chain_reordering)
    #Also, assuming the location_lat_long_pairs are given in WGS84 format
    #Basically, we take the ordered road, check which part of the road is closest to the given location, 
    #and plot the traffic data, on it the location closest to the road, as axvspans

    #0th step: Check if there are separate groups in the road_gdf
    if 'group' in road_gdf.columns: #or .keys()
        if len(road_gdf['group'].unique()) > 1:
            print("The input has multiple groups, which might require more analysis")

    #Turn the location dict into a GeoDataFrame: this way it is easy to transform the CRS
    gdf_loc = create_location_gdf_with_crs(location_lat_long_pairs, road_gdf.crs)
    #Turn back to dict: faster to access
    location_point_pairs = dict(zip(gdf_loc['Location'], gdf_loc['geometry']))
    #Initialize the closest segment and distance for each location
    location_min_distance_segment = {key: {'segment_order':0, 'segment_index': road_gdf.iloc[0].name, 'distance': road_gdf['geometry'].iloc[0].distance(value)} for key, value in location_point_pairs.items()}

    #Iterate through the road and find the closest segment to each location
    for i, (index, row) in enumerate(road_gdf.iterrows()):
        road_geom = row['geometry']
        for loc, loc_point in location_point_pairs.items():
            distance = road_geom.distance(loc_point)
            if distance < location_min_distance_segment[loc]['distance']:
                location_min_distance_segment[loc]['segment_order'] = i
                location_min_distance_segment[loc]['segment_index'] = index
                location_min_distance_segment[loc]['distance'] = distance

    #Plot the traffic on each segment, with spans for the locations
    plt.plot(list(range(len(road_gdf))),road_gdf['anf'],  label='Traffic')

    cmap = colors.ListedColormap(plt.cm.Set3.colors).reversed()
    for i, (loc, segment_data) in enumerate(location_min_distance_segment.items()):
        #segment = road_gdf.iloc[segment_data['segment_order']]
        color = cmap(i % cmap.N)  #Likely never more than 12 locations, but just in case
        plt.axvspan(segment_data['segment_order']-location_radius*len(road_gdf), segment_data['segment_order']+location_radius*len(road_gdf), color=color, alpha=0.3, label=loc)
        plt.axvline(x=segment_data['segment_order'], color='darkgrey', linestyle='--')

    if title:
        plt.title(title)
    plt.legend()
    plt.show()

def plot_road_traffic_with_given_locations_groups_separately(road_gdf,location_lat_long_pairs, location_radius=0.1, title = None):
    #Turn the location dict into a GeoDataFrame: this way it is easy to transform the CRS
    gdf_loc = create_location_gdf_with_crs(location_lat_long_pairs, road_gdf.crs)
    #Turn back to dict: faster to access
    location_point_pairs = dict(zip(gdf_loc['Location'], gdf_loc['geometry']))
    
    #Take the indexes where the group changes (we assume the order: 0 0 0 0 0 ... 0 1 1 1 ... 1 1 2 2 2...)
    group_separation_index = [0] + list(road_gdf[road_gdf['group'].diff() != 0].index) + [len(road_gdf)]
    group_index_limits = list(zip(group_separation_index[:-1], group_separation_index[1:]))

    #Iterate over unique groups
    for group_num in range(len(road_gdf['group'].unique())):
        #Subset the DataFrame for the current group
        road_gdf_group = road_gdf[road_gdf['group'] == group_num]
        
        #Initialize the closest segment and distance for each location
        location_min_distance_segment = {key: {'segment_order':0, 'segment_index': road_gdf_group.iloc[0].name, 'distance': road_gdf_group['geometry'].iloc[0].distance(value)} for key, value in location_point_pairs.items()}

        #Iterate through the road and find the closest segment to each location
        for row_num, (index, row) in enumerate(road_gdf_group.iterrows()):
            road_geom = row['geometry']
            for loc, loc_point in location_point_pairs.items():
                distance = road_geom.distance(loc_point)
                if distance < location_min_distance_segment[loc]['distance']:
                    location_min_distance_segment[loc]['segment_order'] = index
                    location_min_distance_segment[loc]['segment_index'] = index
                    location_min_distance_segment[loc]['distance'] = distance

        #Plot the traffic on each segment, with spans for the locations
        plt.plot(road_gdf_group.index, road_gdf_group['anf'], color='black')
        
        cmap = colors.ListedColormap(plt.cm.Set3.colors).reversed()
        for loc_num, (loc, segment_data) in enumerate(location_min_distance_segment.items()):
            color = cmap(loc_num % cmap.N)  #Likely never more than 12 locations, but just in case
            plt.axvspan(segment_data['segment_order']-location_radius*len(road_gdf_group), segment_data['segment_order']+location_radius*len(road_gdf_group), color=color, alpha=0.3, label=loc)
            plt.axvline(x=segment_data['segment_order'], color='darkgrey', linestyle='--')
            ymin = road_gdf_group['anf'].min(); ymax = road_gdf_group['anf'].max()
            plt.text(segment_data['segment_order'], ymin+loc_num*(ymax-ymin)/(len(location_min_distance_segment)), f"{loc_num}. dist: {segment_data['distance']:.2f}", rotation=90, verticalalignment='bottom')

        if title:
            plt.title(f"{title} (Group {group_num})")
        plt.legend()
        plt.show()