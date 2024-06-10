from helper_functions import computing_functions
from helper_functions import map_functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import networkx as nx
from sklearn.metrics import mean_squared_error, mean_absolute_error

################ Map plotting functions ################

def plot_roads_and_locations(gdf, geoposition_dict, road_name_list, location_list, with_text = True, fontsize=15):
    if type(gdf)==list:
        gdf = pd.concat(gdf)
    roads_gdf = gdf[gdf['kszam'].isin(road_name_list)]
    ax=roads_gdf.plot(color='black', linewidth=1)
    
    if with_text: #Route's name
        for _, route_gdf in roads_gdf.groupby('kszam'):
            plt.text(route_gdf['geometry'].iloc[0].centroid.x, route_gdf['geometry'].iloc[0].centroid.y, route_gdf['kszam'].iloc[0], fontsize=fontsize, color='red')

    loc_gdf = map_functions.create_location_gdf_with_crs(map_functions.LUT_geopositions(location_list,geoposition_dict), gdf.crs)
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

    loc_gdf = map_functions.create_location_gdf_with_crs(map_functions.LUT_geopositions(location_list,geoposition_dict), gdf.crs)
    loc_gdf.plot(ax=ax, color='red', markersize=20)
    
    if with_text: #Location's name
        for _, location in loc_gdf.iterrows():
            plt.text(location['geometry'].x, location['geometry'].y, location['Location'], fontsize=fontsize, color='blue')
    
    #Intersection points across gdframes
    intersection_points, intersection_segments = map_functions.find_intersections_cross(gdfs)
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
    gdf_loc = map_functions.create_location_gdf_with_crs(location_lat_long_pairs, road_gdf.crs)
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
    gdf_loc = map_functions.create_location_gdf_with_crs(location_lat_long_pairs, road_gdf.crs)
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

################ Model plotting functions ################
def plot_odm(odm_2d, locations, plot_type='heatmap', order = None, log_scale=False, half=False, color='blue', x_label_size=None, x_label_rotation=0):
    from collections import defaultdict
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

    elif plot_type == 'scatterplot':
        if log_scale:
            print("Log scale is not supported for scatterplot")
        odm_long = odm_2d[np.triu_indices(odm_2d.shape[0], k = 1)] if half else odm_2d.reshape(-1)
        x = [(locations[i], locations[j]) for i in range(len(locations)) for j in range(i+1, len(locations))] if half else [(loc1, loc2) for loc1 in locations for loc2 in locations]
        y = odm_long
        if order is not None:
            x = order
        scatter = sns.scatterplot(x=[str(i) for i in x], y=y, size=odm_long, color=color)
        #plt.xticks(rotation=45, fontsize='small')
    else:
        print(f"Unknown plot type: {plot_type}")
        return None
    
    plot_params = {}
    plot_params['rotation'] = x_label_rotation
    plot_params['fontsize'] = x_label_size
    plt.xticks(**plot_params)
    return plt

def plot_odm_axis(odm_2d, locations, plot_type='heatmap', order=None, ax=None, log_scale=False, half=False, title=None, color='blue', x_label_size=None, x_label_rotation=0):
    if ax is None:
        fig, ax = plt.subplots()

    if plot_type == 'heatmap':
        if order is not None:
            original_order = locations
            current_order = [original_order.index(i) for i in order]
            odm_2d = odm_2d[current_order][:, current_order]
            locations = order
        if not log_scale:
            sns.heatmap(odm_2d, xticklabels=locations, yticklabels=locations, ax=ax)
        else:
            sns.heatmap(odm_2d, xticklabels=locations, yticklabels=locations, norm=colors.LogNorm(), ax=ax)
    elif plot_type == 'scatterplot':
        if log_scale:
            print("Log scale is not supported for scatterplot")
        odm_long = odm_2d[np.triu_indices(odm_2d.shape[0], k = 1)] if half else odm_2d.reshape(-1)
        x = [(locations[i], locations[j]) for i in range(len(locations)) for j in range(i+1, len(locations))] if half else [(loc1, loc2) for loc1 in locations for loc2 in locations]
        y = odm_long
        if order is not None:
            x = order
        scatter = sns.scatterplot(x=[str(i) for i in x], y=y, size=odm_long, color=color, ax=ax)
    else:
        print(f"Unknown plot type: {plot_type}")  # potential change to raise ValueError
    if title is not None:
        ax.set_title(title)

    plot_params = {}
    plot_params['rotation'] = x_label_rotation
    plot_params['fontsize'] = x_label_size
    plt.xticks(**plot_params)
    return plt

def plot_models(models_dict, x_labels=None):
    """
    The dictionary should have the model names as keys and the ordered O-D values as values.
    """
    fig, ax = plt.subplots(figsize=(20, 7))

    for label, model in models_dict.items():
        if model.ndim != 1:
            print(f"Model {label} is not 1D, converting to 1D")
            model = model.reshape(-1)
        ax.plot(model, label=label)

        if x_labels is not None:
            min_idx = np.argmin(model)
            max_idx = np.argmax(model)
            ax.annotate(f'{x_labels[min_idx]}', (min_idx, model[min_idx]), textcoords="offset points", xytext=(-10,-10), ha='center')
            ax.annotate(f'{x_labels[max_idx]}', (max_idx, model[max_idx]), textcoords="offset points", xytext=(-10,10), ha='center')

    plt.legend()
    plt.show()

def evaluate_model(predicted, real, model_name="model", verbose=True):
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error

    if type(predicted) in [list, pd.Series]:
        predicted = np.array(predicted)
    if type(real) in [list, pd.Series]:
        real = np.array(real)
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

def plot_symmetric_models_by_real_values(x, models_dict, colors_list=None, names=None, limits=[0, 10**5]):
    # Assuming array inputs
    if x.ndim != 1:
        print("Assuming x is a matrix, taking the upper triangle")
        x = list(x[np.triu_indices(x.shape[0], k = 1)])

    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    plt.plot(limits, limits, color='black', label='prediction=real value line')

    error_comparisons = {}
    for i, (label, model) in enumerate(models_dict.items()):
        if model.ndim != 1:
            print(f"Assuming {label} is a matrix, taking the upper triangle")
            y = list(model[np.triu_indices(model.shape[0], k = 1)])
        else:
            y = model
        if colors_list is not None:
            ax.scatter(x, y, label=label, color=colors_list[i])
        else:
            ax.scatter(x, y, label=label)

        mse = mean_squared_error(x, y)
        mae = mean_absolute_error(x, y)
        error_comparisons[label] = {'MSE': mse, 'MAE': mae}

        #Find the index of the largest difference
        diff = np.abs(np.array(x) - np.array(y))
        max_diff_idx = np.argmax(diff)
        if names is not None:
            ax.annotate(f'{names[max_diff_idx]}', (x[max_diff_idx], y[max_diff_idx]), textcoords="offset points", xytext=(-10,10), ha='center')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Real O-D traffic values', fontsize=15)
    plt.ylabel('Predicted values', rotation=90, fontsize=15)
    plt.title('Models predicted O-D pair values vs correct values line', fontsize=18)
    plt.xlim(limits)
    plt.ylim(limits)
    plt.legend(fontsize=14)

    for model, errors in error_comparisons.items():
        print(f'Error comparisons for {model}: MSE = {errors["MSE"]}, MAE = {errors["MAE"]}')

    if len(models_dict) > 1:
        # Compare how many times one model was closer to the real value than the others
        for i, (label1, model1) in enumerate(models_dict.items()):
            if model1.ndim != 1:
                m1 = list(model1[np.triu_indices(model1.shape[0], k = 1)])
            for j, (label2, model2) in enumerate(models_dict.items()):
                if model2.ndim != 1:
                        m2 = list(model2[np.triu_indices(model2.shape[0], k = 1)])
                if i < j:
                    closer_count = np.sum(np.abs(np.array(x) - np.array(m1)) < np.abs(np.array(x) - np.array(m2)))
                    print(f'{label1} was closer to the real value than {label2} {closer_count} times')
                    print(f'{label2} was closer to the real value than {label1} {len(x) - closer_count} times')
    plt.show()