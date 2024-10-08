import helper_functions
import warnings
import argparse
import pickle
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import copy

# Purposefully avoiding classes for simplicity

def construct_model_args(model_name, flow_traffic_data = None, tessellation = None, **kwargs):
    """
    Check and handle the given parameters, return a simplified model parameters dictionary.

    if model_name == 'gravity':
        Args:
            flows_df (pd.DataFrame): Traffic data, columns: 'origin', 'destination', 'flow'.
            tessellation (gpd.GeoDataFrame): Location ID, geometry (e.g shapely point).
            deterrence (float): The deterrence parameter.
            friction (float): The friction parameter.
            model_params (dict): The parameter dict of the gravity model.
                Example: {'deterrence_func_type': "power_law", 'deterrence_func_args': [-2.0]}
    """

    if model_name == 'gravity':
        #Handle necessary parameters
        if tessellation is None:
            raise ValueError('Error: The gravity model requires a tessellation: a GeoDataFrame\
                              of locations with location ID, population and geometry.')
        if type(tessellation) not in [str, gpd.GeoDataFrame]:
            raise ValueError('Error: The tessellation parameter must be a GeoDataFrame, \
                             or a string with the path of the file that contains it.')
        if type(tessellation) == str:
            #Assume filename
            try:
                tessellation = gpd.read_file(tessellation)
            except:
                raise ValueError('Error: Could not read the tessellation file given as GeoDataFrame.\
                                 Might be a wrong file path, or format.')
        if 'tileID' not in tessellation.columns:
            raise ValueError('Error: The tessellation GeoDataFrame must have a tile_ID column, with this name.')
        if 'geometry' not in tessellation.columns:
            raise ValueError('Error: The tessellation GeoDataFrame must have a geometry column, with this name.')
        if 'population' not in tessellation.columns:
            raise ValueError('Error: The tessellation GeoDataFrame must have a population column, with this name.')

        if flow_traffic_data not in [str, pd.DataFrame]:
            raise ValueError('Error: The traffic data parameter must be a pandasDataFrame, or a \
                             string with the path of the file that contains it. Try converting it.')
        if type(flow_traffic_data) == str:
            #Assume filename
            try:
                flow_traffic_data = pd.read_csv(flow_traffic_data)
            except:
                raise ValueError('Error: Could not read the flow_traffic_data file given as DataFrame.\
                                     Might be a wrong file path, or file format.')

        #Gather optional parameters
        gravity_type="singly constrained" #This is assumed for now, might be changed later
        deterrence_func_type = kwargs.get('deterrence_func_type', "power_law")
        deterrence_func_args = kwargs.get('deterrence_func_args', [-2.0])
        origin_exp = kwargs.get('origin_exp', 1.0)
        destination_exp = kwargs.get('destination_exp', 1.0)
        tot_outflows = tessellation['tot_outflow'] if 'tot_outflow' in tessellation.columns else None

        #TODO Add conversions
        model_params = {'gravity_type': gravity_type, # 'singly constrained'
                        'deterrence_func_type': deterrence_func_type,
                        'deterrence_func_args': deterrence_func_args,
                        'origin_exp': origin_exp,
                        'destination_exp': destination_exp,
                        'tot_outflows': tot_outflows}
        return flow_traffic_data, tessellation, model_params
    
    if model_name == 'bell':
        #Redirect to Bell modified model (modified with loss function)
        print("Redirecting to Bell modified model (modified with loss function).\
              Should have already been redirected in the run_model function.")
        flow_traffic_data, tessellation, args = construct_model_args('bell_modified',
                                                                             flow_traffic_data,
                                                                             tessellation,
                                                                             **kwargs)
        return flow_traffic_data, tessellation, args

    if (model_name == 'bell_modified') | (model_name == 'bell_L1'):
        """
        The difference between the two is in the loss function used. \
        The latter is an approximation of L1 loss, the former is an O(L*log(L)) loss. \
        bell_modified is the default, and it grows at the as the objective function grows.

        Args:
            initial_odm_df (pd.DataFrame): The initial ODM (for some models) in a DataFrame form: columns
                have to be 'origin', 'destination', 'flow'. Recommended to use for consistency. If it is
                not provided and initial_odm_vector is None, the gravity model will be used to estimate it.
                Default is None.
            
            #TODO: Take out initial_odm_vector
            initial_odm_vector (numpy.ndarray | array_like): Unrecommended (see above): The initial ODM
                (for some models) in a vectorized form. Recommended usage is to use initial_odm_df.
                If neither of the two are provided, the gravity model will be used to estimate them.
                Default is None.

            q (float): The q parameter of the Bell model. Default is None. If given, must be sorted in
                the same order as the initial ODM DF. If None, the initial ODM is used to estimate it.

            c (float): The loss function coefficient. Default is 0.05.

            upper_bound (float | int): The upper bound for the ODM values. Default is 100000.

            network (networkx.Graph or DiGraph): A network to use for the Bell model, traffic as weights.
                    Node names should be the locations, a possible attribute of nodes is 'ignore'.
                    Edges should have the traffic data as edge weights. Possible attribute is 'time'.
            
            hidden_locations (list): The irrelevant locations/nodes in the network. Default is None,
                    but if a network is given, it is assumed that the nodes with 'ignore' attribute
                    are hidden locations. If both hidden_locations and ignored nodes are given,
                    the union of the two is used.
                    Not yet implemented to work without a network.

            find_locations (dict): Given a dictionary where keys are geographical locations (ideally
                    points), and value includes location names + radius, the model will find the nodes
                    in the network that are within the given radius and combine them into one node.
                    The intended use is without a given network, only with the traffic GeoDataFrame,
                    from which a network is constructed. In this network, the added nodes (road 
                    intersections) which are close to a certain location, within some radius, are
                    combined into one node, representing the location with the given name.
                    Default is None.
                    
            P_algorithm (str): The algorithm to use for computing the P matrix. Currently, options
                    are 'shortest_path' and 'shortest_time'. Default is 'shortest_path'.

            extra_paths (dict): Given a dictionary where keys are tuples of nodes (pairs of locations),
                    and values are lists of paths (lists of nodes), the model will use these paths too
                    when computing the P matrix based on shortest paths. Default is None.

            #TODO: Adding the P matrix explicitly as a parameter, if it is already computed.

        Output:
            flow_traffic_data, tessellation, model_params: The flow data and tessellation (can differ
            from the original: if read from a file, the returned version already contains the correct
            format), and the needed model parameters.
            Intended to be used with the run_*X*_model functions.
        """
        
        #Gather parameters
        initial_odm_df = kwargs.get('initial_odm_df', None)
        initial_odm_vector = kwargs.get('initial_odm_vector', None)
        q = kwargs.get('q', None)
        c = kwargs.get('c', 0.05)
        upper_bound = kwargs.get('upper_bound', 100000)
        network = kwargs.get('network', None)
        hidden_locations = kwargs.get('hidden_locations', None)
        removed_nodes = kwargs.get('removed_nodes', None)
        find_locations = kwargs.get('find_locations', None)
        P_algorithm = kwargs.get('P_algorithm', 'shortest_path')
        extra_paths = kwargs.get('extra_paths', None)
        return_before_optimization = kwargs.get('return_before_optimization', False)
        #TODO: Adding the P matrix explicitly as a parameter, if it is already computed.

        #Handle necessary parameters based on if network is given
        if type(network) == str:
            #Assume filename
            with open(network, 'rb') as f:
                network = pickle.load(f)
        elif (type(network) not in [nx.Graph, nx.DiGraph]) and (network is not None):
            raise ValueError('Error: The network parameter must be a networkx Graph or DiGraph, or \
                             a string with the path of the (g)pickle file that contains the network.')

        if type(flow_traffic_data) == str:
            #Assume filename
            try:
                flow_traffic_data = gpd.read_file(flow_traffic_data)
            except:
                try:
                    flow_traffic_data = pd.read_csv(flow_traffic_data)
                except:
                    raise ValueError('Error: Could not read the flow_traffic_data file given as \
                                     DataFrame or GeoDataFrame. Might be a wrong file path, or format.')
        elif type(flow_traffic_data) not in [pd.DataFrame, gpd.GeoDataFrame] and flow_traffic_data is not None:
            raise ValueError('Error: The traffic data parameter must be a GeoDataFrame, or a DataFrame, \
                             or a string with the path of the file that contains it.')

        if type(tessellation) == str:
            #Assume filename
            try:
                tessellation = gpd.read_file(tessellation)
            except:
                raise ValueError('Error: Could not read the tessellation file given as GeoDataFrame.\
                                 Might be a wrong file path, or format.')
        elif type(tessellation) is not gpd.GeoDataFrame and tessellation is not None:
            raise ValueError('Error: The tessellation parameter must be a GeoDataFrame, or a string \
                             with the path of the file that contains it.')

        if type(initial_odm_df) == str:
            #Assume filename
            try:
                initial_odm_df = pd.read_csv(initial_odm_df)
            except:
                raise ValueError('Error: Could not read the initial_odm_df file given as DataFrame.\
                                 Might be a wrong file path, or format.')
        elif type(initial_odm_df) is not pd.DataFrame and initial_odm_df is not None:
            raise ValueError('Error: The initial_odm_df parameter must be a DataFrame, or a string \
                             with the path of the file that contains it.')

        if network is None:
            #TODO
            raise ValueError('Error: Model not yet implemented without input network')
            if flow_traffic_data is None:
                raise ValueError('Error: If a network is not explicitly given, the Bell model\
                                requires traffic data given by flow_traffic_data.')
                # TODO: Do this without using node geographical locations as names (much less issues)
        else:
            for edge in network.edges:
                if 'weight' not in network.edges[edge]:
                    raise ValueError('Error: Network must have edge weights, representing traffic data.')
                if 'time' not in network.edges[edge] and P_algorithm == 'shortest_time':
                    if 'time (min)' in network.edges[edge]:
                        network.edges[edge]['time'] = network.edges[edge]['time (min)']
                    else:
                        warnings.warn('Warning: The network edges do not have a time attribute, but P_algorithm is set to\
                                  "shortest_time". Each value will be computed if possible or set to time with 50 km/h.')

            all_nodes_ignored = True
            for node in network.nodes:
                if 'ignore' not in network.nodes[node]:
                    network.nodes[node]['ignore'] = True
                else:
                    if network.nodes[node]['ignore'] not in [True, False]:
                        raise ValueError('Error: The ignore attribute of nodes must be True or False.')
                    if network.nodes[node]['ignore'] == True:
                        all_nodes_ignored = False
            if all_nodes_ignored:
                warnings.warn('Warning: All nodes in the network are initially ignored: ignoring none instead.')
                for node in network.nodes:
                    network.nodes[node]['ignore'] = False
            
            if hidden_locations is None:
                #This relies on the previous step.
                hidden_locations = [node for node in network.nodes if network.nodes[node]['ignore'] == True]
            else:
                #TODO Check if this works as intended
                hidden_locations = list(set(hidden_locations
                                            + [node for node in network.nodes if network.nodes[node]['ignore'] == True]))
            #TODO Do the same for removed_nodes

            if find_locations is not None:
                warnings.warn('Warning: find_locations parameter may not be used, because a network is given.')
        
        #P matrix computation parameters
        if P_algorithm not in ['shortest_path', 'shortest_time']:
            raise ValueError('Error: P_algorithm must be either "shortest_path" or "shortest_time".')
        else:
            if (P_algorithm != 'shortest_path') and (extra_paths is not None):
                warnings.warn('Warning: extra_paths parameter is not yet implemented to be used \
                              with P_algorithm != "shortest_path".')
        if (P_algorithm == 'shortest_time'):
            #Check if the network has time attribute in the edges
            missing_time_edges = [edge for edge in network.edges if 'time' not in network.edges[edge]]
            if len(missing_time_edges) > 0:
                print("No time attribute in some network edges, try adding time attribute")
                for edge in missing_time_edges:
                    speed = network.edges[edge]['speed'] if 'speed' in network.edges[edge] else 50
                    if 'distance' in network.edges[edge]:
                        network.edges[edge]['time'] = network.edges[edge]['distance'] / speed
                    else:
                        if 'geometry' in network.edges[edge]:
                            geometry = network.edges[edge]['geometry'] #Assuming LineString or MultiLineString
                            if geometry.type == 'LineString':
                                network.edges[edge]['distance'] = geometry.length
                            elif geometry.type == 'MultiLineString':
                                network.edges[edge]['distance'] = sum([line.length for line in geometry])
                            else:
                                raise ValueError('Error: No time or distance of road, and geometry type not recognized.')
                            network.edges[edge]['time'] = network.edges[edge]['distance'] / speed
                        else:
                            raise ValueError('Error: No time or distance of road, and no geometry attribute.')

        #ODM computation parameters
        if initial_odm_df is None:
            #Run gravity model to estimate initial ODM vector
            if tessellation is None:
                raise ValueError('Error: If the Bell model is not given an initial ODM DataFrame\
                                 it requires a tessellation to estimate it, which is missing.')
            if flow_traffic_data is None:
                raise ValueError('Error: If the Bell model is not given an initial ODM DataFrame\
                                    it requires traffic data to estimate it, which is missing.')
            #Run gravity model to estimate initial ODM vector
            warnings.warn('Warning: No initial ODM given, running gravity model to get an \
                          initial ODM vector.')
            g_flow_traffic_data, g_tessellation, g_arg_dict = construct_model_args('gravity',
                                                                                    flow_traffic_data,
                                                                                    tessellation,
                                                                                    **kwargs)
            initial_odm_df = run_gravity_model(g_flow_traffic_data, g_tessellation, **g_arg_dict)
        else:
            if ('origin' not in initial_odm_df.columns) or ('destination' not in initial_odm_df.columns):
                raise ValueError('Error: The initial_odm_df must have columns "origin" and "destination".')
            if 'flow' not in initial_odm_df.columns:
                raise ValueError('Error: The initial_odm_df must have a column "flow".')
            if initial_odm_vector is not None:
                warnings.warn('Warning: Both initial_odm_df and initial_odm_vector are given.\
                              Using initial_odm_df, ignoring the vector.')
        
        if q is None:
            q = (initial_odm_df['flow']+0.001) / np.sum(initial_odm_df['flow']+0.001) #Avoid division by zero
        else:
            try:
                q = np.array(q)
            except:
                raise ValueError('Error: Could not convert q to an np.array, please try giving the\
                                 variable as a numpy array or a list.')
            if q.shape != initial_odm_df['flow'].shape: #Could use .size instead of .shape
                raise ValueError('Error: q must have the same size (shape) as the (initial) ODM.')

        try:
            c = float(c)
        except:
            raise ValueError('Error: The loss function coefficient c must be a number.')
        if c < 0:
            warnings.warn('Warning: The loss function coefficient c is negative, this may lead to bad results.')

        try:
            upper_bound = float(upper_bound)
        except:
            raise ValueError('Error: The upper bound for the ODM values must be a number.')
        if upper_bound < 0:
            raise ValueError('Error: The upper bound for the ODM values must be a positive number.')
        
        if type(return_before_optimization) is not bool:
            warnings.warn('Warning: The return_before_optimization parameter must be a boolean.\
                          Setting it to False.')
            return_before_optimization = False
        #Copying
        flow_traffic_data = copy.deepcopy(flow_traffic_data)
        tessellation = copy.deepcopy(tessellation)
        initial_odm_df = initial_odm_df.copy()
        model_params = {'initial_odm_df': initial_odm_df, 'network': network, 'q': q, 'c': c,
                        'upper_bound': upper_bound,'hidden_locations': hidden_locations,
                        'removed_nodes': removed_nodes, 'find_locations': find_locations,
                        'P_algorithm': P_algorithm, 'extra_paths': extra_paths,
                        'initial_odm_vector': initial_odm_vector,
                        'return_before_optimization': return_before_optimization}
        
        return flow_traffic_data, tessellation, model_params

    raise ValueError(f'Error: Model {model_name} not found.Only \
                     gravity, bell, bell_modified, bell_L1 are valid model name inputs.')

def run_gravity_model(flows_df, tessellation, gravity_type="singly constrained",
                      deterrence_func_type = "power_law", deterrence_func_args = [-2.0],
                      origin_exp = 1.0, destination_exp = 1.0, tot_outflows = None):
    """
    Run the gravity model with given data and parameters.

    Note: This function uses the skmob library to run the gravity model.
        Currently, this is intended to be used with models, where both directions are
        represented in the input. (These can be equal, e.g. for a symmetric model.)

    Args:
        flows_df (pd.DataFrame): Traffic data, columns: 'origin', 'destination', 'flow'.
        tessellation (gpd.GeoDataFrame): Location ID, geometry (e.g shapely point).
        gravity_type (str): The type of gravity model to use. Default is 'singly constrained'.
        deterrence_func_type (str): The type of deterrence function to use. Default is 'power_law'.
        deterrence_func_args (list): The arguments of the deterrence function. Default is [-2.0].
        origin_exp (float): The origin exponent. Default is 1.0.
        destination_exp (float): The destination exponent. Default is 1.0.
        tot_outflows (pd.Series): The total outflows from each location. Default is None.
    """
    print("Importing the needed libraries.")
    #Import here, because skmob isn't a necessary dependency
    import skmob
    from skmob.utils import utils, constants
    #from skmob.models import gravity
    from skmob.models.gravity import Gravity

    print("Loading the given data into a FlowDataFrame.")
    #load real flows into a FlowDataFrame
    fdf = skmob.FlowDataFrame(flows_df,
                                tessellation=tessellation,
                                tile_id='tile_ID',
                                )
    fdf['origin'] = fdf['origin'].astype('int64')
    fdf['destination'] = fdf['destination'].astype('int64')
    fdf.tessellation[constants.TILE_ID] = fdf.tessellation[constants.TILE_ID].astype('int64')

    if tot_outflows is None:
        # compute the total outflows from each location of the tessellation
        #For now: excluding self loops #fdf['origin'] != fdf['destination']

        #This may exclude some locations, as they may not have any outflows
        print("Assuming each location has outflows. If not, the model skips those locations by mistake.\n\
              Designed for input that has flows from each location to another (in both direction).\n\
              Computing total outflows from each location (didn't find prior total outflows).")
        tot_outflows = fdf.groupby(by='origin', axis=0)[['flow']].sum().fillna(0)
    
    print("Adding total outflows per location to the tessellation.")
    tessellation = tessellation.merge(tot_outflows, left_on='tile_ID', right_on='origin')\
                                        .rename(columns={'flow': constants.TOT_OUTFLOW})
    print("Running the gravity model.")
    gravity_singly = Gravity(gravity_type=gravity_type, deterrence_func_type=deterrence_func_type,
                             deterrence_func_args=deterrence_func_args, origin_exp=origin_exp,
                             destination_exp=destination_exp)
    np.random.seed(0)
    synth_fdf = gravity_singly.generate(tessellation,
                                   tile_id_column='tile_ID',
                                   tot_outflows_column=constants.TOT_OUTFLOW,
                                   relevance_column= 'population',
                                   out_format='flows')
    synth_fdf['origin'] = synth_fdf['origin'].astype('int64')
    synth_fdf['destination'] = synth_fdf['destination'].astype('int64')

    return synth_fdf

def run_bell_model(bell_type, flow_traffic_data, tessellation=None, initial_odm_df = None,
                    q=None, c=0.05, upper_bound=100000, network=None, hidden_locations=None,
                    removed_nodes=None, find_locations=None, P_algorithm='shortest_path',
                    extra_paths=None, initial_odm_vector = None, return_before_optimization=False):
    """
    Run the Bell model with given data and parameters.
    
    Args:
    
        bell_type (str): The type of Bell model to run. Can be 'bell_modified' or 'bell_L1'.
        
        flow_traffic_data (gpd.GeoDataFrame): Traffic data on roads, with columns 'origin', 'destination',
            'flow', and 'geometry'. If a network is given, this parameter is not needed. Default is None.
            
        tessellation (gpd.GeoDataFrame): Location ID, geometry (e.g shapely point), and population data.
            Only required when the model needs to estimate the initial ODM, which it does with a gravity
            model. In this case, the tot_outflow column can be given, optionally. Default is None.

        initial_odm_df (pd.DataFrame): The initial ODM (for some models) in a DataFrame form: columns
            have to be 'origin', 'destination', 'flow'. Recommended to use for consistency. If it is
            not provided and initial_odm_vector is None, the gravity model will be used to estimate it.
            Default is None.

        #TODO: Remove initial_odm_vector
        initial_odm_vector (numpy.ndarray | array_like): Unrecommended (see above): The initial ODM
            (for some models) in a vectorized form. Recommended usage is to use initial_odm_df.
            If neither of the two are provided, the gravity model will be used to estimate them.
            Default is None.

        q (float): The q parameter of the Bell model. Default is None.

        c (float): The loss function coefficient. Default is 0.05.

        upper_bound (float | int): The upper bound for the ODM values. Default is 100000.
        #TODO: Add lower bound

        network (networkx.Graph or DiGraph): A network to use for the Bell model.

        hidden_locations (list): The irrelevant locations/nodes in the network. Default is None,

        find_locations (dict): Given a dictionary where keys are geographical locations (ideally
            points), and value includes location names + radius, the model will find the nodes
            in the network that are within the given radius and combine them into one node.
            The intended use is without a given network, only with the traffic GeoDataFrame,
            from which a network is constructed. In this network, the added nodes (road
            intersections) which are close to a certain location, within some radius, are
            combined into one node, representing the location with the given name.
            Default is None.

        P_algorithm (str): The algorithm to use for computing the P matrix. Currently, options
            are 'shortest_path' and 'shortest_time'. Default is 'shortest_path'.

        extra_paths (dict): Given a dictionary where keys are tuples of nodes (pairs of locations),
            and values are lists of paths (lists of nodes), the model will use these paths too
            when computing the P matrix based on shortest paths. Default is None.

        #TODO: Adding the P matrix explicitly as a parameter, if it is already computed.
                
    Output:
        The O-D matrix as a pandas.DataFrame.
    """
    if bell_type == 'bell':
        #Redirect to Bell modified model (modified with loss function)
        print("Redirecting to Bell modified model (modified with loss function).\
              Should have already been redirected in the run_model function.")
        odm_ = run_bell_model(bell_type='bell_modified', flow_traffic_data=flow_traffic_data, 
                              tessellation = tessellation, initial_odm_df = initial_odm_df,
                              q = q, c = c, upper_bound = upper_bound, network = network,
                              hidden_locations = hidden_locations, find_locations = find_locations,
                              P_algorithm=P_algorithm, extra_paths = extra_paths, 
                              initial_odm_vector = initial_odm_vector)
        #All procedures are done in the previous call, just return the ODM
        return odm_
    
    #Assuming otherwise
    if bell_type == 'bell_modified':
        warnings.warn('Warning: The Bell modified (entropic absolute error loss) model is slow.')
        loss_func = 'modified'
        objective_function = helper_functions.F_Bell_modified_optimize
        objective_function_gradient = helper_functions.F_Bell_modified_optimize_gradient
    elif bell_type == 'bell_L1':
        loss_func = 'L1'
        objective_function = helper_functions.F_Bell_L1_approximation_optimize
        objective_function_gradient = helper_functions.F_Bell_L1_approximation_optimize_gradient
    else:
        raise ValueError('Error: Invalid Bell model type, only "bell_modified" or "bell_L1" are accepted\
                         ("bell" redirects to "bell_modified"). This error should have been raised earlier,\
                         in the construct_model_args function, please check what caused the issue.')
    
    if network is None:
        raise ValueError('Error: Not yet implemented to run the Bell model without a network.\
                         This should have been caught earlier, in the construct_model_args function.')
        #Relevant variables for the network: flow_traffic_data, hidden_locations, find_locations, extra_paths
        #First, construct the network 
        #TODO
        #Then, if find_locations is given, look for nodes near locations and combine the nodes into one,
            #ignore = False for the new node, the node's name will be the location name.
            #If it is not given, continue.
        #If given hidden_locations, ignore = True for those nodes. If not given, either all nodes are
            #in the analysis, or if find_locations is given, only the "location nodes" are not ignored.
            #Finally, update hidden_locations as the set of ignored nodes.
    
    #Assuming we now have a network
    #(Assuming that "ignore" attribute also was used in the function construct_model_args to get 
        #hidden_locations, so that step is not repeated here)

    #P matrix computation
    print(f"Computing the P matrix based on method: {P_algorithm}. Sizes: roads: {network.number_of_edges()},\n\
              nodes: {network.number_of_nodes()}, among which are not hidden: {len(list(set(network.nodes) - set(hidden_locations)))}")
    if P_algorithm not in ['shortest_path', 'shortest_time']:
        raise ValueError('Error: Currently, only "shortest_path" and "shortest_time" are implemented for P_algorithm.')
    if P_algorithm == 'shortest_time':
        #TODO
        max_selected_paths = 3
        time_threshold = None
        v, P, odm_blueprint, extra = helper_functions.v_P_odmbp_shortest_times(network,
                                                                                max_selected_paths=max_selected_paths,
                                                                                time_threshold=time_threshold,
                                                                                hidden_locations = hidden_locations,
                                                                                removed_nodes=removed_nodes,
                                                                                extra_paths_dict = extra_paths,
                                                                                verbose=True)
    if P_algorithm == 'shortest_path':
        #Compute the P matrix based on shortest paths
        v, P, odm_blueprint, extra = helper_functions.v_P_odmbp_shortest_paths(network,
                                                                               removed_nodes=removed_nodes,
                                                                               hidden_locations = hidden_locations,
                                                                               extra_paths_dict = extra_paths,
                                                                               verbose=True)
    print(f"Computed the P matrix based on {P_algorithm}. Size of road traffic vector: {v.shape[0]},\
                size of the ODM vector: {odm_blueprint.shape[0]}.")

    #Something to consider: reducing the P matrix to maximum rank size
    print("Removing roads from the P matrix that are full of 0 values (unused between O-D locations).")
    P_,v_ = helper_functions.remove_full_zero_rows(P,v)
    print(f"Removed {P.shape[0] - P_.shape[0]} rows from the P matrix and from the v vector.")
    P = P_; v = v_
    print("Finding dependent and independent rows in the P matrix.")
    dep, indep = helper_functions.find_dependent_rows_simplified(P, verbose=True, return_independent=True)
    P_dep = P[dep, :]; v_dep = v[dep]
    P_indep = P[indep, :]; v_indep = v[indep]
    print(f"Rows in the P matrix: dependent: {len(dep)}, independent: {len(indep)}.")

    #Run the Bell model
    #Synch the initial_odm_df and q with the odm_blueprint
    print("Synchronizing the initial ODM with the blueprint ODM.")
    initial_odm_sorted, order = helper_functions.sort_odm_loc_names_df(initial_odm_df, extra['location_pairs'], return_ordering=True)
    odm_initial = np.array(initial_odm_sorted['flow'])
    print(initial_odm_sorted.shape, odm_initial.shape, q.shape, len(order), np.max(order), np.min(order))
    q = q[order]
    if return_before_optimization:
        return (v, P, odm_blueprint, extra, q, initial_odm_sorted, order, P_dep, v_dep, P_indep, v_indep)
    #Print sizes for everything
    print(f"Sizes: odm_initial: {odm_initial.shape}, q: {q.shape}, P_dep: {P_dep.shape}, P_indep: {P_indep.shape}, v_dep: {v_dep.shape}, v_indep: {v_indep.shape}")
    print("Running the SciPy optimizer.")
    odm = helper_functions.optimize_odm(model_function = objective_function, odm_initial = odm_initial,
                                        model_derivative=objective_function_gradient, runs=101,
                                        constraints_linear = helper_functions.odm_linear_constraint(P_indep, v_indep),
                                        model_func_args = {'q': q, 'c': c, 'P_modified_loss': P_dep, 'v_modified_loss': v_dep},
                                        bounds = helper_functions.bounds(0.001, upper_bound), verbose=False, return_last=True
                                        )
    odm_df = pd.DataFrame({'origin': initial_odm_sorted['origin'],
                           'destination': initial_odm_sorted['destination'],
                           'flow': odm})
    return odm_df
        

def run_model(model_name, flow_traffic_data=None, tessellation=None, output_filename=None, **kwargs):
    """
    Run a model with given data and save the output to a file.

    Args:
        model_name (str): The name of the model to run. Can be:
            'gravity', 'bell', 'bell_modified', 'bell_L1'. ('bell_L2' isn't implemented.)

        flow_traffic_data (pd.DataFrame | gpd.GeoDataFrame | str (filename)):
            Could be a DataFrame with columns 'origin', 'destination', 'flow' for the gravity model.
            For Bell models, either given a GeoDataFrame with the traffic data on roads, or with
            the network parameter a graph is given, which stores the traffic data as edge attributes.
            In that case, flow_traffic_data is not needed (can be None).
        
        output_filename (str): The name of the output file. (Do not include the file extension.)
            If None, the name will be ODM_{model_name}_{current_date_time}.
        
        tessellation (gpd.GeoDataFrame | str (filename)): Location ID, geometry (e.g shapely point),
            and population data. tot_outflow is optional, computed if needed. Only needed for gravity.
            See odm_gravity.py for an example.
        
        kwargs (dict): Additional optional arguments to pass to the model.
            
            initial_odm_vector (numpy.ndarray | array_like): The initial ODM (for some models)
                in a vectorized form. If it is not provided but the model
                requires it, the gravity model will be used to estimate it.
            
            output_format (str): The format to save the output in. 
                Default is 'csv', other options are 'json' and 'txt'.

            Other arguments: Parameters of the given model. For example:
                q for the Bell model, or deterrence for the gravity model.
                See the construct_model_args function for more details.
            
    Output:
        Saves the output to a file with the given output_filename. Returns the O-D matrix as pandas.DataFrame.
    """

    if model_name == 'gravity':
        print("Constructing the gravity model arguments.")
        flow_traffic_data, tessellation, arg_dict = construct_model_args('gravity', tessellation=tessellation,
                                                                         flows_df = flow_traffic_data, **kwargs)
        print("Running the gravity model.")
        odm_df = run_gravity_model(flows_df = flow_traffic_data, tessellation = tessellation, **arg_dict)

    if model_name == 'bell':
        """Redirect to Bell modified model (modified with loss function)"""
        print("Redirecting to Bell modified model (modified with loss function).")
        #Safest (future-proof) way to do this is to re-run this function with the modified model name.
        #Might need to be changed later, to safeproof recursive calls.
        odm_ = run_model('bell_modified', flow_traffic_data, tessellation, output_filename, **kwargs)
        #All procedures are done in the previous call, just return the ODM
        return odm_

    if model_name == 'bell_modified':
        print("Constructing the Bell modified (absolute entropic loss) model arguments.")
        flow_traffic_data, tessellation, arg_dict = construct_model_args('bell_modified', tessellation=tessellation,
                                                                         flows_df = flow_traffic_data, **kwargs)
        print("Running the Bell modified model.")
        odm_df = run_bell_model('bell_modified', flow_traffic_data, tessellation, **arg_dict)

    if model_name == 'bell_L1':
        print("Constructing the Bell L1 (approximation of L1 loss) model arguments.")
        flow_traffic_data, tessellation, arg_dict = construct_model_args('bell_L1', tessellation=tessellation,
                                                                         flows_df = flow_traffic_data, **kwargs)
        print("Running the Bell L1 model.")
        odm_df = run_bell_model('bell_L1', flow_traffic_data, tessellation, **arg_dict)

    if 'return_before_optimization' in kwargs:
        if kwargs['return_before_optimization']:
            return odm_df
    odm_df = (odm_df.astype({'flow': 'int'})).reset_index(drop=True)
    #Save the output, return the ODM
    print("Saving the output to a file.")
    if output_filename is None:
        #This may be changed to not output anything.
        import datetime
        output_filename = f'ODM_{model_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    output_format = kwargs.get('output_format', 'csv')
    if output_format== 'csv':
        odm_df.to_csv(output_filename + '.csv')
    elif output_format == 'json':
        odm_df.to_json(output_filename + '.json')
    elif output_format == 'txt':
        with open(output_filename + '.txt', 'w') as f:
            #Other option is to use to_csv, with a defined separator
            f.write(odm_df.to_string())
    else:
        raise ValueError('Error: Invalid output format. Choose from "csv", "json" or "txt".')

    return odm_df
            
def parser(args=None):
    """
    Parse command line arguments, or arguments passed in as a dictionary.

    args is used when running the script from another Python script, or Jupyter notebook

    Example:
    args = {'model': 'gravity', 'data': flow_DF, 'tessellation': tess_GDF,
            'output_filename': 'ODM_2022_Hungary', 'kwargs': 'output_format=csv,origin_exp=1.0'}
    model, data, output_filename, kwargs = parser(args)

    Otherwise, command line arguments are used and args=None.
    In this case, 'argparse' is used to parse the arguments.
    
    Example:
    python model_run.py -m bell_modified -d data/roads.geojson -t data/tessellation.shp 
                        -o ODM_2022_Hungary -k output_format=csv,origin_exp=1.0
    """
    if args is None: #Assume command line arguments
        warnings.warn('Warning: This function accepts only string arguments now, if called from command line.\
                    This can be an issue with non-string arguments such as dictionaries, or arrays.\
                    Commonly, such an argument is initial_odm_vector.')
        parser_ = argparse.ArgumentParser(description='Run a model with given data and save the output to a file.')
        parser_.add_argument('-m', '--model', type=str, required=True, help='The name of the model to run.')
        parser_.add_argument('-d', '--data', type=str, required=False, help='The data to use with the model.')
        parser_.add_argument('-t', '--tessellation', type=str, required=False, help='The tessellation to use with the model.')
        parser_.add_argument('-o', '--output_filename', type=str, required=False, help='The name of the output file.')
        parser_.add_argument('-k', '--kwargs', type=str, default='', help='Additional optional arguments as key=value pairs separated by commas.')
        args = vars(parser_.parse_args())
    #else:#Assume e.g. Jupyter notebook run, with dictionary args
    if 'kwargs' in args and isinstance(args['kwargs'], str):
        kv_pairs = args['kwargs'].split(',')
        try:
            args['kwargs'] = {key: value for key, value in (pair.split('=') for pair in kv_pairs)}
        except ValueError:
            raise ValueError("kwargs must be in the format 'key1=value1,key2=value2,...'")
    return args['model'], args.get('data'), args.get('tessellation'), args.get('output_filename'), args.get('kwargs', {})

if __name__ == '__main__':
    model, data, tessellation, output_filename, kwargs = parser()
    
    run_model(model, data, tessellation, output_filename, **kwargs)
