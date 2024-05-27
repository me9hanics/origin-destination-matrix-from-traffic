import warnings
import argparse
import helper_functions
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx

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
        The difference between the two is in the loss function used.
        The latter is an approximation of L1 loss, the former is an O(L*log(L)) loss.
        bell_modified is the default, and it grows at the as the objective function grows.

        Args:
            initial_odm_vector (numpy.ndarray | list): The initial ODM (for some models)

            q (float): The q parameter of the Bell model. Default is None.

            network (networkx.Graph or DiGraph): A network to use for the Bell model.
                    Node names should be the locations, a possible attribute of nodes is 'ignore'.
                    Edges should have the traffic data as edge weights. Possible attribute is 'time'.
            
            hidden_locations (list): The irrelevant locations/nodes in the network. Default is None,
                    but if a network is given, it is assumed that the nodes with 'ignore' attribute
                    are hidden locations. If both hidden_locations and ignored nodes are given,
                    the union of the two is used.
                    Not yet implemented to work without a network.

            find_locations (dict): Given a dictionary where keys are geographical locations (ideally
                    points), and value includes location names + radius, 
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

        Output:
            flow_traffic_data, tessellation, model_params: The flow data and tessellation (can differ
            from the original: if read from a file, the returned version already contains the correct
            format), and the needed model parameters.
            Intended to be used with the run_*X*_model functions.
        """
        
        q = kwargs.get('q', None)
        initial_odm_vector = kwargs.get('initial_odm_vector', None)
        network = kwargs.get('network', None)
        hidden_locations = kwargs.get('hidden_locations', None)
        find_locations = kwargs.get('find_locations', None)
        P_algorithm = kwargs.get('P_algorithm', 'shortest_path')
        extra_paths = kwargs.get('extra_paths', None)

        #Handle necessary parameters based on if network is given
        if type(network) == str:
            #Assume filename
            network = nx.read_gpickle(network)
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
                    raise ValueError('Error: Could not read the flow_traffic_data file given as\
                                     DataFrame or GeoDataFrame. Might be a wrong file path, or format.')

        if network is None:
            raise ValueError('Error: Model not yet implemented without input network')
            if flow_traffic_data is None:
                raise ValueError('Error: If a network is not explicitly given, the Bell model\
                                requires traffic data given by flow_traffic_data.')
        else:
            if hidden_locations is None:
                #TODO Check if this works as intended
                hidden_locations = [node for node in network.nodes if 'ignore' in network.nodes[node]]
            else:
                #TODO Check if this works as intended
                hidden_locations = list(set(hidden_locations
                                            + [node for node in network.nodes if 'ignore' in network.nodes[node]]))
                
            if find_locations is not None:
                warnings.warn('Warning: hidden_locations parameter may not be used, because a network is given.')
        
        #P matrix computation parameters
        if P_algorithm not in ['shortest_path', 'shortest_time']:
            raise ValueError('Error: P_algorithm must be either "shortest_path" or "shortest_time".')
        else:
            if (P_algorithm != 'shortest_path') and (extra_paths is not None):
                warnings.warn('Warning: extra_paths parameter is not yet implemented to be used \
                              with P_algorithm != "shortest_path".')

        #ODM computation parameters
        if initial_odm_vector is None:
            #Run gravity model to estimate initial ODM vector
            if tessellation is None:
                raise ValueError('Error: If the Bell model is not given an initial ODM vector\
                                 it requires a tessellation to estimate it, which is missing.')
            if flow_traffic_data is None:
                raise ValueError('Error: If the Bell model is not given an initial ODM vector\
                                    it requires traffic data to estimate it, which is missing.')
            #Run gravity model to estimate initial ODM vector
            warnings.warn('Warning: No initial ODM given, running gravity model to get an \
                          initial ODM vector.')
            g_flow_traffic_data, g_tessellation, g_arg_dict = construct_model_args('gravity',
                                                                                    flow_traffic_data,
                                                                                    tessellation,
                                                                                    **kwargs)
            initial_odm_vector = run_gravity_model(g_flow_traffic_data, g_tessellation, **g_arg_dict)
        
        model_params = {'q': q, 'initial_odm_vector': initial_odm_vector, 'network': network,
                        'hidden_locations': hidden_locations, 'find_locations': find_locations,
                        'P_algorithm': P_algorithm, 'extra_paths': extra_paths}
        
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
    #Import here, because skmob isn't a necessary dependency
    import skmob
    from skmob.utils import utils, constants
    #from skmob.models import gravity
    from skmob.models.gravity import Gravity

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
    
    tessellation = tessellation.merge(tot_outflows, left_on='tile_ID', right_on='origin')\
                                        .rename(columns={'flow': constants.TOT_OUTFLOW})
    #Run the gravity model
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

def run_bell_model(bell_type, flow_traffic_data, tessellation=None, network=None, initial_odm_vector = None, q=None, output_format='csv'):
    if bell_type == 'bell':
        #Redirect to Bell modified model (modified with loss function)
        print("Redirecting to Bell modified model (modified with loss function).\
              Should have already been redirected in the run_model function.")
        odm_ = run_bell_model('bell_modified', flow_traffic_data, tessellation, output_filename, **kwargs)
        #All procedures are done in the previous call, just return the ODM
        return odm_
    
    #Assuming otherwise
    if bell_type == 'bell_modified':
        loss_func = 'modified'
    elif bell_type == 'bell_L1':
        loss_func = 'L1'
    else:
        raise ValueError('Error: Invalid Bell model type, only "bell_modified" or "bell_L1" are accepted\
                         ("bell" redirects to "bell_modified"). This error should have been raised earlier,\
                         in the construct_model_args function, please check what caused the issue.')
    
    pass
        

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
            
            initial_odm_vector (numpy.ndarray | list): The initial ODM (for some models)
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
        flow_traffic_data, tessellation, arg_dict = construct_model_args('gravity', tessellation=tessellation,
                                                                         flows_df = flow_traffic_data, **kwargs)
        odm_df = run_gravity_model(flows_df = flow_traffic_data, tessellation = tessellation, **arg_dict)

    if model_name == 'bell':
        """Redirect to Bell modified model (modified with loss function)"""
        #Safest (future-proof) way to do this is to re-run this function with the modified model name.
        #Might need to be changed later, to safeproof recursive calls.
        odm_ = run_model('bell_modified', flow_traffic_data, tessellation, output_filename, **kwargs)
        #All procedures are done in the previous call, just return the ODM
        return odm_

    if model_name == 'bell_modified':
        flow_traffic_data, tessellation, arg_dict = construct_model_args('bell_modified', tessellation=tessellation,
                                                                         flows_df = flow_traffic_data, **kwargs)
        odm_df = run_bell_model('bell_modified', flow_traffic_data, tessellation, output_filename, **arg_dict)

    if model_name == 'bell_L1':
        flow_traffic_data, tessellation, arg_dict = construct_model_args('bell_L1', tessellation=tessellation,
                                                                         flows_df = flow_traffic_data, **kwargs)
        odm_df = run_bell_model('bell_L1', flow_traffic_data, tessellation, output_filename, **arg_dict)

    #Save the output, return the ODM
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
    args = {'model': 'gravity', 'data': 'KSH', 'output': 'ODM_2022_Hungary', 'kwargs': 'output_format=csv,key2=value2'}
    model, data, output_filename, kwargs = parser(args)

    Otherwise, command line arguments are used and args=None
    
    Example:
    python model_run.py -m bell_modified -d data/roads.geojson -o ODM_2022_Hungary -k output_format=csv,key2=value2
    """
    if args is None: #Assume command line arguments
        warnings.warn('Warning: This function accepts only string arguments now, if called from command line.\
                    This can be an issue with non-string arguments such as dictionaries, or arrays.\
                    Commonly, such an argument is initial_odm_vector.')
        parser_ = argparse.ArgumentParser(description='Run a model with given data and save the output to a file.')
        parser_.add_argument('-m', '--model', type=str, required=True, help='The name of the model to run.')
        parser_.add_argument('-d', '--data', type=str, required=True, help='The data to use with the model.')
        parser_.add_argument('-o', '--output', type=str, required=True, help='The name of the output file.')
        parser_.add_argument('-k', '--kwargs', type=str, default='', help='Additional optional arguments as key=value pairs separated by commas.')
        args = vars(parser_.parse_args())
    else:#Assume e.g. Jupyter notebook run, with dictionary args
        #Will need to change this, function will be called directly with various args, without key 'kwargs'
        if 'kwargs' in args and isinstance(args['kwargs'], str):
            kv_pairs = args['kwargs'].split(',')
            try:
                args['kwargs'] = {key: value for key, value in (pair.split('=') for pair in kv_pairs)}
            except ValueError:
                raise ValueError("kwargs must be in the format 'key1=value1,key2=value2,...'")
    return args['model'], args['data'], args['output'], args.get('kwargs', {})

if __name__ == '__main__':
    model, data, output_filename, kwargs = parser()
    run_model(model, data, output_filename, **kwargs)
