import argparse
import helper_functions
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx

# Purposefully avoiding classes for simplicity

def construct_model_args(model_name, flow_traffic_data = None, tessellation = None, **kwargs):
    """
    Construct a model object with given parameters.

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
        if tessellation is None:
            raise ValueError('Error: The gravity model requires a tessellation: a GeoDataFrame\
                              of locations with location ID, population and geometry.')
        if type(tessellation) == str:
            #Assume filename
            tessellation = gpd.read_file(tessellation)
        
        if 'tileID' not in tessellation.columns:
            raise ValueError('Error: The tessellation GeoDataFrame must have a tile_ID column, with this name.')
        if 'geometry' not in tessellation.columns:
            raise ValueError('Error: The tessellation GeoDataFrame must have a geometry column, with this name.')
        if 'population' not in tessellation.columns:
            raise ValueError('Error: The tessellation GeoDataFrame must have a population column, with this name.')

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
        return model_params
    
    if model_name == 'bell':
        #Redirect to Bell modified model (modified with loss function)
        print("Redirecting to Bell modified model (modified with loss function).\
              Should have already been redirected in the run_model function.")
        args = construct_model_args('bell_modified', flow_traffic_data, tessellation, **kwargs)

    if model_name == 'bell_modified':
        pass

    if model_name == 'bell_L1':
        pass

    raise ValueError(f'Error: Model {model_name} not found.Only \
                     gravity, bell, bell_modified, bell_L1 are valid model names.')

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
        
        output_filename (str): The name of the output file.
        
        tessellation (gpd.GeoDataFrame | str (filename)): Location ID, geometry (e.g shapely point),
            and population data. tot_outflow is optional, computed if needed.
            See odm_gravity.py for an example.
        
        kwargs (dict): Additional optional arguments to pass to the model.
            
            initial_odm (numpy.ndarray | list): The initial ODM (for some models)
                in a vectorized form. If it is not provided but the model
                requires it, the gravity model will be used to estimate it.
            
            output_format (str): The format to save the output in. 
                Default is 'csv', other options are 'json' and 'txt'.

            network (networkx.Graph or DiGraph): A network to use for the Bell model.
                Nodes should be the locations, a possible attribute of nodes is 'ignore'.
                Edges should have the traffic data as edge weights. Possible attributes
                are 'time'

            Other arguments: Parameters of the given model. For example:
                q for the Bell model, or deterrence for the gravity model.
            
    """
    
    #TODO
    if model_name == 'gravity':
        arg_dict = construct_model_args('gravity', tessellation=tessellation, flows_df = flow_traffic_data, **kwargs)
        odm_df = run_gravity_model(flows_df = flow_traffic_data, tessellation = tessellation, **arg_dict)

    if model_name == 'bell':
        """Redirect to Bell modified model (modified with loss function)"""
        #Safest (future-proof) way to do this is to re-run this function with the modified model name.
        odm_ = run_model('bell_modified', flow_traffic_data, tessellation, output_filename, **kwargs)
        return odm_

    if model_name == 'bell_modified':
        pass

    if output_filename is None:
        import datetime
        output_filename = f'ODM_{model_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    if kwargs.get('output_format', 'csv') == 'csv':
        odm_df.to_csv(output_filename + '.csv')
    elif kwargs.get('output_format', 'csv') == 'json':
        odm_df.to_json(output_filename + '.json')
    elif kwargs.get('output_format', 'csv') == 'txt':
        with open(output_filename + '.txt', 'w') as f:
            #Other option is to use to_csv, with a defined separator
            f.write(odm_df.to_string())
            
def parser(args=None):
    """
    Parse command line arguments, or arguments passed in as a dictionary.

    args is used when running the script from another Python script, or Jupyter notebook

    Example:
    args = {'model': 'gravity', 'data': 'KSH', 'output': 'ODM_2022_Hungary', 'kwargs': 'output_format=csv,key2=value2'}
    model, data, output_filename, kwargs = parser(args)

    Otherwise, command line arguments are used and args=None
    
    Example:
    python model_run.py -m gravity -d KSH -o ODM_2022_Hungary -k output_format=csv,key2=value2
    """
    if args is None: #Assume command line arguments
        parser_ = argparse.ArgumentParser(description='Run a model with given data and save the output to a file.')
        parser_.add_argument('-m', '--model', type=str, required=True, help='The name of the model to run.')
        parser_.add_argument('-d', '--data', type=str, required=True, help='The data to use with the model.')
        parser_.add_argument('-o', '--output', type=str, required=True, help='The name of the output file.')
        parser_.add_argument('-k', '--kwargs', type=str, help='Additional optional arguments as key=value pairs separated by commas.')
        args = vars(parser_.parse_args())
    else: #Assume e.g. Jupyter notebook run, with dictionary args
        #Will need to change this, function will be called directly with various args, without key 'kwargs'
        if 'kwargs' in args and isinstance(args['kwargs'], str):
            kv_pairs = args['kwargs'].split(',')
            args['kwargs'] = {key: value for key, value in (pair.split('=') for pair in kv_pairs)}

    return args['model'], args['data'], args['output'], args.get('kwargs', {})

if __name__ == '__main__':
    model, data, output_filename, kwargs = parser()
    run_model(model, data, output_filename, **kwargs)
