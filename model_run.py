import argparse
import sys

def run_model(model_name, data, output_filename, **kwargs):
    """
    Run a model with given data and save the output to a file.

    Args:
        model_name (str): The name of the model to run.
        data (str): The data to use with the model.
        output_filename (str): The name of the output file.
        kwargs (dict): Additional optional arguments to pass to the model.
            output_format (str): The format to save the output in. Default is 'csv', other options are 'json' and 'txt'.
    """
    
    #TODO


    pass

def parser(args=None):
    """
    Parse command line arguments, or arguments passed in as a list of strings.

    args is used when running the script from another Python script, or Jupyter notebook

    Example:
    args = ['-m', 'gravity', '-d', 'KSH', '-o', 'ODM_2022_Hungary', '-k', 'output_format=csv,key2=value2']
    model, data, output_filename, kwargs = cmd_parser(args)

    Otherwise, command line arguments are used and args=None
    
    Example:
    python model_run.py -m gravity -d KSH -o ODM_2022_Hungary -k output_format=csv,key2=value2
    """
    parser = argparse.ArgumentParser(description='Run a model with given data and save the output to a file.')
    parser.add_argument('-m', '--model', type=str, required=True, help='The name of the model to run.')
    parser.add_argument('-d', '--data', type=str, required=True, help='The data to use with the model.')
    parser.add_argument('-o', '--output', type=str, required=True, help='The name of the output file.')
    parser.add_argument('-k', '--kwargs', type=str, help='Additional optional arguments as key=value pairs separated by commas.')

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    kwargs = {}
    if args.kwargs:
        kv_pairs = args.kwargs.split(',')
        for pair in kv_pairs:
            key, value = pair.split('=')
            kwargs[key] = value

    return args.model, args.data, args.output, kwargs

if __name__ == '__main__':
    model, data, output_filename, kwargs = parser()
    run_model(model, data, output_filename, **kwargs)