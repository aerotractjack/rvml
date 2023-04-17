import os
import shutil
import subprocess
import pathlib
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def load_yaml(path):
    ''' helper function to load YAML config file and return contents as dict '''
    with open(path, "r") as fp:
        return load(fp, Loader=Loader)
    
def copy_config_to_output_dir(output_dir, config_uri):
    ''' Place a copy of the training hyperparams in the prediction results dir 
    so we can reference them in the future '''
    dst = os.path.join(output_dir, "pipeline-config.json")
    shutil.copyfile(config_uri, dst)

def make_output_dir(tif_uri):
    ''' Create the output directory for the results.geojson file '''
    parts = tif_uri.split("Data")
    outdir = os.path.join(parts[0], "Modeling-Jack")
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    return outdir

def make_output_uri(outdir):
    ''' Build the uri/path for the predictions results file'''
    outuri = os.path.join(outdir, "results.json")
    return outuri

def pre_pipeline(tif_uri, train_config_uri):
    ''' Run all of the pre-processing steps before running a prediction '''
    output_dir = make_output_dir(tif_uri)
    copy_config_to_output_dir(output_dir, train_config_uri)
    output_uri = make_output_uri(output_dir)
    return output_uri

def batch_predict(input_config):
    ''' Run predictions on multiple images '''
    prediction_list = input_config["predict_list"][0]
    bundle_uri = os.path.join(input_config["model_bundle_uri"], "bundle", "model-bundle.zip")
    train_config_uri = os.path.join(input_config["model_bundle_uri"], "pipeline-config.json")
    L = len(prediction_list)
    for n, tif_uri in enumerate(prediction_list):
        output_uri = pre_pipeline(tif_uri, train_config_uri)
        cmd = ["rastervision", "predict", bundle_uri, tif_uri, output_uri]
        print(" ".join(cmd))
        print(f"Starting: {n+1}/{L}")
        subprocess.run(cmd)
        print(f"Done: {n+1}/{L}\n")
        
if __name__ == "__main__":
    import sys
    input_config = load_yaml(sys.argv[1])
    batch_predict(input_config)
