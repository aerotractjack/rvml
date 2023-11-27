import requests
import subprocess
from pathlib import Path 
import geopandas as gpd 
import os 
import yaml

storage_url = "http://192.168.1.35:7111"

def filter_train_points(train_data):
    out = {"aoi": [], "points": []}
    for i in range(len(train_data["points"])):
        p = train_data["points"][i]
        a = train_data["aoi"][i]
        df = gpd.read_file(p, driver="GeoJSON")
        if df.shape[0] == 0:
            continue
        if not os.path.exists(p) or not os.path.exists(a):
            continue
        out["aoi"].append(a)
        out["points"].append(p)
    return out

def _fetch_paths(client_id, project_id, stand_id):
    entry = {"CLIENT_ID": client_id, "PROJECT_ID": project_id, "STAND_ID": stand_id}
    body = {"entry": entry, "filetype": ""}
    paths = {}
    for ft in ["max_res_ortho", "training_data_and_boundary", "validation_data_and_boundary"]:
        body["filetype"] = ft
        req = requests.post(storage_url + "/filepath", json=body)
        paths[ft] = req.json()['filepath']
    paths["training_data_and_boundary"] = filter_train_points(paths["training_data_and_boundary"])
    return paths

def fetch_paths(client_id, project_id, stand_ids):
    out = []
    for sid in stand_ids:
        out.append(_fetch_paths(client_id, project_id, sid))
    return out

def format_for_rv(data_paths, output_root_uri, epochs):
    with open("autogen_yaml/default.yaml", "r") as fp:
        dflt = yaml.safe_load(fp)
    training_list = []
    validation_list = []
    for p in data_paths:
        ortho = p["max_res_ortho"]
        for i in range(len(p["training_data_and_boundary"]["points"])):
            rec = [ortho, p["training_data_and_boundary"]["points"][i],
                    p["training_data_and_boundary"]["aoi"][i]]
            training_list.append(rec)
        for i in range(len(p["validation_data_and_boundary"]["points"])):
            rec = [ortho, p["validation_data_and_boundary"]["points"][i],
                    p["validation_data_and_boundary"]["aoi"][i]]
            validation_list.append(rec)
            break
    dflt["training_list"] = training_list
    dflt["validation_list"] = validation_list
    dflt["output_root_uri"] = output_root_uri
    dflt["solver_config"]["num_epochs"] = int(epochs)
    with open("autogen_yaml/autogen_input.yaml", "w") as fp:
        yaml.dump(dflt, fp)

def main(client_id, project_id, stand_ids, output_root_uri, epochs):
    out = fetch_paths(client_id, project_id, stand_ids)
    format_for_rv(out, output_root_uri, epochs)

if __name__ == "__main__":
    import argparse

    DEFAULT_CLIENT_ID = 10050
    DEFAULT_PROJECT_ID = 101042
    DEFAULT_STAND_IDS = [100, 101, 102, 103, 104, 105, 106, 107, 108]
    #DEFAULT_STAND_IDS = [101, 102]
    DEFAULT_OUTPUT = "Manulife_WA"
    DEFAULT_EPOCHS = 5

    if not isinstance(DEFAULT_STAND_IDS, list):
        DEFAULT_STAND_IDS = [DEFAULT_STAND_IDS]

    parser = argparse.ArgumentParser(description="Input client/project/stand and ML params.")
    parser.add_argument('--client_id', '-cid', type=int, default=DEFAULT_CLIENT_ID, help='The client ID (an integer)')
    parser.add_argument('--project_id', '-pid', type=int, default=DEFAULT_PROJECT_ID, help='The project ID (an integer)')
    parser.add_argument('--stand_ids', '-sids', type=int, nargs='+', default=DEFAULT_STAND_IDS, help='A list of stand IDs (integers)')
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUTPUT, help='Output file name (a path)')
    parser.add_argument('--epochs', '-e', type=int, default=DEFAULT_EPOCHS, help='Number of epochs (an integer)')

    args = parser.parse_args()

    paths = fetch_paths(args.client_id, args.project_id, args.stand_ids)
    format_for_rv(paths, args.output, args.epochs)