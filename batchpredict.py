import os
from os.path import join
import pathlib
import subprocess
import cProfile
import time
from enforce_grid import enforce_grid
from pathlib import Path
import subprocess
import os
from pathlib import Path

def downsample_tif(input_path, output_path):
    try:
        subprocess.run([
            'gdal_translate',
            '-outsize', '50%', '50%',
            input_path,
            output_path
        ], check=True)
        print(f'Successfully downsampled {input_path} to {output_path}')
    except subprocess.CalledProcessError as e:
        print(f'Error downsampling {input_path}: {e}')

def generate_output_path(input_path):
    # Example: generate output path by appending '_downsampled' to the filename
    directory = Path(input_path).parent
    file_name = Path(input_path).name
    new_file_name = file_name.split(".")[0] + "_downsampled.tif"
    output_filename = os.path.join(directory,new_file_name)
    return output_filename

def main():
    input_file = 'tif_path_list.txt'
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        input_path = line.strip()
        if input_path:
            output_path = generate_output_path(input_path)
            downsample_tif(input_path, output_path)

def batch_predict():
    #tifpathlist.txt might need to be command line arg later
    with open('tifpathlist.txt','r') as text:
        prediction_list = [line.rstrip('\n') for line in text]

    bundle_list = [#'/home/aerotract/GoBag/Migrate/dev/PreTrainedModels/DFModels/ThirdYearDF/Resnet18/bundle/model-bundle.zip',
                    #'/home/aerotract/GoBag/Migrate/dev/PreTrainedModels/DFModels/ThirdYearDF/Resnet34/bundle/model-bundle.zip', 
                    # '/home/aerotract/NAS/main/ml_storage/PreTrainedModels/DFModels/FiveYearDF/Resnet152_2/bundle/model-bundle.zip',
                    # '/home/aerotract/NAS/main/ml_storage/PreTrainedModels/DFModels/FiveYearDF/Resnet152_2/bundle/model-bundle.zip',
                    '/home/aerotract/NAS/main/ml_storage/PreTrainedModels/potlatch_multiclass_spring2024_NEW/bundle/model-bundle.zip'
                    ]

    for bundle_uri in bundle_list:
        for number,tif in enumerate(prediction_list):
            
            print(f'starting project {number+1} of {len(prediction_list)}')
            
            model_name = bundle_uri.split('/')[-3]
            split_path = tif.split('Data')
            output_directory = os.path.join('/',split_path[0],'Modeling',model_name+'-products'+str(time.time())+".json")
            
            print(f"downsampling {tif}")
            tif_downsampled = generate_output_path(tif)
            downsample_tif(tif, tif_downsampled)
            print(output_directory)
            print(tif_downsampled)
            print(bundle_uri)

            print(f"processing {tif_downsampled}")
            subprocess.run(['rastervision','predict',bundle_uri,tif_downsampled,output_directory])
            
            enforce_grid
            enforce_output = os.path.join(Path(output_directory).parent,"enforced_grid.geojson")
            new_enforce_output = os.path.join(Path(output_directory).parent,"enforced_grid2.geojson")

            wait = os.path.exists(output_directory)
            while not wait:
                
                time.sleep(10)
                wait = os.path.exists(output_directory)

            enforce_grid(output_directory,enforce_output)
            enforce_grid(enforce_output,new_enforce_output)

     
if __name__ == "__main__":
    batch_predict()
