import os
from os.path import join
import pathlib
import subprocess
import cProfile
import time
from enforce_grid import enforce_grid
from pathlib import Path

def batch_predict():
    #tifpathlist.txt might need to be command line arg later
    with open('tifpathlist.txt','r') as text:
        prediction_list = [line.rstrip('\n') for line in text]

    bundle_list = [#'/home/aerotract/GoBag/Migrate/dev/PreTrainedModels/DFModels/ThirdYearDF/Resnet18/bundle/model-bundle.zip',
                    #'/home/aerotract/GoBag/Migrate/dev/PreTrainedModels/DFModels/ThirdYearDF/Resnet34/bundle/model-bundle.zip', 
                    # '/home/aerotract/NAS/main/ml_storage/PreTrainedModels/DFModels/FiveYearDF/Resnet152_2/bundle/model-bundle.zip',
                    # '/home/aerotract/NAS/main/ml_storage/PreTrainedModels/DFModels/FiveYearDF/Resnet152_2/bundle/model-bundle.zip',
                    '/home/aerotract/NAS/main/ml_storage/PreTrainedModels/Siskiyou_3/bundle/model-bundle.zip'
                    ]

    for bundle_uri in bundle_list:
        for number,tif in enumerate(prediction_list):
            
            print(f'starting project {number+1} of {len(prediction_list)}')
            
            model_name = bundle_uri.split('/')[-3]
            split_path = tif.split('Data')
            output_directory = os.path.join('/',split_path[0],'Modeling',model_name+'-products'+str(time.time())+".json")
            
            print(output_directory)
            print(tif)
            print(bundle_uri)

            subprocess.run(['rastervision','predict',bundle_uri,tif,output_directory])
            
            #enforce_grid
            enforce_output = os.path.join(Path(output_directory).parent,"enforced_grid.geojson")
            wait = os.path.exists(output_directory)
            while not wait:
                time.sleep(10)
                wait = os.path.exists(output_directory)

            enforce_grid(output_directory,enforce_output)

        
if __name__ == "__main__":
    batch_predict()