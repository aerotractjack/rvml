import os
from os.path import join
import pathlib
import subprocess

def batch_predict():
    #tifpathlist.txt might need to be command line arg later
    with open('tifpathlist.txt','r') as text:
        prediction_list = [line.rstrip('\n') for line in text]

    bundle_list = [#'/home/aerotract/GoBag/Migrate/dev/PreTrainedModels/DFModels/ThirdYearDF/Resnet18/bundle/model-bundle.zip',
                    #'/home/aerotract/GoBag/Migrate/dev/PreTrainedModels/DFModels/ThirdYearDF/Resnet34/bundle/model-bundle.zip', 
                    '/home/aerotract/GoBag/Migrate/dev/PreTrainedModels/DFModels/ThirdYearDF/Resnet50_10Epoch/bundle/model-bundle.zip'
                    #'/home/aerotract/GoBag/Migrate/dev/PreTrainedModels/DFModels/ThirdYearDF/Resnet152/bundle/model-bundle.zip',
                    ]

    for bundle_uri in bundle_list:
        for number,tif in enumerate(prediction_list):
            
            print(f'starting project {number+1} of {len(prediction_list)}')
            
            model_name = bundle_uri.split('/')[-3]
            print(model_name)
            split_path = tif.split('/')
            output_directory = os.path.join('/',split_path[0],split_path[1],split_path[2],split_path[3],split_path[4],split_path[5],split_path[6],split_path[7],'Modeling',model_name+'-products.json')
            
            print(output_directory)
            print(tif)
            print(bundle_uri)

            subprocess.run(['rastervision','predict',bundle_uri,tif,output_directory])

        
if __name__ == "__main__":
    batch_predict()