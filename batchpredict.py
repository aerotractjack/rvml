import os
from os.path import join
import pathlib
import subprocess

def batch_predict():
    #tifpathlist.txt might need to be command line arg later
    with open('tifpathlist.txt','r') as text:
        prediction_list = [line.rstrip('\n') for line in text]

    bundle_uri = '/home/aerotract/GoBag/Migrate/dev/PreTrainedModels/DFModels/ThirdYearDF/Resnet50/bundle/model-bundle.zip'


    for number,tif in enumerate(prediction_list):
        
        print(f'starting project {number+1} of {len(prediction_list)}')
        
        #create output directory
        
        
        output_directory = os.path.join(pathlib.Path(tif).parent.parent.parent,'Modeling','products.json')
        print(output_directory)
        print(tif)
        print(bundle_uri)

        subprocess.run(['rastervision','predict',bundle_uri,tif,output_directory])

        
if __name__ == "__main__":
    batch_predict()