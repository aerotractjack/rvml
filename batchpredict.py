import os
from os.path import join
import pathlib
from pprint import pprint
import yaml
import tqdm
import rastervision.pipeline
from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import (get_scene_info,save_image_crop)
from torch.utils.data import DataLoader
import albumentations as A
import torch 
from rastervision.pytorch_learner import (
    ObjectDetectionRandomWindowGeoDataset,
    ObjectDetectionSlidingWindowGeoDataset,
    ObjectDetectionVisualizer)

#tifpathlist.txt might need to be command line arg later
with open('tifpathlist.txt','r') as text:
    prediction_list = [line.rstrip('\n') for line in text]

bundle_uri = '/home/aerotract/RasterVision/RunModelWithScenesOutput/train/model-bundle.zip'

with open('input.yaml') as yamlconfig:
    configYaml = yaml.load(yamlconfig,Loader=yaml.Loader)
#pprint(configYaml)
    

class_config = ClassConfig(names=configYaml['class_config']['names'], colors=configYaml['class_config']['colors'])



for number,tif in enumerate(prediction_list):
    
    print(f'starting project {number} of {len(prediction_list)}')
    
    #create output directory
    output_directory = os.path.join(pathlib.Path(tif).parent.parent.parent.parent,'Modeling')
    print(output_directory)
    
    print("creating learner....")
    learner = ObjectDetectionLearner.from_model_bundle(model_bundle_uri=bundle_uri,
                                                    training = False,
                                                    output_dir = output_directory,
                                                    )
    
    print("creating prediction dataset...")
    Dataset = ObjectDetectionSlidingWindowGeoDataset.from_uris(
        class_config=class_config,
        image_uri=tif,
        size=configYaml['window_config']['img_sz'],
        stride=configYaml['window_config']['img_sz'],
        transform=A.Resize(configYaml['window_config']['max_windows'], configYaml['window_config']['max_windows']))

    print("starting prediction.....")
    predictions = learner.predict_dataset(dataset= Dataset,numpy_out=True,progress_bar=True)

    pred_labels = ObjectDetectionLabels.from_predictions(Dataset.windows,
                                                        predictions)

    pred_labels.save(
        uri=output_directory,
        crs_transformer=Dataset.scene.raster_source.crs_transformer,
        class_config=class_config)