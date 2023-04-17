import matplotlib as plt
plt.use("agg")
from rastervision.core.data import ClassConfig, DatasetConfig, SceneConfig
from rastervision.core.data.raster_source.rasterio_source_config import RasterioSourceConfig
from rastervision.core.rv_pipeline.object_detection_config import ObjectDetectionPredictOptions
from rastervision.pytorch_learner import ObjectDetectionGeoDataWindowConfig
from rastervision.pytorch_backend.pytorch_object_detection_config import PyTorchObjectDetectionConfig
from rastervision.core.data.label_source.object_detection_label_source_config import ObjectDetectionLabelSourceConfig
from rastervision.core.data.vector_source.geojson_vector_source_config import GeoJSONVectorSourceConfig
from rastervision.core.data.vector_transformer.class_inference_transformer_config import ClassInferenceTransformerConfig
from rastervision.pytorch_learner.learner_config import (
    GeoDataWindowMethod,
    SolverConfig,
    Backbone,
    ExternalModuleConfig
)
from rastervision.core.rv_pipeline import (
    ObjectDetectionConfig,
    ObjectDetectionChipOptions,
)
from rastervision.pytorch_learner.object_detection_learner_config import (
    ObjectDetectionGeoDataConfig,
    ObjectDetectionModelConfig
)
import os
import time
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import geopandas
import albumentations as A

def load_yaml(kw):
    ''' helper function to load YAML config file and return contents as dict '''
    path = kw["config"]
    with open(path, "r") as fp:
        return load(fp, Loader=Loader)
    
def set_env(input_config):
    ''' Set environment variables (if any) given in config YAML file'''
    envvars = input_config.get("env", {})
    if not envvars.get("apply", False):
        return
    for k, v in envvars.items():
        os.environ[k] = str(v)

def _shp_to_geojson(file_group_list):
    ''' Given an input list of files, convert any SHP files to GeoJSON and replace
     their SHP path with the GeoJSON path in the list '''
    for k, group in enumerate(file_group_list):
        for i, file in enumerate(group):
            if file.split(".")[-1] != "shp":
                continue
            fileshp = geopandas.read_file(file).dropna().to_crs("EPSG:4326")
            jsonpath = file.split(".")[0] + ".geojson"
            print(fileshp, "->", jsonpath)
            fileshp.to_file(jsonpath, driver="GeoJSON", crs="EPSG:4326")
            group[i] = jsonpath
        file_group_list[k] = group
    return file_group_list

def shp_to_geojson(input_config):
    ''' Convert training and validation SHP files to GeoJSON '''
    input_config["training_list"] = _shp_to_geojson(input_config["training_list"])
    input_config["validation_list"] = _shp_to_geojson(input_config["validation_list"])
    return input_config

def pre_pipeline(kw):
    ''' Run the pre-pipeline steps '''
    cfg = load_yaml(kw)
    set_env(cfg)
    cfg = shp_to_geojson(cfg)
    return cfg

def get_model_cfg(bbkw, class_config):
    ''' Return instance of model defined in config YAML file'''
    src = bbkw["source"]
    if src == "rastervision":
        name = bbkw["name"]
        backbone = getattr(Backbone, name)
        return ObjectDetectionModelConfig(backbone=backbone)
    elif src == "external":
        external_def = ExternalModuleConfig(
            github_repo=bbkw["github_repo"],
            name=bbkw["name"],
            entrypoint=bbkw["entrypoint"],
            force_reload=bbkw.get("force_reload", True),
            entrypoint_kwargs={
                "num_classes": len(class_config) + 1,
                "pretrained": bbkw.get("pretrained", False),
                "pretrained_backbone": bbkw.get("pretrained_backbone", True)
            }
        )
        return ObjectDetectionModelConfig(external_def=external_def)
    else:
        raise NotImplementedError("valid Backbone sources: {rastervision, external}")

def make_scene(image_uri: str, label_uri: str, aoi_uri=None) -> SceneConfig:
    '''' Define a Scene with image and labels from the given URIs. '''
    scene_id = label_uri.split('/')[-3]+str(time.time())
    raster_source = RasterioSourceConfig(
            uris=[image_uri], channel_order=[0, 1, 2])
    label_source = ObjectDetectionLabelSourceConfig(
        vector_source=GeoJSONVectorSourceConfig(
                uri=label_uri,
                ignore_crs_field=True,
                transformers=[ClassInferenceTransformerConfig(default_class_id=0)]
        )
    )
    aoi_uris = [aoi_uri] if aoi_uri is not None else []
    return SceneConfig(
        id=scene_id,
        raster_source=raster_source,
        label_source=label_source,
        aoi_uris=aoi_uris
    )

def get_config(runner, **kw) -> ObjectDetectionConfig:
    input_config = pre_pipeline(kw)
    output_root_uri = input_config["output_root_uri"]
    class_config = ClassConfig(
        names=input_config["class_config"]["names"],
        colors=input_config["class_config"]["colors"]
    )
    
    #list of tuples where each tuple is (image_uri, label_uri, aoi_uri)
    training_list = input_config["training_list"]
    validation_list = input_config["validation_list"]
    
    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene(*stand) for stand in training_list],
        validation_scenes=[make_scene(*stand) for stand in validation_list]
    )

    chip_options = ObjectDetectionChipOptions(**input_config["chip_options"])
    
    chip_sz = input_config["window_config"]["chip_sz"]
    img_sz = input_config["window_config"]["img_sz"]    
    window_opts = ObjectDetectionGeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=chip_sz,
            size_lims=(chip_sz, chip_sz + 1),
            max_windows=input_config["window_config"]["max_windows"],
            clip=True,
            neg_ratio=chip_options.neg_ratio,
            ioa_thresh=chip_options.ioa_thresh)

    data = ObjectDetectionGeoDataConfig(
            scene_dataset=scene_dataset,
            window_opts=window_opts,
            img_sz=img_sz,
            num_workers=input_config["num_workers"],
            #base_transform=A.to_dict(A.flip()),
            #aug_transform={},
            augmentors=['RGBShift', 'RandomRotate90', 'HorizontalFlip', 'VerticalFlip']#'GaussianBlur', 'GaussNoise', 'Blur',
            )

    predict_options = ObjectDetectionPredictOptions(
        **input_config["predict_options"]
    )

    model_cfg = get_model_cfg(input_config["backbone_config"], class_config)
    
    backend = PyTorchObjectDetectionConfig(
        data=data,
        model=model_cfg,
        solver=SolverConfig(**input_config["solver_config"]),
        log_tensorboard=input_config["tensorboard"]["log"],
        run_tensorboard=input_config["tensorboard"]["run"]
    )

    return ObjectDetectionConfig(
        root_uri=output_root_uri,
        dataset=scene_dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options,
        predict_options=predict_options)
