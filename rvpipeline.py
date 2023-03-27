from os.path import join
from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def load_yaml(path):
    with open(path, "r") as fp:
        return load(fp, Loader=Loader)

def get_config(runner, **kw) -> ObjectDetectionConfig:
    input_config = load_yaml(kw["yaml"])
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
        train_scenes=[make_scene(*stand,class_config) for stand in training_list],
        validation_scenes=[make_scene(*stand,class_config) for stand in validation_list]
    )

    chip_sz = input_config["chip_sz"]
    img_sz = input_config["img_sz"]

    chip_options = ObjectDetectionChipOptions(**input_config["chip_options"])
    
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
            augmentors=[])
    
    predict_options = ObjectDetectionPredictOptions(
        **input_config["predict_options"]
    )
    
    backend = PyTorchObjectDetectionConfig(
        data=data,
        model=ObjectDetectionModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(**input_config["solver_config"]),
        log_tensorboard=True,
        run_tensorboard=False)

    return ObjectDetectionConfig(
        root_uri=output_root_uri,
        dataset=scene_dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options,
        predict_options=predict_options)


def make_scene(image_uri: str, label_uri: str, aoi_uri: str,
               class_config: ClassConfig) -> SceneConfig:
    """Define a Scene with image and labels from the given URIs."""
    scene_id = label_uri.split('/')[-3]
    raster_source = RasterioSourceConfig(
            uris=[image_uri], channel_order=[0, 1, 2])
    # configure transformation of vector data into Object Detection labels
    label_source = ObjectDetectionLabelSourceConfig(
        # object detection labels must be rasters, so rasterize the geoms
        vector_source=GeoJSONVectorSourceConfig(
                uri=label_uri,
                ignore_crs_field=True,
                transformers=[
                   ClassInferenceTransformerConfig(default_class_id=0)]
        )
    )
    return SceneConfig(
        id=scene_id,
        raster_source=raster_source,
        label_source=label_source,
        aoi_uris=[aoi_uri]
    )