from rastervision.core.data import ClassConfig
from rastervision.pytorch_learner import (
    ObjectDetectionGeoDataConfig,
    ObjectDetectionRandomWindowGeoDataset,
    ObjectDetectionSlidingWindowGeoDataset,
    SolverConfig,
    ObjectDetectionLearner,
    ObjectDetectionLearnerConfig)
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from rastervision.pytorch_learner.object_detection_utils import TorchVisionODAdapter
from rastervision.core.data import ObjectDetectionLabels
import time
import os

class RVMLTrainer:

    def __init__(self, files, bundle):
        self.tr_img = files.get("train_image_uri")
        self.tr_aoi = files.get("train_aoi_uri")
        self.tr_lab = files.get("train_label_uri")
        self.val_img = files.get("val_image_uri")
        self.val_aoi = files.get("val_aoi_uri")
        self.val_lab = files.get("val_label_uri")
        self.pred_img = files.get("pred_image_uri")
        self.pred_aoi = files.get("pred_aoi_uri")
        self.pred_lab = files.get("pred_label_uri")
        self.bundle = bundle
        self.cc = None
        self.default_class_id = None
        self.trds = None
        self.vds = None
        self.pds = None
        self.learner = None

    def create_class_config(self, names, colors, default=None):
        cc = ClassConfig(names=names, colors=colors)
        self.cc = cc
        if default is None:
            self.default_class_id = self.cc.get_class_id(names[-1])
        if isinstance(default, int):
            self.default_class_id = default
        elif isinstance(default, str):
            self.default_class_id = self.cc.get_class_id(default)

    def create_train_ds(self, **kw):
        trds = ObjectDetectionRandomWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=self.tr_img,
            aoi_uri=self.tr_aoi,
            label_vector_uri=self.tr_lab,
            label_vector_default_class_id=self.default_class_id,
            size_lims=kw.pop("size_lims", (550,600)),
            out_size=kw.pop("out_size", 256),
            max_windows=kw.pop("max_windows", None),
            **kw
        )
        self.trds = trds
        return trds

    def create_val_ds(self, **kw):
        vds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=self.val_img,
            aoi_uri=self.val_aoi,
            label_vector_uri=self.val_lab,
            label_vector_default_class_id=self.default_class_id,
            size=kw.pop("size", 325),
            stride=kw.pop("stride", 325),
            **kw
        )
        self.vds = vds
        return vds
    
    def create_pred_ds(self, **kw):
        pds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
            class_config=self.cc,
            image_uri=self.pred_img,
            size=kw.pop("size", 325),
            stride=kw.pop("stride", 10000),
            **kw
        )
        self.pds = pds
        return pds
    
    def build_model(self):
        model = fasterrcnn_resnet50_fpn_v2(num_classes=len(self.cc)+1)
        model = TorchVisionODAdapter(model)
        return model

    def create_learner(self, **kw):
        data_cfg = ObjectDetectionGeoDataConfig(
            class_names=self.cc.names,
            class_colors=self.cc.colors,
            num_workers=0,
        )
        solver_cfg = SolverConfig(
            batch_sz=kw.get("solver_cfg_kw", {}).pop("batch_sz", 2),
            lr=kw.get("solver_cfg_kw", {}).pop("lr", 3e-2),
            **kw.pop("solver_cfg_kw", {})
        )
        learner_cfg = ObjectDetectionLearnerConfig(
            data=data_cfg,
            solver=solver_cfg,
            **kw.pop("learner_cfg_kw", {}),
        )
        model = self.build_model()
        learner = ObjectDetectionLearner(
            cfg=learner_cfg,
            output_dir=kw.pop("output", self.bundle),
            model=model,
            train_ds=self.trds,
            valid_ds=self.vds,
            **kw.pop("learner_kw", {})
        )
        self.learner = learner
        return learner
    
    def train(self, **kw):
        self.learner.log_data_stats()
        self.learner.train(**kw.pop("train_kw", {}))
        self.learner.save_model_bundle()

    def predict(self, **kw):
        model = self.build_model()
        predictor = ObjectDetectionLearner.from_model_bundle(
            os.path.join(self.bundle, "model-bundle.zip"),
            training=False,
            model=model,
            output_dir="./test-pred-out"
        )
        pred_generator = predictor.predict_dataset(
            self.pds,
            raw_out=True,
            numpy_out=True,
            progress_bar=True,
        )
        pred_labels = ObjectDetectionLabels.from_predictions(
            self.pds.windows,
            pred_generator,
        )
        pred_labels.save("preds.json", crs_transformer=self.pds.scene.raster_source.crs_transformer, class_config=self.cc)
        return pred_labels
        

if __name__ == "__main__":
    files = {
        "train_image_uri": "data/input.tif", 
        "train_label_uri": "data/labels.json", 
        "train_aoi_uri": "data/aoi.json", 
        "val_image_uri": "data/input.tif", 
        "val_label_uri": "data/labels.json", 
        "val_aoi_uri": "data/aoi.json", 
        "pred_image_uri": "data/input.tif", 
    }
    bundle = f"outputs/{time.monotonic():0.0f}-train"

    ml = RVMLTrainer(files, bundle)

    print("creating class config...")
    ml.create_class_config(["DF"], ["red"])
    print("creating train ds...")
    ml.create_train_ds(max_windows=5)
    print("creating val ds...")
    ml.create_val_ds()
    print("creating pred ds...")
    ml.create_pred_ds()
    print("creating learner...")
    ml.create_learner()
    print("training...")
    ml.train(train_kw={"epochs": 1})
    print("predicting...")
    labels = ml.predict()
    print(labels)