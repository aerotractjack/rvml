from os.path import join
from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *


def get_config(runner) -> ObjectDetectionConfig:
    output_root_uri = '/home/aerotract/RasterVision/RunModelWithScenesOutput'
    class_config = ClassConfig(
        names=['DF'], colors=['red'])
    
    #list of tuples where each tuple is (image_uri, label_uri, aoi_uri)
    training_list = [
                    
                        # ('/home/aerotract/NAS/main/Clients/Giustina/Winter2023/103_WardsButteSx/03_WardsButteSx_02122023/Data/ortho/03WardsButteSxGR02122023_Orthomosaic_WedMar08194203915827/03WardsButteSxGR02122023_Orthomosaic_export_WedMar08194203915827.tif',
                        # '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/103_WardsButteSx/03_WardsButteSx_02122023/Training/WardsButte_trees_poly.geojson',
                        # '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/103_WardsButteSx/03_WardsButteSx_02122023/Training/WardsButte_sample_boundary.geojson')
                        # ,
                        # ('/home/aerotract/NAS/main/Clients/Giustina/Winter2023/104_FinalSalem/04_FinalSalem_02122023/Data/ortho/04FinalSalemGR02122023_Orthomosaic_WedMar08195655345878/04FinalSalemGR02122023_Orthomosaic_export_WedMar08195655345878.tif',
                        #  '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/104_FinalSalem/04_FinalSalem_02122023/Training/FinalSalem_tree_polys.geojson',
                        #  '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/104_FinalSalem/04_FinalSalem_02122023/Training/finalSalem_sample_Boundary.geojson')
                        # ,
                        # ('/home/aerotract/NAS/main/Clients/Giustina/Winter2023/105_WinStormGTF/05_WinStormGTF_02122023/Data/ortho/05WinstormGR02122023_Orthomosaic_WedMar08201410773708/05WinstormGR02122023_Orthomosaic_export_WedMar08201410773708.tif',
                        #  '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/105_WinStormGTF/05_WinStormGTF_02122023/Training/Tree_poly.geojson',
                        #  '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/105_WinStormGTF/05_WinStormGTF_02122023/Training/Sample_Boundary.geojson')
                        # ,
                        # ('/home/aerotract/NAS/main/Clients/Giustina/Winter2023/106_Winfire/06_Winfire_02122023/Data/ortho/06WinfireGR02122023_Orthomosaic_WedMar08213557063397/06WinfireGR02122023_Orthomosaic_export_WedMar08213557063397.tif',
                        #  '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/106_Winfire/06_Winfire_02122023/Training/Tree_poly.geojson',
                        #  '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/106_Winfire/06_Winfire_02122023/Sample_Boundary.geojson')
                        
                        # ('/home/aerotract/NAS/main/Clients/Giustina/Winter2023/107_Pitfall/07_Pitfall_02122023/Data/ortho/07PitfallGR02122023_Orthomosaic_MonMar13183928000094/07PitfallGR02122023_Orthomosaic_export_MonMar13183928000094.tif',
                        #  '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/107_Pitfall/07_Pitfall_02122023/Training/Trees.geojson',
                        #  '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/107_Pitfall/07_Pitfall_02122023/Training/Boundary.geojson')
                        # ,
                        ('/home/aerotract/NAS/main/Clients/Giustina/Winter2023/108_Moffit/08_Moffit_0212023/Data/ortho/08MoffitGR02122023_Orthomosaic_export_FriMar10183649417525.tif',
                         '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/108_Moffit/08_Moffit_0212023/Training/Trees.geojson',
                         '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/108_Moffit/08_Moffit_0212023/Training/Boundary.geojson' )
                    ]
    
    
    validation_list = [
                        ('/home/aerotract/NAS/main/Clients/Giustina/Winter2023/100_230OG/00_230OG_02022023/Data/ortho/230OG_Orthomosaic_WedMar08181810258552/230OG_Orthomosaic_export_WedMar08181810258552.tif',
                         '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/100_230OG/00_230OG_02022023/Training/Tree_polygons_00_230OG.geojson',
                         '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/100_230OG/00_230OG_02022023/Training/border_polygon_00_230OG.geojson')
                        
                        # ,('/home/aerotract/NAS/main/Clients/Giustina/Winter2023/112_Wiley520Junction/12_Wiley520Junction_02022023/Data/ortho/Wiley520Junction_Orthomosaic_export_MonMar13192503043098.tif',
                        #  '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/112_Wiley520Junction/12_Wiley520Junction_02022023/Training/tree_points.geojson',
                        #  '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/112_Wiley520Junction/12_Wiley520Junction_02022023/Training/Boundary.geojson')
                      ]
    
    # val_image_uri = '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/100_230OG/00_230OG_02022023/Data/ortho/230OG_Orthomosaic_WedMar08181810258552/230OG_Orthomosaic_export_WedMar08181810258552.tif'
    # val_aoi_uri = '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/100_230OG/00_230OG_02022023/Training/border_polygon_00_230OG.geojson'
    # val_label_uri = '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/100_230OG/00_230OG_02022023/Training/Tree_polygons_00_230OG.geojson'

    # train_image_uri = '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/103_WardsButteSx/03_WardsButteSx_02122023/Data/ortho/03WardsButteSxGR02122023_Orthomosaic_WedMar08194203915827/03WardsButteSxGR02122023_Orthomosaic_export_WedMar08194203915827.tif'
    # train_aoi_uri = '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/103_WardsButteSx/03_WardsButteSx_02122023/Training/WardsButte_sample_boundary.geojson'
    # train_label_uri = '/home/aerotract/NAS/main/Clients/Giustina/Winter2023/103_WardsButteSx/03_WardsButteSx_02122023/Training/WardsButte_trees_poly.geojson'

    # train_scene = make_scene(train_image_uri, train_label_uri,train_aoi_uri,
    #                          class_config)
    # val_scene = make_scene(val_image_uri, val_label_uri,val_aoi_uri,
    #                        class_config)
    
    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene(*stand,class_config) for stand in training_list],
        validation_scenes=[make_scene(*stand,class_config) for stand in validation_list])

    # Use the PyTorch backend for the SemanticSegmentation pipeline.
    chip_sz = 450
    img_sz = chip_sz

    chip_options = ObjectDetectionChipOptions(neg_ratio=1.0, ioa_thresh=0.8)
    
    window_opts = ObjectDetectionGeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=chip_sz,
            size_lims=(chip_sz, chip_sz + 1),
            max_windows=450,
            clip=True,
            neg_ratio=chip_options.neg_ratio,
            ioa_thresh=chip_options.ioa_thresh)

    data = ObjectDetectionGeoDataConfig(
            scene_dataset=scene_dataset,
            window_opts=window_opts,
            img_sz=img_sz,
            num_workers = 16,
            augmentors=[])
    
    predict_options = ObjectDetectionPredictOptions(
        merge_thresh=0.1, score_thresh=0.8)
    

    backend = PyTorchObjectDetectionConfig(
        data=data,
        model=ObjectDetectionModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=10,
            test_num_epochs=2,
            batch_sz=8,
            one_cycle=True),
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
                ))

    return SceneConfig(
        id=scene_id,
        raster_source=raster_source,
        label_source=label_source,
        aoi_uris=[aoi_uri]
    )