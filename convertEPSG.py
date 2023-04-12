import os

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def convert_tif_3857_to_4326(input_path, output_path):
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'EPSG:4326',
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.nearest)

if __name__ == "__main__":
    
    tif_list = ['/home/aerotract/NAS/Clients/Weyerhauser/Springfield_Spring2022/P22-749-P1/P22-749-P1_03282022/Data/ortho/WCSF08903282022P1.tif',
                '/home/aerotract/NAS/Clients/Weyerhauser/Springfield_Spring2022/P22-748-P1/P22-748-P1_03242022/Data/ortho/WCSF08803242022P1.tif',
                '/home/aerotract/NAS/Clients/Weyerhauser/Springfield_Spring2022/P22-746-P1/P22-746-P1_03242022/Data/ortho/WCSF08703242022P1.tif',
                '/home/aerotract/NAS/Clients/Weyerhauser/Springfield_Spring2022/P22-659-P1/P22-659-P1_03312022/Data/ortho/WCSF09403312022P1.tif',
                '/home/aerotract/NAS/Clients/Weyerhauser/Springfield_Spring2022/P22-651-P1/P22-651-P1_05172022/Data/ortho/WCSF11105172022P1.tif',
                '/home/aerotract/NAS/Clients/Weyerhauser/Springfield_Spring2022/P22-637-P1/P22-637_P1_05232022/Data/ortho/WCSF11605232022P1.tif'
                ]

    for tif in tif_list:
        output_tif,tail = os.path.split(tif)
        tail = tail.split('.')[0]
        output_tif = os.path.join('/home/aerotract/GoBag/Migrate/dev/DataBase/FirstYearDF',tail+'4326'+'.tif')
        print(output_tif)
        convert_tif_3857_to_4326(tif,output_tif)