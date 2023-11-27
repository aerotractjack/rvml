import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
from geopy.distance import geodesic
from geopy.point import Point as GeoPoint
import os
from pathlib import Path
import time

def enforce_grid(input_path: str, output_path: str):
    """
    Function to enforce grid distances between the polygons.

    Parameters:
    - input_path (str): Path to the input GeoJSON file
    - output_path (str): Path where the updated GeoJSON will be saved

    Returns:
    None
    """
    # Load the GeoJSON file into a GeoDataFrame
    gdf = gpd.read_file(input_path)
    gdf = gdf.to_crs(epsg=4326)
   
    # Compute centroids for each polygon
    gdf['centroid'] = gdf['geometry'].centroid

    to_remove = []

    # Loop over each polygon's centroid
    for idx, row in tqdm(gdf.iterrows()):
        # Compute the distance from the current centroid to all other centroids
        distances = gdf['centroid'].distance(row['centroid'])
       
        # Sort the distances (excluding the distance to itself which will be zero)
        nearest_idxs = distances.nsmallest(2).index[1:]
       
        # If the distance to the nearest centroid is less than 2.1 meters
        point = Point(gdf["centroid"][nearest_idxs].iloc[0].y,gdf["centroid"][nearest_idxs].iloc[0].x)
        ref = Point(gdf["centroid"][idx].y,gdf["centroid"][idx].x)
       
        # Extracting coordinates from Shapely Point objects
        point_coords = (point.x, point.y)
        ref_coords = (ref.x, ref.y)

        # Calculate the distance in meters using geodesic
        distance = geodesic(point_coords, ref_coords).meters
        if distance < 2.1:
            # Compare the scores and remove the polygon with the lower score
            current_score = row['score']
            nearest_score = gdf.loc[nearest_idxs[0], 'score']
           
            if current_score <= nearest_score:
                to_remove.append(idx)
            else:
                to_remove.append(nearest_idxs[0])

    # Drop polygons with lower scores based on the conditions above
    gdf.drop(index=to_remove, inplace=True)
    gdf = gdf.drop(columns=['centroid'])

    # Save the updated GeoDataFrame to the specified output path
    gdf.to_file(output_path, driver='GeoJSON')

if __name__ == "__main__":
    #tifpathlist.txt might need to be command line arg later
    with open('tifpathlist.txt','r') as text:
        prediction_list = [line.rstrip('\n') for line in text]

    
    for number,tif in enumerate(prediction_list):
        
        print(f'starting project {number+1} of {len(prediction_list)}')
        
        split_path = tif.split('Data')
        output_directory = tif
        
        print(output_directory)
        
        #enforce_grid
        enforce_output = os.path.join(Path(output_directory).parent,"enforced_grid.geojson")
        wait = os.path.exists(output_directory)
        while not wait:
            time.sleep(10)
            wait = os.path.exists(output_directory)

        enforce_grid(output_directory,enforce_output)