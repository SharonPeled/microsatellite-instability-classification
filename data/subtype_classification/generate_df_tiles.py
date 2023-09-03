from glob import glob 
import pandas as pd 
import os
from src.configs import Configs


tile_sizes = [224, 512, 1024]
for tile_size in tile_sizes:
    print(f'Starting tile size {tile_size}.')
    paths = glob(f'{Configs.DATA_FOLDER}/processed_tiles_{tile_size}/*/*.jpg')
    print(f'Found {len(paths)} tiles.')
    df_tiles = pd.DataFrame({'tile_path': paths})
    df_tiles['slide_uuid'] = df_tiles.tile_path.apply(lambda p: os.path.basename(os.path.dirname(p)))
    df_mc = pd.read_csv('df_manifest_all_dx_cohort_combined.tsv', sep='\t')
    df_tiles_merged = df_tiles.merge(df_mc, how='inner', left_on='slide_uuid', right_on='id')
    print(f'After merge: {len(df_tiles_merged)} tiles.')
    if tile_size != 224:
        df_tiles_merged.to_csv(f'df_processed_tile_paths_{tile_size}.csv')
    else:
        df_tiles_merged.to_csv(f'df_processed_tile_paths_{tile_size}_reduced.csv')
print('Finished')












