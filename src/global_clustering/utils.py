import numpy as np


def get_surrounding_similar_cluster_percentage(df_slide, row, col, c):
    tot_equal = 0
    tot_surrounding = 0.0
    surrounding_offsets = ((0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1))
    for offset in surrounding_offsets:
        row += offset[0]
        col += offset[1]
        try:
            tot_equal += c == df_slide.loc[row, col]
            tot_surrounding += 1
        except KeyError:
            pass
    return tot_equal/tot_surrounding


def get_tile_and_slide_avg_clusters_alignment(df_metadata):
    slide_avgs = []
    tile_avgs = []
    for slide_uuid, df_slide in df_metadata.groupby('slide_uuid'):
        df_slide.set_index(['row', 'col'], inplace=True)
        for (row, col), c in df_slide['c'].iteritems():
            p = get_surrounding_similar_cluster_percentage(df_slide, row, col ,c)
            tile_avgs.append(p)
        slide_avgs.append(np.mean(tile_avgs[:-len(df_slide)]))
    return slide_avgs, tile_avgs














