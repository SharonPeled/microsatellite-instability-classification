import pandas as pd
import numpy as np
from src.configs import Configs


def reduction_with_limit(df, col, limit=1000):
    limit = int(max(len(df)*0.8, limit))
    return df.loc[df[col].nlargest(limit).index]


r_per_slide = {}


def reduction_to_target(df, col, target=1500):
    slide_uuid = df.slide_uuid.iloc[0]
    if slide_uuid not in r_per_slide.keys():
        #  first time
        if len(df) <= target:
            r_per_slide[slide_uuid] = 1.0
        else:
            e = 5  # number of epochs with reduction
            a = len(df)  # original value
            r = (target / a) ** (1 / e)
            r_per_slide[slide_uuid] = r
    r = r_per_slide[slide_uuid]
    limit = int(max(len(df) * r, target))
    return df.loc[df[col].nlargest(limit).index]


def reduction_to_target_per_class(df, col, target=1000):
    slide_uuid = df.slide_uuid.iloc[0]
    y = 1 if df.subtype.iloc[0] == 'CIN' else 0
    if slide_uuid not in r_per_slide.keys():
        #  first time
        if len(df) <= target:
            r_per_slide[slide_uuid] = 1.0
        else:
            e = 2  # number of epochs with reduction
            a = len(df)  # original value
            r = (target / a) ** (1 / e)
            r_per_slide[slide_uuid] = r
    r = r_per_slide[slide_uuid]
    limit = int(max(len(df) * r, target))
    if y == 1:
        return df.loc[df[col].nlargest(limit).index]
    else:
        return df.loc[df[col].nsmallest(limit).index]


def reduction_to_target_per_end(df, col, target=750):
    slide_uuid = df.slide_uuid.iloc[0]
    y = 1 if df.subtype.iloc[0] == 'CIN' else 0
    if slide_uuid not in r_per_slide.keys():
        #  first time
        if len(df) <= target:
            r_per_slide[slide_uuid] = 1.0
        else:
            e = 2  # number of epochs with reduction
            a = len(df)  # original value
            r = (target / a) ** (1 / e)
            r_per_slide[slide_uuid] = r
    r = r_per_slide[slide_uuid]
    limit = int(max(len(df) * r, target))
    index = np.concatenate([df[col].nlargest(limit).index.values,
                            df[col].nsmallest(limit).index.values])
    return df.loc[index]


def get_number_of_balanced_tiles_per_class_cohort(df, target_per_cohort_per_class, target_per_class,
                                                  num_used_tiles_per_cls, balanced_cohorts):
    num_tile_y = df.groupby(['y']).tile_path.count()
    num_tile_y_c = df.groupby(['y', 'cohort']).agg({'tile_path': 'count', 'slide_uuid': 'nunique'}).reset_index()
    df_cls_list = []
    for cls in [0, 1]:
        num_tile_cls = num_tile_y_c[num_tile_y_c.y == cls].reset_index(drop=True)
        if num_tile_cls.empty:
            continue
        used_tiles_per_c = (num_used_tiles_per_cls.loc[cls] // len(balanced_cohorts))
        num_tile_cls['num_tiles_using'] = num_tile_cls.tile_path.apply(
            lambda t: min(t, target_per_cohort_per_class - used_tiles_per_c))
        num_tile_cls['remain'] = num_tile_cls.tile_path.apply(
            lambda t: max(0, t - (target_per_cohort_per_class - used_tiles_per_c)))
        num_tile_cls['res'] = num_tile_cls.tile_path.apply(
            lambda t: max(0, target_per_cohort_per_class - used_tiles_per_c - t))
        if num_tile_y.loc[cls] - (target_per_class - num_used_tiles_per_cls.loc[cls]) < 0:
            print(f'All cohorts are maxed out for y={cls}.')
        else:
            sum_res_cls = num_tile_cls.res.sum()
            while sum_res_cls > 0:
                num_tiles_per_boosted_c = max(1, sum_res_cls // (num_tile_cls.remain > 0).sum())
                for i, row in num_tile_cls.iterrows():
                    if row['remain'] > 0 and sum_res_cls > 0:
                        sum_res_cls -= min(num_tile_cls.iloc[i]['remain'], num_tiles_per_boosted_c)
                        num_tile_cls['num_tiles_using'].iloc[i] += min(num_tile_cls.iloc[i]['remain'],
                                                                       num_tiles_per_boosted_c)
                        num_tile_cls['remain'].iloc[i] = max(0,
                                                             num_tile_cls.iloc[i]['remain'] - num_tiles_per_boosted_c)
        df_cls_list.append(num_tile_cls)
    return pd.concat(df_cls_list, ignore_index=True).drop(columns=['remain', 'res', 'tile_path']).rename(
        columns={'slide_uuid': 'num_slides'})


def get_number_of_balanced_tiles_per_class_cohort_slide(df, num_tile_per_class_per_cohort, balanced_cohorts):
    num_tile_per_class_per_cohort[
        'num_tiles_per_slide'] = num_tile_per_class_per_cohort.num_tiles_using // num_tile_per_class_per_cohort.num_slides
    num_tile_per_class_per_cohort = num_tile_per_class_per_cohort.drop(columns=['num_tiles_using', 'num_slides'])
    num_tile_per_slide = df.groupby(['y', 'cohort', 'slide_uuid']).agg({'tile_path': 'count'}).reset_index()
    num_tile_per_slide = num_tile_per_slide.merge(num_tile_per_class_per_cohort, how='inner', on=['y', 'cohort'])
    num_tile_per_slide['num_tiles_using'] = num_tile_per_slide.apply(
        lambda row: min(row['tile_path'], row['num_tiles_per_slide']), axis=1)
    num_tile_per_slide['remain'] = num_tile_per_slide.apply(
        lambda row: max(0, row['tile_path'] - row['num_tiles_per_slide']), axis=1)
    num_tile_per_slide['res'] = num_tile_per_slide.apply(
        lambda row: max(0, row['num_tiles_per_slide'] - row['tile_path']), axis=1)
    df_cls_list = []
    for cls in [0, 1]:
        for c in balanced_cohorts:
            # print(cls, c)
            num_tile_cls = num_tile_per_slide[
                (num_tile_per_slide.y == cls) & (num_tile_per_slide.cohort == c)].reset_index(drop=True)
            if num_tile_cls.empty:
                continue
            sum_res_cls = num_tile_cls.res.sum()
            while sum_res_cls > 0:
                num_tiles_per_boosted_slide = max(1, sum_res_cls // (num_tile_cls.remain > 0).sum())
                for i, row in num_tile_cls.iterrows():
                    if row['remain'] > 0 and sum_res_cls > 0:
                        sum_res_cls -= min(num_tile_cls.iloc[i]['remain'], num_tiles_per_boosted_slide)
                        num_tile_cls['num_tiles_using'].iloc[i] += min(num_tile_cls.iloc[i]['remain'],
                                                                       num_tiles_per_boosted_slide)
                        num_tile_cls['remain'].iloc[i] = max(0, num_tile_cls.iloc[i][
                            'remain'] - num_tiles_per_boosted_slide)
            df_cls_list.append(num_tile_cls)
    return pd.concat(df_cls_list, ignore_index=True)


def get_num_tiles_y_full(df_f, target_all_full, full_cohort):
    print(f'target_all_full: {target_all_full}')
    if df_f.empty:
        return pd.DataFrame()
    target_per_class = target_all_full // 2
    num_tiles_y = df_f.groupby('y').agg({'tile_path': 'count', 'slide_uuid': 'nunique'})
    num_tiles_y['num_tiles_using'] = target_per_class
    if num_tiles_y.tile_path.loc[0] > target_per_class and num_tiles_y.tile_path.loc[1] < target_per_class:
        res = target_per_class - num_tiles_y.tile_path.loc[1]
        num_tiles_y.num_tiles_using.loc[0] = min(target_per_class + res, num_tiles_y.tile_path.loc[0])
    if num_tiles_y.tile_path.loc[1] > target_per_class and num_tiles_y.tile_path.loc[0] < target_per_class:
        res = target_per_class - num_tiles_y.tile_path.loc[0]
        num_tiles_y.num_tiles_using.loc[1] = min(target_per_class + res, num_tiles_y.tile_path.loc[1])
    num_tiles_y.num_tiles_using = num_tiles_y.apply(
        lambda row: row['num_tiles_using'] if row['tile_path'] > target_per_class else row['tile_path'], axis=1)
    num_tiles_y_c = num_tiles_y.reset_index()

    num_tiles_y_c['cohort'] = full_cohort
    num_tiles_y_c = num_tiles_y_c.drop(columns=['tile_path']).rename(columns={'slide_uuid': 'num_slides'})
    num_tiles_y_c = num_tiles_y_c[['y', 'cohort', 'num_slides', 'num_tiles_using']]
    return num_tiles_y_c


def get_balanced_tiles(df, fp_f, fp_all, full_cohort, balanced_cohorts):
    print(f'Total size: {len(df)}')
    print(f'Full cohort: {full_cohort}, balanced cohorts: {balanced_cohorts}.')
    print(f'Per cohort and class statistics:')
    print(df.groupby(['y', 'cohort'], as_index=False).agg({'tile_path': 'count', 'slide_uuid': 'nunique'}))

    df_f = df[df.cohort == full_cohort]
    df_b = df[df.cohort.isin(balanced_cohorts)]

    num_tile_per_class_per_cohort_full = get_num_tiles_y_full(df_f, int(len(df) * fp_f), full_cohort)
    num_tile_y_full = num_tile_per_class_per_cohort_full.groupby('y').num_tiles_using.sum()

    target_all = int(len(df) * fp_all)
    target_per_class = target_all // 2
    target_per_cohort_per_class = target_per_class // len(balanced_cohorts)
    num_tile_per_class_per_cohort = get_number_of_balanced_tiles_per_class_cohort(df_b, target_per_cohort_per_class,
                                                                                  target_per_class, num_tile_y_full,
                                                                                  balanced_cohorts)
    num_tile_per_class_per_cohort = pd.concat([num_tile_per_class_per_cohort, num_tile_per_class_per_cohort_full],
                                              ignore_index=True)
    print(num_tile_per_class_per_cohort)

    num_tile_per_class_per_cohort_per_slide = get_number_of_balanced_tiles_per_class_cohort_slide(df,
                                                                                                  num_tile_per_class_per_cohort,
                                                                                                  balanced_cohorts + [
                                                                                                      full_cohort, ])
    return num_tile_per_class_per_cohort_per_slide


# df_balanced_tiles = get_balanced_tiles(df_train_b, fp_f=0.15, fp_all=0.35, full_cohort='CRC', balanced_cohorts=['STAD', 'UCEC'])


def calc_reduction_factor(total, target, num_epochs_with_reduction):
    if total <= target:
        return 1.0
    else:
        return (target / total) ** (1 / num_epochs_with_reduction)


class ReductionObject:
    def __init__(self, df):
        self.df = df
        self.balanced_cohorts = Configs.SC_ITER_ARGS['balanced_cohorts']
        self.num_tile_per_slide = get_balanced_tiles(df, fp_all=Configs.SC_ITER_ARGS['final_reduction_all'],
                                                     fp_f=Configs.SC_ITER_ARGS['final_reduction_tuned'],
                                                     full_cohort=Configs.SC_ITER_ARGS['tune_cohort'],
                                                     balanced_cohorts=self.balanced_cohorts)
        self.num_tile_per_slide['reduction_per_epoch'] = self.num_tile_per_slide.num_tiles_using

    def reduction_to_target_per_class(self, df, col):
        slide_uuid = df.slide_uuid.iloc[0]
        slide_row = self.num_tile_per_slide[self.num_tile_per_slide.slide_uuid==slide_uuid]
        r = calc_reduction_factor(total=slide_row['tile_path'].values[0], target=slide_row['num_tiles_using'].values[0],
                                  num_epochs_with_reduction=Configs.joined['NUM_EPOCHS']-1)
        y = 1 if df.subtype.iloc[0] == 'CIN' else 0
        limit = int(len(df) * r)
        if y == 1:
            return df.loc[df[col].nlargest(limit).index]
        else:
            return df.loc[df[col].nsmallest(limit).index]

    def apply_reduction(self, df, col):
        return self.reduction_to_target_per_class(df, col)











