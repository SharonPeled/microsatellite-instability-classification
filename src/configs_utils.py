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


def print_balance_details(df, target_per_class, target_per_cohort_per_class, balanced_cohorts):
    print(f'Number of tiles per class: {target_per_class}')
    num_tile_y = df.groupby(['y']).tile_path.count()
    print(num_tile_y.to_dict())
    if ((num_tile_y - target_per_class) < 0).any():
        print(f'Warninig - Classes are not balanced!')
    else:
        print('Classes are balanced.')
    print(f'Number of tiles per class per cohort: {target_per_cohort_per_class}')
    num_tile_y_c = df.groupby(['y', 'cohort']).tile_path.count()
    print(num_tile_y_c.to_dict())
    if ((num_tile_y_c - target_per_cohort_per_class) < 0).any():
        print(f'Warninig - Cohorts are not balanced!')
    else:
        print('Cohorts are balanced.')


def get_number_of_balanced_tiles_per_class_cohort(df, target_per_cohort_per_class, target_per_class):
    num_tile_y = df.groupby(['y']).tile_path.count()
    num_tile_y_c = df.groupby(['y', 'cohort']).agg({'tile_path': 'count', 'slide_uuid': 'nunique'}).reset_index()
    num_tile_y_c['num_tiles_using'] = num_tile_y_c.tile_path.apply(lambda t: min(t, target_per_cohort_per_class))
    num_tile_y_c['remain'] = num_tile_y_c.tile_path.apply(lambda t: max(0, t - target_per_cohort_per_class))
    num_tile_y_c['res'] = num_tile_y_c.tile_path.apply(lambda t: max(0, target_per_cohort_per_class - t))
    df_cls_list = []
    for cls in [0, 1]:
        num_tile_cls = num_tile_y_c[num_tile_y_c.y == cls].reset_index(drop=True)
        if num_tile_cls.empty:
            continue
        if num_tile_y.loc[cls] - target_per_class < 0:
            print(f'All cohorts are maxed out for y={cls}.')
        else:
            sum_res_cls = num_tile_cls.res.sum()
            while sum_res_cls > 0:
                num_tiles_per_boosted_c = sum_res_cls // (num_tile_cls.remain > 0).sum()
                for i, row in num_tile_cls.iterrows():
                    if row['remain'] > 0:
                        sum_res_cls -= min(num_tile_cls.iloc[i]['remain'], num_tiles_per_boosted_c)
                        num_tile_cls['num_tiles_using'].iloc[i] += min(num_tile_cls.iloc[i]['remain'],
                                                                       num_tiles_per_boosted_c)
                        num_tile_cls['remain'].iloc[i] = max(0,
                                                             num_tile_cls.iloc[i]['remain'] - num_tiles_per_boosted_c)
        df_cls_list.append(num_tile_cls)
    return pd.concat(df_cls_list, ignore_index=True).drop(columns=['remain', 'res', 'tile_path']).rename(columns={'slide_uuid': 'num_slides'})


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


def get_balanced_tiles(df, fp, balanced_cohorts):
    target_all = int(len(df) * fp)
    target_per_class = target_all // 2
    target_per_cohort_per_class = target_per_class // len(balanced_cohorts)
    print_balance_details(df, target_per_class, target_per_cohort_per_class, balanced_cohorts)
    num_tile_per_class_per_cohort = get_number_of_balanced_tiles_per_class_cohort(df, target_per_cohort_per_class,
                                                                                  target_per_class)
    num_tile_per_class_per_cohort_per_slide = get_number_of_balanced_tiles_per_class_cohort_slide(df,
                                                                                                  num_tile_per_class_per_cohort,
                                                                                                  balanced_cohorts)
    return num_tile_per_class_per_cohort_per_slide


# df_balanced_tiles = get_balanced_tiles(df_train[df_train.cohort != 'ESCA'], fp=0.5,
#                                        balanced_cohorts=['CRC', 'STAD', 'UCEC'])


def calc_reduction_factor(total, target, num_epochs_with_reduction):
    if total <= target:
        return 1.0
    else:
        return (target / total) ** (1 / num_epochs_with_reduction)


class ReductionObject:
    def __init__(self, df):
        self.df = df
        self.fp = Configs.SC_ITER_ARGS['final_reduction']
        self.balanced_cohorts = Configs.SC_ITER_ARGS['balanced_cohorts']
        self.num_tile_per_slide = get_balanced_tiles(df, self.fp, self.balanced_cohorts)
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











