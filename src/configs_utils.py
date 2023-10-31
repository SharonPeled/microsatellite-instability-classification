



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
