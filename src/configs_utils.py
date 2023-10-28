



def reduction_with_limit(df, col, limit=1000):
    return df.loc[df[col].nlargest(limit).index]
