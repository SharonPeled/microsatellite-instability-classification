



def reduction_with_limit(df, col, limit=1000):
    limit = int(max(len(df)*0.8, limit))
    return df.loc[df[col].nlargest(limit).index]
