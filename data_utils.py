import pandas as pd
def load_data(path: str) -> pd.DataFrame:
    """
    Load raw retail inventory dataset from disk.
    """
    return pd.read_csv(path)




def filter_store_product(df, store_id, product_id):
    """
    Filter dataset for selected store and product.
    Keeps all columns (not just Units Sold).
    """

    df["Date"] = pd.to_datetime(df["Date"])

    filtered = df[
        (df["Store ID"] == store_id) &
        (df["Product ID"] == product_id)
    ].sort_values("Date")

    return filtered



    
    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter store & product
    ts_df = df[
        (df['Store ID'] == store_id) &
        (df['Product ID'] == product_id)
    ][['Date', 'Units Sold']]
    
    # Sort chronologically
    ts_df = ts_df.sort_values('Date').reset_index(drop=True)
    
    return ts_df
