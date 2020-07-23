from sklearn import preprocessing
import pandas as pd


def fill_na_mean(columns, dataframe):
    """
    :param columns: columns of which na values have to be filled with mean
    :param dataframe: dataframe of interest
    """
    for column in columns:
        dataframe[column].fillna(int(dataframe[column].mean()), inplace=True)


def fill_na_zero(columns, dataframe):
    """
    :param columns: columns of which na values have to be filled with mean
    :param dataframe: dataframe of interest
    """
    for column in columns:
        dataframe[column].fillna("0", inplace=True)


def from_categorical_to_numerical(dataframe):
    _list = []
    for col in dataframe.columns:
        if type(dataframe[col][0]) == type('str'):
            _list.append(col)

    dataframe[_list] = dataframe[_list].astype('str')

    le = preprocessing.LabelEncoder()
    for li in _list:
        le.fit(list(set(dataframe[li])))
        dataframe[li] = le.transform(dataframe[li])

def subset_by_iqr(df, column, whisker_width=1.5):
    """Remove outliers from a dataframe by column, including optional
       whiskers, removing rows for which the column value are
       less than Q1-1.5IQR or greater than Q3+1.5IQR.
    Args:
        df (`:obj:pd.DataFrame`): A pandas dataframe to subset
        column (str): Name of the column to calculate the subset from.
        whisker_width (float): Optional, loosen the IQR filter by a
                               factor of `whisker_width` * IQR.
    Returns:
        (`:obj:pd.DataFrame`): Filtered dataframe
    """
    # Calculate Q1, Q2 and IQR
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    # Apply filter with respect to IQR, including optional whiskers
    filter = (df[column] >= q1 - whisker_width * iqr) & (df[column] <= q3 + whisker_width * iqr)
    return df.loc[filter]

def return_submission_csv(prediction):
    id_list = range(1461, 2920)
    df = pd.DataFrame({'Id': id_list, 'SalePrice': prediction.flatten()})
    df = df.set_index('Id')
    df.to_csv('data/log_submission_test.csv')
    return df
