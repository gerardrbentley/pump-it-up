import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prep_for_model(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df["installer"] = clean_installer(df["installer"])
    df["funder"] = clean_funder(df["funder"])
    nonzero_lat_lon = df[(df["latitude"] != 0) | (df["longitude"] != 0)]
    region_lat_means = (
        nonzero_lat_lon[["latitude", "region_code"]].groupby("region_code").mean()
    )
    region_lon_means = (
        nonzero_lat_lon[["longitude", "region_code"]].groupby("region_code").mean()
    )
    df = fill_mean_lon(df, region_lon_means)
    df = fill_mean_lat(df, region_lat_means)
    df = drop_unused_columns(df)

    df, encoders = labelencode_values(df)
    return df, encoders


def clean_installer(series: pd.Series, other_percentile: float = 0.9) -> pd.Series:
    s = (
        series.str.lower()
        .fillna("0")
        .str.replace("0", "other")
        .str.replace("[^a-zA-Z]", "", regex=True)
    )
    counts = s.value_counts()
    cutoff = counts.quantile(other_percentile)
    others = counts[counts < cutoff]
    s[s.isin(others.index)] = "other"
    return s


def clean_funder(series: pd.Series, other_percentile: float = 0.9) -> pd.Series:
    s = (
        series.str.lower()
        .fillna("0")
        .str.replace("0", "other")
        .str.replace("[^a-zA-Z]", "", regex=True)
    )
    counts = s.value_counts()
    cutoff = counts.quantile(other_percentile)
    others = counts[counts < cutoff]
    s[s.isin(others.index)] = "other"
    return s


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.drop(
        [
            "scheme_management",
            "quantity_group",
            "water_quality",
            "payment_type",
            "extraction_type",
            "waterpoint_type_group",
            "region_code",
            "date_recorded",
            "recorded_by",
        ],
        axis=1,
    )
    return output_df


def clean_lat(s: pd.Series, mean_lons: pd.Series):
    if s["latitude"] != 0:
        return s["latitude"]
    else:
        return float(mean_lons.loc[int(s["region_code"])])


def fill_mean_lat(dataframe: pd.DataFrame, mean_lats: pd.Series) -> pd.DataFrame:
    df = dataframe.copy()
    df["latitude"] = df[["latitude", "region_code"]].apply(
        lambda x: clean_lat(x, mean_lats), axis=1
    )
    return df


def clean_lon(s: pd.Series, mean_lons: pd.Series):
    if s["longitude"] != 0:
        return s["longitude"]
    else:
        return float(mean_lons.loc[int(s["region_code"])])


def fill_mean_lon(dataframe: pd.DataFrame, mean_lons: pd.Series) -> pd.DataFrame:
    df = dataframe.copy()
    df["longitude"] = df[["longitude", "region_code"]].apply(
        lambda x: clean_lon(x, mean_lons), axis=1
    )
    return df


CATEGORICAL_FEATURES = "funder, installer, wpt_name, basin, subvillage, region, lga, ward, public_meeting, scheme_name, permit, extraction_type_group, extraction_type_class, management, management_group, payment, quality_group, quantity, source, source_type, source_class, waterpoint_type".split(
    ", "
)


def labelencode_values(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    encoders = {}

    for col in CATEGORICAL_FEATURES:
        labelencoder = LabelEncoder()
        df[col] = labelencoder.fit_transform(df[col]).astype("int")
        encoders[col] = labelencoder
    return df, encoders


def labelencode_labels(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    labelencoder = LabelEncoder()
    df["status_group"] = labelencoder.fit_transform(df["status_group"]).astype("int")
    return df, labelencoder


def labelencode_data(values: pd.DataFrame, labels: pd.DataFrame) -> tuple:
    return labelencode_values(values), labelencode_labels(labels)
