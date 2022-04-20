from pathlib import Path

import lightgbm as lgb
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from src.data.load_dataset import load_full_dataset
from src.features.build_features import (
    clean_funder,
    clean_installer,
    drop_unused_columns,
    fill_mean_lat,
    fill_mean_lon,
    labelencode_labels,
    prep_for_model,
)
from src.models.predict_model import make_prediction

header_text = "Water Pumps"
home = "Project Home"
data = "Data Sources"
eda = "Exploratory Analysis"
pandas_profiling = "Pandas Profiling"
features = "Feature Engineering"
training = "Model Training"
data_dir = Path("data")
models_dir = Path("models")
reports_dir = Path("reports")


def render_home():
    st.subheader("Project Home Page")
    st.write("Water Pump functionality prediction. Based on DrivenData challenge")

    st.subheader("Make a prediction")
    models = Path("models").glob("*")
    latest_path = max(models, key=lambda p: p.stat().st_ctime)
    bst = lgb.Booster(model_file=str(latest_path))
    train_values, train_labels, test_values = load_full_dataset()
    with st.expander("Raw Input"):
        st.write(test_values)
    test_values, encoders = prep_for_model(test_values)
    st.subheader("Input:")
    st.write(test_values)
    preds = make_prediction(bst, test_values)
    t, label_encoder = labelencode_labels(train_labels)
    decoded_preds = pd.DataFrame(
        label_encoder.inverse_transform(preds),
        columns=["status_group"],
        index=test_values.index,
    )
    st.subheader("Prediction:")
    st.write(decoded_preds)


def render_data():
    st.subheader("Data Source Information")
    st.write("The following data were gathered from the following sources:")
    problem_report = reports_dir / "problem_description.md"
    st.write(problem_report.read_text())


@st.cache(allow_output_mutation=True, persist=True)
def prep_analysis(df: pd.DataFrame):
    pr = ProfileReport(df, title="Water Pumps Profiling Report", minimal=True)
    return pr


def render_pandas_profiling():
    train_values, train_labels, test_values = load_full_dataset()
    train_data = train_values.join(train_labels, on="id")
    analysis = prep_analysis(train_data)
    st_profile_report(analysis)


def render_eda():
    train_values, train_labels, test_values = load_full_dataset()
    st.subheader("Training Labels")
    st.bar_chart(train_labels.value_counts("status_group"))
    st.write(
        """\
There are many fewer examples of "functional needs repair" and "non functional" than "functional".
We can use a model that can account for these class imbalances, augment the dataset by oversampling the under-represented classes, or filter the dataset by undersampling the over-represented classes.
    """
    )
    with st.expander("Raw Data"):
        st.write(train_labels)
    train_data = train_values.join(train_labels, on="id")
    st.subheader("Training Data Exploratory Analysis")
    with st.expander("Raw Data"):
        st.write(train_data)
    st.subheader("Labels on Map")
    choice = st.selectbox(
        "Which Labels to View on Map", train_data["status_group"].unique()
    )
    st.map(train_data[train_data["status_group"] == choice])

    st.subheader("Breakdown of Interesting Label Counts")
    st.subheader("Waterpoints by quantity group (how much water)")
    st.plotly_chart(
        px.bar(
            train_data.groupby(
                ["quantity_group", "status_group"], as_index=False
            ).size(),
            x="quantity_group",
            y="size",
            color="status_group",
        )
    )
    st.subheader("Waterpoints by quality status")
    st.plotly_chart(
        px.bar(
            train_data.groupby(
                ["quality_group", "status_group"], as_index=False
            ).size(),
            x="quality_group",
            y="size",
            color="status_group",
        )
    )
    st.subheader('Waterpoints by quality status excluding "good"')
    st.plotly_chart(
        px.bar(
            train_data[train_data["quality_group"] != "good"]
            .groupby(["quality_group", "status_group"], as_index=False)
            .size(),
            x="quality_group",
            y="size",
            color="status_group",
        )
    )
    st.subheader("Waterpoints by waterpoint type")
    st.plotly_chart(
        px.bar(
            train_data.groupby(
                ["waterpoint_type_group", "status_group"], as_index=False
            ).size(),
            x="waterpoint_type_group",
            y="size",
            color="status_group",
        )
    )
    st.subheader("Waterpoints by Construction Year excluding missing")
    st.plotly_chart(
        px.bar(
            train_data[train_data["construction_year"] != 0]
            .groupby(["construction_year", "status_group"], as_index=False)
            .size(),
            x="size",
            y="construction_year",
            color="status_group",
            orientation="h",
        )
    )
    st.subheader("Funder groups funding more than 500 waterpoints excluding missing")
    funders = (
        train_data[train_data["funder"] != "0"]
        .groupby("funder", as_index=False)
        .size()
        .query("size > 500")
    )
    st.plotly_chart(
        px.bar(
            train_data[train_data["funder"].isin(funders["funder"])]
            .groupby(["funder", "status_group"], as_index=False)
            .size(),
            x="size",
            y="funder",
            color="status_group",
            orientation="h",
        )
    )
    st.subheader("Waterpoints by basin")
    st.plotly_chart(
        px.bar(
            train_data.groupby(["basin", "status_group"], as_index=False).size(),
            x="size",
            y="basin",
            color="status_group",
            orientation="h",
        )
    )
    st.subheader("Waterpoints by Payment Type")
    st.plotly_chart(
        px.bar(
            train_data.groupby(["payment_type", "status_group"], as_index=False).size(),
            x="size",
            y="payment_type",
            color="status_group",
            orientation="h",
        )
    )

    st.subheader(
        "Total static head (amount water available to waterpoint) investigation"
    )
    st.write(
        "[discussion link](https://community.drivendata.org/t/interpreting-amount-tsh/338)"
    )
    with st.expander("Raw data"):
        st.write(train_data.groupby(["amount_tsh"], as_index=False).size())
    st.write(
        "50 percentile ignoring 0's",
        train_data["amount_tsh"][train_data["amount_tsh"] > 0].quantile(),
    )
    st.write(
        "75 percentile ignoring 0's",
        train_data["amount_tsh"][train_data["amount_tsh"] > 0].quantile(0.75),
    )
    st.write("50 percentile", train_data["amount_tsh"].quantile())
    st.write("75 percentile", train_data["amount_tsh"].quantile(0.75))
    st.plotly_chart(
        px.bar(
            train_data.groupby(["amount_tsh", "status_group"], as_index=False).size(),
            x="amount_tsh",
            y="size",
            color="status_group",
        )
    )
    st.plotly_chart(
        px.bar(
            train_data[
                (train_data["amount_tsh"] > 0) & (train_data["amount_tsh"] < 10000)
            ]
            .groupby(["amount_tsh", "status_group"], as_index=False)
            .size(),
            x="amount_tsh",
            y="size",
            color="status_group",
        )
    )


def render_features():
    st.subheader("Feature Engineering Process")
    st.write("The following transformations were applied to the following datasets:")
    train_values, train_labels, test_values = load_full_dataset()
    st.subheader("Clean Installers and break outliers into 'other' group")
    st.write("Initial 'installers'")
    st.write(len(train_values["installer"].unique()), " Number of installers")
    st.write(train_values["installer"].value_counts())
    s = clean_installer(train_values["installer"])
    st.write("Transformed data")
    st.write(len(s.unique()), " Number of installers")
    st.write(s.value_counts())
    st.subheader("Clean Funders and break outliers into 'other' group")
    st.write("Initial 'funders'")
    st.write(len(train_values["funder"].unique()), " Number of funders")
    st.write(train_values["funder"].value_counts())
    s = clean_funder(train_values["funder"])
    st.write("Transformed data")
    st.write(len(s.unique()), " Number of funders")
    st.write(s.value_counts())

    st.subheader("Filling lat and lon with region means")
    nonzero_lat_lon = train_values[
        (train_values["latitude"] != 0) | (train_values["longitude"] != 0)
    ]
    zero_lat_lon = train_values[
        (train_values["latitude"] == 0) | (train_values["longitude"] == 0)
    ]
    st.write("Data with 0 lat or lon: ", len(zero_lat_lon), " records")
    st.write(zero_lat_lon)
    region_lat_means = (
        nonzero_lat_lon[["latitude", "region_code"]].groupby("region_code").mean()
    )
    region_lon_means = (
        nonzero_lat_lon[["longitude", "region_code"]].groupby("region_code").mean()
    )
    st.write(
        "region_means",
        nonzero_lat_lon[["latitude", "longitude", "region_code"]]
        .groupby("region_code", as_index=False)
        .mean(),
    )
    st.write(len(train_values))
    df = fill_mean_lon(train_values, region_lon_means)
    df = fill_mean_lat(df, region_lat_means)
    zero_lat_lon = df[(df["latitude"] == 0) | (df["longitude"] == 0)]
    st.write(
        "After Transformation data with 0 lat or lon: ", len(zero_lat_lon), " records"
    )
    st.write(df)

    st.subheader("Drop duplicate and constant columns")
    st.write("initial columns")
    st.write(train_values.columns)
    st.write("Transformed columns")
    df = drop_unused_columns(df)
    st.write(df.columns)


def render_training():
    st.subheader("Model Training Overview")
    st.write(
        "The following model and hyperparameters were tested to generate the final model:"
    )
    for sub_path in (
        x for x in models_dir.iterdir() if x.is_file() and not x.name.startswith(".")
    ):
        st.subheader(sub_path.name)
        st.write("Size in bytes: ", len(sub_path.read_bytes()))


display_page = st.sidebar.radio(
    "View Page:", (home, data, pandas_profiling, eda, features, training)
)
st.header(header_text)

if display_page == home:
    render_home()
elif display_page == pandas_profiling:
    render_pandas_profiling()
elif display_page == eda:
    render_eda()
elif display_page == data:
    render_data()
elif display_page == features:
    render_features()
elif display_page == training:
    render_training()
