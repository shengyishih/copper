"""\u9285\u7b94\u88fd\u7a0b\u8cc7\u6599\u7570\u5e38\u5075\u6e2c Streamlit \u61c9\u7528\u7a0b\u5f0f\u3002

\u555f\u52d5\u65b9\u5f0f\uff1astreamlit run 05_app.py
\u8acb\u5c07\u6a21\u578b\u6a94\u8207\u8cc7\u6599 CSV \u653e\u5728\u6b64\u7a0b\u5f0f\u540c\u4e00\u500b\u8cc7\u6599\u593e\u3002
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Resolve bundled files relative to this script, not the launch directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_anomaly.joblib"
DATA_PATH = BASE_DIR / "copper_foil_cleaned.csv"
REFERENCE_PATH = BASE_DIR / "copper_foil_transformed.csv"

st.set_page_config(page_title="\u9285\u7b94\u88fd\u7a0b\u7570\u5e38\u5075\u6e2c", page_icon=chr(0x1F50E), layout="wide")


@st.cache_resource
def load_model(model_path: str):
    """\u8f09\u5165\u8a13\u7df4\u597d\u7684 Pipeline \u8207\u6b04\u4f4d\u4e2d\u7e7c\u8cc7\u6599\u3002"""
    return joblib.load(model_path)


@st.cache_data
def load_csv(csv_path: str) -> pd.DataFrame:
    """\u8f09\u5165 UTF-8 CSV\uff1b\u6642\u9593\u6b04\u4f4d\u7531\u9801\u9762\u4f9d\u9700\u6c42\u8f49\u63db\u3002"""
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def page_visualization() -> None:
    """\u7b2c 1 \u9801\uff1a\u7528\u5e7e\u500b\u4e3b\u8981\u88fd\u7a0b\u6b04\u4f4d\u6aa2\u8996\u6b77\u53f2\u8cc7\u6599\u3002"""
    st.title("\u88fd\u7a0b\u8cc7\u6599\u8996\u89ba\u5316")
    st.caption("\u67e5\u770b\u9285\u7b94\u539a\u5ea6\u3001\u88fd\u7a0b\u53c3\u6578\u8207\u7570\u5e38\u5206\u5e03\u3002\u5716\u8868\u7528\u65bc\u63a2\u7d22\u8cc7\u6599\uff0c\u4e0d\u4ee3\u8868\u56e0\u679c\u95dc\u4fc2\u3002")

    if not DATA_PATH.exists():
        st.error(f"\u627e\u4e0d\u5230\u8cc7\u6599\u6a94\uff1a{DATA_PATH.name}\u3002\u8acb\u5c07 CSV \u653e\u5728 05_app.py \u540c\u4e00\u8cc7\u6599\u593e\u3002")
        return

    data = load_csv(str(DATA_PATH))
    if "Timestamp" in data.columns:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")

    total_rows = len(data)
    anomaly_count = int(data["Is_Anomaly"].sum()) if "Is_Anomaly" in data.columns else 0
    anomaly_rate = anomaly_count / total_rows if total_rows else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("\u6a23\u672c\u6578", f"{total_rows:,}")
    col2.metric("\u7570\u5e38\u6a23\u672c", f"{anomaly_count:,}")
    col3.metric("\u7570\u5e38\u6bd4\u4f8b", f"{anomaly_rate:.1%}")

    st.subheader("\u5e73\u5747\u539a\u5ea6\u8da8\u52e2")
    if {"Timestamp", "Thick_Mean_um"}.issubset(data.columns):
        trend = data[["Timestamp", "Thick_Mean_um"]].dropna().sort_values("Timestamp")
        st.line_chart(trend, x="Timestamp", y="Thick_Mean_um", use_container_width=True)
    else:
        st.info("\u8cc7\u6599\u7f3a\u5c11 Timestamp \u6216 Thick_Mean_um\uff0c\u7121\u6cd5\u986f\u793a\u539a\u5ea6\u8da8\u52e2\u3002")

    left, right = st.columns(2)
    with left:
        st.subheader("\u539a\u5ea6\u5206\u5e03")
        if "Thick_Mean_um" in data.columns:
            values = pd.to_numeric(data["Thick_Mean_um"], errors="coerce").dropna()
            if not values.empty:
                hist = pd.cut(values, bins=min(20, max(5, values.nunique()))).value_counts(sort=False)
                # Convert IntervalIndex to ordinary text labels for Altair's categorical axis.
                hist_data = pd.DataFrame({
                    "\u539a\u5ea6\u5340\u9593": hist.index.astype(str),
                    "\u6a23\u672c\u6578": hist.to_numpy(),
                })
                st.bar_chart(
                    hist_data,
                    x="\u539a\u5ea6\u5340\u9593",
                    y="\u6a23\u672c\u6578",
                    use_container_width=True,
                )
        else:
            st.info("\u8cc7\u6599\u7f3a\u5c11 Thick_Mean_um\u3002")

    with right:
        st.subheader("\u73ed\u5225\u7570\u5e38\u6bd4\u4f8b")
        if {"Shift", "Is_Anomaly"}.issubset(data.columns):
            rate_by_shift = data.groupby("Shift", dropna=False)["Is_Anomaly"].mean().sort_values(ascending=False)
            st.bar_chart(rate_by_shift.rename("\u7570\u5e38\u6bd4\u4f8b"), use_container_width=True)
        else:
            st.info("\u8cc7\u6599\u7f3a\u5c11 Shift \u6216 Is_Anomaly\u3002")

    st.subheader("\u96fb\u6d41\u5bc6\u5ea6\u8207\u5e73\u5747\u539a\u5ea6")
    if {"Current_ASD", "Thick_Mean_um"}.issubset(data.columns):
        scatter_cols = ["Current_ASD", "Thick_Mean_um"]
        if "Is_Anomaly" in data.columns:
            scatter_cols.append("Is_Anomaly")
        scatter = data[scatter_cols].apply(pd.to_numeric, errors="coerce").dropna()
        st.scatter_chart(
            scatter,
            x="Current_ASD",
            y="Thick_Mean_um",
            color="Is_Anomaly" if "Is_Anomaly" in scatter.columns else None,
            use_container_width=True,
        )
    else:
        st.info("\u8cc7\u6599\u7f3a\u5c11 Current_ASD \u6216 Thick_Mean_um\u3002")

    st.subheader("\u4e3b\u8981\u88fd\u7a0b\u53c3\u6578\u6458\u8981")
    summary_cols = [
        col for col in [
            "Line_Speed_mpm", "Current_ASD", "Bath_Temp_C",
            "Cu_Conc_gL", "Additive_ppm", "Jet_Pressure_bar",
            "Edge_Current_Bias", "Thick_Mean_um", "Thick_Range_um",
        ]
        if col in data.columns
    ]
    if summary_cols:
        st.dataframe(data[summary_cols].describe().T, use_container_width=True)


def page_prediction() -> None:
    """\u7b2c 2 \u9801\uff1a\u8f38\u5165\u88fd\u7a0b\u53c3\u6578\uff0c\u8f09\u5165\u6a21\u578b\u4e26\u9032\u884c\u55ae\u7b46\u7570\u5e38\u9810\u6e2c\u3002"""
    st.title("\u7570\u5e38\u9810\u6e2c")
    st.caption("\u8f38\u5165\u88fd\u7a0b\u689d\u4ef6\u5f8c\uff0c\u6a21\u578b\u5c07\u56de\u50b3\u7570\u5e38\u985e\u5225\u8207\u4f30\u8a08\u6a5f\u7387\u3002")

    if not MODEL_PATH.exists():
        st.error(f"\u627e\u4e0d\u5230\u6a21\u578b\u6a94\uff1a{MODEL_PATH.name}\u3002\u8acb\u5148\u8a13\u7df4\u6a21\u578b\u4e26\u5c07 joblib \u653e\u5728\u6b64\u8cc7\u6599\u593e\u3002")
        return

    try:
        artifact = load_model(str(MODEL_PATH))
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        target_name = artifact.get("target", "Is_Anomaly")
    except Exception as exc:
        st.error(f"\u6a21\u578b\u8f09\u5165\u5931\u6557\uff1a{exc}")
        return

    # \u53c3\u8003\u8a13\u7df4\u8cc7\u6599\u7684\u7bc4\u570d\u8207\u4e2d\u4f4d\u6578\u8a2d\u5b9a\u8868\u55ae\u9810\u8a2d\u503c\u3002
    reference = None
    if REFERENCE_PATH.exists():
        try:
            reference = load_csv(str(REFERENCE_PATH))
        except Exception:
            reference = None

    process_labels = {
        "Line_Speed_mpm": "\u7dda\u901f (m/min)",
        "Current_ASD": "\u96fb\u6d41\u5bc6\u5ea6 (A/dm\u00b2)",
        "Bath_Temp_C": "\u69fd\u6db2\u6eab\u5ea6 (\u00b0C)",
        "Cu_Conc_gL": "\u9285\u96e2\u5b50\u6fc3\u5ea6 (g/L)",
        "Additive_ppm": "\u6dfb\u52a0\u5291 (ppm)",
        "Jet_Pressure_bar": "\u5674\u6d41\u58d3\u529b (bar)",
        "Edge_Current_Bias": "\u908a\u7de3\u96fb\u6d41\u504f\u5dee",
    }
    time_fields = {"Timestamp_hour", "Timestamp_dayofweek", "Timestamp_month"}
    shift_columns = [col for col in feature_columns if col.startswith("Shift_")]
    process_fields = [
        col for col in feature_columns
        if col not in time_fields and col not in shift_columns
    ]

    st.info("\u8acb\u8f38\u5165\u88fd\u7a0b\u53c3\u6578\u4e26\u9078\u64c7\u6642\u9593\u8207\u73ed\u5225\u3002\u9810\u6e2c\u7d50\u679c\u4f9b\u5206\u6790\u53c3\u8003\uff0c\u5be6\u969b\u88fd\u7a0b\u5224\u5b9a\u4ecd\u61c9\u642d\u914d\u73fe\u5834\u898f\u7bc4\u3002")
    with st.form("prediction_form"):
        st.subheader("\u88fd\u7a0b\u53c3\u6578")
        values = {}
        controls = st.columns(2)
        for index, feature in enumerate(process_fields):
            label = process_labels.get(feature, feature)
            default = 0.0
            low, high = -1_000_000.0, 1_000_000.0
            if reference is not None and feature in reference.columns:
                numeric = pd.to_numeric(reference[feature], errors="coerce").dropna()
                if not numeric.empty:
                    low, high = float(numeric.min()), float(numeric.max())
                    default = float(numeric.median())
            if high <= low:
                high = low + 1.0
            with controls[index % 2]:
                values[feature] = st.number_input(
                    label,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(np.clip(default, low, high)),
                    step=0.01,
                    format="%.3f",
                    help=f"\u8a13\u7df4\u8cc7\u6599\u89c0\u6e2c\u7bc4\u570d\uff1a{low:.3f} \u81f3 {high:.3f}",
                )

        st.subheader("\u6642\u9593\u8207\u73ed\u5225")
        time_cols = st.columns(3)
        if "Timestamp_hour" in feature_columns:
            with time_cols[0]:
                values["Timestamp_hour"] = st.selectbox("\u5c0f\u6642", list(range(24)), index=8)
        if "Timestamp_dayofweek" in feature_columns:
            weekdays = ["\u9031\u4e00", "\u9031\u4e8c", "\u9031\u4e09", "\u9031\u56db", "\u9031\u4e94", "\u9031\u516d", "\u9031\u65e5"]
            with time_cols[1]:
                values["Timestamp_dayofweek"] = st.selectbox(
                    "\u661f\u671f", list(range(7)), format_func=lambda n: weekdays[n], index=0
                )
        if "Timestamp_month" in feature_columns:
            with time_cols[2]:
                values["Timestamp_month"] = st.selectbox("\u6708\u4efd", list(range(1, 13)), index=9)

        if shift_columns:
            # \u73ed\u5225 one-hot \u6b04\u4f4d\u5fc5\u9808\u6070\u6709\u4e00\u500b\u503c\u70ba 1\u3002
            shift_names = [col.removeprefix("Shift_") for col in shift_columns]
            selected_shift = st.selectbox("\u73ed\u5225", shift_names)
            for col, shift_name in zip(shift_columns, shift_names):
                values[col] = int(shift_name == selected_shift)

        submitted = st.form_submit_button("\u57f7\u884c\u7570\u5e38\u9810\u6e2c", type="primary", use_container_width=True)

    if submitted:
        # \u4f9d\u6a21\u578b\u8a13\u7df4\u6642\u7684\u6b04\u4f4d\u540d\u7a31\u8207\u9806\u5e8f\u5efa\u7acb\u8f38\u5165\uff0c\u907f\u514d\u6b04\u4f4d\u932f\u4f4d\u3002
        input_data = pd.DataFrame(
            [[values.get(col, 0) for col in feature_columns]],
            columns=feature_columns,
        )
        try:
            prediction = int(model.predict(input_data)[0])
            probability = float(model.predict_proba(input_data)[0, 1])
        except Exception as exc:
            st.error(f"\u9810\u6e2c\u5931\u6557\uff1a{exc}")
            return

        result_col, probability_col = st.columns(2)
        if prediction == 1:
            result_col.error("\u9810\u6e2c\u7d50\u679c\uff1a\u7570\u5e38")
        else:
            result_col.success("\u9810\u6e2c\u7d50\u679c\uff1a\u6b63\u5e38")
        probability_col.metric("\u7570\u5e38\u6a5f\u7387", f"{probability:.1%}")
        st.caption(f"\u76ee\u6a19\u6b04\u4f4d\uff1a{target_name}\uff1b\u5206\u985e\u95be\u503c\u6cbf\u7528\u6a21\u578b\u9810\u8a2d\u503c 0.5\u3002")
        st.write("\u672c\u6b21\u8f38\u5165\u7279\u5fb5")
        st.dataframe(input_data, use_container_width=True)


st.sidebar.title("\u9285\u7b94\u88fd\u7a0b\u5206\u6790")
page = st.sidebar.radio(
    "\u529f\u80fd\u5206\u9801",
    ["\u8cc7\u6599\u8996\u89ba\u5316", "\u7570\u5e38\u9810\u6e2c"],
    help="\u9078\u64c7\u8cc7\u6599\u63a2\u7d22\u6216\u6a21\u578b\u9810\u6e2c\u529f\u80fd\u3002",
)
if page == "\u8cc7\u6599\u8996\u89ba\u5316":
    page_visualization()
else:
    page_prediction()
