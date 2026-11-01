import pandas as pd
import streamlit as st

from fraud_xai.data.loader import load_table
from fraud_xai.data.schema import (Schema, binarize_target, class_balance, guess_amount, guess_target,
                                   guess_time, parse_time, suggest_drop_columns)
from fraud_xai.data.synthetic import make_demo_dataset
from fraud_xai.viz import plots
from ui.state import S, reset_downstream

s = S()
st.title("📥 Data")

src = st.radio("Source", ["Upload a file", "Demo dataset", "Local path"], horizontal=True)
if src == "Upload a file":
    up = st.file_uploader("CSV, TSV, Parquet or Excel", type=["csv", "tsv", "txt", "parquet", "xlsx", "xls"])
    if up is not None and up.name != s.filename:
        with st.spinner("Reading file..."):
            s.df, s.filename = load_table(up, up.name), up.name
        reset_downstream("data")
elif src == "Demo dataset":
    n = st.slider("Rows", 5_000, 100_000, 30_000, 5_000)
    if st.button("Generate demo data"):
        s.df, s.filename = make_demo_dataset(n), f"demo_{n}"
        reset_downstream("data")
else:
    path = st.text_input("Path on the machine running Streamlit", placeholder="data/transaction_dataset.csv")
    if st.button("Load") and path:
        try:
            s.df, s.filename = load_table(path), path
            reset_downstream("data")
        except Exception as e:
            st.error(f"Could not read file: {e}")

df: pd.DataFrame | None = s.df
if df is None:
    st.info("Load a dataset to begin. Any tabular fraud dataset works (Ethereum, Sparkov, PaySim, BAF, ...).")
    st.stop()

st.success(f"**{s.filename}** · {df.shape[0]:,} rows × {df.shape[1]} columns")
st.dataframe(df.head(50), width="stretch", height=220)

# ---------------- schema -----------------
st.subheader("Describe your dataset")
cols = list(df.columns)
tag = f"{s.filename}_{df.shape}"
c1, c2, c3 = st.columns(3)
target = c1.selectbox("Target column (fraud flag)", cols, index=cols.index(guess_target(df)), key=f"t_{tag}")
labels = sorted(df[target].dropna().unique().tolist(), key=str)
default_pos = 1 if 1 in labels else (labels[-1] if labels else None)
pos = c2.selectbox("Value meaning FRAUD", labels, index=labels.index(default_pos) if default_pos in labels else 0, key=f"p_{tag}")
none = ["— none —"]
gt, ga = guess_time(df, target), guess_amount(df)
time_col = c3.selectbox("Time column (chronological split)", none + cols,
                        index=(cols.index(gt) + 1) if gt else 0, key=f"tm_{tag}")
c4, c5 = st.columns(2)
amount_col = c4.selectbox("Amount column (business cost)", none + cols,
                          index=(cols.index(ga) + 1) if ga else 0, key=f"a_{tag}")
entity_col = c5.selectbox("Customer / card / account id (behavioural features)", none + cols, key=f"e_{tag}")
drop = st.multiselect("Columns to drop (ids, hashes, leakage)", [c for c in cols if c != target],
                      default=suggest_drop_columns(df, target), key=f"d_{tag}")

norm = lambda v: None if v == none[0] else v
schema = Schema(target, pos, norm(time_col), norm(amount_col), norm(entity_col), drop)
if schema != s.schema:
    if s.schema is not None:           # schema edited -> everything downstream is stale
        reset_downstream("prep")
        s.prepared = None
    s.schema = schema

y = binarize_target(df[target], pos)
cb = class_balance(y)
m1, m2, m3 = st.columns(3)
m1.metric("Transactions", f"{cb['n']:,}")
m2.metric("Frauds", f"{cb['fraud']:,}")
m3.metric("Fraud rate", f"{cb['fraud_rate']:.3%}")
if cb["fraud"] < 30:
    st.warning("Very few frauds: results will be unstable (cf. dataset #7 in the catalogue).")
if schema.time_col is None:
    st.warning("No time column: a stratified random split will be used (signalled as a limitation).")
if {schema.time_col, schema.entity_col, schema.amount_col} & set(drop):
    st.caption("ℹ️ A time / amount / id column is in the drop list: it is still used to build features, "
               "but the raw id/time columns never reach the model.")

# ---------------- EDA -----------------
st.subheader("Exploration")
EDA_MAX = 200_000
dfe = df.sample(EDA_MAX, random_state=0) if len(df) > EDA_MAX else df
ye = y.loc[dfe.index]
tabs = st.tabs(["Balance", "Amounts", "Time", "Categories & hour", "Numeric summary"])
with tabs[0]:
    st.plotly_chart(plots.class_balance_fig(y), width="stretch")
with tabs[1]:
    if schema.amount_col:
        st.plotly_chart(plots.amount_by_class_fig(dfe[schema.amount_col], ye), width="stretch")
    else:
        st.caption("Select an amount column above.")
with tabs[2]:
    if schema.time_col:
        t = parse_time(dfe[schema.time_col])
        st.plotly_chart(plots.rate_over_time_fig(t, ye), width="stretch")
    else:
        st.caption("Select a time column above.")
with tabs[3]:
    cat_cols = [c for c in dfe.columns if c != target and not pd.api.types.is_numeric_dtype(dfe[c])
                and 1 < dfe[c].nunique() <= 60]
    if schema.time_col and pd.api.types.is_datetime64_any_dtype(parse_time(dfe[schema.time_col])):
        hour = parse_time(dfe[schema.time_col]).dt.hour
        st.plotly_chart(plots.rate_by_category_fig(hour, ye, 24, "Fraud rate by hour"), width="stretch")
    if cat_cols:
        pick = st.selectbox("Categorical variable", cat_cols)
        st.plotly_chart(plots.rate_by_category_fig(dfe[pick], ye), width="stretch")
with tabs[4]:
    num = dfe.select_dtypes("number")
    st.dataframe(num.describe().T, width="stretch")
    miss = dfe.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        st.caption("Share of missing values (imputed later with TRAIN medians)")
        st.dataframe(miss.rename("missing").to_frame().style.format("{:.1%}"), width="stretch")
