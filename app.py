

import io
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ─────────────────────────── Page Config ────────────────────────────────────

st.set_page_config(
    page_title="📊 Data Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Constants ──────────────────────────────────────

FALLBACK_DATASET_URL = (
    "https://raw.githubusercontent.com/nicholasgasior/next-direction/"
    "master/netflix_titles.csv"
)

NETFLIX_COLUMNS = {
    "show_id": str,
    "type": str,
    "title": str,
    "director": str,
    "cast": str,
    "country": str,
    "date_added": str,
    "release_year": "Int64",
    "rating": str,
    "duration": str,
    "listed_in": str,
    "description": str,
}

PALETTE = [
    "#E50914", "#B81D24", "#F5C518", "#00BCD4", "#4CAF50",
    "#FF9800", "#9C27B0", "#3F51B5", "#009688", "#795548",
]

# ─────────────────────────── Helpers ────────────────────────────────────────

def styled_header(title: str, subtitle: str = "") -> None:
    """Render a styled title + optional subtitle."""
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")


def kpi_card(col, label: str, value, delta=None, icon: str = "") -> None:
    """Render a KPI metric card in the given Streamlit column."""
    with col:
        st.metric(label=f"{icon} {label}", value=value, delta=delta)


def fmt_number(n) -> str:
    """Format a large number with commas."""
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return str(n)


# ─────────────────────────── Data Loading ────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_builtin_netflix() -> pd.DataFrame:
    """
    Load the Netflix titles dataset.
    Order of attempts:
      1. Kaggle API (if credentials present)
      2. GitHub raw fallback URL
    Returns a cleaned DataFrame or empty DataFrame on failure.
    """
    # --- attempt 1: Kaggle API ---
    try:
        import kaggle  # noqa: PLC0415
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "shivamb/netflix-shows",
            path="/tmp/netflix_kaggle",
            unzip=True,
            quiet=True,
        )
        df = pd.read_csv("/tmp/netflix_kaggle/netflix_titles.csv")
        return clean_dataframe(df)
    except Exception:
        pass

    # --- attempt 2: fallback GitHub mirror ---
    try:
        resp = requests.get(FALLBACK_DATASET_URL, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        return clean_dataframe(df)
    except Exception:
        return pd.DataFrame()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame:
    - Drop full duplicates
    - Strip whitespace from string columns
    - Coerce numeric columns
    - Fill missing categoricals with 'Unknown'
    """
    df = df.copy()

    # Remove exact duplicate rows
    df.drop_duplicates(inplace=True)

    # Strip string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Coerce numeric columns
    for col in df.select_dtypes(include="object").columns:
        try:
            converted = pd.to_numeric(df[col], errors="ignore")
            if converted.dtype != object:
                df[col] = converted
        except Exception:
            pass

    # Fill missing categoricals
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

    # Netflix-specific: parse release_year safely
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

    # Netflix-specific: parse date_added to year
    if "date_added" in df.columns:
        df["year_added"] = pd.to_datetime(
            df["date_added"], errors="coerce"
        ).dt.year

    return df


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load and clean a user-uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        return clean_dataframe(df)
    except Exception as exc:
        st.error(f"❌ Could not read file: {exc}")
        return pd.DataFrame()


# ─────────────────────────── Sidebar Controls ────────────────────────────────

def build_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render all sidebar controls and return the filtered DataFrame.
    Supports:
    - Dropdown filters for low-cardinality string columns
    - Slider filters for numeric columns
    - Sort control
    """
    st.sidebar.header("🎛️ Dashboard Controls")
    st.sidebar.markdown("---")

    filtered_df = df.copy()

    # ── Dropdown filters ────────────────────────────────────────────────────
    st.sidebar.subheader("📂 Categorical Filters")

    # Identify good categorical columns (string, low cardinality, not IDs)
    cat_cols = [
        c for c in df.select_dtypes(include="object").columns
        if df[c].nunique() <= 50 and df[c].nunique() > 1
        and c.lower() not in ("show_id", "title", "description", "cast",
                               "director", "date_added", "duration")
    ]

    active_cat_filters: dict[str, list] = {}
    for col in cat_cols[:6]:  # cap at 6 dropdowns for UX
        unique_vals = sorted(df[col].dropna().unique().tolist())
        selected = st.sidebar.multiselect(
            f"Filter by **{col.replace('_', ' ').title()}**",
            options=unique_vals,
            default=[],
            key=f"cat_{col}",
        )
        if selected:
            active_cat_filters[col] = selected

    # Apply categorical filters
    for col, vals in active_cat_filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(vals)]

    # ── Numeric / Slider filters ────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔢 Numeric Filters")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols[:4]:  # cap at 4 sliders
        col_min = int(df[col].min()) if not pd.isna(df[col].min()) else 0
        col_max = int(df[col].max()) if not pd.isna(df[col].max()) else 1
        if col_min == col_max:
            continue
        selected_range = st.sidebar.slider(
            f"**{col.replace('_', ' ').title()}** range",
            min_value=col_min,
            max_value=col_max,
            value=(col_min, col_max),
            key=f"num_{col}",
        )
        filtered_df = filtered_df[
            (filtered_df[col] >= selected_range[0])
            & (filtered_df[col] <= selected_range[1])
        ]

    # ── Sort control ────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔃 Sort Options")
    sort_col = st.sidebar.selectbox(
        "Sort by column",
        options=["(none)"] + df.columns.tolist(),
        key="sort_col",
    )
    sort_asc = st.sidebar.radio(
        "Sort direction",
        options=["Ascending ↑", "Descending ↓"],
        index=0,
        key="sort_dir",
    )

    if sort_col != "(none)":
        filtered_df = filtered_df.sort_values(
            by=sort_col,
            ascending=(sort_asc == "Ascending ↑"),
        )

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"🔎 **{len(filtered_df):,}** of **{len(df):,}** rows visible after filters."
    )

    return filtered_df


# ─────────────────────────── Dataset Overview ────────────────────────────────

def show_dataset_overview(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Render dataset shape, column info, and summary statistics."""
    st.subheader("📋 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Total Rows", fmt_number(df.shape[0]), icon="📄")
    kpi_card(c2, "Columns", df.shape[1], icon="📊")
    kpi_card(c3, "Filtered Rows", fmt_number(len(filtered_df)), icon="🔍")
    kpi_card(c4, "Missing Values", fmt_number(df.isnull().sum().sum()), icon="❓")

    with st.expander("📌 Column Information", expanded=False):
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.values,
            "Non-Null Count": df.notnull().sum().values,
            "Unique Values": df.nunique().values,
            "Missing %": (df.isnull().mean() * 100).round(2).values,
        })
        st.dataframe(col_info, use_container_width=True)

    with st.expander("📈 Summary Statistics", expanded=False):
        st.dataframe(filtered_df.describe(include="all"), use_container_width=True)


# ─────────────────────────── KPI Metrics ─────────────────────────────────────

def show_kpis(filtered_df: pd.DataFrame) -> None:
    """Compute and render KPI cards for numeric columns."""
    st.subheader("💡 Key Performance Indicators")

    num_cols = filtered_df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("No numeric columns available for KPI computation.")
        return

    # Show KPIs for up to 4 numeric columns
    display_cols = num_cols[:4]
    cols = st.columns(len(display_cols))

    for i, col in enumerate(display_cols):
        series = filtered_df[col].dropna()
        label = col.replace("_", " ").title()
        total = series.sum()
        avg = series.mean()
        mx = series.max()
        mn = series.min()

        with cols[i]:
            st.markdown(f"**{label}**")
            m1, m2 = st.columns(2)
            m1.metric("Total", fmt_number(total))
            m2.metric("Avg", f"{avg:,.1f}")
            m3, m4 = st.columns(2)
            m3.metric("Max ↑", fmt_number(mx))
            m4.metric("Min ↓", fmt_number(mn))


# ─────────────────────────── Top / Bottom Records ────────────────────────────

def show_top_bottom(filtered_df: pd.DataFrame) -> None:
    """Show top 5 and bottom 5 records."""
    st.subheader("📑 Top & Bottom Records")
    tab_top, tab_bottom = st.tabs(["🏆 Top 5 Rows", "📉 Bottom 5 Rows"])

    with tab_top:
        st.dataframe(filtered_df.head(5), use_container_width=True)

    with tab_bottom:
        st.dataframe(filtered_df.tail(5), use_container_width=True)


# ─────────────────────────── Visualizations ──────────────────────────────────

def show_visualizations(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Render all charts with interactive column selection."""
    st.subheader("📊 Visualizations")

    cat_cols = [
        c for c in filtered_df.select_dtypes(include="object").columns
        if filtered_df[c].nunique() <= 30 and filtered_df[c].nunique() > 1
    ]
    num_cols = filtered_df.select_dtypes(include="number").columns.tolist()

    if not cat_cols and not num_cols:
        st.warning("⚠️ No suitable columns found for visualization.")
        return

    # ── Chart configuration row ─────────────────────────────────────────────
    st.markdown("##### ⚙️ Chart Column Selection")
    cfg1, cfg2, cfg3 = st.columns(3)

    with cfg1:
        bar_col = st.selectbox(
            "Bar / Pie chart — categorical column",
            options=cat_cols if cat_cols else ["N/A"],
            key="bar_col",
        )
    with cfg2:
        line_col = (
            st.selectbox(
                "Line chart — time / numeric X axis",
                options=num_cols if num_cols else ["N/A"],
                key="line_col",
            )
            if num_cols
            else None
        )
    with cfg3:
        hist_col = (
            st.selectbox(
                "Histogram — numeric column",
                options=num_cols if num_cols else ["N/A"],
                key="hist_col",
            )
            if num_cols
            else None
        )

    st.markdown("---")

    # ── Row 1: Bar + Pie ────────────────────────────────────────────────────
    r1c1, r1c2 = st.columns(2)

    # Bar chart
    with r1c1:
        st.markdown("**📊 Bar Chart — Top 10 by Count**")
        if bar_col and bar_col != "N/A":
            bar_data = (
                filtered_df[bar_col]
                .value_counts()
                .head(10)
                .reset_index()
            )
            bar_data.columns = [bar_col, "Count"]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(
                bar_data[bar_col].astype(str),
                bar_data["Count"],
                color=PALETTE[: len(bar_data)],
            )
            ax.invert_yaxis()
            ax.set_xlabel("Count")
            ax.set_title(
                f"Top 10 {bar_col.replace('_', ' ').title()} by Count",
                fontsize=12,
            )
            ax.tick_params(axis="y", labelsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Highlight max
            if not bar_data.empty:
                mx_row = bar_data.loc[bar_data["Count"].idxmax()]
                st.success(
                    f"🏆 **Highest:** {mx_row[bar_col]}  →  {fmt_number(mx_row['Count'])} entries"
                )
        else:
            st.info("Select a categorical column.")

    # Pie chart
    with r1c2:
        st.markdown("**🥧 Pie Chart — Distribution**")
        if bar_col and bar_col != "N/A":
            pie_data = filtered_df[bar_col].value_counts().head(8)

            fig, ax = plt.subplots(figsize=(7, 4))
            wedges, texts, autotexts = ax.pie(
                pie_data.values,
                labels=pie_data.index.astype(str),
                autopct="%1.1f%%",
                colors=PALETTE[: len(pie_data)],
                startangle=140,
                pctdistance=0.82,
            )
            for at in autotexts:
                at.set_fontsize(8)
            ax.set_title(
                f"Distribution of {bar_col.replace('_', ' ').title()}",
                fontsize=12,
            )
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Select a categorical column.")

    # ── Row 2: Line + Histogram ─────────────────────────────────────────────
    r2c1, r2c2 = st.columns(2)

    # Line chart (trend over numeric/time axis)
    with r2c1:
        st.markdown("**📈 Line Chart — Trend Over Time**")
        if line_col and line_col != "N/A" and line_col in filtered_df.columns:
            trend = (
                filtered_df.groupby(line_col)
                .size()
                .reset_index(name="Count")
                .sort_values(line_col)
            )
            if len(trend) > 1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(
                    trend[line_col].astype(str),
                    trend["Count"],
                    marker="o",
                    color=PALETTE[0],
                    linewidth=2,
                    markersize=5,
                )
                ax.fill_between(
                    trend[line_col].astype(str),
                    trend["Count"],
                    alpha=0.15,
                    color=PALETTE[0],
                )
                ax.set_xlabel(line_col.replace("_", " ").title())
                ax.set_ylabel("Count")
                ax.set_title(
                    f"Count Trend over {line_col.replace('_', ' ').title()}",
                    fontsize=12,
                )
                tick_step = max(1, len(trend) // 10)
                ax.set_xticks(ax.get_xticks()[::tick_step])
                plt.xticks(rotation=45, ha="right", fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Not enough unique values for a trend line.")
        else:
            st.info("Select a numeric column for the X axis.")

    # Histogram
    with r2c2:
        st.markdown("**📉 Histogram — Numeric Distribution**")
        if hist_col and hist_col != "N/A" and hist_col in filtered_df.columns:
            series = filtered_df[hist_col].dropna()
            if len(series) > 1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(
                    series,
                    bins=min(40, series.nunique()),
                    color=PALETTE[2],
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax.set_xlabel(hist_col.replace("_", " ").title())
                ax.set_ylabel("Frequency")
                ax.set_title(
                    f"Distribution of {hist_col.replace('_', ' ').title()}",
                    fontsize=12,
                )
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Highlight stats
                st.info(
                    f"Mean: **{series.mean():.2f}** | Median: **{series.median():.2f}** "
                    f"| Std: **{series.std():.2f}**"
                )
            else:
                st.info("Not enough data for a histogram.")
        else:
            st.info("Select a numeric column.")

    # ── Row 3: Heatmap ──────────────────────────────────────────────────────
    st.markdown("**🔥 Correlation Heatmap**")
    num_df = filtered_df.select_dtypes(include="number")
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        fig, ax = plt.subplots(
            figsize=(min(14, corr.shape[1] * 1.5), min(10, corr.shape[0] * 1.2))
        )
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax,
            annot_kws={"size": 9},
        )
        ax.set_title("Correlation Matrix (lower triangle)", fontsize=13)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("ℹ️ Need at least 2 numeric columns for a correlation heatmap.")


# ─────────────────────────── Full Data Table ─────────────────────────────────

def show_full_table(filtered_df: pd.DataFrame) -> None:
    """Render the full filtered data table with download button."""
    st.subheader("📋 Filtered Data Table")

    # Highlight max values in numeric columns
    num_cols = filtered_df.select_dtypes(include="number").columns.tolist()

    def highlight_extremes(col):
        if col.name not in num_cols:
            return [""] * len(col)
        styles = [""] * len(col)
        if col.max() != col.min():
            max_idx = col.idxmax()
            min_idx = col.idxmin()
            styles[filtered_df.index.get_loc(max_idx)] = (
                "background-color: #c6efce; color: #276221"
            )
            styles[filtered_df.index.get_loc(min_idx)] = (
                "background-color: #ffc7ce; color: #9c0006"
            )
        return styles

    try:
        styled = filtered_df.style.apply(highlight_extremes, axis=0)
        st.dataframe(styled, use_container_width=True, height=350)
    except Exception:
        st.dataframe(filtered_df, use_container_width=True, height=350)

    # Download button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="filtered_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ─────────────────────────── Main App ────────────────────────────────────────

def main() -> None:
    # ── Header ──────────────────────────────────────────────────────────────
    styled_header(
        "📊 Netflix Data Analytics Dashboard",
        "Explore, filter, and visualize your dataset with ease.",
    )

    # ── Data Source Selection ────────────────────────────────────────────────
    st.sidebar.header("📁 Data Source")
    data_source = st.sidebar.radio(
        "Choose a data source",
        options=["📺 Netflix Dataset (auto-load)", "📂 Upload your own CSV"],
        key="data_source",
    )

    df_raw = pd.DataFrame()

    if data_source == "📺 Netflix Dataset (auto-load)":
        with st.spinner("⏳ Loading Netflix dataset…"):
            df_raw = load_builtin_netflix()

        if df_raw.empty:
            st.error(
                "❌ Could not load the Netflix dataset automatically. "
                "Please upload a CSV file using the sidebar."
            )
        else:
            st.success(
                f"✅ Netflix dataset loaded successfully — "
                f"**{len(df_raw):,} rows × {df_raw.shape[1]} columns**."
            )

    else:
        uploaded = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="Upload any CSV file to explore it interactively.",
        )
        if uploaded is not None:
            with st.spinner("⏳ Reading your file…"):
                df_raw = load_uploaded_file(uploaded)
            if not df_raw.empty:
                st.success(
                    f"✅ File loaded — "
                    f"**{len(df_raw):,} rows × {df_raw.shape[1]} columns**."
                )
        else:
            st.info(
                "👈 Upload a CSV file from the sidebar to get started. "
                "Alternatively, switch to the Netflix dataset above."
            )

    if df_raw.empty:
        st.stop()

    # ── Sidebar Filters → Filtered DataFrame ────────────────────────────────
    filtered_df = build_sidebar(df_raw)

    if filtered_df.empty:
        st.warning(
            "⚠️ No data matches the current filters. "
            "Please adjust your filter selections."
        )
        st.stop()

    # ── Dashboard Sections ───────────────────────────────────────────────────
    show_dataset_overview(df_raw, filtered_df)
    st.markdown("")
    show_kpis(filtered_df)
    st.markdown("")
    show_top_bottom(filtered_df)
    st.markdown("")
    show_visualizations(df_raw, filtered_df)
    st.markdown("")
    show_full_table(filtered_df)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:gray; font-size:0.85rem;'>"
        "📊 Data Analytics Dashboard · Built with Streamlit, Pandas, Seaborn & Matplotlib"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
