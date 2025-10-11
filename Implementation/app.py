import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import geopandas as gpd

st.set_page_config(page_title="Urban Greening Explorer", layout="wide")

# -----------------------------
# Loaders
# -----------------------------
@st.cache_data
def load_trees_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
    df.columns = [c.upper() for c in df.columns]
    if "GEO_POINT_2D" in df.columns:
        parts = df["GEO_POINT_2D"].astype(str).str.split(",", n=1, expand=True)
        df["LATITUDE"] = pd.to_numeric(parts[0], errors="coerce")
        df["LONGITUDE"] = pd.to_numeric(parts[1], errors="coerce")
    if "DATE_PLANTED" in df.columns:
        df["DATE_PLANTED"] = pd.to_datetime(df["DATE_PLANTED"], errors="coerce")
    if "DIAMETER" in df.columns:
        df["DIAMETER"] = pd.to_numeric(df["DIAMETER"], errors="coerce")
    return df

@st.cache_data
def load_local_areas(path: str) -> gpd.GeoDataFrame:
    # Directly read the GeoJSON/Shape file
    gdf = gpd.read_file(path)
    # Normalize column naming
    if "name" in gdf.columns and "NAME" not in gdf.columns:
        gdf = gdf.rename(columns={"name": "NAME"})
    # Ensure projection is standardized
    return gdf.to_crs(4326)

# -----------------------------
# Spatial Join & Metrics
# -----------------------------
def attach_neighbourhood(trees_df: pd.DataFrame, areas_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    gtrees = gpd.GeoDataFrame(
        trees_df.copy(),
        geometry=gpd.points_from_xy(trees_df["LONGITUDE"], trees_df["LATITUDE"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gtrees, areas_gdf[["NAME", "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={"NAME": "LOCAL_AREA"})
    return pd.DataFrame(joined.drop(columns=["index_right"]))

def compute_area_km2(areas_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    metric = areas_gdf.to_crs(26910)
    areas_gdf["AREA_KM2"] = metric.geometry.area / 1e6
    return areas_gdf

def summarize_neighbourhoods(trees_with_area: pd.DataFrame, areas_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    grp = trees_with_area.groupby("LOCAL_AREA", dropna=False).agg(
        trees=("LOCAL_AREA", "count"),
        unique_species=("SPECIES_NAME", lambda s: s.dropna().nunique()),
        avg_diameter=("DIAMETER", "mean"),
        oldest_year=("DATE_PLANTED", lambda s: int(pd.to_datetime(s, errors="coerce").dt.year.min())
                     if pd.to_datetime(s, errors="coerce").notna().any() else np.nan),
        newest_year=("DATE_PLANTED", lambda s: int(pd.to_datetime(s, errors="coerce").dt.year.max())
                     if pd.to_datetime(s, errors="coerce").notna().any() else np.nan),
    ).reset_index()
    areas_small = areas_gdf[["NAME", "AREA_KM2"]].rename(columns={"NAME": "LOCAL_AREA"})
    out = grp.merge(areas_small, on="LOCAL_AREA", how="left")
    out["trees_per_km2"] = out["trees"] / out["AREA_KM2"]
    return out

# Main App
DATA_TREES = "data/public-trees.csv"
DATA_AREAS = "data/local_area_boundaries.geojson"

st.title("🌳 Urban Greening Explorer – Vancouver")

with st.spinner("Loading datasets..."):
    trees = load_trees_from_csv(DATA_TREES)
    areas = compute_area_km2(load_local_areas(DATA_AREAS))
    trees = attach_neighbourhood(trees, areas)

# Summary stats
st.header("Neighbourhood Summary (Exploratory Analysis)")
summary = summarize_neighbourhoods(trees, areas)
st.dataframe(summary, use_container_width=True)

# Exploratory charts
st.subheader("Species Counts (Top 15)")
top_species = trees["COMMON_NAME"].fillna("Unknown").value_counts().nlargest(15).reset_index()
top_species.columns = ["Common Name", "Count"]
chart1 = alt.Chart(top_species).mark_bar().encode(
    x="Count:Q", y=alt.Y("Common Name:N", sort="-x")
).properties(height=350)
st.altair_chart(chart1, use_container_width=True)

st.subheader("Planting Year Histogram")
year_series = pd.to_datetime(trees["DATE_PLANTED"], errors="coerce").dt.year.dropna().astype(int)
if not year_series.empty:
    chart2 = alt.Chart(pd.DataFrame({"Year": year_series})).mark_bar().encode(
        x="Year:O", y="count()"
    ).properties(height=350)
    st.altair_chart(chart2, use_container_width=True)
else:
    st.info("No valid planting years available.")

st.caption("Data: City of Vancouver Open Data (Public Trees & Local Area Boundaries).")