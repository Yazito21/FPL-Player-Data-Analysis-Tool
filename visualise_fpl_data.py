# =============================
# Fantasy Premier League Stats Dashboard
# Built with Streamlit + Plotly + Pandas
# Author: Yazito21
# Description:
#   This tool allows users to upload FPL player stats (CSV or Excel),
#   filter players by multiple attributes, and visualize stats 
#   with scatter plots, league summaries, and radar charts.
# =============================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# --- Page Config ---
# Sets Streamlit page title and makes layout use full browser width
st.set_page_config(page_title="Fantasy Premier League Stats Dashboard", layout="wide")

# --- Inject Custom Theme (CSS Styling) ---
# Streamlit does not natively allow full styling, so we inject HTML <style> overrides
st.markdown(
    """
    <style>
    /* Background + text colors */
    .stApp {
        background-color: #00C49A; /* Teal green */
        color: black;
    }

    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background-color: #360D3A; /* Dark purple */
        color: white;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #360D3A !important; /* Dark purple */
    }

    /* Buttons */
    div.stButton > button {
        background-color: #E90052;
        color: white !important;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #ff2a6d;
        color: white !important;
    }

    /* Data table */
    .stDataFrame {
        background-color: white;
        color: black;
    }

    /* Fix black text in top bar + uploader */
    header[data-testid="stHeader"] * {
        color: white !important;
    }
    div[data-testid="stFileUploader"] button {
        color: white !important;
    }

    /* Hide default Streamlit footer + menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions for Centered Titles ---
def centered_subheader(text):
    """Display a subheader (H2) centered on the page."""
    st.markdown(f"<h2 style='text-align: center;'>{text}</h2>", unsafe_allow_html=True)

def centered_title(text):
    """Display a title (H1) centered on the page."""
    st.markdown(f"<h1 style='text-align: center;'>{text}</h1>", unsafe_allow_html=True)

# --- Premier League Logo (centered) ---
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg" width="500">
    </div>
    """,
    unsafe_allow_html=True
)

centered_title("Fantasy Premier League Stats Visualisation Tool")

# --- Upload Spreadsheet ---
# User uploads either CSV or Excel file containing player stats
uploaded_file = st.file_uploader("Upload your Premier League stats file", type=["csv", "xlsx"])

if uploaded_file:
    # --- Read Data ---
    # Skip first 2 rows (common in FPL exports where headers start at row 3)
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, skiprows=2)
    else:
        df = pd.read_excel(uploaded_file, skiprows=2)

    st.success("File uploaded successfully!")

    # --- Clean Data ---
    df.columns = [str(c).strip() for c in df.columns]  # strip spaces from headers
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # drop "Unnamed" empty columns
    df = df.apply(pd.to_numeric, errors='ignore')  # convert numbers if possible

    # Detect numeric columns for plotting
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # Filter by Team and Player
    with st.sidebar.expander("Team / Player Selection", expanded=True):
        if "Team" in df.columns:
            teams = st.multiselect("Filter by Team", df["Team"].dropna().unique())
            if teams:
                df = df[df["Team"].isin(teams)]

        if "Player" in df.columns:
            players = st.multiselect("Filter by Player", df["Player"].dropna().unique())
            if players:
                df = df[df["Player"].isin(players)]

    # --- Group Filters into Categories ---
    filter_groups = {
        "General Info": ["Position", "Nationality", "Age"],
        "General Stats": ["Minutes", "Appearances", "Starts"],
        "Attack Stats": ["Goals", "Assists", "Shots"],
        "Defensive Stats": ["Tackles", "Interceptions", "Clearances"],
    }

    used_cols = set(["Player", "Team", "Points", "ICT"])
    for group_name, columns in filter_groups.items():
        with st.sidebar.expander(group_name, expanded=False):
            for col in columns:
                if col in df.columns:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) > 1:
                        selected_vals = st.multiselect(f"Filter by {col}", unique_vals, default=[])
                        if selected_vals:
                            df = df[df[col].isin(selected_vals)]
                    used_cols.add(col)

    # --- Catch-all filter for leftover columns ---
    leftover_cols = [c for c in df.columns if c not in used_cols]
    if leftover_cols:
        with st.sidebar.expander("Other", expanded=False):
            for col in leftover_cols:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) > 1:
                    selected_vals = st.multiselect(f"Filter by {col}", unique_vals, default=[])
                    if selected_vals:
                        df = df[df[col].isin(selected_vals)]

    # --- Font Size Controls (for charts) ---
    st.sidebar.header("Font Size Controls")
    axis_font_size = st.sidebar.slider("Axis Font Size", 8, 24, 12)
    legend_font_size = st.sidebar.slider("Legend Font Size", 8, 24, 12)

    # --- Scatter Plot: Points vs ICT ---
    if "Points" in df.columns and "ICT" in df.columns:
        centered_subheader("Visualisation: Points vs ICT")
        x_axis = st.sidebar.selectbox("X-axis", numeric_cols, index=numeric_cols.index("ICT") if "ICT" in numeric_cols else 0)
        y_axis = st.sidebar.selectbox("Y-axis", numeric_cols, index=numeric_cols.index("Points") if "Points" in numeric_cols else 1)
        
        # Plot scatter chart
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color="Team" if "Team" in df.columns else None,
            hover_data=["Player"] if "Player" in df.columns else None
        )
        fig.update_layout(
            font=dict(size=axis_font_size),
            legend=dict(font=dict(size=legend_font_size))
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- League Summary Table ---
    if "Team" in df.columns and numeric_cols:
        centered_subheader("League Summary (Average Stats per Team)")
        team_summary = df.groupby("Team")[numeric_cols].mean().reset_index()
        st.dataframe(team_summary)

    # --- Radar Chart: Player Comparison ---
    centered_subheader(" Player Comparison (Radar Chart)")
    if "Player" in df.columns and numeric_cols:
        st.markdown("### Player Filters")

        # Position filter
        if "Position" in df.columns:
            all_positions = df["Position"].dropna().unique().tolist()
            positions_selected = st.multiselect("Filter by Position", all_positions, default=[])
            if positions_selected:
                df = df[df["Position"].isin(positions_selected)]

        # Price filter
        if "Price" in df.columns:
            min_price, max_price = float(df["Price"].min()), float(df["Price"].max())
            price_range = st.slider("Filter by Price Range", min_price, max_price, (min_price, max_price))
            df = df[(df["Price"] >= price_range[0]) & (df["Price"] <= price_range[1])]

        # Player selection
        players_selected = st.multiselect(
            "Select players to compare",
            df["Player"].unique(),
            default=df["Player"].unique()[:5]
        )

        # Show full stats for selected players
        if players_selected:
            st.markdown("### Full Stats for Selected Players")
            st.dataframe(df[df["Player"].isin(players_selected)].reset_index(drop=True).round(2))

        # --- Preset stat groups by position (ATT, MID, DEF, GK) ---
        st.markdown("### Stats Selection")
        presets = {
            "ATT": ["Points", "ICT", "Threat", "GI/90", "xGI/90", "G/A Weight"],
            "MID": ["Points", "ICT", "GI/90", "xGI/90", "DefCon/90", "G/A Weight"],
            "DEF": ["Points", "ICT", "Influence", "xGI/90", "DefCon/90", "GC"],
            "GK": ["Points", "Saves/90", "CS/90"],
        }

        preset_choice = st.radio("Choose preset (you can still add/remove stats)", ["Custom"] + list(presets.keys()), horizontal=True)

        if preset_choice != "Custom":
            default_stats = [s for s in presets[preset_choice] if s in numeric_cols]
        else:
            # fallback default stats
            default_stats = ["% Owned", "Points", "ICT", "Form", "xGI/90", "DefCon/90"] if set(["% Owned", "Points", "ICT", "Form", "xGI/90", "DefCon/90"]).issubset(numeric_cols) else numeric_cols[:5]

        # Final stat selection
        stats_selected = st.multiselect("Select stats to compare", numeric_cols, default=default_stats)
        normalize = st.checkbox("Normalize stats to 0–100 scale", value=True)

        # --- Build Radar Chart ---
        if players_selected and stats_selected:
            if normalize:
                # Scale values to range 0–100 for easier comparison
                scaler = MinMaxScaler(feature_range=(0, 100))
                norm_data = pd.DataFrame(
                    scaler.fit_transform(df[stats_selected]), 
                    columns=stats_selected, 
                    index=df.index
                )
                df_norm = pd.concat([df[["Player", "Team"]], norm_data], axis=1)
            else:
                df_norm = df.copy()
    
            fig = go.Figure()
            table_data = []

            # Colors for different players
            colors = [
                "rgba(231, 76, 60, 0.33)",   # Red
                "rgba(46, 204, 113, 0.33)",  # Green
                "rgba(52, 152, 219, 0.33)",  # Blue
                "rgba(241, 196, 15, 0.33)",  # Yellow
                "rgba(155, 89, 182, 0.33)",  # Purple
                "rgba(230, 126, 34, 0.33)"   # Orange
            ]
    
            for i, p in enumerate(players_selected):
                # Normalized values for chart
                values = df_norm[df_norm["Player"] == p][stats_selected].mean().values.flatten().tolist()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=stats_selected,
                    fill='toself',
                    name=p,
                    line=dict(color=colors[i % len(colors)]),
                    fillcolor=colors[i % len(colors)]
                ))
    
                # Raw values for table (unscaled)
                raw_values = df[df["Player"] == p][stats_selected].mean().round(2).tolist()
                table_data.append([p] + raw_values)
    
            # Chart settings
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, tickfont=dict(size=axis_font_size))),
                legend=dict(font=dict(size=legend_font_size)),
                font=dict(size=axis_font_size),
                showlegend=True,
                margin=dict(l=50, r=50, t=50, b=50)
            )
    
            # Show radar chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Show stats table below chart
            # st.write("Player Stats Table")
            st.dataframe(pd.DataFrame(table_data, columns=["Player"] + stats_selected), use_container_width=True)
    
        else:
            st.info("Select at least one player and one stat to generate the radar chart.")

else:
    st.info("Upload a CSV or Excel file to get started.")
