import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import requests
from bs4 import BeautifulSoup

def scrape_cricbuzz_bio(player_id):
    url = f"https://www.cricbuzz.com/profiles/{player_id}/"
    r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    bio = {}

    # Full name
    h1 = soup.find("h1", class_="cb-font-40")
    bio["fullname"] = h1.text.strip() if h1 else None

    # Country
    country_div = soup.find("div", class_="cb-font-12 text-gray")
    bio["country"] = country_div.text.strip() if country_div else None

    # Other info — born, batting style, bowling style, role
    info_divs = soup.find_all("div", class_="cb-col cb-col-67")
    for d in info_divs:
        txt = d.get_text(separator=" ").strip()
        if "Born" in txt:
            bio["born"] = txt.replace("Born", "").strip()
        if "Batting style" in txt:
            bio["battingStyle"] = txt.replace("Batting style", "").strip()
        if "Bowling style" in txt:
            bio["bowlingStyle"] = txt.replace("Bowling style", "").strip()
        if "Role" in txt:
            bio["playingRole"] = txt.replace("Role", "").strip()

    return bio

MATCHES_CSV = "data/IPL_Matches_2008_2025.csv"
BALLS_CSV   = "data/IPL_Ball_by_Ball_2008_2025.csv"

st.set_page_config(page_title="CricketLive Analytics", layout="wide")

@st.cache_resource
def load_data():
    matches = pd.read_csv(MATCHES_CSV)
    balls = pd.read_csv(BALLS_CSV)
    return matches, balls

matches, balls = load_data()

st.sidebar.title("CricketLive")

with st.sidebar.expander("Navigation", expanded=True):
    page = st.radio(
        "Select Page",
        [
            "Dashboard",
            "Matches",
            "Players",
            "Player vs Bowler",
            "Top Partnerships",
            "Win Predictor"
        ]
    )

# ================== Resources Section ==================
with st.sidebar.expander("Resources / Repo"):
    st.markdown("[GitHub Repository](https://github.com/Chran19/analytics-app)")
    st.markdown("[Documentation](https://drive.google.com/drive/folders/1Yny-ELnDTXn3HladqR23asdZwGBnno-I)")
    st.markdown("[Project Demo(Google Colab)](https://colab.research.google.com/drive/17UoY_XTxSWofQS2iqScEhR6HJ5_PJg4N)")


# ======================================================================
# DASHBOARD
# ======================================================================
if page == "Dashboard":
    st.title("IPL Analytics Dashboard")

    # =================== KPI CARDS ===================
    total_matches = len(matches)
    total_teams = matches["Team1"].nunique()
    total_seasons = matches["Season"].nunique()
    total_venues = matches["Venue"].nunique()

    st.markdown("""
    <style>
    .kpi-card {background:#0e1b2a; color:#d6e4ee; padding:20px; border-radius:12px;
                border:1px solid #21384c; text-align:center; transition:0.3s; margin-bottom:10px;}
    .kpi-card:hover {background:#162b42; transform: scale(1.05);}
    .kpi-value {font-size:28px; font-weight:600; color:#3eb4ff;}
    .kpi-label {font-size:14px; margin-top:5px;}
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip([c1, c2, c3, c4],
                               [total_matches, total_teams, total_seasons, total_venues],
                               ["Total Matches", "Total Teams", "Seasons", "Venues"]):
        col.markdown(f"<div class='kpi-card'><div class='kpi-value'>{val}</div><div class='kpi-label'>{label}</div></div>", unsafe_allow_html=True)

    # =================== Wins by Team ===================
    st.subheader("Wins by Team")
    win_count = matches["WinningTeam"].value_counts().reset_index()
    win_count.columns = ["Team", "Wins"]

    fig = px.bar(
        win_count,
        x="Team",
        y="Wins",
        color="Wins",
        color_continuous_scale="Blues",
        hover_data={"Team": True, "Wins": True},
        text="Wins",
        title="Team-wise Wins in IPL"
    )
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(yaxis=dict(title="Wins"), xaxis=dict(title="Team"))
    st.plotly_chart(fig, use_container_width=True)

    # =================== Matches per Season ===================
    st.subheader("Matches per Season")
    season_count = matches.groupby("Season").size().reset_index(name="Matches")
    fig = px.line(
        season_count,
        x="Season",
        y="Matches",
        markers=True,
        color_discrete_sequence=["#3eb4ff"],
        hover_data={"Season": True, "Matches": True},
        title="Number of Matches Across Seasons"
    )
    fig.update_layout(yaxis=dict(title="Matches Played"), xaxis=dict(title="Season"))
    st.plotly_chart(fig, use_container_width=True)

    # =================== Most Successful Venues ===================
    st.subheader("Most Successful Venues")
    venue_win = matches["Venue"].value_counts().reset_index()
    venue_win.columns = ["Venue", "Matches"]

    fig = px.bar(
        venue_win.head(15),  # top 15 venues
        x="Venue",
        y="Matches",
        color="Matches",
        color_continuous_scale="Viridis",
        hover_data={"Venue": True, "Matches": True},
        text="Matches",
        title="Top Venues by Matches Played"
    )
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(yaxis=dict(title="Matches"), xaxis=dict(title="Venue"))
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# MATCHES
# ======================================================================
elif page == "Matches":
    st.title("Match Analytics")

    selected_team = st.selectbox("Select Team", sorted(matches["Team1"].unique()))
    filtered = matches[(matches["Team1"] == selected_team) | (matches["Team2"] == selected_team)]

    col1, col2 = st.columns(2)
    col1.metric("Matches Played", len(filtered))
    col2.metric("Win Percentage", round((sum(filtered["WinningTeam"] == selected_team) / len(filtered)) * 100, 2))

    # =================== Win Distribution ===================
    st.subheader("Win Distribution")
    win_count = filtered["WinningTeam"].value_counts().reset_index()
    win_count.columns = ["Team", "Wins"]
    fig = px.pie(
        win_count,
        names="Team",
        values="Wins",
        title=f"Win Split for {selected_team}",
        hover_data=["Wins"],
        labels={"Wins": "Number of Wins"},
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05]*len(win_count))
    st.plotly_chart(fig, use_container_width=True)

    # =================== Runs Across Matches ===================
    st.subheader("Runs Across Matches")
    runs = balls.groupby(["ID", "BattingTeam"])["total_run"].sum().reset_index()
    team_runs = runs[runs["BattingTeam"] == selected_team]

    fig = px.bar(
        team_runs,
        x="ID",
        y="total_run",
        hover_data={"ID": True, "total_run": True, "BattingTeam": True},
        labels={"ID": "Match ID", "total_run": "Runs Scored"},
        color="total_run",
        color_continuous_scale="Blues",
        title=f"Runs Scored by {selected_team} in Matches"
    )
    fig.update_traces(hovertemplate='<b>Match ID:</b> %{x}<br><b>Runs:</b> %{y}<br><b>Team:</b> %{customdata[2]}')
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# PLAYERS
# ======================================================================
elif page == "Players":
    st.title("Player Analytics Dashboard")

    import os
    import requests
    from functools import lru_cache
    from bs4 import BeautifulSoup

    RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "Your api key here")
    IMAGE_PATH = "images"   # local folder containing player images (e.g., images/Jos Buttler.jpg)

    @lru_cache(maxsize=400)
    def search_player_id(name):
        url = f"https://cricbuzz-cricket.p.rapidapi.com/players/search?plrN={name}"
        headers = {
            "x-rapidapi-host": "cricbuzz-cricket.p.rapidapi.com",
            "x-rapidapi-key": RAPIDAPI_KEY
        }
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        if "player" not in data or len(data["player"]) == 0:
            return None
        return data["player"][0]["id"]

    @lru_cache(maxsize=400)
    def fetch_player_profile(pid):
        if pid is None:
            return None
        url = f"https://cricbuzz-cricket.p.rapidapi.com/stats/v1/player/{pid}"
        headers = {
            "x-rapidapi-host": "cricbuzz-cricket.p.rapidapi.com",
            "x-rapidapi-key": RAPIDAPI_KEY
        }
        r = requests.get(url, headers=headers, timeout=10)
        return r.json() if r.status_code == 200 else None

    # ---------- SCRAPER FALLBACK ----------
    def scrape_cricbuzz_bio(player_id):
        url = f"https://www.cricbuzz.com/profiles/{player_id}/"
        r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        bio = {}

        h1 = soup.find("h1", class_="cb-font-40")
        bio["fullname"] = h1.text.strip() if h1 else None

        country_div = soup.find("div", class_="cb-font-12 text-gray")
        bio["country"] = country_div.text.strip() if country_div else None

        info_divs = soup.find_all("div", class_="cb-col cb-col-67")
        for d in info_divs:
            txt = d.get_text(separator=" ").strip()
            if "Born" in txt:
                bio["born"] = txt.replace("Born", "").strip()
            if "Batting style" in txt:
                bio["battingStyle"] = txt.replace("Batting style", "").strip()
            if "Bowling style" in txt:
                bio["bowlingStyle"] = txt.replace("Bowling style", "").strip()
            if "Role" in txt:
                bio["playingRole"] = txt.replace("Role", "").strip()
        return bio

    players = sorted(balls["batter"].dropna().unique())
    player = st.selectbox("Select Player", players)
    player_data = balls[balls["batter"] == player].copy()

    pid = search_player_id(player)
    profile = fetch_player_profile(pid) if pid else None

    # Use scraper fallback if API bio is missing
    bio_keys = ["fullname", "country", "playingRole", "battingStyle", "bowlingStyle", "born"]
    if pid and (not profile or not any(profile.get(k) for k in bio_keys)):
        scraped = scrape_cricbuzz_bio(pid)
        if scraped:
            profile = scraped

    img_url = profile.get("image") if profile and "image" in profile else None
    local_path = os.path.join(IMAGE_PATH, f"{player}.jpg")

    st.markdown("<div style='display:flex; align-items:center; gap:20px; margin-bottom:25px;'>", unsafe_allow_html=True)
    if img_url:
        st.markdown(
            f"<img src='{img_url}' style='width:135px; height:135px; border-radius:10px; border:2px solid #224567;'>",
            unsafe_allow_html=True
        )
    elif os.path.exists(local_path):
        st.image(local_path, width=135)
    else:
        st.markdown(
            "<div style='width:135px; height:135px; border-radius:10px; border:2px solid #224567; "
            "background:#101b2a; display:flex; align-items:center; justify-content:center; "
            "color:#5e8db9;'>No Image</div>",
            unsafe_allow_html=True
        )
    st.markdown(f"<h2 style='margin:0; padding:0; color:#ffffff;'>{player}</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ========= PLAYER BIO =========
    st.subheader("Player Info / Biography")
    if profile:
        bio_parts = []
        name = profile.get("fullname") or player
        country = profile.get("country")
        role = profile.get("playingRole")
        batting_style = profile.get("battingStyle")
        bowling_style = profile.get("bowlingStyle")
        dob = profile.get("born")
        started = profile.get("debut")

        if dob:
            bio_parts.append(f"{name} was born on {dob}.")
        if country:
            bio_parts.append(f"They represent {country} in international cricket.")
        if role:
            bio_parts.append(f"Primarily plays as a {role}.")
        if batting_style:
            bio_parts.append(f"Batting style: {batting_style}.")
        if bowling_style:
            bio_parts.append(f"Bowling style: {bowling_style}.")
        if started:
            bio_parts.append(f"Made their international debut in {started}.")

        bio_text = " ".join(bio_parts)
        if not bio_text:
            bio_text = "Biography information is not available for this player."

        st.markdown(f"<p style='color:#d6e4ee; line-height:1.6;'>{bio_text}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#d6e4ee;'>Biography information is not available for this player.</p>", unsafe_allow_html=True)





    # ========= CORE STATS =========
    runs = int(player_data["batsman_run"].sum()) if len(player_data) else 0
    balls_faced = len(player_data)
    matches = player_data["ID"].nunique()
    dismissals = int(player_data[player_data["isWicketDelivery"] == 1].shape[0]) if "isWicketDelivery" in player_data.columns else 0
    strike_rate = round((runs / balls_faced * 100), 2) if balls_faced else 0
    avg = round((runs / dismissals), 2) if dismissals else runs
    dot_balls = int((player_data["batsman_run"] == 0).sum())

    st.markdown("""
    <style>
    .stat-card {
        background:#0e1b2a; color:#d6e4ee; padding:18px; border-radius:12px;
        border:1px solid #21384c; text-align:center; transition:0.25s;
    }
    .stat-card:hover { background:#162b42; transform: scale(1.05); border-color:#2f80c7; }
    .stat-value { font-size:26px; font-weight:600; color:#3eb4ff; }
    .stat-label { font-size:14px; color:#aaccee; margin-top:4px; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    vals = [runs, balls_faced, matches, strike_rate, avg, dot_balls]
    labels = ["Runs", "Balls", "Matches", "Strike Rate", "Avg", "Dot Balls"]

    for col, v, lbl in zip([c1, c2, c3, c4, c5, c6], vals, labels):
        col.markdown(f"<div class='stat-card'><div class='stat-value'>{v}</div><div class='stat-label'>{lbl}</div></div>", unsafe_allow_html=True)

    # ========= TIMELINE =========
    st.subheader("Match-wise Run Timeline")
    if len(player_data) > 0:
        tline = player_data.groupby("ID")["batsman_run"].sum().reset_index()
        fig = px.line(tline, x="ID", y="batsman_run", markers=True, color_discrete_sequence=["#3eb4ff"])
        st.plotly_chart(fig, use_container_width=True)

    # ========= PHASES =========
    st.subheader("Performance Across Innings Phases")
    if "overs" in player_data.columns:
        def phase(over):
            over = int(over)
            if over <= 5: return "Powerplay"
            if over <= 15: return "Middle"
            return "Death"
        player_data["Phase"] = player_data["overs"].apply(phase)
        phase_df = player_data.groupby("Phase")["batsman_run"].sum().reset_index()
        fig = px.bar(phase_df, x="Phase", y="batsman_run", text_auto=True, color="batsman_run", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    # ========= BOWLER MATCHUPS =========
    st.subheader("Bowler Matchups")
    if len(player_data) and "bowler" in player_data.columns:
        bowler_summary = player_data.groupby("bowler")["batsman_run"].sum().reset_index().sort_values("batsman_run", ascending=False)
        top_bowlers = bowler_summary.head(12).reset_index(drop=True)
        cols = st.columns(3)
        for i, r in top_bowlers.iterrows():
            col = cols[i % 3]
            col.markdown(
                f"<div style='background:#0e1b2a;padding:12px;border-radius:10px;border:1px solid #1c3348;margin-bottom:10px;'>"
                f"<b>{r['bowler']}</b><br>"
                f"<span style='font-size:22px;color:#3eb4ff;'>{int(r['batsman_run'])} Runs</span>"
                "</div>",
                unsafe_allow_html=True
            )

    # ========= EXPORT =========
    csv = player_data.to_csv(index=False).encode()
    st.download_button("Download Player Data", csv, f"{player}_data.csv", "text/csv")


# ======================================================================
# PLAYER vs BOWLER HEAD-TO-HEAD
# ======================================================================
elif page == "Player vs Bowler":
    st.title("Batter vs Bowler – Head-to-Head Analytics")

    balls["Season"] = balls["ID"].map(matches.set_index("ID")["Season"])

    # ====================== Filters ======================
    seasons = sorted(balls["Season"].dropna().unique())
    season = st.selectbox("Select Season", ["All"] + seasons)

    flt = balls.copy()
    if season != "All":
        flt = flt[flt["Season"] == season]

    teams = sorted(flt["BattingTeam"].dropna().unique())
    team = st.selectbox("Select Team (optional)", ["All"] + teams)
    if team != "All":
        flt = flt[(flt["BattingTeam"] == team)]


    batters = sorted(flt["batter"].dropna().unique())
    batter = st.selectbox("Select Batter", batters)

    bowlers = sorted(flt["bowler"].dropna().unique())
    bowler = st.selectbox("Select Bowler", bowlers)

    h2h = flt[(flt["batter"] == batter) & (flt["bowler"] == bowler)]

    if h2h.empty:
        st.warning("No direct confrontation recorded between the selected Batter and Bowler.")
        st.stop()

    wicket_col = next(
        (c for c in ["isWicketDelivery", "isWicket", "wicket"] if c in flt.columns),
        None
    )

    runs = int(h2h["batsman_run"].sum())
    balls_faced = len(h2h)
    dismissals = h2h[h2h[wicket_col] == 1].shape[0] if wicket_col else 0
    strike_rate = round((runs / balls_faced * 100), 2) if balls_faced else 0
    average = round((runs / dismissals), 2) if dismissals else "Not Dismissed"
    dot_balls = int((h2h["batsman_run"] == 0).sum())

    # ==================== KPI Card Styling =====================
    st.markdown("""
    <style>
        .kpi-card {padding:15px;border-radius:10px;background:#112233;
                   border:1px solid #1b3958;color:#d8e6ee;text-align:center;
                   transition:0.25s;}
        .kpi-card:hover {background:#173650;transform:scale(1.03);border-color:#3087c3;}
        .kpi-value {font-size:28px;font-weight:600;color:#33b7ff;}
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f"<div class='kpi-card'><div class='kpi-value'>{runs}</div>Runs</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='kpi-card'><div class='kpi-value'>{balls_faced}</div>Balls Faced</div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='kpi-card'><div class='kpi-value'>{dismissals}</div>Times Out</div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='kpi-card'><div class='kpi-value'>{strike_rate}</div>Strike Rate</div>", unsafe_allow_html=True)
    col5.markdown(f"<div class='kpi-card'><div class='kpi-value'>{dot_balls}</div>Dot Balls</div>", unsafe_allow_html=True)

    # ==================== Match-wise Runs Timeline ====================
    timeline = h2h.groupby("ID")["batsman_run"].sum().reset_index()
    fig = px.line(
        timeline, x="ID", y="batsman_run", markers=True,
        title=f"Match-wise Runs: {batter} vs {bowler}",
        color_discrete_sequence=["#33b7ff"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==================== Aggression Matrix (Runs by Over) ====================
    if "over" in h2h.columns:
        over_agg = h2h.groupby("over")["batsman_run"].sum().reset_index()
        fig = px.bar(
            over_agg,
            x="over", y="batsman_run", text_auto=True,
            title="Over-wise Aggression Pattern",
            color="batsman_run",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==================== Boundary Pattern Scatter ====================
    boundaries = h2h[h2h["batsman_run"].isin([4, 6])]
    if len(boundaries) > 0:
        st.subheader("Boundary Breakdown")
        ball_col = next(
            (c for c in ["ball", "ballnumber", "overs_ball"] if c in boundaries.columns),
            None
        )
        if ball_col:
            boundaries["BallIndex"] = boundaries[ball_col]
        else:
            boundaries["BallIndex"] = range(len(boundaries))

        fig = px.scatter(
            boundaries,
            x="BallIndex", y="batsman_run",
            size="batsman_run", color="batsman_run",
            title="Boundary Placement Pattern",
            color_continuous_scale="Viridis",
            hover_data=["ID", "batsman_run", "Season"]
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==================== Match Log Table ====================
    st.subheader("Ball-by-Ball Log (Filtered)")
    show_cols = ["ID", "Season", "batsman_run", "batter", "bowler"]
    if wicket_col:
        show_cols.append(wicket_col)
    st.dataframe(h2h[show_cols].reset_index(drop=True))

    # ==================== Export CSV ====================
    csv = h2h.to_csv(index=False).encode()
    st.download_button(
        "Download Head-to-Head Dataset",
        csv,
        "h2h_batter_bowler.csv",
        mime="text/csv"
    )




# ======================================================================
# TOP PARTNERSHIPS
# ======================================================================
elif page == "Top Partnerships":
    st.title("Top Batting Partnerships")

    balls["Season"] = balls["ID"].map(matches.set_index("ID")["Season"])
    balls["ball_count"] = 1

    partnerships = balls.groupby(
        ["Season", "ID", "BattingTeam", "batter", "non-striker"]
    ).agg(
        Runs=("total_run", "sum"),
        Balls=("ball_count", "count")
    ).reset_index()

    partnerships["StrikeRate"] = (partnerships["Runs"] / partnerships["Balls"]) * 100

    # ======================= Filters (Dynamic) ==========================
    seasons = sorted(partnerships["Season"].unique())
    season = st.selectbox("Select Season", ["All"] + seasons)

    flt = partnerships.copy()
    if season != "All":
        flt = flt[flt["Season"] == season]

    teams = sorted(flt["BattingTeam"].unique())
    team = st.selectbox("Select Team", ["All"] + teams)

    if team != "All":
        flt = flt[flt["BattingTeam"] == team]

    batters = sorted(flt["batter"].unique())
    batter = st.selectbox("Select Batter", ["All"] + batters)

    if batter != "All":
        flt = flt[flt["batter"] == batter]

    non_strikers = sorted(flt["non-striker"].unique())
    non_striker = st.selectbox("Select Non-Striker", ["All"] + non_strikers)

    if non_striker != "All":
        flt = flt[flt["non-striker"] == non_striker]

    flt = flt.sort_values(by="Runs", ascending=False)

    # ======================= KPI Stats Grid =============================
    if len(flt) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Highest Partnership Runs", int(flt["Runs"].max()))
        c2.metric("Best Strike Rate", round(flt["StrikeRate"].max(), 2))
        c3.metric("Most Balls Faced", int(flt["Balls"].max()))
        duo = flt.groupby(["batter", "non-striker"])["Runs"].mean().idxmax()
        c4.metric("Most Consistent Duo", f"{duo[0]} + {duo[1]}")

    # ======================= Hover Style CSS =============================
    st.markdown("""
        <style>
        .partnership-card {
            padding: 14px;
            border-radius: 10px;
            background: #112233;
            border: 1px solid #1b3b5a;
            margin-bottom: 12px;
            color: #d7e5ec;
            transition: 0.25s;
        }
        .partnership-card:hover {
            background: #16324b;
            transform: scale(1.015);
            border-color: #2b74a5;
        }
        .partnership-runs {
            font-size: 28px;
            color: #29b8ff;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    # ======================= Partnership Cards Grid =====================
    st.subheader("Featured Partnerships")
    cards = flt.head(12).reset_index(drop=True)
    cols = st.columns(3)

    for i, r in cards.iterrows():
        idx = i % 3
        cols[idx].markdown(
            f"""
            <div class='partnership-card'>
                <b>{r['batter']} + {r['non-striker']}</b><br>
                <span class='partnership-runs'>{r['Runs']} Runs</span><br>
                Strike Rate: {round(r['StrikeRate'],2)} | Balls: {r['Balls']}<br>
                <span style='font-size:13px;color:#92a9b1'>
                    Season {r['Season']} – {r['BattingTeam']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ======================= MVP Score Leaderboard ======================
    flt["MVP"] = flt["Runs"] + (flt["StrikeRate"] * 0.5) - (flt["Balls"] * 0.1)
    st.subheader("Top 10 MVP Partnerships (Quality Score)")
    mvp = flt.sort_values(by="MVP", ascending=False).head(10)
    st.dataframe(mvp.reset_index(drop=True))

    # ======================= Runs Bar Chart =============================
    st.subheader("Best Partnerships – Runs")
    fig = px.bar(
        flt.head(15),
        x="Runs",
        y="batter",
        color="non-striker",
        orientation="h",
        title="Top 15 Partnerships by Runs",
        hover_data=["Season", "BattingTeam", "Balls", "StrikeRate"],
        color_discrete_sequence=px.colors.sequential.Agsunset
    )
    st.plotly_chart(fig, use_container_width=True)

    # ======================= Timeline Chart =============================
    st.subheader("Match-by-Match Partnership Timeline")
    if len(flt) > 0:
        timeline = flt.groupby("ID")["Runs"].sum().reset_index()
        fig = px.line(
            timeline, x="ID", y="Runs", markers=True,
            title="Partnership Runs Across Matches",
            color_discrete_sequence=["#29b8ff"]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No timeline for selected filters")

    # ======================= Highest Strike Rate ========================
    st.subheader("Highest Strike-Rate Partnerships")
    best_sr = flt.sort_values(by="StrikeRate", ascending=False).head(10)
    fig = px.bar(
        best_sr,
        x="StrikeRate",
        y="batter",
        color="non-striker",
        orientation="h",
        title="Top 10 Strike-Rate Partnerships",
        hover_data=["Runs", "Balls", "BattingTeam", "Season"],
        color_discrete_sequence=px.colors.sequential.Magenta
    )
    st.plotly_chart(fig, use_container_width=True)

    # ======================= Export Button ===============================
    csv = flt.to_csv(index=False).encode()
    st.download_button(
        "Download Filtered Partnerships Dataset",
        csv,
        "partnerships.csv",
        mime="text/csv"
    )


# ======================================================================
# WIN PREDICTOR – MACHINE LEARNING
# ======================================================================
elif page == "Win Predictor":
    st.title("Win Predictor – Machine Learning Model (Pre-Trained)")

    from models.predict import predict_winner
    import pickle

    with open("models/label_encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    le_team = enc["team"]
    le_venue = enc["venue"]

    teams = sorted(le_team.classes_)
    venues = sorted(le_venue.classes_)

    t1 = st.selectbox("Team 1", teams)
    t2 = st.selectbox("Team 2", teams)
    venue = st.selectbox("Venue", venues)
    toss_winner = st.selectbox("Toss Won By", [t1, t2])

    st.subheader("Live / Estimated Match Stats")
    total_runs = st.number_input("Runs Scored", 0, 300, 160)
    wickets = st.number_input("Wickets Down", 0, 10, 3)
    runrate = st.number_input("Run Rate", 0.0, 20.0, 8.0, step=0.1)

    if st.button("Predict Result"):
        winner, prob = predict_winner(t1, t2, venue, toss_winner, total_runs, wickets, runrate)
        st.success(f"Prediction: {winner} is most likely to win")

        prob_df = pd.DataFrame({"Team": le_team.classes_, "Probability": prob})
        fig = px.pie(prob_df, names="Team", values="Probability", title="Win Probability")
        st.plotly_chart(fig, use_container_width=True)


    # ================= LIVE MATCH MODE =================
    st.divider()
    st.subheader("Live Match Win Probability (during match)")

    over = st.slider("Overs Completed", 0.0, 20.0, 10.0, 0.1)
    runs = st.number_input("Total Runs", 0, 300, 80)
    wicket = st.number_input("Wickets Down", 0, 10, 2)

    if st.button("Calculate Live Win Chance"):
        crr = runs / over if over > 0 else 0
        live_score = (crr * 5) - (wicket * 2)
        live_prob_t1 = 50 + live_score
        live_prob_t2 = 100 - live_prob_t1
        fig = px.pie({"Team": [t1, t2], "Probability": [live_prob_t1, live_prob_t2]}, names="Team", values="Probability", title="Live Win Probability")
        st.plotly_chart(fig, use_container_width=True)
