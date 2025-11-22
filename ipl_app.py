import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

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
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Matches", "Players", "Player vs Bowler", "Top Partnerships", "Win Predictor"]
)

# ======================================================================
# DASHBOARD
# ======================================================================
if page == "Dashboard":
    st.title("IPL Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches", len(matches))
    col2.metric("Total Teams", matches["Team1"].nunique())
    col3.metric("Total Seasons", matches["Season"].nunique())
    col4.metric("Total Venues", matches["Venue"].nunique())

    st.subheader("Wins by Team")
    win_count = matches["WinningTeam"].value_counts().reset_index()
    win_count.columns = ["Team", "Wins"]
    fig = px.bar(win_count, x="Team", y="Wins", color="Team")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matches per Season")
    season_count = matches.groupby("Season").size().reset_index(name="Matches")
    fig = px.line(season_count, x="Season", y="Matches", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most Successful Venues")
    venue_win = matches["Venue"].value_counts().reset_index()
    venue_win.columns = ["Venue", "Matches"]
    fig = px.bar(venue_win, x="Venue", y="Matches")
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

    st.subheader("Win Distribution")
    fig = px.pie(filtered, names="WinningTeam", title=f"Win Split for {selected_team}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Runs Across Matches")
    runs = balls.groupby(["ID", "BattingTeam"])["total_run"].sum().reset_index()
    team_runs = runs[runs["BattingTeam"] == selected_team]
    fig = px.bar(team_runs, x="ID", y="total_run")
    st.plotly_chart(fig, use_container_width=True)
# ======================================================================
# PLAYERS
# ======================================================================
elif page == "Players":
    st.title("Player Analytics Dashboard")

    players = sorted(balls["batter"].dropna().unique())
    player = st.selectbox("Select Player", players)

    player_data = balls[balls["batter"] == player]

    total_runs = int(player_data["batsman_run"].sum())
    balls_faced = len(player_data)
    matches_played = player_data["ID"].nunique()

    dismissals_df = player_data[player_data["isWicketDelivery"] == 1]
    dismissals = len(dismissals_df)

    strike_rate = (total_runs / balls_faced * 100) if balls_faced else 0
    batting_avg = (total_runs / dismissals) if dismissals else total_runs

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Runs", total_runs)
    col2.metric("Balls", balls_faced)
    col3.metric("Matches", matches_played)
    col4.metric("Strike Rate", round(strike_rate, 2))
    col5.metric("Batting Avg", round(batting_avg, 2))

    # Timeline chart (Match-by-match progression)
    timeline = player_data.groupby("ID")["batsman_run"].sum()
    fig = px.line(timeline, markers=True, title=f"{player} – Match-wise Run Timeline")
    st.plotly_chart(fig, use_container_width=True)

    # Dismissal breakdown chart
    if len(dismissals_df) > 0:
        dismissal_types = dismissals_df["kind"].value_counts()
        fig = px.bar(
            dismissal_types,
            title="Dismissal Breakdown",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # Most productive bowler faced
    bowler_runs = player_data.groupby("bowler")["batsman_run"].sum().sort_values(ascending=False)
    st.subheader("Most Runs Scored Against Bowlers")
    st.dataframe(bowler_runs.reset_index().rename(columns={0: "Runs"}).head(10))
# ======================================================================
# PLAYER vs BOWLER HEAD-TO-HEAD
# ======================================================================
elif page == "Player vs Bowler":
    st.title("Batter vs Bowler – Head-to-Head Analytics")

    batters = sorted(balls["batter"].dropna().unique())
    bowlers = sorted(balls["bowler"].dropna().unique())

    batter = st.selectbox("Select Batter", batters)
    bowler = st.selectbox("Select Bowler", bowlers)

    h2h = balls[(balls["batter"] == batter) & (balls["bowler"] == bowler)]

    if h2h.empty:
        st.warning("No direct confrontation recorded between the selected Batter and Bowler.")
        st.stop()

    wicket_col = next(
        (col for col in ["isWicketDelivery", "isWicket", "wicket"] if col in balls.columns),
        None
    )

    runs = int(h2h["batsman_run"].sum())
    balls_faced = len(h2h)
    dismissals = h2h[h2h[wicket_col] == 1].shape[0] if wicket_col else 0
    strike_rate = (runs / balls_faced * 100) if balls_faced else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runs", runs)
    col2.metric("Balls Faced", balls_faced)
    col3.metric("Times Out", dismissals)
    col4.metric("Strike Rate", round(strike_rate, 2))

    # Match-by-match confrontation
    timeline = h2h.groupby("ID")["batsman_run"].sum()
    if len(timeline) > 0:
        fig = px.line(
            timeline, markers=True,
            title=f"Match-wise Runs – {batter} vs {bowler}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Length-wise aggression (extra pro)
    if "ball" in h2h.columns:
        over_distribution = h2h.groupby("over")["batsman_run"].sum()
        fig = px.bar(over_distribution, title="Over-wise Scoring Pattern", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    # Boundary analysis
    boundary_df = h2h[h2h["batsman_run"].isin([4, 6])]
    if len(boundary_df) > 0:
        st.subheader("Boundaries Hit Against the Bowler")

        ball_col = next(
            (col for col in ["ball", "ballnumber", "ball_id", "overs_ball"] if col in boundary_df.columns),
            None
        )

        display_cols = ["ID", "batsman_run"]
        if ball_col:
            display_cols.append(ball_col)

        st.dataframe(boundary_df[display_cols].reset_index(drop=True))



# ======================================================================
# TOP PARTNERSHIPS
# ======================================================================
elif page == "Top Partnerships":
    st.title("Top Batting Partnerships")

    # Add season column from matches dataset
    balls["Season"] = balls["ID"].map(matches.set_index("ID")["Season"])

    # Calculate total runs + balls faced per partnership
    balls["ball_count"] = 1
    partnerships = balls.groupby(
        ["Season", "ID", "BattingTeam", "batter", "non-striker"]
    ).agg(
        Runs=("total_run", "sum"),
        Balls=("ball_count", "count")
    ).reset_index()

    partnerships["StrikeRate"] = (partnerships["Runs"] / partnerships["Balls"]) * 100

    seasons = sorted(partnerships["Season"].unique())
    season = st.selectbox("Select Season", ["All"] + seasons)

    teams = sorted(partnerships["BattingTeam"].unique())
    team = st.selectbox("Select Team", ["All"] + teams)

    batters = sorted(partnerships["batter"].unique())
    batter = st.selectbox("Select Batter (optional)", ["All"] + batters)

    non_strikers = sorted(partnerships["non-striker"].unique())
    non_striker = st.selectbox("Select Non-Striker (optional)", ["All"] + non_strikers)

    flt = partnerships.copy()
    if season != "All": flt = flt[flt["Season"] == season]
    if team != "All": flt = flt[flt["BattingTeam"] == team]
    if batter != "All": flt = flt[flt["batter"] == batter]
    if non_striker != "All": flt = flt[flt["non-striker"] == non_striker]

    flt = flt.sort_values(by="Runs", ascending=False)

    st.subheader("Top Partnerships Ranked (by Runs)")
    st.dataframe(flt.head(25).reset_index(drop=True))

    fig = px.bar(
        flt.head(15),
        x="Runs",
        y="batter",
        color="non-striker",
        orientation="h",
        title="Best Partnerships – Runs",
        hover_data=["BattingTeam", "Season", "Balls", "StrikeRate"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # =============== Partnership Timeline Graph (Match by Match) ===============
    st.subheader("Partnership Timeline (Match by Match)")
    if len(flt) > 0:
        timeline = flt.groupby("ID")["Runs"].sum().reset_index()
        fig = px.line(
            timeline,
            x="ID",
            y="Runs",
            markers=True,
            title="Partnership Runs Across Matches"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data for timeline with current filter.")

    # =============== Biggest Strike Rate Partnerships ===============
    st.subheader("Biggest Strike-Rate Partnerships")
    best_sr = flt.sort_values(by="StrikeRate", ascending=False).head(10)
    st.dataframe(best_sr.reset_index(drop=True))

    fig = px.bar(
        best_sr,
        x="StrikeRate",
        y="batter",
        color="non-striker",
        orientation="h",
        title="Highest Strike-Rate Partnerships",
        hover_data=["Runs", "Balls", "BattingTeam", "Season"]
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# WIN PREDICTOR – MACHINE LEARNING
# ======================================================================
elif page == "Win Predictor":
    st.title("Win Predictor – Elite Version with Live Match Mode & Model Comparison")

    df = matches.dropna(subset=["Team1", "Team2", "WinningTeam"])
    le_team = LabelEncoder()
    le_venue = LabelEncoder()

    df["T1"] = le_team.fit_transform(df["Team1"])
    df["T2"] = le_team.transform(df["Team2"])
    df["VenueEnc"] = le_venue.fit_transform(df["Venue"])
    df["TossWin"] = (df["TossWinner"] == df["Team1"]).astype(int)
    df["Winner"] = le_team.transform(df["WinningTeam"])

    X = df[["T1", "T2", "VenueEnc", "TossWin"]]
    y = df["Winner"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=450, max_depth=12),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
        "Logistic Regression": LogisticRegression(max_iter=2000)
    }

    model_accuracy = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        model_accuracy[name] = accuracy_score(y_test, m.predict(X_test))

    best_model = max(model_accuracy, key=model_accuracy.get)
    model = models[best_model]

    st.info(f"Best model selected automatically: **{best_model}**")

    # ========= User Inputs =========
    teams = sorted(df["Team1"].unique())
    venues = sorted(df["Venue"].unique())

    t1 = st.selectbox("Team 1", teams)
    t2 = st.selectbox("Team 2", teams)
    venue = st.selectbox("Venue", venues)
    toss_winner = st.selectbox("Toss Won By", [t1, t2])

    if st.button("Predict Result"):
        t1_enc = le_team.transform([t1])[0]
        t2_enc = le_team.transform([t2])[0]
        venue_enc = le_venue.transform([venue])[0]
        toss_flag = 1 if toss_winner == t1 else 0

        inp = np.array([[t1_enc, t2_enc, venue_enc, toss_flag]])
        prob = model.predict_proba(inp)[0]
        prediction = le_team.inverse_transform([np.argmax(prob)])[0]

        st.success(f"Prediction: {prediction} is likely to win the match")

        # ====================== Graphs ======================
        # 1. Win Probability Pie Chart
        prob_df = pd.DataFrame({"Team": le_team.inverse_transform(np.arange(len(prob))), "Probability": prob})
        fig = px.pie(prob_df, names="Team", values="Probability", title="Win Probability")
        st.plotly_chart(fig, use_container_width=True)

        # 2. Feature Importance
        feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_ if hasattr(model, "feature_importances_") else [0]*len(X.columns)})
        fig = px.bar(feat_imp, x="Feature", y="Importance", title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

        # 3. Model Comparison Graph
        acc_df = pd.DataFrame({"Model": model_accuracy.keys(), "Accuracy": model_accuracy.values()})
        fig = px.bar(acc_df, x="Model", y="Accuracy", title="Accuracy of Trained Models")
        st.plotly_chart(fig, use_container_width=True)

        # 4. Historical Win Comparison of Teams
        h2h = df[(df["Team1"].isin([t1, t2])) & (df["Team2"].isin([t1, t2]))]
        head_to_head = h2h["WinningTeam"].value_counts().reset_index()
        head_to_head.columns = ["Team", "Wins"]
        fig = px.bar(head_to_head, x="Team", y="Wins", color="Team", title="Historical Head-to-Head Wins")
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
