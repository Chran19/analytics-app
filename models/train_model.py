import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

MATCHES = "../data/IPL_Matches_2008_2025.csv"
BALLS = "../data/IPL_Ball_by_Ball_2008_2025.csv"

def train_model():
    matches = pd.read_csv(MATCHES)
    balls = pd.read_csv(BALLS)

    balls["ball_count"] = 1
    over_stats = balls.groupby("ID").agg(
        TotalRuns=("total_run", "sum"),
        Wickets=("isWicketDelivery", "sum"),
        Boundaries=("batsman_run", lambda x: ((x == 4) | (x == 6)).sum()),
        BallsPlayed=("ball_count", "count")
    ).reset_index()
    over_stats["RunRate"] = over_stats["TotalRuns"] / over_stats["BallsPlayed"].replace(0, 1) * 6

    df = matches.merge(over_stats, on="ID", how="inner")
    df = df.dropna(subset=["Team1", "Team2", "WinningTeam", "Venue", "TossWinner"]).reset_index(drop=True)

    le_team = LabelEncoder()
    le_venue = LabelEncoder()

    df["T1"] = le_team.fit_transform(df["Team1"])
    df["T2"] = le_team.transform(df["Team2"])
    df["VenueEnc"] = le_venue.fit_transform(df["Venue"])
    df["TossWin"] = (df["TossWinner"] == df["Team1"]).astype(int)
    df["Winner"] = le_team.transform(df["WinningTeam"])
    df["Margin"] = df["Margin"].fillna(0)

    # separation of margin types – increases signal
    df["MarginRuns"] = np.where(df["WonBy"] == "Runs", df["Margin"], 0)
    df["MarginWkts"] = np.where(df["WonBy"] == "Wickets", df["Margin"], 0)

    # smarter venue bias (prob win per team at venue)
    venue_team_bias = (
    df.groupby(["Venue", "WinningTeam"]).size() / df.groupby("Venue").size()).rename("VenueTeamBias")
    # convert encoded Winner back to team name temporarily to match the index format
    df["WinnerName"] = le_team.inverse_transform(df["Winner"])
    df = df.join(venue_team_bias, on=["Venue", "WinnerName"])
    df["VenueTeamBias"] = df["VenueTeamBias"].fillna(df["VenueTeamBias"].mean())

    FEATURES = [
        "T1", "T2", "VenueEnc", "TossWin",
        "TotalRuns", "Wickets", "RunRate",
        "Boundaries", "BallsPlayed",
        "MarginRuns", "MarginWkts",
        "VenueTeamBias"
    ]

    X = df[FEATURES]
    y = df["Winner"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
        )

    model = XGBClassifier(
        n_estimators=700,
        max_depth=13,
        learning_rate=0.04,
        subsample=0.92,
        colsample_bytree=0.92,
        eval_metric="mlogloss"
        )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)


    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump({"team": le_team, "venue": le_venue}, open("label_encoders.pkl", "wb"))

    return model.score(X_test, y_test)

if __name__ == "__main__":
    acc = train_model()
    print("Training Completed | Accuracy:", acc)
