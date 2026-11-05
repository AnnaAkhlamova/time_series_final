"""Helpers shared by all notebooks, so every model is scored in exactly the same way."""
import numpy as np
import pandas as pd

H = 48          # hours to forecast
M = 24          # the benchmark's season: one day
SURNAME = "AKHLAMOVA"
DYNREG_DAYS = 60
CHRONOS_CONTEXT = 8192      # the longest context the model accepts; see notebook 03
CHRONOS_QUANTILES = [0.025, 0.5, 0.975]


def load_data():
    data = pd.read_csv("train.csv", parse_dates=["ds"])
    weather_fc = pd.read_csv("weather_forecast.csv", parse_dates=["ds"])
    assert len(weather_fc) == H
    assert weather_fc["ds"].iloc[0] == data["ds"].iloc[-1] + pd.Timedelta(hours=1)
    return data, weather_fc


# ---- Metrics and benchmark: the same definitions as the course template ----

def rmse(actual, forecast) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def mape(actual, forecast) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)


def winkler(actual, lower, upper, alpha: float = 0.05) -> float:
    actual, lower, upper = (np.asarray(a, float) for a in (actual, lower, upper))
    below = np.clip(lower - actual, 0, None)
    above = np.clip(actual - upper, 0, None)
    return float(np.mean((upper - lower) + 2 / alpha * (below + above)))


def score(actual, forecast, lower, upper) -> dict:
    actual, lower, upper = (np.asarray(a, float) for a in (actual, lower, upper))
    inside = (actual >= lower) & (actual <= upper)
    return {"RMSE": rmse(actual, forecast), "MAPE": mape(actual, forecast),
            "Winkler": winkler(actual, lower, upper), "coverage": float(inside.mean())}


def snaive(history, h: int = H, m: int = M):
    # The benchmark: repeat the last season, with the seasonal-naive 95% interval.
    history = np.asarray(history, float)
    point = np.array([history[-m + (i % m)] for i in range(h)])
    sigma = np.std(history[m:] - history[:-m], ddof=1)
    k = np.arange(h) // m
    half = 1.96 * sigma * np.sqrt(k + 1)
    return point, np.clip(point - half, 0, None), point + half


def check_submission(sub: pd.DataFrame, future: pd.Series) -> None:
    assert list(sub.columns) == ["ds", "forecast", "lower_95", "upper_95"], "wrong columns"
    assert len(sub) == H, f"need {H} rows, got {len(sub)}"
    assert (pd.to_datetime(sub["ds"]).to_numpy() == future.to_numpy()).all(), "the hours do not match"
    assert sub[["forecast", "lower_95", "upper_95"]].notna().all().all(), "missing values"
    assert (sub["lower_95"] >= 0).all(), "a lower bound is negative"
    assert (sub["lower_95"] <= sub["forecast"]).all() and (sub["forecast"] <= sub["upper_95"]).all(), \
        "the forecast must lie inside its interval"
    print("format OK")


# ---- Validation ----

def origins(data: pd.DataFrame, windows: int, h: int = H):
    # Rolling origins without overlap. Window 1 is the last 48 hours of the history, and every
    # window starts at 01:00, like the hidden horizon, so step 25-48 matches the private hours.
    for k in range(1, windows + 1):
        end = len(data) - (k - 1) * h
        yield k, data.iloc[:end - h], data.iloc[end - h:end]


def window_table(bt: pd.DataFrame, columns: list, metric: str = "RMSE", first: int = 1, last: int = H):
    # One row per model: the metric in each window, the mean, and direct wins against the benchmark.
    # We count head-to-head wins, because "best of the table" changes when the list of models changes.
    part = bt[bt["step"].between(first, last)]
    rows = {}
    for c in columns:
        values = {}
        for k, g in part.groupby("window"):
            if metric == "RMSE":
                values[k] = rmse(g["actual"], g[c])
            elif metric == "MAPE":
                values[k] = mape(g["actual"], g[c])
            else:
                values[k] = winkler(g["actual"], g[f"{c} lo"], g[f"{c} hi"])
        rows[c] = values
    table = pd.DataFrame(rows).T
    table.columns = [f"w{k}" for k in table.columns]
    windows = list(table.columns)
    table["mean"] = table[windows].mean(axis=1)
    table["beats benchmark"] = table[windows].lt(table.loc["benchmark", windows]).sum(axis=1)
    return table


# ---- Dynamic regression: the Session 1 model ----

def make_X(df: pd.DataFrame) -> np.ndarray:
    # Hour dummies for the daily shape, hour x cloud so cloud can act differently by hour, and three
    # weather columns. There is no level term: the level comes only from the training window.
    hour = pd.get_dummies(df["ds"].dt.hour, prefix="h").astype(float)
    hour_cloud = hour.mul(df["cloud_cover"].to_numpy(), axis=0).add_prefix("cloud_")
    return pd.concat([hour, hour_cloud, df[["wind_speed", "temperature", "humidity"]]], axis=1).to_numpy(float)


def dynreg(train_df: pd.DataFrame, future_df: pd.DataFrame, h: int = H):
    # OLS, then AR(1) on the residuals: a regression with AR errors, written out to stay fast.
    X, Xf, yt = make_X(train_df), make_X(future_df), train_df["y"].to_numpy(float)
    beta = np.linalg.lstsq(X, yt, rcond=None)[0]
    res = yt - X @ beta
    phi = np.sum(res[1:] * res[:-1]) / np.sum(res[:-1] ** 2)
    point = np.clip(Xf @ beta + res[-1] * phi ** np.arange(1, h + 1), 0, None)
    # The spread depends strongly on the hour, so each hour gets its own residual sd.
    sd_by_hour = pd.Series(res).groupby(train_df["ds"].dt.hour.to_numpy()).std()
    half = 1.96 * sd_by_hour.reindex(future_df["ds"].dt.hour).to_numpy()
    return point, np.clip(point - half, 0, None), point + half


# ---- Chronos-2 ----

def load_chronos():
    import torch
    from chronos import Chronos2Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)


def chronos_predict(pipe, past_df, future_df, features: list, context: int = CHRONOS_CONTEXT,
                    log: bool = False, h: int = H) -> dict:
    # Chronos-2 is pretrained: predict_df only reads the context, it does not fit on our data.
    ctx = past_df[["ds", "y"] + features].iloc[-context:].assign(item_id="portfolio")
    if log:
        ctx["y"] = np.log1p(ctx["y"])
    fut = future_df[["ds"] + features].assign(item_id="portfolio") if features else None
    pred = pipe.predict_df(ctx, future_df=fut, prediction_length=h, quantile_levels=CHRONOS_QUANTILES,
                           id_column="item_id", timestamp_column="ds", target="y").sort_values("ds")
    undo = np.expm1 if log else (lambda v: v)
    # Quantiles stay valid after undoing the log, so the log run uses the 0.5 quantile as its point.
    median = np.clip(undo(pred["0.5"].to_numpy()), 0, None)
    point = median if log else np.clip(pred["predictions"].to_numpy(), 0, None)
    return {"median": median, "point": point,
            "lo": np.minimum(np.clip(undo(pred["0.025"].to_numpy()), 0, None), point),
            "hi": np.maximum(undo(pred["0.975"].to_numpy()), point)}
