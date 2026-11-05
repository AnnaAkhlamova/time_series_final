# MATH840 final project

An hour-by-hour forecast of electricity generation for the 48 hours after `train.csv`, with a 95%
prediction interval. The final model is Chronos-2, a pretrained foundation model, with all weather columns and 8192 hours of context.
The point forecast is the model's own. The interval is the model's own, with its half-width capped.

## How to run

Tested on Windows with Python 3.13. The first Chronos run needs internet access, because the weights
download from Hugging Face. No API key is needed, and a CPU is enough.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
(if by chance smth went wrong with requirements.txt.  download libraries as pip install and name of the library

numpy==2.5.4
pandas==2.3.3
matplotlib==3.11.2
statsmodels==0.15.0
statsforecast==2.1.1
chronos-forecasting==2.3.2
torch==2.15.0
ipykernel==7.4.0
)

To produce the final file, run `05_final_forecast.ipynb` from top to bottom. It writes
`AKHLAMOVA_project.csv` and takes under a minute once the weights are downloaded. It reads
`backtest_30_windows.csv`, which is in the repository. To rebuild that file, delete it and run
`04_chronos_backtest.ipynb`.

## Files

| File | What it does | Time on CPU |
|---|---|---|
| `common.py` | Metrics, benchmark, validation windows, DynReg, Chronos wrapper | |
| `01_eda.ipynb` | Level, seasonality, weather, autocorrelation, and what they mean for the models | under 1 min |
| `02_classical_models.ipynb` | ETS, SARIMA, AutoSARIMA, MSTL, DynReg, and a rebuild of the Session 1 file | about 2 min |
| `03_chronos_configs.ipynb` | Context length, weather columns and log target compared | about 8 min |
| `04_chronos_backtest.ipynb` | 30 windows against the benchmark, blends, interval checks, residuals | about 6 min |
| `05_final_forecast.ipynb` | Final forecast and submission file | under 1 min |
| `report.ipynb` | The report | seconds |
| `backtest_30_windows.csv` | Output of notebook 04, input of notebook 05 | |
| `chronos_configs.csv` | Output of notebook 03 | |
| `AKHLAMOVA_session1.csv` | Session 1 file, rebuilt by notebook 02 | |
| `AKHLAMOVA_project.csv` | Final submission | |
| `train.csv`, `weather_forecast.csv` | Course data | |

## Validation in short

Each window is a 48-hour forecast from an earlier point in the history, with no overlap between windows.
Day 2 is scored on its own, because the private ranking uses hours 25-48. The windows use actual weather,
since no archived forecasts exist, so their scores are optimistic.

## Use of LLMs

I used Claude Code throughout the project: to suggest models and validation (with my corrections because I wanted to try smth from the course and not the completely new model), to write code and debug it and to draft this README, the notes in the notebooks and the report, which I then edited (I think with notes it's easier to understand what is going on). Early on it reported results and conclusions that turned out to be wrong, such
as how `cloud_cover` relates to generation, so I checked its claims against the outputs. Decisions on what to
submit were mine.
