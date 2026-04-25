# Gold CNN-BiLSTM Flask Live Simulation GUI

This folder contains a complete Flask API and browser dashboard for the fixed-model rolling one-step-ahead gold forecasting simulation.

The dashboard lets you:

1. Enter a today/anchor date.
2. Click **Load Simulation**.
3. Click **Next Day Predict** repeatedly.
4. Watch the actual vs. predicted graph move over time.
5. See RMSE and R² update after every revealed forecast.
6. Download the revealed simulation log as CSV.

## Important required data file

The trained model, scalers, metadata, and historical data are already included in this bundle. The uploaded bundle did **not** contain the completed new-data CSV for the requested test window, so the Flask app includes an upload box.

Required local path:

```text
data/new/gold_RRL_interpolate_2025_05_01_to_2025_11_26.csv
```

Template path:

```text
data/new/gold_RRL_interpolate_2025_05_01_to_2025_11_26_TEMPLATE.csv
```

Required columns:

```text
Date, Gold_Futures, Silver_Futures, Crude_Oil_Futures,
UST10Y_Treasury_Yield, Federal_Funds_Rate,
Employment_Pop_Ratio, gepu, gpr_daily
```

The CSV must contain the actual aligned values for the simulation period from `2025-05-01` to `2025-11-26`.

## Windows: step-by-step

1. Extract the zip file.
2. Open the extracted folder.
3. Double-click:

```text
setup_env.bat
```

4. Double-click:

```text
run_flask_gui.bat
```

5. Open this address in your browser:

```text
http://127.0.0.1:5000
```

6. If the app says the new-data CSV is missing, upload the completed CSV using the upload box.
7. Enter the today/anchor date, for example:

```text
2025-05-01
```

8. Click **Load Simulation**.
9. Click **Next Day Predict** repeatedly.

## Mac/Linux: step-by-step

```bash
bash setup_env.sh
bash run_flask_gui.sh
```

Then open:

```text
http://127.0.0.1:5000
```

## API endpoints

### Check status

```http
GET /api/status
```

Returns whether the required CSV exists, the expected paths, and available API routes.

### Download template

```http
GET /api/template
```

Downloads the required CSV template.

### Upload completed new-data CSV

```http
POST /api/upload-new-data
```

Form field name:

```text
file
```

The app saves the uploaded CSV as:

```text
data/new/gold_RRL_interpolate_2025_05_01_to_2025_11_26.csv
```

### Start simulation

```http
POST /api/start
Content-Type: application/json
```

Example body:

```json
{
  "seed": 2,
  "start_date": "2025-05-01",
  "mode": "anchor"
}
```

`mode = "anchor"` means the entered date is treated as today's date, so the first click predicts the next available forecast day.

`mode = "forecast"` means the entered date is treated as the forecast date itself.

### Reveal the next forecast row

```http
POST /api/next
Content-Type: application/json
```

Example body:

```json
{
  "run_id": "RUN_ID_FROM_API_START"
}
```

Returns one new row plus RMSE, R², and updated chart data.

### Reset current simulation

```http
POST /api/reset
Content-Type: application/json
```

Example body:

```json
{
  "run_id": "RUN_ID_FROM_API_START"
}
```

## How to explain this in defense

This GUI served as a fixed-model rolling one-step-ahead simulation. The trained CNN-BiLSTM model was loaded once and was not retrained during the simulation. Each button click revealed the next available out-of-sample forecast row, compared the predicted gold price with the actual gold price, and updated RMSE and R². This showed how model performance evolved across the unseen test period rather than reporting only one static final metric.
