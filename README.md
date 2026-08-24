# CardioLens — Heart Disease Predictor

## Train the models locally

Install training dependencies:

```bash
pip install -r training_requirements.txt
```

Then:

```bash
python train_models.py
```

This creates the files that the deployed app needs:

```text
models/
  logistic_regression.joblib
  random_forest.joblib
  metadata.json

data/
  heart_disease.csv
```

The production API does **not** download Kaggle data or retrain models. It only loads these saved artifacts.

## Test locally

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
python -m uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Check:

```text
http://127.0.0.1:8000/api/health
```

You want:

```json
{"status":"ok","modelsLoaded":true}
```

In another terminal:

```bash
pnpm install
pnpm dev
```

Open:

```text
http://localhost:3000
```

For local frontend -> backend calls, create `.env.local`:

```text
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## Deploy to Vercel

1. Run `python train_models.py` locally.
2. Confirm the three files exist in `models/`.
3. Commit/push the entire project to GitHub, including the model artifacts.
4. Import the repository into Vercel.
5. Deploy with the normal Next.js settings.

The Vercel deployment uses `/api/predict` and loads the saved `.joblib` artifacts.

## Important

The model output is an educational machine-learning result, not a clinically validated diagnosis.
