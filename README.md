# Reel-IQ
Instagram Reel Virality Predictor - XGBoost + Groq API + Flask

# ReelIQ — Instagram Reel Virality Predictor

Predicts whether your reel will go Bad, Mid, Viral or Explosive before you post it.

Built on 300 rows of real personal content data- 50 manually collected reels, 250 synthetically generated. Validated by 40 Million+ organic views.

## Features
- Virality prediction across 4 tiers-Bad / Mid / Viral / Explosive
- Groq LLM hook analysis - scores your hook 1-10 with specific feedback
- AI Hook Rewriter - 3 stronger versions of your hook instantly
- Posting time heatmap - best days and times based on real data
- Reel scorecard - letter grades across 5 factors

## Tech Stack
Python, XGBoost, Scikit-learn, Flask, Groq API, HTML/CSS/JS

## Model
XGBoost classifier with hyperparameter tuning and class-weight balancing
Test accuracy: 81.7% across 4 performance tiers

## Run Locally
pip install -r requirements.txt
Add GROQ_API_KEY to .env file
python app_v2.py
Open http://localhost:5050

## Built by
Vivaan Jasapara - Data Science Student, NMIMS Mumbai
