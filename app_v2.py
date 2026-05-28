import os, json, joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
@app.route("/")
def index():
    return send_from_directory(".", "index.html")
CORS(app)

model     = joblib.load("model_final.pkl")
encoders  = joblib.load("encoders_final.pkl")
le_target = joblib.load("target_encoder_final.pkl")
feat_cols = joblib.load("feature_cols_final.pkl")
with open("model_meta_final.json") as f:
    meta = json.load(f)

def analyze_hook(hook_text):
    api_key = os.getenv("GROQ_API_KEY","")
    if not api_key:
        return None, "No API key"
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = f"""You are a brutally honest Gen Z content coach who scores Instagram Reel hooks.

You grew up scrolling reels. You know EXACTLY what makes someone stop or swipe.
Your scores must be SPREAD OUT — not everything is a 7 or 8.

STRICT scoring rules:
- Score 1-3: Generic, seen a million times, zero reason to stop. ("study with me", "day in my life", plain facts)
- Score 4-5: Has some angle but missing the pull. Mildly interesting but forgettable.
- Score 6-7: Clear hook, genuine curiosity gap or relatability. Most people would pause.
- Score 8-9: Stops the scroll. Unexpected, deeply relatable, or creates urgent need to watch.
- Score 10: Instant stop. Unforgettable. Once in 100 hooks.

BE STRICT. Most hooks are 4-6. A 7 is actually good. An 8 means it's genuinely strong.
DO NOT give 8+ unless the hook is truly remarkable.

Hook to analyze: "{hook_text}"

Context: Indian college student content, Hinglish is fine, Gen Z audience 18-22.

Ask yourself:
- Would a stranger stop scrolling for this? Or only someone who knows the creator?
- Is there a genuine curiosity gap or is it just a topic?
- Is it specific or vague?
- Does it trigger instant emotion or just mild interest?

Reply EXACTLY in this format, nothing else:
SCORE: [number 1-10, decimals allowed like 6.5]
STRENGTH: [one sentence, be specific about what works]
WEAKNESS: [one sentence, be honest about what's missing]
TIP1: [one specific improvement for this exact hook]
TIP2: [one tip about delivery, visual, or timing]
PREDICTED_SKIP: [number 20-70, lower = stronger hook. Most hooks are 38-55]"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=300, temperature=0.4,
        )
        raw = response.choices[0].message.content.strip()
        result = {"score":5.0,"strength":"","weakness":"","tip1":"","tip2":"","skip":45.0}
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try: result["score"] = float(line.split(":",1)[1].strip())
                except: pass
            elif line.startswith("STRENGTH:"): result["strength"] = line.split(":",1)[1].strip()
            elif line.startswith("WEAKNESS:"): result["weakness"] = line.split(":",1)[1].strip()
            elif line.startswith("TIP1:"): result["tip1"] = line.split(":",1)[1].strip()
            elif line.startswith("TIP2:"): result["tip2"] = line.split(":",1)[1].strip()
            elif line.startswith("PREDICTED_SKIP:"):
                try: result["skip"] = float(line.split(":",1)[1].strip())
                except: pass
        return result, None
    except Exception as e:
        return None, str(e)

@app.route("/meta")
def get_meta():
    return jsonify(meta)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    hook_text      = data.get("hook_text","")
    selected_niches= data.get("niches",[])
    selected_hooks = data.get("hooks",[])
    selected_frames= data.get("frames",["Face"])
    posting_day    = data.get("posting_day","Thursday")
    trending_audio = data.get("trending_audio","Yes")
    reel_length    = float(data.get("reel_length",9))

    hook_result, err = analyze_hook(hook_text)
    hook_score = hook_result["score"] if hook_result else 5.0
    skip_rate  = max(20.0, min(70.0, 70.0 - hook_score * 4.0))

    niche_tags = meta["niche_tags"]
    hook_tags  = meta["hook_tags"]
    frame_tags = meta["frame_tags"]

    row = {}
    for tag in niche_tags:
        row[f"niche_{tag.lower()}"] = 1.0 if tag in selected_niches else 0.0
    for tag in hook_tags:
        row[f"hook_{tag.lower()}"] = 1.0 if tag in selected_hooks else 0.0
    for tag in frame_tags:
        row[f"frame_{tag.lower()}"] = 1.0 if tag in selected_frames else 0.0

    le_day   = encoders["posting_day"]
    le_audio = encoders["trending_audio"]
    day_enc   = le_day.transform([posting_day])[0]   if posting_day   in le_day.classes_   else 0
    audio_enc = le_audio.transform([trending_audio])[0] if trending_audio in le_audio.classes_ else 1
    row["posting_day_enc"]     = float(day_enc)
    row["trending_audio_enc"]  = float(audio_enc)
    row["reel_length_seconds"] = reel_length
    row["skip_rate"]           = skip_rate

    X = pd.DataFrame([row])[feat_cols]
    pred_idx   = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    pred_label = le_target.classes_[pred_idx]
    confidence = round(float(pred_proba[pred_idx]) * 100, 1)
    proba_dict = {cls: round(float(pred_proba[i])*100,1) for i,cls in enumerate(le_target.classes_)}

    if hook_score <= 3:
        pred_label = "Bad"; confidence = min(confidence, 55.0)
    elif hook_score <= 4.5:
        if pred_label in ["Viral","Explosive"]:
            pred_label = "Mid"; confidence = min(confidence, 60.0)
    elif hook_score <= 6:
        if pred_label == "Explosive":
            pred_label = "Viral"; confidence = min(confidence, 70.0)
    elif hook_score >= 8.5:
        if pred_label == "Mid":
            pred_label = "Viral"

    tips = []
    if hook_score < 5:
        tips.append("🎣 Hook score below 5 — fix this before posting. Weak hook = high skip rate = algorithm kills the reel in the first 30 mins.")
    if reel_length > 15:
        tips.append(f"⏱️ Your reel is {int(reel_length)}s. Explosive reels on this account average under 12s. Every extra second risks losing the viewer.")
    if "Gym" in selected_niches:
        tips.append("🏋️ Gym content consistently underperforms on student accounts. Try adding a College or Student angle to increase relatability.")
    if "Cinematic" in selected_niches and not any(h in selected_hooks for h in ["Emotional","Nostalgia","Relatable"]):
        tips.append("🎬 Cinematic content only performs with Emotional, Nostalgic or Relatable hooks. Without one of these it almost always goes Mid or Bad.")
    if posting_day in ["Monday","Tuesday","Wednesday"]:
        tips.append(f"📅 {posting_day} is a weaker posting day. Thursday, Friday and Sunday show significantly higher Explosive rates on this account.")
    if trending_audio == "No" and any(n in selected_niches for n in ["Student","Funny","College"]):
        tips.append("🎵 Trending audio boosts early reach for Student/Funny/College content. Worth finding a trending sound before posting.")
    if not tips:
        tips.append("✅ Everything looks solid. Focus on execution — the first 1.5 seconds is everything.")

    return jsonify({
        "prediction": pred_label,
        "confidence": confidence,
        "probabilities": proba_dict,
        "hook": hook_result,
        "hook_score": hook_score,
        "skip_rate": round(skip_rate, 1),
        "tips": tips,
        "error": err
    })
@app.route("/heatmap")
def heatmap():
    try:
        df = pd.read_csv("full_dataset_v2.csv")
        niche = request.args.get("niche", "Student")
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        time_slots = ["Morning (6-11am)","Afternoon (11am-4pm)","Evening (4-8pm)","Night (8pm-12am)"]

        def time_bucket(t):
            try:
                h = float(str(t).replace(":",".")[:5])
                if h < 11: return "Morning (6-11am)"
                elif h < 16: return "Afternoon (11am-4pm)"
                elif h < 20: return "Evening (4-8pm)"
                else: return "Night (8pm-12am)"
            except: return "Evening (4-8pm)"

        df["time_slot"] = df["posting_time"].apply(time_bucket)
        perf_map = {"Bad":0,"Mid":1,"Viral":2,"Explosive":3}
        df["perf_score"] = df["performance"].map(perf_map).fillna(1)

        niche_cols = [c for c in df.columns if c.startswith("niche_")]
        niche_key = f"niche_{niche.lower()}"
        if niche_key in df.columns:
            filtered = df[df[niche_key]==1]
            if len(filtered) < 3:
                filtered = df
        else:
            filtered = df

        grid = {}
        for day in days:
            grid[day] = {}
            for slot in time_slots:
                sub = filtered[(filtered["posting_day"]==day)&(filtered["time_slot"]==slot)]
                grid[day][slot] = round(float(sub["perf_score"].mean()),2) if len(sub)>0 else 1.0

        return jsonify({"grid": grid, "days": days, "slots": time_slots})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/rewrite", methods=["POST"])
def rewrite_hook():
    data = request.json
    hook_text = data.get("hook_text","")
    niche = data.get("niche","Student")
    hook_score = data.get("hook_score", 5.0)
    api_key = os.getenv("GROQ_API_KEY","")
    if not api_key:
        return jsonify({"error":"No API key"})
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = f"""You are a top-tier Instagram Reel hook writer for Indian Gen Z (18-22 year olds).

The user's original hook scored {hook_score}/10. Rewrite it into 3 MUCH stronger versions.

Original hook: "{hook_text}"
Content niche: {niche}

Rules for rewriting:
- Write for Indian Gen Z — Hinglish is welcome, keep it real and natural
- Each version must use a DIFFERENT hook style: one curiosity gap, one shock/unexpected twist, one deeply relatable
- First 3-5 words are everything — start with the strongest word possible
- No corporate language, no "let me tell you", no "POV:" overuse
- Create instant FOMO or "I need to watch this" feeling
- Keep it under 12 words each — shorter is stronger
- Think: would a complete stranger stop scrolling for this?

Reply EXACTLY in this format, nothing else:
VERSION1: [curiosity gap version]
VERSION2: [shock/unexpected twist version]
VERSION3: [deeply relatable version]
REASON1: [one line why version 1 works]
REASON2: [one line why version 2 works]
REASON3: [one line why version 3 works]"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=400, temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()
        result = {"v1":"","v2":"","v3":"","r1":"","r2":"","r3":""}
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("VERSION1:"): result["v1"] = line.split(":",1)[1].strip()
            elif line.startswith("VERSION2:"): result["v2"] = line.split(":",1)[1].strip()
            elif line.startswith("VERSION3:"): result["v3"] = line.split(":",1)[1].strip()
            elif line.startswith("REASON1:"): result["r1"] = line.split(":",1)[1].strip()
            elif line.startswith("REASON2:"): result["r2"] = line.split(":",1)[1].strip()
            elif line.startswith("REASON3:"): result["r3"] = line.split(":",1)[1].strip()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
if __name__ == "__main__":
    app.run(port=5050, debug=False)
