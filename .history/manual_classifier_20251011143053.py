from typing import Dict, Tuple
from emotion_predictor import create_prediction_dict

# List of required emotion keys (exactly as you provided)
REQUIRED_KEYS = [
    'example_very_unclear', 'admiration', 'amusement', 'anger', 'annoyance',
    'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise'
]

EMOJIS = ["😄","😢","😠","😨","😮","🤢","❤","🤩","🤔","😐"]

# For readability: create mapping per emotion that distributes influence across emojis.
# Each emotion maps to a dict of emoji -> weight (float). Weights are handcrafted heuristics.
EMOTION_CONTRIBS = {
    # Unclear/ambiguous pushes toward neutral/thinking/surprised
    'example_very_unclear': {"😐":0.6,"🤔":0.25,"😮":0.15},

    'admiration': {"🤩":0.5,"😄":0.25,"❤":0.15,"😐":0.10},
    'amusement': {"😄":0.6,"🤩":0.25,"😐":0.15},
    'anger': {"😠":0.85,"😐":0.10,"😮":0.05},
    'annoyance': {"😠":0.6,"😐":0.25,"😢":0.05,"😮":0.10},
    'approval': {"😄":0.45,"❤":0.3,"🤩":0.15,"😐":0.10},
    'caring': {"❤":0.5,"😄":0.2,"😢":0.15,"😐":0.15},
    'confusion': {"🤔":0.55,"😮":0.25,"😐":0.20},
    'curiosity': {"🤔":0.6,"😮":0.25,"😄":0.15},
    'desire': {"❤":0.45,"🤩":0.25,"😄":0.15,"😐":0.15},
    'disappointment': {"😢":0.6,"😐":0.25,"😠":0.05,"😮":0.10},
    'disapproval': {"😠":0.6,"😐":0.3,"😢":0.05,"🤔":0.05},
    'disgust': {"🤢":0.9,"😐":0.05,"😠":0.05},
    'embarrassment': {"😐":0.4,"😢":0.3,"🤔":0.2,"😮":0.1},
    'excitement': {"🤩":0.6,"😄":0.3,"😮":0.1},
    'fear': {"😨":0.8,"😢":0.1,"😐":0.1},
    'gratitude': {"❤":0.5,"😄":0.3,"😐":0.2},
    'grief': {"😢":0.9,"😐":0.05,"❤":0.05},
    'joy': {"😄":0.8,"🤩":0.15,"❤":0.05},
    'love': {"❤":0.85,"😄":0.1,"🤩":0.05},
    'nervousness': {"😨":0.5,"😐":0.3,"😢":0.2},
    'optimism': {"😄":0.6,"🤩":0.25,"❤":0.15},
    'pride': {"😄":0.5,"🤩":0.35,"❤":0.15},
    'realization': {"😮":0.5,"🤔":0.4,"😐":0.1},
    'relief': {"😄":0.5,"😨":0.2,"😐":0.3},
    'remorse': {"😢":0.6,"😐":0.25,"❤":0.15},
    'sadness': {"😢":0.9,"😐":0.1},
    'surprise': {"😮":0.8,"🤔":0.15,"😐":0.05}
}

def validate_and_fill(emotions: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure all required keys exist. Missing keys are treated as 0. Values clipped to [0,1].
    """
    filled = {}
    for k in REQUIRED_KEYS:
        v = emotions.get(k, 0.0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        # clip
        if v < 0: v = 0.0
        if v > 1: v = 1.0
        filled[k] = v
    return filled

def classify_emotions(emotions: Dict[str, float], explain: bool=False) -> Tuple[str, Dict[str, float]]:
    """
    Classify the input emotion dictionary into one emoji.
    Returns (emoji, scores_dict). If explain==True, returns full per-emoji scores.
    Uses all parameters in REQUIRED_KEYS and handcrafted weights.
    """
    e = validate_and_fill(emotions)

    # compute base scores by summing emotion contributions
    scores = {emoji: 0.0 for emoji in EMOJIS}
    for emo_name, value in e.items():
        contrib = EMOTION_CONTRIBS.get(emo_name, {})
        # If an emotion definition doesn't mention an emoji, it contributes 0 to that emoji.
        for emoji, w in contrib.items():
            scores[emoji] += w * value

    # --- Heuristic overrides / adjustments (explicit rules to resolve very clear cases) ---
    # If disgust is extremely high, prefer 🤢.
    if e['disgust'] >= 0.65:
        scores = {k: v * 0.1 for k, v in scores.items()}  # damp everything
        scores["🤢"] = max(scores["🤢"], 1.0 * e['disgust'] * 2.0)  # force high score
    # If love is very strong, prefer ❤
    if e['love'] >= 0.7:
        scores["❤"] += 1.0 * e['love']
    # If anger is very strong, boost 😠 heavily
    if e['anger'] >= 0.6:
        scores["😠"] += 0.8 * e['anger']
    # If joy/excitement extremely high -> prefer 🤩 (starstruck) when admiration/excitement dominate
    if (e['excitement'] >= 0.7 or e['admiration'] >= 0.7) and e['joy'] >= 0.4:
        scores["🤩"] += 0.9 * max(e['excitement'], e['admiration'])

    # If sadness/grief/remorse dominate strongly -> push toward 😢
    sadness_dom = max(e['sadness'], e['grief'], e['remorse'], e['disappointment'])
    if sadness_dom >= 0.65:
        scores["😢"] += 0.9 * sadness_dom

    # If fear + nervousness dominate -> push 😨
    if (e['fear'] + e['nervousness']) / 2.0 >= 0.6:
        scores["😨"] += 0.8 * max(e['fear'], e['nervousness'])

    # Normalize (optional) to make scores comparable -- but ranking only matters, not absolute scaling.
    # But keep simple: ensure non-negative and small smoothing
    for k in scores:
        if scores[k] < 0:
            scores[k] = 0.0
    # Tie-breaker: in case of equal scores, prefer an order that makes sense (priority list)
    priority = EMOJIS  # left-to-right priority: 😄,😢,😠,😨,😮,🤢,❤,🤩,🤔,😐

    # Choose the emoji with highest score; deterministic tie-breaking by priority
    best_emoji = max(priority, key=lambda emoji: (scores.get(emoji, 0.0), -priority.index(emoji)))
    if explain:
        # return both the emoji and a sorted score breakdown
        sorted_scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
        return best_emoji, sorted_scores
    else:
        return best_emoji, scores

# Example usage / quick tests
if __name__ == "__main__":
    examples = [
        ({'joy':1.0, 'excitement':0.8, 'admiration':0.2, **{k:0.0 for k in REQUIRED_KEYS if k not in ['joy','excitement','admiration']}}, "Happy/excited -> 😄/🤩"),
        ({'sadness':0.9, 'grief':0.6, **{k:0.0 for k in REQUIRED_KEYS if k not in ['sadness','grief']}}, "Very sad -> 😢"),
        ({'disgust':0.8, 'anger':0.1, **{k:0.0 for k in REQUIRED_KEYS if k not in ['disgust','anger']}}, "Disgust -> 🤢"),
        ({'love':0.9, 'caring':0.7, 'desire':0.5, **{k:0.0 for k in REQUIRED_KEYS if k not in ['love','caring','desire']}}, "Love/caring -> ❤"),
        ({'fear':0.8, 'nervousness':0.7, **{k:0.0 for k in REQUIRED_KEYS if k not in ['fear','nervousness']}}, "Fearful -> 😨"),
        ({'curiosity':0.9, 'confusion':0.6, 'realization':0.2, **{k:0.0 for k in REQUIRED_KEYS if k not in ['curiosity','confusion','realization']}}, "Thinking/curious -> 🤔 or 😮"),
        ({'example_very_unclear':0.9, 'confusion':0.5, **{k:0.0 for k in REQUIRED_KEYS if k not in ['example_very_unclear','confusion']}}, "Very unclear -> 😐 or 🤔"),
    ]

    for inp, desc in examples:
        emoji, breakdown = classify_emotions(inp, explain=True)
        print(f"{desc}\n -> Predicted: {emoji}\n -> Score breakdown: {breakdown}\n")

while True:
    prompt = input("enter your prompt: ")
    if prompt == "exit":
        break
    emotion_dict = classify_emotions(create_prediction_dict(prompt))
    print("\n\t",("\n\t".join(str(emotion_dict).split(",")))[1:-1])  # name a cleaner line of code