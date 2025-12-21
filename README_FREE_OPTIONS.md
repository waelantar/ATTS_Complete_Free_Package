# 🆓 ATTS Experiment - FREE Options (No Credit Card!)

Choose the option that works best for you:

---

## Option 1: Google Gemini API ⭐ RECOMMENDED

**Cost:** FREE (60 requests/minute)
**Difficulty:** Easy
**Quality:** High (Gemini 1.5 Flash)

### Setup (5 minutes):
```bash
# 1. Get free API key (no credit card!)
#    Go to: https://aistudio.google.com/app/apikey
#    Click "Create API Key"

# 2. Install
pip install google-generativeai pandas tqdm

# 3. Set your key
export GOOGLE_API_KEY="your-key-here"

# 4. Run!
python atts_experiment_free.py
```

**File:** `atts_experiment_free.py`

---

## Option 2: Ollama (Local, 100% Offline)

**Cost:** FREE forever
**Difficulty:** Medium (requires download)
**Quality:** Good (depends on model)

### Setup (15 minutes):
```bash
# 1. Download Ollama
#    Go to: https://ollama.ai/download
#    Install for your OS

# 2. Pull a model (in terminal)
ollama pull llama3.2       # 2GB, good balance
# OR
ollama pull phi3           # 1.7GB, smaller
# OR  
ollama pull mistral        # 4GB, better quality

# 3. Install Python package
pip install ollama pandas tqdm

# 4. Run!
python atts_experiment_local.py --model llama3.2
```

**File:** `atts_experiment_local.py`

**Pros:** No internet needed after setup, completely private
**Cons:** Slower, needs decent computer (8GB+ RAM)

---

## Option 3: Manual Testing (No Code!)

**Cost:** FREE
**Difficulty:** Very Easy
**Quality:** High (use any AI)

### How:
1. Open ChatGPT/Claude/Gemini in browser (free accounts)
2. Follow the guide step by step
3. Record results in spreadsheet
4. Takes ~30-40 minutes

**File:** `ATTS_Manual_Testing_Guide.md`

**Pros:** No coding, use any AI
**Cons:** Manual work, slower

---

## Option 4: Other Free APIs

### Groq (Free, Fast)
- Get key: https://console.groq.com/keys
- Very fast inference
- Uses Llama/Mixtral models

### Together AI (Free Credits)
- Get key: https://api.together.ai
- $5 free credits for new users
- Many model options

### Hugging Face (Free Tier)
- Get key: https://huggingface.co/settings/tokens
- Limited free inference
- Open source models

---

## 📊 Quick Comparison

| Option | Cost | Setup Time | Quality | Speed |
|--------|------|------------|---------|-------|
| Gemini API | Free | 5 min | ⭐⭐⭐⭐ | Fast |
| Ollama Local | Free | 15 min | ⭐⭐⭐ | Medium |
| Manual Testing | Free | 0 min | ⭐⭐⭐⭐ | Slow |
| Groq | Free | 5 min | ⭐⭐⭐⭐ | Very Fast |

---

## 🚀 My Recommendation

**If you want quick results:** Use Gemini API
**If you want privacy/offline:** Use Ollama
**If you hate coding:** Use Manual Testing

---

## Files Included

```
📁 Your Files:
├── atts_experiment_free.py      # Gemini version
├── atts_experiment_local.py     # Ollama version
├── ATTS_Manual_Testing_Guide.md # No-code guide
├── math_problems.json           # Test dataset
└── README_FREE_OPTIONS.md       # This file
```

Good luck! 🎉
