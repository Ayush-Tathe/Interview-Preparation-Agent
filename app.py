"""
Interview Preparation Agent
Adaptive mock-interview agent with NLP scoring (TF-IDF + concept coverage), hints,
follow-ups, a skill dashboard and history. Web UI built with Gradio.

Run:  python app.py
"""
import re, sys, time
from datetime import datetime

import numpy as np
import pandas as pd
import gradio as gr
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# 1. KNOWLEDGE BASE (4 roles x 7 questions)
# =====================================================================
NUM_QUESTIONS = 6

def Q(q, qtype, diff, topic, keys, model):
    # keys = key concepts; "a|b" means either word counts as a match
    return {"q": q, "type": qtype, "difficulty": diff, "topic": topic, "keys": keys, "model": model}

QUESTION_BANK = {
 "Software Engineer": [
  Q("What is the difference between a stack and a queue?", "technical", 1, "Data Structures",
    ["lifo|last in", "fifo|first in", "push|pop|enqueue|dequeue", "plate|line|example"],
    "A stack is LIFO (last in, first out) like a pile of plates; a queue is FIFO (first in, first out) like a line of people."),
  Q("What is the difference between SQL and NoSQL databases?", "technical", 1, "Databases",
    ["schema", "table", "document|key-value|key value", "scal", "structured|flexible"],
    "SQL databases store structured data in tables with a fixed schema; NoSQL stores flexible data such as documents or key-value pairs and scales horizontally more easily."),
  Q("Explain how a hash map works and its average time complexity.", "technical", 2, "Data Structures",
    ["hash function|hash", "bucket|index", "o(1)|constant", "collision", "o(n)|worst"],
    "It uses a hash function to map keys to buckets, giving average O(1) insert and lookup; collisions can degrade it to O(n)."),
  Q("What are the four pillars of object-oriented programming?", "technical", 2, "OOP",
    ["encapsulation", "abstraction", "inheritance", "polymorphism"],
    "Encapsulation, abstraction, inheritance and polymorphism, for example a Shape class with Circle and Square subclasses."),
  Q("What is the difference between REST and GraphQL?", "technical", 3, "Web & APIs",
    ["endpoint", "single|one endpoint", "over-fetch|overfetch|exact|only the fields", "query", "http|resource"],
    "REST has many fixed endpoints; GraphQL has one endpoint where the client asks for exactly the fields it needs."),
  Q("What happens when you type a URL in a browser and press Enter?", "technical", 3, "Web & APIs",
    ["dns", "ip", "tcp|tls|https|connection", "request", "server", "render"],
    "DNS resolves the domain to an IP, the browser opens a TCP/TLS connection, sends an HTTP request, the server responds, and the browser renders the page."),
  Q("Tell me about a time you disagreed with a teammate.", "behavioral", 2, "Behavioral", [],
    "Use STAR: the Situation and Task, the Action you took to reach a compromise, and the measurable Result."),
 ],
 "Data Analyst": [
  Q("What is the difference between INNER JOIN and LEFT JOIN in SQL?", "technical", 1, "SQL",
    ["inner", "left", "matching", "null", "all rows|every row"],
    "INNER JOIN keeps only matching rows; LEFT JOIN keeps all rows from the left table and fills NULL where nothing matches."),
  Q("When would you use the mean, the median or the mode?", "technical", 1, "Statistics",
    ["mean", "median", "mode", "outlier|skew", "symmetric|categor"],
    "Mean for symmetric data, median when there are outliers or skew, mode for the most common category."),
  Q("How would you handle missing data in a dataset?", "technical", 2, "Data Cleaning",
    ["why|reason|cause|pattern", "drop|remove|delete", "impute|mean|median|fill", "flag|indicator", "trade-off|tradeoff|bias"],
    "Find out why it is missing, then drop, impute (mean/median/mode) or flag it, and explain the trade-offs."),
  Q("What is an outlier and how would you deal with it?", "technical", 2, "Data Cleaning",
    ["box plot|boxplot|iqr|z-score|zscore", "error|mistake|real|genuine", "cap|winsor|remove|keep|transform", "investigate|check|verify"],
    "A value far from the rest; detect it with a box plot or IQR, investigate whether it is an error or real, then keep, cap or remove it."),
  Q("Explain the difference between correlation and causation.", "technical", 3, "Statistics",
    ["together|relationship|move", "cause|causation|produces", "experiment|randomi|controlled", "confound|third variable|lurking", "example|ice cream"],
    "Correlation means two variables move together; causation means one produces the change in the other, which usually needs a controlled experiment."),
  Q("How would you design an A/B test for a new web page?", "technical", 3, "Experimentation",
    ["metric|kpi", "random", "control", "variant|treatment", "sample size|power", "significan|p-value|p value"],
    "Define a metric, randomly split users into control and variant, decide the sample size, run for a fixed period and test for statistical significance."),
  Q("Tell me about a time your analysis changed a decision.", "behavioral", 2, "Behavioral", [],
    "Use STAR: the business question, your analysis, and the decision and measurable impact."),
 ],
 "HR": [
  Q("How do you stay organised when managing many priorities?", "technical", 1, "Planning",
    ["urgent|important|priorit", "list|tool|planner|app", "calendar|schedule", "example|deadline"],
    "Use a method such as urgent/important, a task list and a calendar, and give a real example."),
  Q("What do you look for when screening a resume?", "technical", 1, "Recruitment",
    ["skill", "experience", "achievement|measurable|impact", "progression|growth", "format|clear", "job description|requirement|match"],
    "Relevant skills and experience, measurable achievements, career progression, clear formatting and a match with the job description."),
  Q("How would you handle confidential information?", "technical", 2, "Ethics",
    ["need-to-know|need to know", "policy|compliance|law", "secure|lock|password|access", "outside|discuss|share", "trust"],
    "Share only on a need-to-know basis, follow company policy and never discuss it outside proper channels."),
  Q("How would you handle an employee who is repeatedly late?", "technical", 2, "Employee Relations",
    ["private|one-on-one|one on one|conversation", "reason|understand|cause", "policy", "plan|improvement", "document|record", "follow up|follow-up|review"],
    "Talk privately to understand the reason, restate the policy, agree on an improvement plan, document it and follow up."),
  Q("How would you design a fair, structured interview process?", "technical", 3, "Recruitment",
    ["same question|consistent|standard", "rubric|scoring|criteria", "bias", "train", "document|record", "job-related|competenc"],
    "Ask every candidate the same questions, score with a rubric, train interviewers to avoid bias and document decisions."),
  Q("How would you reduce employee turnover in a company?", "technical", 3, "Retention",
    ["exit interview|survey|feedback", "cause|reason|root", "onboarding", "pay|salary|compensation", "career|growth|development", "manager|culture|engagement", "track|metric|retention rate"],
    "Find the causes through exit interviews and surveys, then improve onboarding, pay fairness, career growth and manager quality, and track retention."),
  Q("Describe a time you resolved a conflict between two people.", "behavioral", 2, "Behavioral", [],
    "Use STAR: stay neutral, listen to both sides, focus on the issue and show the agreed outcome."),
 ],
 "Marketing": [
  Q("What is the difference between B2B and B2C marketing?", "technical", 1, "Fundamentals",
    ["business", "consumer", "cycle|longer|shorter", "emotion|logic|rational", "decision"],
    "B2B sells to businesses with longer, logic-driven sales cycles; B2C sells to consumers with shorter, emotion-driven decisions."),
  Q("What is SEO and why does it matter?", "technical", 1, "SEO",
    ["search", "rank", "organic|free", "traffic", "keyword", "long-term|long term|content|backlink"],
    "Search Engine Optimization improves a site's ranking in search results, bringing free, long-term, high-intent traffic."),
  Q("How do you measure the success of a marketing campaign?", "technical", 2, "Analytics",
    ["goal|objective", "kpi|metric", "conversion", "roi|return", "cost per|cpa|cac", "baseline|benchmark|compare"],
    "Set goals first, then track KPIs such as conversion rate, cost per acquisition, ROI and engagement, and compare them with a baseline."),
  Q("Explain the marketing funnel.", "technical", 2, "Fundamentals",
    ["awareness", "interest", "consideration", "conversion", "loyalty|retention", "content|metric"],
    "Awareness, interest, consideration, conversion and loyalty; each stage needs different content and metrics."),
  Q("How would you launch a new product with a small budget?", "technical", 3, "Strategy",
    ["target audience|niche|segment", "low-cost|low cost|social|email|organic", "partnership|influencer|referral", "test", "scale|iterate", "measure|track"],
    "Define a narrow target audience, use low-cost channels such as social media, email and partnerships, test small, then scale what works."),
  Q("What are CAC and LTV, and why does their ratio matter?", "technical", 3, "Analytics",
    ["acquisition", "lifetime", "ratio|3:1", "revenue|profit", "healthy|sustain|profitab", "spend|cost"],
    "Customer Acquisition Cost is what you spend to win a customer; Lifetime Value is the revenue they bring. An LTV:CAC ratio around 3:1 signals a healthy business."),
  Q("Tell me about a campaign that did not go as planned.", "behavioral", 2, "Behavioral", [],
    "Use STAR: the goal, what went wrong, the action you took to fix it, and what you learned and the result."),
 ],
}

# NLP: TF-IDF vectoriser trained on all model answers (used for similarity scoring)
VEC = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit(
    [x["model"] for r in QUESTION_BANK.values() for x in r])

# =====================================================================
# 2. EVALUATOR (NLP + rules) AND REFLEX
# =====================================================================
STAR = {
  "Situation": ["situation", "context", "background", "when i", "in my last", "at my", "project", "during"],
  "Task":      ["task", "goal", "responsib", "objective", "my role", "needed to", "had to"],
  "Action":    ["action", "i decided", "i led", "i organi", "i built", "i created", "i ran", "i talked",
                "i set up", "i proposed", "i implemented", "i took", "i worked", "i started"],
  "Result":    ["result", "outcome", "as a result", "improved", "reduced", "increased", "achieved",
                "learned", "saved", "%", "percent"],
}
REASON_WORDS = ["because", "for example", "for instance", "trade-off", "however", "whereas",
                "which", "so that", "therefore", "while"]
FILLERS = ["you know", "basically", "like,", "kind of", "sort of"]

def evaluate_answer(answer, q):
    """Score 0-10 = Content (0-5) + Detail (0-3) + Clarity (0-2)."""
    low = answer.lower()
    words = len(answer.split())
    has_number = bool(re.search(r"\d|o\(|%|percent", low))
    tips = []

    if q["type"] == "behavioral":
        found = [k for k, syn in STAR.items() if any(s in low for s in syn)]
        missed = [k for k in STAR if k not in found]
        sim = None
        content = 5 * len(found) / 4                                   # STAR coverage
        detail = (2 if has_number else 0) + (1 if re.search(r"\b(i|my|we|our)\b", low) else 0)
        if missed: tips.append("Add the missing STAR parts: " + ", ".join(missed) + ".")
        if not has_number: tips.append("Add a measurable result (a number or %).")
    else:
        found = [k.split("|")[0] for k in q["keys"] if any(a in low for a in k.split("|"))]
        missed = [k.split("|")[0] for k in q["keys"] if k.split("|")[0] not in found]
        coverage = len(found) / len(q["keys"])                         # concept coverage
        sim = float(cosine_similarity(VEC.transform([answer]), VEC.transform([q["model"]]))[0, 0])
        content = 5 * (0.65 * coverage + 0.35 * min(1.0, sim / 0.45))  # coverage + TF-IDF similarity
        marks = sum(1 for w in REASON_WORDS if w in low)
        detail = min(2, marks) + (1 if has_number else 0)
        if missed: tips.append("Key ideas missing: " + ", ".join(missed[:3]) + ".")
        if marks == 0: tips.append("Explain your reasoning (because, however, for example).")

    if 25 <= words <= 200: clarity = 2
    elif 15 <= words < 25 or 200 < words <= 300: clarity = 1
    else: clarity = 0
    if words < 15: tips.append("Too short - add detail and an example.")
    elif words > 300: tips.append("Too long - be more concise.")

    total = round(min(10, content + detail + clarity), 1)
    followup = None
    if total < 7 and missed:
        followup = (f"What was the {missed[0]} in your story?" if q["type"] == "behavioral"
                    else f"You did not mention '{missed[0]}'. How does it fit into your answer?")
    return {"total": total, "content": round(content, 1), "detail": detail, "clarity": clarity,
            "found": found, "missed": missed, "similarity": sim,
            "feedback": " ".join(tips) or "Solid answer.", "followup": followup}

def reflex(answer, seconds):
    """Instant rule-based reaction (sense, then react)."""
    low = answer.lower()
    if seconds > 180:
        return "TIME ALERT: that took over 3 minutes - practise wrapping up faster."
    if re.match(r"\s*(um|uh)\b", low) or any(f in low for f in FILLERS):
        return "COACH: filler words spotted - pause instead of saying them."
    if len(answer.split()) < 10:
        return "PROMPT: can you elaborate on that?"
    return "OK: answer length looks healthy."

# =====================================================================
# 3. THE AGENT (no UI code here)
# =====================================================================
class InterviewAgent:
    def __init__(self, role, num_questions=6, mode="Adaptive"):
        self.role = role
        self.pool = QUESTION_BANK[role]                          # KNOWLEDGE
        self.num_questions = min(num_questions, len(self.pool))
        self.mode = mode                                         # Adaptive / Easy / Medium / Hard
        self.level = 1                                           # BELIEF about the candidate (1-3)
        self.level_trace = [1]
        self.asked = set()                                       # MEMORY
        self.history = []                                        # MEMORY: scores per question
        self.review = []                                         # MEMORY: for the study plan
        self.topic_scores = {}                                   # MEMORY: skill map

    def choose_question(self):
        """DECIDE: closest difficulty to target, prefer uncovered topics, force 1 behavioral."""
        options = [q for q in self.pool if q["q"] not in self.asked]
        if not options:
            return None
        remaining = self.num_questions - len(self.history)
        behavioral = [q for q in options if q["type"] == "behavioral"]
        if remaining == 1 and behavioral and not any(h["Type"] == "behavioral" for h in self.history):
            options = behavioral
        target = {"Easy": 1, "Medium": 2, "Hard": 3}.get(self.mode, self.level)
        return min(options, key=lambda q: (abs(q["difficulty"] - target),
                                           len(self.topic_scores.get(q["topic"], []))))

    def update_level(self, score):
        """LEARN: raise belief after a great answer, lower it after a weak one."""
        if score >= 8:
            self.level = min(3, self.level + 1)
        elif score <= 3:
            self.level = max(1, self.level - 1)

    def hint(self, q, n):
        if q["type"] == "behavioral":
            return "Structure it as Situation, Task, Action, Result and add one number."
        keys = q["keys"]
        return f"Topic: {q['topic']}. Try mentioning '{keys[min(n, len(keys) - 1)].split('|')[0]}'."

    def process(self, q, answer, seconds, hints=0):
        """ACT + LEARN: react, score, remember, update belief."""
        res = evaluate_answer(answer, q)
        final = max(0, round(res["total"] - hints, 1))           # each hint costs 1 point
        n = len(self.history) + 1
        self.asked.add(q["q"])
        self.topic_scores.setdefault(q["topic"], []).append(final)
        self.history.append({"Question": n, "Topic": q["topic"], "Type": q["type"], "Level": q["difficulty"],
                             "Score": final, "Content": res["content"], "Detail": res["detail"],
                             "Clarity": res["clarity"], "Hints": hints, "Time (s)": round(seconds)})
        self.review.append({"n": n, "topic": q["topic"], "q": q["q"], "score": final,
                            "missed": res["missed"], "model": q["model"]})
        if self.mode == "Adaptive":
            self.update_level(final)
        self.level_trace.append(self.level)
        res.update(score=final, hints=hints, reaction=reflex(answer, seconds), model=q["model"])
        return res

    def finished(self):
        return len(self.history) >= self.num_questions

# =====================================================================
# 4. GRADIO UI
# =====================================================================
SESSIONS = []   # every completed attempt while the app is running

CSS = """
.ia-header{background:linear-gradient(90deg,#1d3557,#2a9d8f);color:#fff;padding:20px 24px;border-radius:12px;font-family:Arial,sans-serif}
.ia-header h2{margin:0;font-size:26px;color:#fff}
.ia-header p{margin:4px 0 0;opacity:.9;font-size:14px;color:#fff}
.ia-card{background:#f8fafc;border:1px solid #dbe4ee;border-left:6px solid #2a9d8f;border-radius:10px;padding:14px 18px;font-family:Arial,sans-serif;color:#1d3557;margin-top:8px}
.ia-card h3{margin:8px 0 0;font-size:19px;color:#1d3557}
.ia-tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;color:#fff;margin-right:6px}
.ia-bar-bg{background:#e5e9f0;border-radius:8px;height:12px;width:100%;margin:6px 0}
.ia-bar{height:12px;border-radius:8px}
.ia-line{margin:6px 0;font-size:14px;line-height:1.5;color:#1d3557}
.ia-warn{background:#fff3cd;color:#664d03;padding:10px 14px;border-radius:8px;font-family:Arial,sans-serif;margin-top:8px}
.ia-hint{background:#e8f4fd;color:#0b4f6c;padding:10px 14px;border-radius:8px;font-family:Arial,sans-serif;margin-top:8px}
.ia-badge{background:#eaf4f4;color:#1d3557;border:1px solid #cfe3e3;padding:8px 14px;border-radius:8px;font-family:Arial,sans-serif;font-size:14px;margin:8px 0}
.ia-chip{display:inline-block;padding:2px 9px;border-radius:12px;font-size:12px;margin:2px 4px 2px 0}
.ia-chip.ok{background:#d8f3ec;color:#14665a}
.ia-chip.bad{background:#fde8e4;color:#9b2c1d}
.ia-stats{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;font-family:Arial,sans-serif}
.ia-stat{flex:1;min-width:130px;background:#fff;border:1px solid #dbe4ee;border-radius:10px;padding:12px;text-align:center;color:#1d3557;font-size:13px}
.ia-stat b{display:block;font-size:22px}
.ia-card b, .ia-card h3, .ia-card i, .ia-card div, .ia-card span{color:#1d3557}
.ia-stat b{color:#1d3557 !important}
.ia-warn b, .ia-warn{color:#664d03 !important}
.ia-hint, .ia-hint b{color:#0b4f6c !important}
.ia-badge, .ia-badge b{color:#1d3557 !important}
.ia-tag{color:#fff !important}
.ia-chip.ok{color:#14665a !important}
.ia-chip.bad{color:#9b2c1d !important}
"""

LEVEL_NAMES = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}
DIFF_NAMES  = {1: "Easy", 2: "Medium", 3: "Hard"}

def score_color(s):
    return "#2a9d8f" if s >= 8 else "#e9c46a" if s >= 5 else "#e76f51"

def progress_html(done, total):
    pct = int(done / total * 100)
    return (f'<div class="ia-line">Progress: <b>{done}/{total}</b></div>'
            f'<div class="ia-bar-bg"><div class="ia-bar" style="width:{pct}%;background:#457b9d"></div></div>')

def question_card(n, total, q):
    tcolor = "#457b9d" if q["type"] == "technical" else "#8e5ea2"
    return f"""<div class="ia-card">
      <span class="ia-tag" style="background:{tcolor}">{q['type'].title()}</span>
      <span class="ia-tag" style="background:#6c757d">{DIFF_NAMES[q['difficulty']]}</span>
      <span class="ia-tag" style="background:#e07a5f">{q['topic']}</span>
      <span style="font-size:13px">Question {n} of {total}</span>
      <h3>{q['q']}</h3></div>"""

def level_badge(level, mode):
    dots = "".join("🟢" if i <= level else "⚪" for i in (1, 2, 3))
    return (f'<div class="ia-badge">🧠 Agent believes your level is: <b>{LEVEL_NAMES[level]}</b> {dots}'
            f' &nbsp;|&nbsp; Mode: <b>{mode}</b></div>')

def mini(label, val, mx, color):
    return (f'<div class="ia-line">{label}: <b>{val}/{mx}</b></div>'
            f'<div class="ia-bar-bg"><div class="ia-bar" style="width:{val / mx * 100}%;background:{color}"></div></div>')

def feedback_card(res):
    c = score_color(res["score"])
    chips = "".join(f'<span class="ia-chip ok">✔ {x}</span>' for x in res["found"]) + \
            "".join(f'<span class="ia-chip bad">✘ {x}</span>' for x in res["missed"])
    sim = "" if res["similarity"] is None else f' &nbsp;|&nbsp; similarity to model answer: {res["similarity"] * 100:.0f}%'
    hint_note = f' (−{res["hints"]} for hints)' if res["hints"] else ""
    follow = (f'<div class="ia-warn">💬 <b>Interviewer follow-up:</b> {res["followup"]}</div>'
              if res["followup"] else "")
    return f"""<div class="ia-card" style="border-left-color:{c}">
      <div class="ia-line"><b>Score: {res['score']}/10</b>{hint_note}{sim}</div>
      <div class="ia-bar-bg"><div class="ia-bar" style="width:{res['score'] * 10}%;background:{c}"></div></div>
      {mini('Content', res['content'], 5, '#457b9d')}
      {mini('Detail', res['detail'], 3, '#e9c46a')}
      {mini('Clarity', res['clarity'], 2, '#2a9d8f')}
      <div class="ia-line">{chips}</div>
      <div class="ia-line"><b>Reflex:</b> {res['reaction']}</div>
      <div class="ia-line"><b>Feedback:</b> {res['feedback']}</div>
      <div class="ia-line"><b>Model answer:</b> {res['model']}</div></div>{follow}"""

def hints_html(st):
    if not st["hints"]:
        return ""
    lines = "<br>".join(f"💡 Hint {i + 1} (-1 point): {h}" for i, h in enumerate(st["hints"]))
    return f'<div class="ia-hint">{lines}</div>'

def draw_charts(df, topic_avg):
    fig = Figure(figsize=(15, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    bars = ax1.bar(df["Question"], df["Score"], color=[score_color(s) for s in df["Score"]])
    ax1.bar_label(bars, fmt="%.1f")
    ax1.set_ylim(0, 11); ax1.set_xlabel("Question"); ax1.set_title("Score per question")

    labels, vals = list(topic_avg), list(topic_avg.values())
    ax2 = fig.add_subplot(1, 3, 2, polar=len(labels) >= 3)
    if len(labels) >= 3:
        ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        ax2.plot(ang + ang[:1], vals + vals[:1], color="#2a9d8f", linewidth=2)
        ax2.fill(ang + ang[:1], vals + vals[:1], color="#2a9d8f", alpha=0.25)
        ax2.set_xticks(ang); ax2.set_xticklabels(labels, fontsize=8); ax2.set_ylim(0, 10)
    else:
        ax2.barh(labels, vals, color="#2a9d8f"); ax2.set_xlim(0, 10)
    ax2.set_title("Skill map by topic", y=1.08)

    ax3 = fig.add_subplot(1, 3, 3)
    bottom = np.zeros(len(df))
    for col, color in [("Content", "#457b9d"), ("Detail", "#e9c46a"), ("Clarity", "#2a9d8f")]:
        ax3.bar(df["Question"], df[col], bottom=bottom, label=col, color=color)
        bottom = bottom + df[col].values
    ax3.set_xlabel("Question"); ax3.set_title("Score breakdown"); ax3.legend(fontsize=8)
    fig.tight_layout()
    return fig

def history_chart(h):
    fig = Figure(figsize=(7, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, len(h) + 1), h["Avg score"], marker="o", color="#2a9d8f")
    ax.set_ylim(0, 10); ax.set_xlabel("Attempt"); ax.set_ylabel("Average score")
    ax.set_title("Improvement across attempts")
    fig.tight_layout()
    return fig

def build_report(st):
    a = st["agent"]
    df = pd.DataFrame(a.history)
    avg = df["Score"].mean()
    verdict = ("Excellent - interview ready." if avg >= 8 else
               "Good foundation - polish structure and detail." if avg >= 5 else
               "Needs practice - use STAR and cover the key concepts.")
    topic_avg = {t: sum(v) / len(v) for t, v in a.topic_scores.items()}
    weakest = min(topic_avg, key=topic_avg.get)
    total_t = int(df["Time (s)"].sum())
    journey = " → ".join(str(x) for x in a.level_trace)

    stats = f"""<div class="ia-card" style="border-left-color:{score_color(avg)}">
      <h3>Final Report - {st['name']} | {a.role} ({a.mode})</h3>
      <div class="ia-stats">
        <div class="ia-stat"><b>{avg:.1f}/10</b>Average score</div>
        <div class="ia-stat"><b>{LEVEL_NAMES[a.level]}</b>Final level<br>(journey {journey})</div>
        <div class="ia-stat"><b>{weakest}</b>Weakest topic ({topic_avg[weakest]:.1f}/10)</div>
        <div class="ia-stat"><b>{total_t // 60}m {total_t % 60}s</b>Total time</div>
        <div class="ia-stat"><b>{int(df['Hints'].sum())}</b>Hints used</div>
      </div>
      <div class="ia-line" style="margin-top:10px"><b>Verdict:</b> {verdict}</div></div>"""

    weak = [r for r in a.review if r["score"] < 6]
    if weak:
        items = "".join(
            f'<div class="ia-line"><b>{r["topic"]}</b> (Q{r["n"]}, {r["score"]}/10): {r["q"]}<br>'
            f'<i>Focus on:</i> {", ".join(r["missed"][:3]) or "structure and detail"}<br>'
            f'<i>Model answer:</i> {r["model"]}</div><hr>' for r in weak)
        plan = f'<div class="ia-card"><h3>📚 Personalised study plan</h3>{items}</div>'
    else:
        plan = '<div class="ia-card"><h3>🎉 No weak areas - every answer scored 6 or more.</h3></div>'

    SESSIONS.append({"Date": datetime.now().strftime("%d %b %H:%M"), "Name": st["name"], "Role": a.role,
                     "Mode": a.mode, "Avg score": round(avg, 1), "Final level": LEVEL_NAMES[a.level]})
    h = pd.DataFrame(SESSIONS)
    hist_fig = history_chart(h) if len(h) > 1 else None
    return (stats, df, draw_charts(df, topic_avg), plan, h, hist_fig, gr.Tabs(selected="dash"))

def render(st, answer_update, feedback="", hint="", answered=False, last=False):
    """Returns the 11 values for OUT, in order."""
    a = st["agent"]
    return (st,
            progress_html(len(a.history), a.num_questions),
            level_badge(a.level, a.mode),
            question_card(st["n"], a.num_questions, st["q"]),
            hint, answer_update, feedback,
            gr.update(interactive=not answered),
            gr.update(interactive=not answered),
            gr.update(visible=answered, value="Finish & See Dashboard" if last else "Next Question"),
            gr.update(visible=True))

def noop(st):
    return (st,) + tuple(gr.update() for _ in range(10))

NO_DASH = tuple(gr.update() for _ in range(7))

def start(name, role, mode):
    a = InterviewAgent(role, NUM_QUESTIONS, mode)
    st = {"agent": a, "name": (name or "").strip() or "Candidate", "hints": [],
          "n": 1, "q": a.choose_question(), "t0": time.time()}
    return render(st, gr.update(value="", interactive=True))

def give_hint(st):
    if not st:
        return noop(st)
    st["hints"].append(st["agent"].hint(st["q"], len(st["hints"])))
    return render(st, gr.update(), hint=hints_html(st))

def submit(answer, st):
    if not st:
        return noop(st)
    if not answer or not answer.strip():
        warn = '<div class="ia-warn">Please type an answer first.</div>'
        return render(st, gr.update(), feedback=warn, hint=hints_html(st))
    a = st["agent"]
    seconds = time.time() - st["t0"]
    res = a.process(st["q"], answer.strip(), seconds, len(st["hints"]))
    return render(st, gr.update(interactive=False), feedback_card(res),
                  hints_html(st), answered=True, last=a.finished())

def next_question(st):
    if not st:
        return noop(st) + NO_DASH
    a = st["agent"]
    if a.finished():
        base = list(noop(st))
        base[9] = gr.update(visible=False)
        return tuple(base) + build_report(st)
    st["n"] += 1
    st["q"] = a.choose_question()
    st["hints"] = []
    st["t0"] = time.time()
    return render(st, gr.update(value="", interactive=True)) + NO_DASH

with gr.Blocks(title="Interview Preparation Agent", css=CSS, theme=gr.themes.Soft(primary_hue="teal")) as demo:
    state = gr.State(None)
    gr.HTML("""<div class="ia-header"><h2>🎤 Interview Preparation Agent</h2>
               <p>Adaptive mock interviews with NLP scoring, hints, follow-ups and a skill dashboard.</p></div>""")

    with gr.Row():
        name_in = gr.Textbox(label="Your name", placeholder="e.g. Priya", scale=2)
        role_in = gr.Dropdown(list(QUESTION_BANK), value="Software Engineer", label="Role", scale=2)
        mode_in = gr.Dropdown(["Adaptive", "Easy", "Medium", "Hard"], value="Adaptive", label="Mode", scale=1)
        start_btn = gr.Button("▶ Start / Restart Interview", variant="primary", scale=1)

    with gr.Tabs() as tabs:
        with gr.Tab("🎤 Interview", id="interview"):
            gr.HTML('<div class="ia-warn">👋 Enter your name, choose a role and a mode, then click <b>Start Interview</b>.</div>')
            with gr.Column(visible=False) as box:
                prog = gr.HTML()
                level_h = gr.HTML()
                question_h = gr.HTML()
                hint_h = gr.HTML()
                answer_in = gr.Textbox(label="Your answer", lines=6, placeholder="Type your answer here, then click Submit...")
                with gr.Row():
                    hint_btn = gr.Button("💡 Hint (-1 pt)")
                    submit_btn = gr.Button("✔ Submit Answer", variant="primary")
                    next_btn = gr.Button("Next Question", visible=False)
                feedback_h = gr.HTML()
        with gr.Tab("📊 Dashboard", id="dash"):
            stats_h = gr.HTML('<div class="ia-warn">Finish an interview to see your dashboard here.</div>')
            table_out = gr.Dataframe(label="Question-wise results", interactive=False)
            plot_out = gr.Plot(label="Charts")
            plan_h = gr.HTML()
        with gr.Tab("📜 History", id="hist"):
            hist_table = gr.Dataframe(label="All attempts in this session", interactive=False)
            hist_plot = gr.Plot(label="Improvement")

    OUT  = [state, prog, level_h, question_h, hint_h, answer_in, feedback_h, hint_btn, submit_btn, next_btn, box]
    DASH = [stats_h, table_out, plot_out, plan_h, hist_table, hist_plot, tabs]

    start_btn.click(start, [name_in, role_in, mode_in], OUT)
    hint_btn.click(give_hint, [state], OUT)
    submit_btn.click(submit, [answer_in, state], OUT)
    next_btn.click(next_question, [state], OUT + DASH)

if __name__ == "__main__":
    in_colab = "google.colab" in sys.modules
    demo.launch(share=in_colab, debug=in_colab)
