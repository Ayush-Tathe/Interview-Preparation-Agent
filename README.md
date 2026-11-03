# Interview Preparation Agent

An adaptive mock-interview agent with NLP-based answer scoring, hints, follow-up questions and a skill dashboard. The web interface is built with Gradio.

## Features
- 4 roles: Software Engineer, Data Analyst, HR, Marketing
- 6 questions per interview, chosen from a pool of 7 per role
- Modes: Adaptive, Easy, Medium, Hard
- NLP scoring: TF-IDF cosine similarity plus key-concept coverage (STAR check for behavioral questions)
- Score out of 10 = Content (5) + Detail (3) + Clarity (2)
- Hints (cost 1 point) and interviewer follow-up questions
- Dashboard: stat cards, score chart, skill radar, score breakdown, personalised study plan
- History tab to track improvement across attempts

## How the agent works
| Step | Where in the code |
|---|---|
| Perceive | `submit()` reads the typed answer and the time taken |
| Decide | `InterviewAgent.choose_question()` picks a question near the candidate's level |
| Act | `evaluate_answer()` scores, `reflex()` reacts, a follow-up is generated |
| Learn | `update_level()` raises the level at a score of 8 or more, lowers it at 3 or less; `topic_scores` builds the skill map |

It is a rule-based agent with memory and adaptation, not a language model.

## Run locally
```bash
git clone https://github.com/<your-username>/interview-preparation-agent.git
cd interview-preparation-agent
pip install -r requirements.txt
python app.py
```
Open the local URL printed in the terminal (usually http://127.0.0.1:7860).

## Run in Google Colab
```python
!pip install -q gradio scikit-learn
!wget -q https://raw.githubusercontent.com/<your-username>/interview-preparation-agent/main/app.py
!python app.py
```
Open the `gradio.live` link printed in the output and keep the cell running.

## Limitations and future work
- TF-IDF measures word overlap, not true meaning. Sentence embeddings or LLM-based scoring would be better.
- History is kept in memory only. A database would save progress between sessions.
- More roles, more questions and voice answers.

## Tech stack
Python, Gradio, scikit-learn, pandas, NumPy, Matplotlib
