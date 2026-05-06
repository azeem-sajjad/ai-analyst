import ollama
import json
import pandas as pd
from config.settings import MODEL

def ask_data_question(df, question):

    prompt = f"""
You are expert data analyst.

Dataset Columns:
{list(df.columns)}

First Rows:
{df.head().to_string()}

Question:
{question}

Return JSON only:

{{
 "answer":"short insight",
 "python_code":"pandas code using df and save result in result",
 "chart_type":"bar/line/pie/none"
}}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role":"user","content":prompt}]
    )

    text = response["message"]["content"]

    try:
        return json.loads(text)
    except:
        return {
            "answer": text,
            "python_code": "",
            "chart_type": "none"
        }