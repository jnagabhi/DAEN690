#!/usr/bin/env python
# coding: utf-8

# In[5]:


# To run the streamlit app use the below commands in terminal
#cd /directory
#streamlit run Model_recommend.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------
# Page Styling
# ------------------------
st.set_page_config(page_title="ML Model Recommendation", layout="wide")

st.markdown("""
    <style>
        .title {
            font-size: 48px;
            font-weight: bold;
            color: white;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 32px;
            font-weight: 600;
            color: white;
            margin-top: 30px;
        }
        .result {
            font-size: 28px;
            font-weight: 600;
            color: white;
            margin-top: 10px;
        }
        div.stButton > button {
            background-color: #007BFF;
            color: white;
            font-weight: bold;
            height: 3em;
            width: 100%;
            border-radius: 10px;
            border: none;
            font-size: 16px;
        }
        div.stButton > button:hover {
            background-color: #007BFF;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">ML Model Recommendation</div>', unsafe_allow_html=True)

# ------------------------
# Load Data
# ------------------------
projects_df = pd.read_csv("Dataset AI Initial Stage(Data).csv", encoding='ISO-8859-1')
models_df = pd.read_csv("Dataset AI Initial Stage(Model Catalog).csv", encoding='ISO-8859-1')

mas_fields = [
    'HIL (Human-in-the-Loop)', 'AF (Algorithm Function)', 'DT (Data Type)',
    'DF (Data Format)', 'DQ (Data Quality)', 'INF (Infrastructure)'
]

severity_levels = {
    'HIL (Human-in-the-Loop)': ['direct', 'indirect', 'auto', 'opaque'],
    'AF (Algorithm Function)': ['descriptive', 'assistive', 'predictive', 'automation'],
    'DT (Data Type)': ['numeric', 'textual', 'image', 'audio', 'geospatial'],
    'DF (Data Format)': ['structured', 'semi-structured', 'unstructured'],
    'DQ (Data Quality)': ['stable', 'semi-stable', 'unstable'],
    'INF (Infrastructure)': ['low', 'medium', 'high']
}

weights = {
    'AF (Algorithm Function)': 0.22, 'DQ (Data Quality)': 0.22,
    'INF (Infrastructure)': 0.18, 'DT (Data Type)': 0.12,
    'DF (Data Format)': 0.12, 'HIL (Human-in-the-Loop)': 0.14
}

# ------------------------
# Helper Functions
# ------------------------
def severity_score(field, value):
    parts = str(value).lower().split('+')
    levels = severity_levels[field]
    scores = [(levels.index(p.strip()) / (len(levels) - 1)) if p.strip() in levels else 0.5 for p in parts]
    return sum(scores) / len(scores) if scores else 0.5

def normalize_mas(row):
    return set(
        v.strip().lower()
        for field in mas_fields
        for v in str(row[field]).split('+')
        if v.strip()
    )

def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

# ------------------------
# TF-IDF Semantic Similarity
# ------------------------
proj_descs = projects_df['ML Model Description'].fillna('').tolist()
model_descs = models_df['ML Model Description'].fillna('').tolist()
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(proj_descs + model_descs)
proj_tfidf = tfidf_matrix[:len(proj_descs)]
model_tfidf = tfidf_matrix[len(proj_descs):]
cos_sim_matrix = cosine_similarity(proj_tfidf, model_tfidf)

# ------------------------
# Streamlit UI
# ------------------------
col1, col2 = st.columns([2, 1])

ai_categories = sorted(projects_df['AI Category'].dropna().unique())
selected_category = col1.selectbox("Select an AI Category:", ai_categories)

filtered_projects = projects_df[projects_df['AI Category'] == selected_category]
project_options = filtered_projects['Project Name'].dropna().unique()
selected_project = col1.selectbox("Select an AI Project:", project_options)

if col1.button("🚀 Recommend Top 3 ML Models"):
    proj_idx = filtered_projects[filtered_projects['Project Name'] == selected_project].index[0]
    proj_row = projects_df.loc[proj_idx]
    proj_mas_set = normalize_mas(proj_row)
    proj_profile = {field: severity_score(field, proj_row[field]) for field in mas_fields}

    candidates = []
    model_profiles = {
        row['ML Model Name']: {field: severity_score(field, row[field]) for field in mas_fields}
        for _, row in models_df.iterrows()
    }

    for model_idx, model_row in models_df.iterrows():
        model_name = model_row['ML Model Name']
        model_mas_set = normalize_mas(model_row)
        model_profile = model_profiles[model_name]

        jaccard = jaccard_similarity(proj_mas_set, model_mas_set)
        tfidf_sim = cos_sim_matrix[proj_idx][model_idx]
        combined_score = 0.4 * jaccard + 0.6 * tfidf_sim

        improvement = sum(proj_profile[f] - model_profile[f] for f in mas_fields)
        candidates.append((model_name, combined_score, improvement, model_profile))

    top_models = sorted(candidates, key=lambda x: x[1], reverse=True)
    selected = [m for m in top_models if m[2] > 0][:3]

    if not selected:
        col1.warning("No suitable model found with improvement.")
    else:
        col1.markdown("<div class='subtitle'>Top 3 Recommended ML Models:</div>", unsafe_allow_html=True)
        for i, (model_name, score, _, _) in enumerate(selected, 1):
            col1.markdown(f"<div class='result'>{i}. {model_name}</div>", unsafe_allow_html=True)


# In[ ]:




