#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load data
projects_df = pd.read_csv("Dataset AI Initial Stage(Data).csv", encoding='ISO-8859-1')
models_df = pd.read_csv("Dataset AI Initial Stage(Model Catalog).csv", encoding='ISO-8859-1')

# MAS fields and risk levels
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

# Helpers
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

# TF-IDF Semantic Similarity
proj_descs = projects_df['ML Model Description'].fillna('').tolist()
model_descs = models_df['ML Model Description'].fillna('').tolist()
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(proj_descs + model_descs)
proj_tfidf = tfidf_matrix[:len(proj_descs)]
model_tfidf = tfidf_matrix[len(proj_descs):]
cos_sim_matrix = cosine_similarity(proj_tfidf, model_tfidf)

# Build model MAS severity profiles
model_profiles = {
    row['ML Model Name']: {field: severity_score(field, row[field]) for field in mas_fields}
    for _, row in models_df.iterrows()
}

# Run selection
output_rows = []

for proj_idx, proj_row in projects_df.iterrows():
    proj_name = proj_row['Project Name']
    proj_mas_set = normalize_mas(proj_row)
    proj_profile = {field: severity_score(field, proj_row[field]) for field in mas_fields}

    candidates = []

    for model_idx, model_row in models_df.iterrows():
        model_name = model_row['ML Model Name']
        model_mas_set = normalize_mas(model_row)
        model_profile = model_profiles[model_name]

        jaccard = jaccard_similarity(proj_mas_set, model_mas_set)
        tfidf_sim = cos_sim_matrix[proj_idx][model_idx]
        combined_score = 0.4 * jaccard + 0.6 * tfidf_sim

        improvement = sum(proj_profile[f] - model_profile[f] for f in mas_fields)
        candidates.append((model_name, combined_score, improvement, model_profile))

    # Sort and filter by improvement
    top_models = sorted(candidates, key=lambda x: x[1], reverse=True)
    selected = [m for m in top_models if m[2] > 0][:3]

    if not selected:
        continue

    # Merge MAS attributes
    combined_mas = defaultdict(list)
    risk_breakdown = defaultdict(list)
    for _, _, _, profile in selected:
        for field in mas_fields:
            levels = severity_levels[field]
            rank = round(profile[field] * (len(levels) - 1))
            label = levels[rank]
            combined_mas[field].append(label)
            risk_breakdown[f"{field}::{label}"].append(profile[field])

    final_mas = {field: '+'.join(sorted(set(values))) for field, values in combined_mas.items()}

    total_risk = 0
    per_category_scores = {}
    for field in mas_fields:
        levels = final_mas[field].split('+')
        avg_severity = sum(severity_score(field, v) for v in levels) / len(levels)
        total_risk += avg_severity * weights[field]

        for label in severity_levels[field]:
            key = f"{field}::{label}"
            if key in risk_breakdown:
                per_category_scores[key] = round(sum(risk_breakdown[key]) / len(risk_breakdown[key]), 2)

    output_rows.append({
        'Project Name': proj_name,
        'Selected Models': '+'.join([m[0] for m in selected]),
        **final_mas,
        **per_category_scores,
        'Overall Risk Score (1-5)': round(1 + total_risk * 4, 2)
    })

# Save result
pd.DataFrame(output_rows).to_csv("Step1_MAS_TFIDF_Scored_Model_Matches.csv", index=False)
print("✅ Step 1 complete. Output saved to 'Step1_MAS_TFIDF_Scored_Model_Matches.csv'")


# In[8]:


import pandas as pd

# Load Step 1 (optimized) and original dataset
step1_df = pd.read_csv("Step1_MAS_TFIDF_Scored_Model_Matches.csv")
projects_df = pd.read_csv("Dataset AI Initial Stage(Data).csv", encoding='ISO-8859-1')

# MAS attributes and risk definitions
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

# Function to calculate severity score for a composite MAS value
def score_mas(field, value):
    levels = severity_levels[field]
    parts = str(value).lower().split('+')
    scores = [(levels.index(p.strip()) / (len(levels) - 1)) if p.strip() in levels else 0.5 for p in parts]
    return sum(scores) / len(scores) if scores else 0.5

# Calculate original risk for each project
original_df = projects_df[['Project Name'] + mas_fields].copy()

def compute_original_risk(row):
    total = 0
    for field in mas_fields:
        total += score_mas(field, row[field]) * weights[field]
    return round(1 + total * 4, 2)

original_df['Original Risk Score (1-5)'] = original_df.apply(compute_original_risk, axis=1)

# Merge with optimized risk
comparison_df = pd.merge(
    original_df[['Project Name', 'Original Risk Score (1-5)']],
    step1_df[['Project Name', 'Overall Risk Score (1-5)']],
    on='Project Name'
)
comparison_df['Risk Change'] = comparison_df['Overall Risk Score (1-5)'] - comparison_df['Original Risk Score (1-5)']

# Save output
comparison_df.to_csv("Step2_Original_vs_MAS_Optimized_Risk_Comparison.csv", index=False)
print("✅ Step 2 complete: comparison saved to 'Step2_Original_vs_MAS_Optimized_Risk_Comparison.csv'")


# In[ ]:




