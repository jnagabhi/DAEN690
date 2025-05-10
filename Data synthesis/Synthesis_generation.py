#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import random

# 1. Load original dataset
df = pd.read_csv("Synthetic_train_data.csv", encoding="ISO-8859-1")  # or use encoding


# 2. Create lookup dictionaries

# Map: AI Category → valid ML Models Used
cat_to_models = df.groupby("AI Category")["ML Model Used"].unique().to_dict()

# Map: ML Model Used → ML Model Description
model_to_desc = df.drop_duplicates("ML Model Used").set_index("ML Model Used")["ML Model Description"].to_dict()

# Map: AI Category → Primary Function, NAICS Code, NAICS Description (fixed for a category)
cat_to_meta = df.drop_duplicates("AI Category").set_index("AI Category")[["Primary Function", "NAICS Code", "NAICS Description"]].to_dict("index")

# Map: ML Model Used → HIL + AF + DT + DF + DQ + INF (group of 6 fields)
model_to_config = df.drop_duplicates("ML Model Used").set_index("ML Model Used")[["HIL (Human-in-the-Loop)", "AF (Algorithm Function)", "DT (Data Type)", "DF (Data Format)", "DQ (Data Quality)", "INF (Infrastructure)"]].to_dict("index")

# Map: NAICS Code → Compliance Cost
naics_to_cost = df.drop_duplicates("NAICS Code").set_index("NAICS Code")["Compliance Cost"].to_dict()

# Map: Tuple of (HIL, AF, DT, DF, DQ, INF) → Daily Data Volume (GB)
config_to_ddv = df.drop_duplicates(subset=["HIL (Human-in-the-Loop)", "AF (Algorithm Function)", "DT (Data Type)", "DF (Data Format)", "DQ (Data Quality)", "INF (Infrastructure)"])
config_to_ddv = config_to_ddv.set_index(["HIL (Human-in-the-Loop)", "AF (Algorithm Function)", "DT (Data Type)", "DF (Data Format)", "DQ (Data Quality)", "INF (Infrastructure)"])["Daily Data Volume (GB)"].to_dict()

# Map: Tuple of (Compliance Cost, Daily Data Volume) → Scope
cost_ddv_to_scope = df.drop_duplicates(subset=["Compliance Cost", "Daily Data Volume (GB)"]).set_index(["Compliance Cost", "Daily Data Volume (GB)"])["Scope"].to_dict()

# 3. Generate synthetic samples
num_samples = 5000
synthetic_rows = []

for _ in range(num_samples):
    # AI Category
    ai_cat = random.choice(list(cat_to_models.keys()))
    
    # ML Model Used (from valid ones for that category)
    ml_model = random.choice(cat_to_models[ai_cat])
    
    # ML Model Description
    ml_desc = model_to_desc[ml_model]

    # Fixed attributes from category
    primary_func = cat_to_meta[ai_cat]["Primary Function"]
    naics_code = cat_to_meta[ai_cat]["NAICS Code"]
    naics_desc = cat_to_meta[ai_cat]["NAICS Description"]

    # Model config
    config = model_to_config[ml_model]
    hil = config["HIL (Human-in-the-Loop)"]
    af = config["AF (Algorithm Function)"]
    dt = config["DT (Data Type)"]
    dfmt = config["DF (Data Format)"]
    dq = config["DQ (Data Quality)"]
    inf = config["INF (Infrastructure)"]

    # Compliance Cost from NAICS
    cost = naics_to_cost[naics_code]

    # Daily Data Volume from (hil, af, dt, dfmt, dq, inf)
    ddv_key = (hil, af, dt, dfmt, dq, inf)
    ddv = config_to_ddv.get(ddv_key, random.choice(df["Daily Data Volume (GB)"]))  # fallback random if key missing

    # Scope based on (cost, ddv)
    scope = cost_ddv_to_scope.get((cost, ddv), random.choice(df["Scope"]))  # fallback if not found

    # Build row
    synthetic_rows.append({
        "AI Category": ai_cat,
        "ML Model Used": ml_model,
        "ML Model Description": ml_desc,
        "Primary Function": primary_func,
        "HIL (Human-in-the-Loop)": hil,
        "AF (Algorithm Function)": af,
        "DT (Data Type)": dt,
        "DF (Data Format)": dfmt,
        "DQ (Data Quality)": dq,
        "INF (Infrastructure)": inf,
        "NAICS Code": naics_code,
        "NAICS Description": naics_desc,
        "Compliance Cost": cost,
        "Daily Data Volume (GB)": ddv,
        "Scope": scope
    })

# 4. Save synthetic dataset
synthetic_df = pd.DataFrame(synthetic_rows)
synthetic_df.to_csv("synthetic_data_mapped.csv", index=False)
print("✅ Synthetic dataset saved as 'synthetic_data_mapped.csv'")


# In[ ]:




