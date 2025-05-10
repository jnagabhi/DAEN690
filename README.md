# DAEN690
DAEN 690 Repository for Spring 2025 - GMU
TEAM A

# 🤖 AI Project Risk Reduction

## 📌 Problem Context

Artificial Intelligence (AI) is rapidly transforming industries such as healthcare, finance, and retail. Despite advancements in powerful, domain-specific models, many organizations continue to struggle with real-world AI adoption. The failure to align AI solutions with practical applications results in underperformance, financial loss, and reduced stakeholder trust.

---

## 🎯 Project Objectives

This project aims to:
- Investigate root causes of AI project failure.
- Perform model evaluation.
- Recommend optimized models based on MAS attributes, semantic similarity, and risk scores.
- Visualize project risk and recommendation insights using a dashboard for AI project managers to make data-driven decisions.

---

## 🧠 Key Features

- **Data Synthesis**: Augments sparse records using CTGAN to support better model mapping.
- **MAS Scoring Framework**: Evaluates original models using normalized attributes like:
  - Algorithm Function (AF)
  - Human-in-the-loop (HIL)
  - Data Quality (DQ)
  - Data Type (DT)
  - Data Format (DF)
  - Infrastructure Availability (INF)
    
- **Model Recommender System**: Recommends top 3 ML models using:
  - Jaccard Similarity
  - Semantic Boosting via TF-IDF & Cosine Similarity
- **Risk Scoring**: Uses a 7-point Likert scale to compare baseline and improved risk profiles.
- **Interactive Dashboard**: Built with **Amazon QuickSight** to display:
  - Key Performance Indicators (KPIs)
  - Outcome distribution
  - Average risk score reduction
  - Individual project risk differences

---


## 🧪 How the ML Model Recommendation System Works

1. **User Input**: The user selects an AI category and a specific AI project.
2. **MAS Evaluation**: The system scores the original project using MAS attributes.
3. **Similarity Matching**:
   - **Jaccard Similarity**: Matches project attributes to catalog models.
   - **TF-IDF with Cosine Similarity**: Boosts semantic relevance.
4. **Model Recommendation**:
   - The top 3 ML models are ranked based on similarity , semantic boosting and risk score.
5. **Dashboard Display**:
   - Visual insights on project outcomes and recommended model performance.

---

## 🛠️ Tech Stack

- **Python** (Pandas, NumPy, Scikit-learn)
- **CTGAN** (Conditional Tabular GAN for synthetic data)
- **Streamlit** (Interactive UI)
- **Amazon QuickSight** (Dashboard visualization)
- **Jupyter Notebook** (Exploratory data analysis and development)

---

## 🚀 Future Enhancements

- Integrate real-time data cost & volume analysis using AWS services.
- Enable dynamic MAS attribute weighting in the dashboard.
- Develop a time-series risk predictor for emerging AI projects.
- Formalize data governance and model documentation standards.
- Establish a model governance committee for audit trails and approvals.

---


Team Members: 
hchandup - Product Owner
sgarnep - Scrum Master
kkalva - Developer
jnagabhi - Developer
ashaik26 - Developer
svalugun - Developer
