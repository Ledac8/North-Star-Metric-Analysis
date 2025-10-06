# 📈 Project 1: North Star Metric (NSM) Correlation Explorer

## Objective
The primary goal of this project is to analytically move beyond the definition of the North Star Metric to identify the **specific, actionable leading indicators** that predict long-term user success. By running correlation analysis on user behaviors against the NSM, we prioritize which actions the Growth team should focus on for onboarding and retention strategies.

## 🌟 The North Star Metric (NSM)
Our defined NSM for this analysis is: **Weekly Active Users (WAU) who complete 30 or more image downloads per week.**

## Methodology
1.  **Data Simulation:** A synthetic dataset of 10,000 users was generated, containing behavioral columns (e.g., feature usage, session time, number of searches) and a binary target column indicating whether the user achieved the NSM in a given week.
2.  **Feature Engineering:** Behavioral data was processed to create quantifiable features (e.g., "Time spent in the first 7 days," "Count of 'Favorite' actions").
3.  **Correlation Analysis:** The **Pearson correlation coefficient** (for linear relationships) and **Spearman's rank correlation** (for monotonic relationships) were calculated using Python's `scipy.stats` library between each behavioral feature and the NSM achievement.
4.  **Hypothesis Generation:** The features were prioritized based on the strength and statistical significance ($P$-Value) of their correlation to create actionable hypotheses for A/B testing.

## Technical Prowess Demonstrated
* **Correlation Analysis:** Effective use of `scipy.stats` to quantify the relationship between multiple independent variables and a binary outcome.
* **Data Preparation:** Handling feature selection and preparing a dataset for statistical analysis.
* **Actionable Reporting:** Translating statistical output into a prioritized, business-ready format.

## Sample Output & Actionable Hypotheses

The table below summarizes the output of the correlation script, prioritized by the strength of the positive correlation.

| Feature Used (Leading Indicator) | Correlation (R) with NSM | P-Value | Actionable Hypothesis |
| :--- | :--- | :--- | :--- |
| **Used "Smart Crop" feature in Week 1** | **+0.78** | **< 0.001** | **H1:** Users exposed to and completing the "Smart Crop" feature tutorial in their first session are 5x more likely to achieve the NSM. Focus onboarding here. |
| Time Spent in Session (Average) | +0.55 | < 0.001 | H2: Increasing average session time through curated content recommendations directly boosts NSM achievement. |
| Number of Searches Performed (Total) | +0.41 | 0.003 | H3: High search volume is a good sign, but doesn't guarantee value; focus should be on search *to-download* conversion. |
| Saved image to "Favorites" in Week 1 | +0.22 | 0.081 | H4: While positive, this action is not a strong enough predictor alone. Don't prioritize over H1 or H2. |

---

### Files in This Folder
* `correlation_explorer.py`: The Python script using Pandas and SciPy for the analysis.
* `simulated_user_data.csv`: The synthetic input data file.
* `analysis_notebook.ipynb`: A Jupyter Notebook detailing the data cleaning and visualization steps (optional, but recommended).
