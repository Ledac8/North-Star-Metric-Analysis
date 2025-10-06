# 👥 Project 2: Customer Segmentation with Clustering

## Objective
The primary goal of this project is to apply unsupervised machine learning (K-Means Clustering) to segment our image generation platform's user base into distinct personas. This segmentation allows for the development of highly **tailored engagement, product, and marketing strategies** for each group.

## Methodology
1.  **Data Simulation:** A synthetic dataset of 100 users was created, simulating usage based on three expected personas: **e-commerce business owners, marketing agencies, and developers (API-centric)**.
2.  **Feature Selection:** Features that differentiate usage were chosen, including `images_generated_monthly`, `api_calls_monthly`, `login_frequency_weekly`, and `templates_used_monthly`.
3.  **Data Preprocessing:** Features were standardized using `StandardScaler` to ensure all metrics contributed equally to the clustering process.
4.  **Clustering:** The **K-Means algorithm** was applied to group the users into 3 clusters.
5.  **Interpretation:** Cluster centroids (the average feature values for each cluster) were analyzed to interpret and label the three distinct user segments.

## Technical Prowess Demonstrated
* **Machine Learning Application:** Direct application of the K-Means algorithm for market segmentation.
* **Data Preprocessing:** Crucial use of data standardization for optimal clustering performance.
* **Domain Interpretation:** Translating feature usage patterns (like high API calls vs. high template use) into actionable business personas.

## Key Takeaways: Customer Segments and Strategies

The analysis successfully identified three distinct segments, each requiring a different approach.

| Cluster ID | High-Value Metrics (Centroids) | Interpreted Persona | Proposed Engagement Strategy |
| :---: | :--- | :--- | :--- |
| **0** | High image generation, high template use, low API calls. | **🛍️ E-commerce Business Owners** | Focus on **bulk image generation**, product photography templates, and platform integrations (e.g., Shopify). |
| **1** | Very high API calls, low login frequency, low template use. | **💻 Developers (API-Centric)** | Focus on **API documentation**, SDKs, and developer support. Promote new API features and stability. |
| **2** | Moderate custom generation time, moderate login frequency. | **🎨 Marketing Agencies** | Focus on **team collaboration features**, client management tools, and diverse creative campaign templates. |

---

## Visualization: Segment Separation

The visualization below shows the clear separation between the three clusters when plotted against API usage and manual image generation.

![Customer Clusters based on API vs. Image Generation](customer_clusters.png)

---

### Files in This Folder
* `customer_segmentation_data.csv`: The simulated dataset used for clustering.
* `customer_segmentation.py`: The Python script for data preprocessing, K-Means clustering, and visualization.
* `customer_clusters.png`: The generated 2D scatter plot visualizing the customer segments.
