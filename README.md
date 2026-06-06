# Classification of Fire Types in India Using MODIS Satellite Data (2021–2023)

## 📌 Project Overview
India witnesses various types of fire incidents annually, including forest fires, agricultural burning, volcanic activity, and other thermal anomalies. Accurate identification of fire sources is crucial for timely disaster response, environmental monitoring, and resource management. 

This project utilizes data from MODIS sensors aboard NASA’s Terra and Aqua satellites, which provide reliable, near real-time thermal anomaly data. The primary challenge addressed in this repository is correctly classifying the type of fire event—whether it stems from vegetation, volcanoes, static land sources, or offshore sources—using satellite-captured features.

## 🎯 Objective
To develop a machine learning classification model that can accurately predict the type of fire using MODIS fire detection data for India from the years 2021 to 2023.

## 🔥 Dataset Summary
The Moderate Resolution Imaging Spectroradiometer (MODIS) is a key NASA sensor capturing Earth observation data at a spatial resolution of 1 km. 

* **Source:** NASA’s FIRMS (Fire Information for Resource Management System)
* **Satellites:** Terra (morning overpasses) and Aqua (afternoon overpasses)
* **Timeframe:** 2021 - 2023
* **Key Parameters:** Latitude, Longitude, Brightness (Temperature), Scan, Track, Acquisition Date/Time, Confidence, Fire Radiative Power (FRP), and Day/Night flag.

## 🛠️ Project Workflow

1.  **Data Loading & Integration:** * Aggregated yearly datasets (2021, 2022, 2023) into a single unified pandas DataFrame comprising over 270,000 records.
2.  **Exploratory Data Analysis (EDA):**
    * Visualized geographical fire distributions using `Folium` and `Seaborn` scatter plots.
    * Analyzed temporal patterns (monthly, daily, hourly fire frequencies).
    * Investigated feature distributions and pair-wise correlations using heatmaps to understand the relationships between Brightness, Confidence, and FRP.
3.  **Data Preprocessing & Cleaning:**
    * **Outlier Treatment:** Applied the Interquartile Range (IQR) method to cap extreme values in continuous features like brightness, scan, track, and FRP.
    * **Feature Engineering:** Extracted temporal features (Year, Month, Day of Week, Day of Year, Hour) from acquisition dates and times.
    * **Encoding:** Performed One-Hot Encoding on categorical variables (`daynight`, `satellite`, `instrument`).
    * **Normalization:** Scaled continuous numerical variables using standard scaling (`StandardScaler`).
4.  **Handling Imbalanced Data:**
    * Addressed significant class imbalances in fire types using Synthetic Minority Over-sampling Technique (SMOTE) to ensure robust model training.
5.  **Model Building (Preparation):**
    * Prepared data pipelines for classification algorithms including Logistic Regression, Random Forest, K-Nearest Neighbors, and XGBoost.

## 💻 Technologies & Libraries Used
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn, Folium
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE)

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/Nishith-062/Fire_Classification_using_MODIS_satellite_data.git](https://github.com/Nishith-062/Fire_Classification_using_MODIS_satellite_data.git)

2. Navigate to the project directory:
```bash
cd Fire_Classification_using_MODIS_satellite_data

```


3. Install the required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn folium xgboost imbalanced-learn

```


4. Open the Jupyter Notebook:
```bash
jupyter notebook Classification_of_Fire_Types_in_India_Using_MODIS_Satellite_Data.ipynb

```


*(Note: Ensure you "Trust Notebook" in your Jupyter environment to properly render the Folium interactive maps).*

## 📚 References

* [NASA FIRMS Documentation](https://earthdata.nasa.gov/earth-observation-data/near-real-time/firms)
* [MODIS Active Fire Product Info (LP DAAC)](https://lpdaac.usgs.gov/)


