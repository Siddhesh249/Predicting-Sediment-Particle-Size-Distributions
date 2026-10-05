# Predicting Sediment Particle Size Distributions

**Team Members:**  
- Siddhesh Dattatray Koditkar (PES2UG24CS502)
- Sumedh B Rao (PES2UG24CS535)   

---

## Project Overview
This project replicates the methodologies presented in the research paper:  
> [**"Machine Learning for Predicting Sediment Particle Size Distributions" — Galen Egan, Stanford University**](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26391077.pdf)

The paper aims to use machine learning to accurately predict suspended marine sediment particle size distributions (median particle diameter $d_{50}$ and variance $\sigma^2$). Accurately predicting these values is critical for modeling coastline erosion, tracking environmental contaminants, and predicting global carbon sequestration rates. 

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Siddhesh249/Predicting-Sediment-Particle-Size-Distributions.git
cd Predicting-Sediment-Particle-Size-Distributions
```

### 2. Install Dependencies
Confirm required libraries are installed using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Generate the Dataset
Since the original dataset from the South San Francisco Bay field campaigns was not available online, we developed a synthetic data generator that faithfully simulates the nonlinear relationships and physical properties described in the paper.
```bash
python3 Dataset_Generator.py
```
This generates `Dataset.csv` containing simulated physical, optical, and biological features.

### 4. Train and Evaluate Models
```bash
python3 Models.py
```
This script trains the models, performs hyperparameter tuning, evaluates performance, and outputs performance metric charts to the `Results/` directory.

---

## Model Architectures & Results

### Random Forest (RF) Regression
Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the average prediction of the individual trees. It naturally handles complex, non-linear relationships and resists overfitting. 
- **Application here:** Evaluated across a varying number of trees, peaking at 400 estimators, using all 8 available features.
- **Results:** Achieved excellent predictive accuracy with an $R^2 \approx 0.84$ for $d_{50}$ and $R^2 \approx 0.91$ for $\sigma^2$.

### Support Vector Regression (SVR)
SVR solves a convex optimization problem to find the optimal margin classifier that minimizes the difference between data points and the hypothesis function.
- **Application here:** As per the paper's feature importance analysis, SVR was trained and tuned using only the top-4 most critical features (Water Salinity $S$, Wave-orbital velocity $u_b$, Particle index of refraction $n_p$, and Water Temperature $T$).
- **Results:** Despite using half the features, SVR achieved highly competitive accuracy with an $R^2 \approx 0.84$ for $d_{50}$ and $R^2 \approx 0.90$ for $\sigma^2$.

---

## Conclusion
Both machine learning algorithms successfully captured the complex, nonlinear particle flocculation dynamics that are notoriously difficult to model with traditional semi-empirical formulas. With testing $R^2$ scores significantly above the $> 0.75$ baseline mentioned in the paper, the approach demonstrates high viability for integration into large-scale sediment transport models.
