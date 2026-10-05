import logging

import numpy as np
import pandas as pd

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_synthetic_sediment_data(n_samples=1648, seed=42, add_missing=True, add_outliers=True):
    """
    Generates a synthetic dataset for predicting sediment particle size distributions.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating synthetic dataset with {n_samples} samples (seed={seed}).")
    np.random.seed(seed)

    # Time index (hourly)
    time = pd.date_range(pd.Timestamp('2018-07-01 00:00'), periods=n_samples, freq='h')

    # Basic feature generation
    S = np.random.uniform(20, 35, n_samples)
    np_index = np.random.normal(loc=1.38, scale=0.03, size=n_samples)

    # Temperature with seasonal component
    hours = np.arange(n_samples)
    T = 16 + 6 * np.sin(2 * np.pi * hours / (24 * 15)) + np.random.normal(0, 1.2, n_samples)

    # AR(1) process helper
    def ar1_process(mu, phi, sigma, n):
        x = np.zeros(n)
        x[0] = np.random.normal(mu, sigma)
        for t in range(1, n):
            x[t] = mu + phi * (x[t - 1] - mu) + np.random.normal(0, sigma)
        return x

    ub = np.clip(ar1_process(mu=0.25, phi=0.7, sigma=0.08, n=n_samples), 0.01, 1.0)
    u = np.clip(ar1_process(mu=0.18, phi=0.6, sigma=0.06, n=n_samples), 0.01, 0.8)

    a676_a650 = np.random.normal(1.0, 0.18, n_samples)
    a450_a676 = np.random.normal(0.9, 0.2, n_samples)
    chl_a = np.random.lognormal(mean=np.log(0.8), sigma=0.9, size=n_samples)

    # Target variables generation based on physical relationships
    d50_base = 5.0 * S + 300.0 * ub + 90.0 * np_index + 2.0 * T
    interaction = 30.0 * ((S - S.mean()) / S.std()) * (ub - ub.mean())
    bio_effect = 12.0 * np.log1p(chl_a)
    spectral_term = 40.0 * (a676_a650 - 1.0) ** 2
    d50_true = d50_base + interaction + bio_effect + spectral_term

    d50_observed = np.maximum(d50_true + np.random.normal(0, np.clip(0.03 * d50_true + 2.0, 1.0, 40.0)), 1.0)

    sigma2_true = 0.12 * (d50_true ** 1.25) + 8.0 * u + 20.0 * (a450_a676 - 0.9)
    sigma2_observed = np.maximum(sigma2_true + np.random.normal(0, np.clip(0.02 * sigma2_true + 1.0, 1.0, 60.0)), 1.0)

    df = pd.DataFrame({
        'time': time,
        'S': S,
        'ub': ub,
        'np': np_index,
        'T': T,
        'a676_a650': a676_a650,
        'a450_a676': a450_a676,
        'chl_a': chl_a,
        'u': u,
        'd50': d50_observed,
        'sigma2': sigma2_observed
    })

    # Injecting Missing values
    if add_missing:
        logger.info("Injecting missing values (2% of feature data).")
        feat_cols = ['S', 'ub', 'np', 'T', 'a676_a650', 'a450_a676', 'chl_a', 'u']
        n_cells = df[feat_cols].size
        n_missing = int(0.02 * n_cells)
        ix = np.random.choice(n_cells, n_missing, replace=False)
        rows = ix // len(feat_cols)
        cols = ix % len(feat_cols)
        for r, c in zip(rows, cols):
            df.at[r, feat_cols[c]] = np.nan

    # Injecting Outliers
    if add_outliers:
        logger.info("Injecting outliers into target variables.")
        n_out = max(1, int(0.005 * n_samples))
        out_idx = np.random.choice(n_samples, n_out, replace=False)
        df.loc[out_idx, 'd50'] *= np.random.uniform(1.5, 3.0, size=n_out)
        df.loc[out_idx, 'sigma2'] *= np.random.uniform(1.5, 2.5, size=n_out)

    # Final sanity clips for features
    df['a676_a650'] = df['a676_a650'].clip(0.4, 1.8)
    df['a450_a676'] = df['a450_a676'].clip(0.3, 1.6)
    df['chl_a'] = df['chl_a'].clip(0.01, 50)

    logger.info("Synthetic dataset generation completed successfully.")
    return df


if __name__ == "__main__":
    output_file = "Dataset.csv"
    data = generate_synthetic_sediment_data()
    data.to_csv(output_file, index=False)
    logger = logging.getLogger(__name__)
    logger.info(f"Dataset successfully saved to {output_file} with shape {data.shape}")
