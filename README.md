# Data Science Application in ICT — 5G Resource Allocation

> **Paper:** *Data Science Application in ICT*  
> **Author:** Dmitrii Savin  
> **Institution:** Institute of Multimedia Information and Communication Technologies, FEI STU in Bratislava  
> **Dataset:** [5G Resource Allocation Dataset](https://kaggle.com/datasets/omarsobhy14/5g-quality-of-service) (Sobhy, 2023)

---

## Overview

This repository contains the source code and LaTeX paper for a study on applying Data Science techniques to optimize **resource allocation in 5G networks**. Using a real-world QoS dataset, the project demonstrates end-to-end data cleaning, feature engineering, and machine learning to predict and analyze how network resources are distributed across different application types.

### Key results

| Metric | Value |
|--------|-------|
| Model  | RandomForestRegressor (200 trees) |
| MAE    | 0.86 % |
| RMSE   | 2.61 % |
| R²     | 0.922  |

---

## Repository structure

```
.
├── main.tex                        # LaTeX paper source
├── analysis.py                     # Main analysis & ML pipeline
├── Quality of Service 5G.csv      # Dataset (Sobhy, 2023 via Kaggle)
├── fig1.png                        # Predicted allocation trends by signal strength
├── fig2.png                        # Resource allocation distribution by app type
├── fig3.png                        # Feature importance
├── fig4.png                        # Predicted vs actual values
├── dashboard.png                   # Summary dashboard (2×2)
├── signal_vs_resource.png          # Signal strength vs allocation scatter
├── latency_vs_resource.png         # Latency vs allocation scatter
├── bandwidth_efficiency.png        # Bandwidth allocation efficiency
└── requirements.txt
```

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/<your-username>/SVOC_2025.git
cd SVOC_2025
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the analysis

```bash
python analysis.py
```

All figures are saved as high-resolution PNGs in the project root.

### 4. Compile the paper

Requires a LaTeX distribution (e.g. TeX Live, MiKTeX).

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Methodology

1. **Data cleaning** — strip unit suffixes (`dBm`, `ms`, `Mbps`, `Kbps`, `%`), parse timestamps, convert to numeric types.
2. **Feature engineering** — bandwidth to Mbps, hour/minute from timestamp, label-encoded application type, bandwidth efficiency ratio.
3. **Modeling** — `RandomForestRegressor` trained on six features: Application Type, Signal Strength, Latency, Required Bandwidth, Hour, Minute.
4. **Evaluation** — MAE, RMSE, R² on a held-out 20% test split.
5. **Visualisation** — 8 publication-quality figures covering distribution, feature importance, prediction quality, bandwidth efficiency, and signal-strength trends.

---

## Figures

| Figure | Description |
|--------|-------------|
| `fig1.png` | Predicted resource allocation vs. signal strength for each app type |
| `fig2.png` | Distribution of actual allocations per app type (strip + mean diamond) |
| `fig3.png` | RandomForest feature importance scores |
| `fig4.png` | Predicted vs. actual scatter, coloured by absolute error |
| `dashboard.png` | Four-panel summary: mean allocation, feature importance, pred vs actual, BW efficiency |

---

## Citation

If you use this code or paper, please cite:

```bibtex
@misc{savin2025svoc,
  author    = {Savin, Dmitrii},
  title     = {Data Science Application in ICT: 5G Resource Allocation},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/dmitthedazed/SVOC_2025}
}
```

---

## References

- Sobhy, O. (2023). *5G Resource Allocation Dataset: Optimizing Band*. Kaggle. https://kaggle.com/datasets/omarsobhy14/5g-quality-of-service
- scikit-learn developers. *RandomForestRegressor*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

---

## License

MIT — see [LICENSE](LICENSE).
