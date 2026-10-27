<div align="center">

# 5G Resource Allocation

Predicting how 5G networks allocate resources across apps with a random forest.

Paper *Data Science Application in ICT* · Dmitrii Savin · FEI STU Bratislava

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white&style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

<img src="figures/dashboard.png" alt="Results dashboard" width="720">

</div>

> [!NOTE]
> Archived. The paper is finished, so this repository is read-only and kept for reference.

## Results

A `RandomForestRegressor` with 200 trees, trained on the [5G Quality of Service dataset](https://kaggle.com/datasets/omarsobhy14/5g-quality-of-service) (Sobhy, 2023), 20% held out for testing:

| R² | MAE | RMSE |
| --- | --- | --- |
| 0.922 | 0.86 % | 2.61 % |

Application type and required bandwidth matter most.

## Run

```sh
git clone https://github.com/dmitthedazed/SVOC_2025.git
cd SVOC_2025
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

Figures are written to `figures/`.

Build the paper (needs TeX Live or MiKTeX):

```sh
cd paper && pdflatex main.tex && pdflatex main.tex
```

<details>
<summary>How it works</summary>

1. **Cleaning.** Strip units (`dBm`, `ms`, `Mbps`, `Kbps`, `%`), parse timestamps, convert to numbers.
2. **Features.** Application type, signal strength, latency, required bandwidth, hour, minute.
3. **Model.** Random forest regressor predicting the resource allocation percentage.
4. **Evaluation.** MAE, RMSE and R² on a 20% test split.

</details>

<details>
<summary>Files</summary>

```text
analysis.py     cleaning, model and figures
data/           5G QoS dataset (Sobhy, 2023)
figures/        generated plots, dashboard.png has them all
paper/          LaTeX source of the paper
```

</details>

## Citation

```bibtex
@misc{savin2025svoc,
  author = {Savin, Dmitrii},
  title  = {Data Science Application in ICT: 5G Resource Allocation},
  year   = {2025},
  url    = {https://github.com/dmitthedazed/SVOC_2025}
}
```

## License

[MIT](LICENSE) © Dmytrii Savin
