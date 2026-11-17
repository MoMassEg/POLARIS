# POLARIS: Polar-Oriented Layered Antipodal Riemannian Isometric Sphere

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets%20%26%20Hub-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Reproducibility](https://img.shields.io/badge/Seeds-10%20Independent%20Runs-purple.svg)](output.txt)
[![Target Venues](https://img.shields.io/badge/Target-IEEE%20TKDE%20%7C%20ACM%20TOIS%20%7C%20Elsevier%20InfSci-orange.svg)](https://github.com/MoMassEg/POLARIS)

---

## 🌟 Overview

**POLARIS** (*Polar-Oriented Layered Antipodal Riemannian Isometric Sphere*) is a geometrically grounded Riemannian representation learning framework designed to resolve fundamental limitations in representation learning for semantics, lexical opposition, and document categorization.

Standard Euclidean and hyperbolic spaces conflate semantic association with true similarity, often clustering antonyms (*e.g., "hot" and "cold"*, *"good" and "bad"*) close together due to shared distribution contexts. Existing spherical models (such as JoSE) mitigate unbounded distance distortion but collapse opposing concepts into identical angular sectors. 

POLARIS introduces a **concentric spherical foliation** over the unit hypersphere $\mathbb{S}^{p-1}$ with **Tri-Zonal Geodesic Energy**, **exact $O(p)$ orthogonal Householder reflections**, **monotonic concentration cascades**, and **bipartite spherical transport** on tangent bundles.

---

## 🔬 Core Architectural Innovations

```
                       [ NORTH POLAR CAP ]
                   Synonyms & Context Attraction
                         cos(u, v) -> +1
                                ▲
                                │  Householder Isometries H(q)
                                │  Monotonic Cascade: κ^(1) > κ^(2) > ... > κ^(L)
     [ EQUATORIAL BELT ] ───────┼─────── [ EQUATORIAL BELT ]
   Orthogonal Decoupling        │        Hubness Suppression
      cos(u, v) -> 0            │           cos(u, v) -> 0
                                ▼
                       [ SOUTH POLAR CAP ]
                      Antipodal Opposition
                         cos(u, v) -> -1
```

1. **Tri-Zonal Spherical Loss Formulation**:
   - **North Polar Cap ($\cos \to +1$)**: Maximizes von Mises-Fisher directional concentration for synonyms and co-occurring distributional context.
   - **Equatorial Belt ($\cos \to 0$)**: Enforces mutual orthogonality for unrelated words, decoupling topological hubness without semantic distortion.
   - **South Antipodal Pole ($\cos \to -1$)**: Drives antonym pairs to exact antipodal opposition ($v^- \approx -u$), minimizing the Antipodal Alignment Score (AAS).

2. **$O(p)$ Orthogonal Householder Reflection Chains**:
   - Computes depth transitions across concentric layers via Householder reflections $H(q) = I - 2qq^\top$, strictly preserving spherical isometry ($\|H(q)x\|_2 = \|x\|_2$) in linear time without expensive matrix inverses or cubic Stiefel retractions.

3. **Monotonic Concentration Cascade**:
   - Automatically learns layered von Mises-Fisher concentrations structured as $\kappa^{(1)} > \kappa^{(2)} > \dots > \kappa^{(L)} > 2.0$, enforcing fine-grained semantic focus in early layers and broader topical integration in deep layers.

4. **Multi-Hop Signed Geodesic Involution**:
   - Formulates 2-hop polarity propagation over Princeton WordNet graphs with strict polysemy contradiction protection, ensuring zero train/test leakage ($\text{Synonyms} \cap \text{Antonyms} = \emptyset$).

5. **Polarity-Gated Negation Reflection**:
   - Incorporates sentence-level negation countdown windows with positive idiom masking (*"not only"*, *"without a doubt"*, etc.) to reflect sentiment tokens to their antipodes during document-level aggregation.

6. **Riemannian Tangential Optimization Engine**:
   - Executes Riemannian unit-normalized tangential descent on $T_x \mathbb{S}^{p-1}$ using retraction $R_x(\xi) = \frac{x + \xi}{\|x + \xi\|_2}$, backed by almost-sure convergence under the Riemannian Robbins-Siegmund theorem (Theorem 2).

---

## 📊 Benchmark Results

All experiments are conducted across **10 independent random seeds** (`[42, 100, 2024, 777, 999, 123, 456, 888, 1010, 2026]`) with 95% Confidence Intervals, paired Student's $t$-tests, and Holm-Bonferroni correction.

### 1. Main Benchmark Comparison (10 Seeds, Mean $\pm$ 95% CI)

| Geometric Space | Model | AntSyn AUC ($\uparrow$) | AAS ($\downarrow 0$) | SimLex-999 $\rho$ ($\uparrow$) | 20NG NMI ($\uparrow$) | 20NG Macro-F1 ($\uparrow$) | Movie Reviews F1 ($\uparrow$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Euclidean+Norm** | Word2Vec+Norm | $0.5173 \pm 0.002$ | $1.4272 \pm 0.001$ | $0.055 \pm 0.011$ | $0.335 \pm 0.005$ | $0.370 \pm 0.004$ | $0.581 \pm 0.010$ |
| **Hyperbolic $\mathbb{B}^p$** | Poincaré-Emb | $0.5011 \pm 0.008$ | $1.4121 \pm 0.001$ | $-0.007 \pm 0.016$ | $0.049 \pm 0.005$ | $0.130 \pm 0.008$ | $0.575 \pm 0.012$ |
| **Euclidean+Pol** | Attract-Repel | $0.8878 \pm 0.002$ | $1.2726 \pm 0.001$ | $0.230 \pm 0.007$ | $0.331 \pm 0.005$ | $0.360 \pm 0.008$ | $0.602 \pm 0.010$ |
| **Spherical ($L=1$)** | JoSE (*NeurIPS '19*) | $0.5280 \pm 0.002$ | $1.4568 \pm 0.001$ | $0.121 \pm 0.005$ | $0.277 \pm 0.005$ | $0.327 \pm 0.009$ | $0.604 \pm 0.017$ |
| **Spherical ($L=3$)** | **POLARIS (Ours)** | **0.9886 $\pm$ 0.0001**$^{***}$ | **0.6108 $\pm$ 0.0003** | **0.435 $\pm$ 0.004**$^{***}$ | **0.354 $\pm$ 0.003** | **0.468 $\pm$ 0.009**$^{***}$ | **0.612 $\pm$ 0.006** |

*Significance against JoSE: $^{***}p < 0.001$ via paired $t$-test with Holm-Bonferroni correction.*

---

### 2. Qualitative Spherical Pole Inspection

Learned unit embeddings in $\mathbb{S}^{99}$ demonstrate precise geometric disentanglement:

| Anchor Word ($u$) | North Pole: Synonyms ($\cos \to +1$) | Equator: Orthogonal ($\cos \approx 0$) | South Pole: Antipodes ($\cos \to -1$) |
|:---|:---|:---|:---|
| **good** | best (+0.992) | power (+0.0001) | stale (-0.977) |
| **love** | adore / cherish (+0.978) | whenever (-0.0001) | hate (-0.969) |
| **happy** | cheerful / willing (+0.887) | sandwich (-0.0000) | unhappy (-0.953) |
| **win** | won / triumph (+0.979) | availability (-0.0001) | lose (-0.971) |
| **strong** | solid / robust (+0.962) | transit (+0.0000) | mindless / weak (-0.909) |
| **smart** | intelligent (+0.937) | routine (-0.0000) | dumb (-0.950) |
| **success** | winner (+0.982) | publication (+0.0000) | failure (-0.965) |
| **increase**| increasing (+0.963) | cabin (+0.0000) | fall / decrease (-0.967) |

---

### 3. Component Ablation Study (10 Seeds)

| Ablation Setting | SimLex $\rho$ | AntSyn AUC | AAS ($\downarrow 0$) | 20NG NMI | RT Movie F1 | Train Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **POLARIS (Full Model, $L=3$)** | **0.435** | **0.9886** | **0.6108** | **0.354** | **0.612** | ~10.5 s |
| w/o Antipodal Zone ($\mathcal{L}_{\text{anti}} = 0$) | 0.213 | 0.7543 | 1.7932 | 0.347 | 0.598 | 8.9 s |
| w/o Equatorial Zone ($\mathcal{L}_{\text{eq}} = 0$) | 0.431 | 0.9881 | 0.6097 | 0.349 | 0.614 | 9.8 s |
| w/o Householder (Dense + $L_2$ Norm) | 0.425 | 0.9859 | 0.6152 | 0.322 | 0.590 | 9.4 s |
| w/o Learnable Kappa (Fixed $\kappa = 1$) | 0.418 | 0.9879 | 0.5987 | 0.287 | 0.614 | 9.9 s |
| w/o Negation Reflection (Unsigned) | 0.434 | 0.9885 | 0.6105 | 0.357 | 0.609 | 10.5 s |

---

## 📂 Repository Structure

```plaintext
POLARIS/
├── main.py                   # Complete, self-contained implementation & benchmark runner
├── notebookf78d214b81.ipynb  # Interactive Jupyter / Kaggle GPU notebook
├── output.txt                # Full raw experimental logs & telemetry for all 10 seeds
├── .gitignore                # Environment, cache, and artifact exclusions
└── README.md                 # Project documentation & publication specifications
```

---

## 🚀 Quick Start

### 1. Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- Hugging Face `datasets` & `huggingface_hub`
- NLTK (`wordnet`, `omw-1.4`)
- Scikit-learn, SciPy, Pandas

Install all dependencies in one command:
```bash
pip install torch torchvision datasets huggingface_hub nltk scikit-learn scipy pandas pyarrow
```

### 2. Running Experiments

Execute the full 10-seed experimental suite (downloads real HF datasets, compiles sparse GPU graphs, evaluates all 5 models and 6 ablations):

```bash
python main.py
```

*Expected runtime: ~3.5 minutes on a modern GPU (NVIDIA T4 / RTX 4000 or better), with peak GPU memory under 750 MB.*

---

## 📐 Mathematical Formulation

### Tri-Zonal Hyperspherical Loss

$$\mathcal{L}_{\text{POLARIS}} = \frac{1}{L} \sum_{l=1}^L \left( \mathcal{L}_{\text{attr}}^{(l)} + \lambda_{\text{anti}} \mathcal{L}_{\text{anti}}^{(l)} + \lambda_{\text{eq}} \mathcal{L}_{\text{eq}}^{(l)} \right)$$

where:

$$\mathcal{L}_{\text{attr}}^{(l)} = -\mathbb{E}_{(u, v) \sim \mathcal{D}_{\text{cooc}}} \left[ w_{uv} \left( \log \sigma(\kappa^{(l)} u_l^\top v) + \log \sigma(-\kappa^{(l)} u_l^\top v_{\text{neg}}) \right) \right] + \lambda_{\text{syn}} \mathcal{L}_{\text{syn}}^{(l)}$$

$$\mathcal{L}_{\text{anti}}^{(l)} = \mathbb{E}_{(u, v^-) \sim \mathcal{E}_{\text{ant}}} \left[ \max(0, \, 0.88 + u_l^\top v^-) + 0.45 (1.0 + u_l^\top v^-)^2 \right]$$

$$\mathcal{L}_{\text{eq}}^{(l)} = \mathbb{E}_{(u, w) \sim \mathcal{D}_{\text{unrel}}} \left[ \max(0, \, |u_l^\top w| - 0.08)^2 + 0.25 (u_l^\top w)^2 \right]$$

### Riemannian Retraction & Descent

Given ambient gradient $\nabla_{\text{euc}} \mathcal{L}$, the Riemannian gradient on the tangent space $T_x \mathbb{S}^{p-1}$ is computed via orthogonal projection:

$$\operatorname{grad} \mathcal{L}(x) = (I - x x^\top) \nabla_{\text{euc}} \mathcal{L}$$

followed by geodesic retraction back to the hypersphere:

$$R_x(\xi) = \frac{x + \xi}{\|x + \xi\|_2}$$

---

## 📖 Citation

If you use POLARIS in your research, please cite:

```bibtex
@article{polaris2026riemannian,
  title   = {POLARIS: Polar-Oriented Layered Antipodal Riemannian Isometric Sphere for Disentangled Lexical and Document Representation},
  author  = {Amer, Mohamed Shabaan and Collaborators},
  journal = {IEEE Transactions on Knowledge and Data Engineering (TKDE) / ACM Transactions on Information Systems (TOIS)},
  year    = {2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
