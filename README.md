# 🔢 The Geometry of Numerical Reasoning

This repository contains the code and experiments for our **NAACL 2025** paper:

> **The Geometry of Numerical Reasoning**  
> El-Shangiti, Hiraoka, AlQuabeh, Heinzerling, Inui

---

## 🧠 Overview

This project investigates how large language models (LLMs) internally represent and manipulate numerical information. We analyze **numerical subspaces** in model activations, revealing geometric structures that support numerical reasoning.

<p align="center">
  <img src="assets/numerical-subspace.png" alt="Numerical Subspace Illustration" width="600"/>
</p>

### 🔬 Key Contributions
- 🔍 Identify **low-dimensional numerical directions** using Partial Least Squares (PLS)
- 📈 Show that numerical representations are **consistent and linear** across layers
- 🔁 Demonstrate **causal effects** through interventions in numerical subspaces
- 🧪 Propose new **evaluation protocols** for assessing LLMs' numerical reasoning

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/your-username/geometry-of-numerical-reasoning.git
cd geometry-of-numerical-reasoning


## 📬 Contact
For questions, feedback, or collaboration ideas:

Ahmed Oumar El-Shangiti
📧 ahmed.oumar@mbzuai.ac.ae

@inproceedings{el-shangiti-etal-2025-geometry,
  title     = "The Geometry of Numerical Reasoning: Language Models Compare Numeric Properties in Linear Subspaces",
  author    = "El-Shangiti, Ahmed Oumar and
               Hiraoka, Tatsuya and
               AlQuabeh, Hilal and
               Heinzerling, Benjamin and
               Inui, Kentaro",
  editor    = "Chiruzzo, Luis and
               Ritter, Alan and
               Wang, Lu",
  booktitle = "Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)",
  month     = apr,
  year      = "2025",
  address   = "Albuquerque, New Mexico",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2025.naacl-short.47/",
  doi       = "10.18653/v1/2025.naacl-short.47",
  pages     = "550--561",
  ISBN      = "979-8-89176-190-2",
  abstract  = "This paper investigates whether large language models (LLMs) utilize numerical attributes encoded in a low-dimensional subspace of the embedding space when answering questions involving numeric comparisons, e.g., Was Cristiano born before Messi? We first identified, using partial least squares regression, these subspaces, which effectively encode the numerical attributes associated with the entities in comparison prompts. Further, we demonstrate causality, by intervening in these subspaces to manipulate hidden states, thereby altering the LLM's comparison outcomes. Experiments conducted on three different LLMs showed that our results hold across different numerical attributes, indicating that LLMs utilize the linearly encoded information for numerical reasoning."
}




