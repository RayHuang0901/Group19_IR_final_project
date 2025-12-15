# Group19_IR_final_project

This project implements a comprehensive Multimodal Information Retrieval (MIR) system tailored for the fashion domain, utilizing the DeepFashion (In-shop) dataset. We address the semantic gap between visual content and textual descriptions through three core tasks: Text-to-Image, Image-to-Text, and Composed Image Retrieval.

By leveraging state-of-the-art Vision-Language Models (VLMs) such as BLIP, OpenCLIP, and FashionSigLIP, along with LLM-augmented strategies, our system aims to improve fine-grained product discovery and retrieval accuracy.

## Installation & PrerequisitesThe project requires Python 3.8+ and PyTorch.

### 1. Clone the repository:

```
git clone https://github.com/YourRepo/Group19_IR_final_project.git
cd Group19_IR_final_project
```

### 2. Install dependencies:

```
pip install -r requirements.txt
```

### 3. Data Setup:Ensure the DeepFashion dataset is structured correctly. (e.g., data/DeepFashion/img_highres/).

## Text-to-Image Retrieval (T2I)

This module evaluates **Text-to-Image (T2I) retrieval** on the DeepFashion (In-shop) dataset.  
Given a textual product description, the goal is to retrieve visually corresponding fashion images from a fixed gallery.

We investigate whether **LLM-assisted query rewriting** improves retrieval accuracy compared to directly encoding raw product descriptions.


### Methodology

#### Baseline Retrieval
We directly encode the original product description (`text` field in DeepFashion) using a frozen vision–language model and retrieve images by similarity in a shared embedding space.

Two retrieval backbones are evaluated:
- **FashionSigLIP (Marqo)** – domain-adapted for fashion retrieval
- **CLIP ViT-B/32 (LAION)** – general-purpose vision–language model

All gallery images are encoded offline and indexed using **FAISS (IndexFlatIP)** for efficient nearest-neighbor search.


#### LLM-assisted Query Rewriting

To reduce noise and improve visual grounding in textual queries, we add an **LLM-based query rewriting step** before retrieval.

Given a raw product description, the LLM rewrites the query into a visually focused form using one of three styles:

- **detail**: one natural English sentence describing appearance (color, fit, cut, length, pattern, material, key design details)
- **keywords**: a comma-separated list of concise visual attributes
- **conversational**: a search-style query that a user might type

We also control the **verbosity** of the rewritten query using prompt-based length constraints:
- `short` (≈ 10–20 words)
- `medium` (≈ 40–50 words)
- `long` (up to ≈ 100 words)
- `default` (no explicit length constraint)

These variants enable controlled experiments on how **rewriting style** and **output length** affect retrieval performance.


### Dataset Subset Construction

To ensure stable evaluation, we sample a constrained subset of **5,000 images** from DeepFashion with the following guarantees:

- Each `(category1, category2, color)` triple appears at least **15** times  
- Each `(category1, category2)` pair contains at least **2** different colors

This ensures there are enough positives per group and prevents degenerate cases during evaluation.


### Evaluation Metrics

We report:
- **Recall@1 / Recall@5 / Recall@10**
- **Precision@1 / Precision@5 / Precision@10**
- **MRR (Mean Reciprocal Rank)**
- **mAP@50**

A retrieved image is considered **relevant** if it matches the query item on:
`category1`, `category2`, and `color`.


### Installation

Create the environment (shared by T2I and CIR):

```
conda env create -f "t2i&c2i_environment.yml"
conda activate IR
```
Set your OpenAI API key:

```
export OPENAI_API_KEY="YOUR_KEY"
```

Usage:

```
cd t2i
python text2image.py
```

## Image-to-Text Retrieval (I2T)

This module focuses on generating dense, attribute-rich captions for fashion images to enable accurate text-based retrieval. We propose an optimized pipeline using BLIP with prompt engineering and strict inference-time constraints.

Directory Structure

```
Image_to_Text_Task/
├── image_to_text.py        # Core generation script (BLIP + Optimization)
├── calculate_recall.py     # Evaluation script (CLIP-based Recall@K)
├── analyze_captions.py     # Statistical analysis of generated text
├── deepfashion_dataset.py  # Data loader for DeepFashion
└── results/                # Generated captions and logs
```

### Methodology

Standard image captioning often yields generic descriptions (e.g., "a white shirt"). To fix this, we applied:

#### 1. Prompt Engineering: Prefixed generation with "a detailed photography of a fashion garment, identifying the fabric, texture, neck style and color :".

#### 2. Inference Constraints:Beam Search: 
- Width = 5 (to explore diverse hypotheses).
- Repetition Penalty: $\lambda=1.5$ (to prevent generic loops).
- Min Length: 25 tokens (to force detailed descriptions).

### Usage

#### Step 1: Generate Captions
Generate descriptions for the DeepFashion test set.
```
cd Image_to_Text_Task
python image_to_text.py
```
_Output:_ results/deepfashion_captions_optimized.csv

#### Step 2: Evaluate (Recall@K)
Compute retrieval performance using a frozen CLIP (ViT-B/32) model as the evaluator.

```
python calculate_recall.py
```

#### Step 3: Analyze QualityAnalyze caption length, vocabulary, and repetition artifacts.

```
python analyze_captions.py
```

## Composed Image Retrieval (C2I)

Given a **reference image** and a **modification text** (e.g., *"change the color to red"*), this module retrieves images that match the **target appearance after modification**.

### Methods
- **Baseline-1 (SigLIP Fusion)**: encode reference image + modification text with SigLIP, then fuse
  `q = α·v_img + (1−α)·v_text` (default `α=0.5`).
- **Baseline-2 (BLIP + Concat, no LLM)**: BLIP captions the reference image, then concatenate  
  `"caption. modification."` and retrieve with SigLIP.
- **Ours (BLIP + LLM)**: BLIP caption → LLM rewrites into a **target-oriented query** → retrieve with SigLIP.  
  Rewrite styles: `detail / keywords / conversational`.  
  Length: `default (no constraint) / short / medium / long`.

### Evaluation Metrics

We report:
- **Recall@1 / Recall@5 / Recall@10**
- **Precision@1 / Precision@5 / Precision@10**
- **MRR (Mean Reciprocal Rank)**
- **mAP@50**

A retrieved image is considered **relevant** if it matches the query item on:
`category1`, `category2`, and `color`.


### Installation

Create the environment (shared by T2I and CIR):

```
conda env create -f "t2i&c2i_environment.yml"
conda activate IR
```
Set your OpenAI API key:

```
export OPENAI_API_KEY="YOUR_KEY"
```

Usage:

```
cd compose2i
python compose2i.py
```
