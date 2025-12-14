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

## Composed Image Retrieval (CIR)
