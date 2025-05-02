# Leveraging NLP for Market Risk Assessment and Decision Support

## Overview

This project uses Natural Language Processing (NLP) and Large Language Models (LLMs) to automate market-entry risk analysis. By analyzing over 12,000 New York Times articles (2010–2020), the system identifies business risks across different geographies and industries. It includes both classification (risk level) and retrieval-augmented generation (RAG) for answering questions about market-entry risk.

- **Research Paper**: [research_paper.pdf](./research_paper.pdf)  
- **Implementation Notebook**: [code.ipynb](./code.ipynb)

## Key Features

- **Risk Classification**  
  Fine-tuned transformer models classify articles into risk levels (High, Medium, Low).

- **Question Answering (QA)**  
  A RAG pipeline retrieves relevant evidence and uses a language model to answer open-ended market-entry questions.

- **Custom Dataset**  
  Preprocessed and labeled data from the NYT archive (2010–2020), with 690 manually annotated articles used for training.

- **Multi-Model Evaluation**  
  Compared classification and QA models using task-specific metrics (F1, ROUGE, METEOR, etc.).

## Dataset

- **Source**: New York Times Business-related articles  
- **Timeframe**: 2010–2020  
- **Categories**: Economy, Technology, Global Business, Automobiles, Your Money, Real Estate  
- **Size**: ~12,000 articles  
- **Annotation**: 690 articles labeled with`risk_level` (High, Medium, Low)

## Tasks

### 1. Risk Classification

- **Models**: BERT, RoBERTa, DeBERTa-v3, FinBERT, T5
- **Best Model**: RoBERTa
- **Evaluation**:

| Model     | Accuracy | Precision | Recall | F1-score |
|-----------|----------|-----------|--------|----------|
| RoBERTa   | 0.75     | 0.754     | 0.75   | 0.747    |
| FinBERT   | 0.71     | 0.72      | 0.70   | 0.71     |
| BERT      | 0.623    | 0.656     | 0.623  | 0.612    |
| T5        | 0.56     | 0.58      | 0.56   | 0.55     |

### 2. Retrieval-Augmented Question Answering (RAG)

- **Retriever**: `multi-qa-mpnet-base-dot-v1` (FAISS index)  
- **QA Models Tested**: GPT-2 (baseline), Mistral-7B, LLaMA 3.1B  
- **Evaluation**:

| Model       | ROUGE-L | METEOR | Faithfulness (NLI) | Perplexity |
|-------------|---------|--------|---------------------|------------|
| Mistral-7B  | 0.51    | 0.38   | 0.79                | 22.7       |
| LLaMA 3.1B  | 0.47    | 0.35   | 0.83                | 25.4       |
| GPT-2       | 0.39    | 0.28   | 0.68                | 31.2       |

**Summary**: Although Mistral excelled slightly more in fluency, semantic relevance, and factual consistency, LLaMA 3.1 's superior precision and faithfulness make it the more suitable choice for a risk-assessment Q&A system, where factual accuracy and relevance are important. GPT-2 was weakest.

## Sample Q&A Outputs

**Q:** Which economic risks have been reported for electric vehicle companies entering the US market?

**A:** 

Risk Type: Economic

Risk Level: High/Medium

1) Dependence on Electricity Sources - The impact of electric vehicles (EVs) on reducing greenhouse gas emissions largely depends upon the region's electrical grid composition. In regions dominated by fossil fuels such as coal, EV charging might result in similar levels of CO₂ emission compared to conventional internal combustion engine vehicles. This uncertainty could negatively affect consumer perception towards EV brands and potentially limit growth opportunities.

## How to Run

This project is implemented in a single notebook:

### Run Locally

1. Clone the repository:

```bash
git clone https://github.com/yourusername/market-risk-nlp.git
cd market-risk-nlp
```

2. Install dependencies

3. Open code.ipynb and run all cells to:
- Load and preprocess the dataset
- Train and evaluate classification models
- Build and query the QA system
