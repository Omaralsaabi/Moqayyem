# Moqayyem: Arabic RAG Evaluation Framework

Moqayyem (مُقَيِّم) is a comprehensive evaluation framework for Retrieval-Augmented Generation (RAG) systems focusing on Arabic language processing. It provides a robust set of metrics and tools to assess both retrieval and generation components while considering Arabic-specific language characteristics.

## Features

- **Comprehensive Evaluation Metrics**
  - Retrieval metrics (relevance, accuracy)
  - Generation metrics (BLEU, ROUGE, faithfulness)
  - Arabic-specific metrics (coherence, semantic similarity)
  - Robustness assessment
  - Performance monitoring

- **Arabic Language Support**
  - Specialized text preprocessing
  - Arabic-aware tokenization
  - Semantic similarity using Arabic language models
  - Support for Arabic text characteristics

- **Flexible Data Handling**
  - Support for multiple data formats (JSON, CSV, YAML)
  - Batch processing capabilities
  - Easy dataset manipulation

## Installation

```bash
# Clone the repository
git clone https://github.com/Omaralsaabi/Moqayyem.git
cd Moqayyem

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install poetry==1.8.4

poetry install
```

## Quick Start

```python
from src.config.config import EvaluationConfig
from src.evaluation.system_evaluator import RAGEvaluator
from src.data.dataset import RAGDataset

# Initialize evaluator
config = EvaluationConfig(language="arabic")
evaluator = RAGEvaluator(config)

# Load dataset
dataset = RAGDataset("path/to/your/data.json")

# Run evaluation
results = evaluator.evaluate_dataset(dataset)

# Save results
evaluator.save_results(results, "evaluation_results")
```

## Project Structure

```
moqayyem/
├── src/
│   ├── config/           # Configuration management
│   ├── data/            # Data handling and loading
│   ├── evaluation/      # Evaluation pipeline
│   ├── metrics/         # Evaluation metrics
│   └── reporting/       # Results reporting and visualization
├── tests/              # Test suite
├── examples/           # Usage examples
├── project.toml    
└── poetry.lock 
```

## Metrics Overview

### Retrieval Metrics
- Query Similarity
- Document Relevance
- Retrieval Accuracy

### Generation Metrics
- BLEU Score (with Arabic adaptations)
- ROUGE Scores (Arabic-aware)
- Semantic Similarity
- Faithfulness

### Arabic-Specific Metrics
- Text Coherence
- Average Word Length
- Sentence Structure Analysis
- Arabic Semantic Similarity

### Robustness Metrics
- Noise Robustness
- Negative Rejection
- Counterfactual Robustness

### Performance Metrics
- Retrieval Latency
- Generation Latency
- System Throughput

## Configuration

The framework can be configured using YAML files:

```yaml
language: arabic
metrics:
  retrieval_metrics:
    - precision
    - recall
    - mrr
  generation_metrics:
    - bleu
    - rouge
    - faithfulness
  arabic_metrics:
    - similarity
    - coherence

model:
  sentence_transformer: "Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet"
  batch_size: 32
  max_length: 512
```

## Data Format

Example JSON input format:
```json
{
  "query": "ما هو تعريف الذكاء الاصطناعي؟",
  "retrieved_docs": [
    "الذكاء الاصطناعي هو محاكاة للذكاء البشري",
    "يتضمن الذكاء الاصطناعي التعلم الآلي"
  ],
  "generated_response": "الذكاء الاصطناعي هو محاكاة لقدرات الإنسان الذهنية",
  "ground_truth": "الذكاء الاصطناعي هو محاكاة للذكاء البشري وقدراته"
}
```

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests/test_generation_metrics.py

# Run with verbosity
python -m unittest -v discover tests
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License - see the LICENSE file for details.

## Acknowledgments

- [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) for Arabic NLP utilities
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) for semantic similarity
- [Omartificial-Intelligence-Space](https://huggingface.co/Omartificial-Intelligence-Space)


## Citation

If you use this framework in your research, please cite:

```bibtex
@software{moqayyem2024,
  title = {Moqayyem: Arabic RAG Evaluation Framework},
  author = {Omar Alsaabi},
  year = {2024},
  url = {https://github.com/Omaralsaabi/Moqayyem.git}
}
```