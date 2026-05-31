from setuptools import setup, find_packages

setup(
    name="llm-forge",
    version="0.1.0",
    description="Full-stack LLM system: pretraining, fine-tuning, RAG, Agent, and Multimodal",
    author="LLM-Forge",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "transformers>=4.36.0",
        "tokenizers>=0.15.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "rag": ["faiss-gpu>=1.7.3", "sentence-transformers>=2.2.0", "PyPDF2>=3.0.0"],
        "eval": ["nltk>=3.8.0", "rouge-score>=0.1.2", "scikit-learn>=1.3.0"],
        "multimodal": ["Pillow>=10.0.0", "timm>=0.9.0"],
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "isort>=5.12.0"],
    },
)
