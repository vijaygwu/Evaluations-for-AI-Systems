from setuptools import setup, find_packages

setup(
    name="evaluations-for-ai-systems",
    version="0.1.0",
    description="Companion code for Book 6: Evaluations for AI Systems",
    author="Vijay Raghavan",
    author_email="vijay@example.com",
    url="https://github.com/vijaygwu/Evaluations-for-AI-Systems",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "llm": ["anthropic>=0.18.0", "openai>=1.0.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
        "all": ["anthropic>=0.18.0", "openai>=1.0.0", "pytest>=7.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
