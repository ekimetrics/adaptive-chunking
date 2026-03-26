# Software Bill of Materials (SBOM)

**Package:** adaptive-chunking 0.1.0
**License:** MIT (SPDX: MIT)
**Copyright:** 2026 Ekimetrics
**Generated:** 2026-03-20

## Core dependencies

| Component | Version | SPDX License | Repository |
|-----------|---------|-------------|------------|
| tiktoken | >=0.9.0 | MIT | https://github.com/openai/tiktoken |
| pandas | >=2.2.3 | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| numpy | * | BSD-3-Clause | https://github.com/numpy/numpy |
| tqdm | >=4.67.1 | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm |
| python-dotenv | >=1.1.0 | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| sentence-transformers | >=3.1 | Apache-2.0 | https://github.com/UKPLab/sentence-transformers |
| spacy | >=3.8.4 | MIT | https://github.com/explosion/spaCy |
| scikit-learn | * | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| scipy | * | BSD-3-Clause | https://github.com/scipy/scipy |
| langdetect | * | MIT | https://github.com/Mimino666/langdetect |

## Optional: `[coref]` extra

| Component | Version | SPDX License | Repository |
|-----------|---------|-------------|------------|
| maverick-coref | * | CC-BY-NC-SA-4.0 | https://github.com/sapienzanlp/maverick-coref |

> **Warning:** CC BY-NC-SA 4.0 prohibits commercial use and requires
> derivative works to use the same license. This component is isolated
> behind an optional extra and is not installed by default.

## Optional: `[parsing]` extra

| Component | Version | SPDX License | Repository |
|-----------|---------|-------------|------------|
| docling | * | MIT | https://github.com/docling-project/docling |
| pymupdf4llm | * | AGPL-3.0-only OR LicenseRef-Artifex-Commercial | https://github.com/pymupdf/PyMuPDF |
| azure-ai-documentintelligence | * | MIT | https://github.com/Azure/azure-sdk-for-python |
| markdownify | * | MIT | https://github.com/matthewwithanm/python-markdownify |

> **Note:** pymupdf4llm is dual-licensed under AGPL-3.0 or a commercial
> Artifex license. Users who cannot comply with AGPL-3.0 terms should
> either obtain a commercial license or use `DoclingParser`/`AzureDIParser`.

## Optional: `[paper]` extra (includes `[parsing]` + `[coref]`)

| Component | Version | SPDX License | Repository |
|-----------|---------|-------------|------------|
| torch | ==2.6.0 | BSD-3-Clause | https://github.com/pytorch/pytorch |
| torchvision | ==0.21.0 | BSD-3-Clause | https://github.com/pytorch/vision |
| langchain | >=0.3.21 | MIT | https://github.com/langchain-ai/langchain |
| langchain-experimental | * | MIT | https://github.com/langchain-ai/langchain |
| stanza | >=1.10.1 | Apache-2.0 | https://github.com/stanfordnlp/stanza |
| nltk | * | Apache-2.0 | https://github.com/nltk/nltk |
| haystack-ai | * | Apache-2.0 | https://github.com/deepset-ai/haystack |
| openai | * | Apache-2.0 | https://github.com/openai/openai-python |
| deepeval | * | Apache-2.0 | https://github.com/confident-ai/deepeval |
| groq | >=0.21.0 | Apache-2.0 | https://github.com/groq/groq-python |
| pydantic | * | MIT | https://github.com/pydantic/pydantic |
| matplotlib | * | PSF-2.0 | https://github.com/matplotlib/matplotlib |
| seaborn | * | BSD-3-Clause | https://github.com/mwaskom/seaborn |
| ipywidgets | * | BSD-3-Clause | https://github.com/jupyter-widgets/ipywidgets |
| markdown | * | BSD-3-Clause | https://github.com/Python-Markdown/markdown |
| tabulate | * | MIT | https://github.com/astanin/python-tabulate |
| ipykernel | >=6.29.5 | BSD-3-Clause | https://github.com/ipython/ipykernel |
