## Discuss-RAG
Code for paper: [Talk Before You Retrieve: Agent-Led Discussions for Better RAG in Medical QA](https://arxiv.org/abs/2504.21252)

## Introduction
Medical question answering (QA) is a reasoning-intensive task that remains challenging for large language models (LLMs) due to hallucinations and outdated domain knowledge. Retrieval-Augmented Generation (RAG) provides a promising post-training solution by leveraging external knowledge. However, existing medical RAG systems suffer from two key limitations: (1) a lack of modeling for human-like reasoning behaviors during information retrieval, and (2) reliance on suboptimal medical corpora, which often results in the retrieval of irrelevant or noisy snippets. To overcome these challenges, we propose **Discuss-RAG**, a **plug-and-play module** designed to enhance the medical QA RAG system through collaborative agent-based reasoning. Our method introduces a summarizer agent that orchestrates a team of medical experts to emulate multi-turn brainstorming, thereby improving the relevance of retrieved content. Additionally, a decision-making agent evaluates the retrieved snippets before their final integration. Experimental results on four benchmark medical QA datasets show that Discuss-RAG consistently outperforms MedRAG, especially significantly improving answer accuracy by up to 16.67% on BioASQ and 12.20% on PubMedQA. 

## Requirement
1. Clone the repo
   ```sh
   git -r clone https://github.com/LLM-VLM-GSL/Discuss-RAG.git
   ```

2. Create a Python Environment and install the required libraries by running
   ```sh
   conda env create -f environment.yml
   conda activate DISCUSS-RAG
   ```

3. Download the medical QA benchmarks: [MMLU-Med](https://arxiv.org/abs/2009.03300), [MedQA-US](https://arxiv.org/abs/2009.13081), [BioASQ](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0564-6) and [PubMedQA](https://aclanthology.org/D19-1259/)
   
## Usage

Using our module is straightforward. The script **main_discuss_rag.py** contains all the necessary functions for performing medical QA as well as calculating accuracy.  For example, to evaluate on **MMLU-Med**, simply run the following commands to start inference and compute the accuracy:

```sh
cd Discuss-RAG
python main_discuss_rag.py
```

**Notification**: In our paper, we evaluate all benchmarks using only [Textbooks](https://arxiv.org/abs/2009.13081) as the corpus. For retrieval, we employ [MedCPT](https://huggingface.co/ncbi/MedCPT-Query-Encoder), and for the LLM, we conduct experiments exclusively with GPT-3.5 (i.e., *gpt-3.5-turbo-0125*).  

Additionally, all necessary prompts are included in `src/template_discuss.py`. For further implementation details, please refer to our manuscript.

## Acknowledgement  
Our work is built upon and inspired by [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG) and [MDAgents](https://github.com/mitmedialab/MDAgents). We sincerely thank the authors of these projects for their valuable contributions to the research community.  

## Cite Us
If you find this repository useful in your research, please cite our works:
```bibtex
@article{dong2025talk,
  title={Talk Before You Retrieve: Agent-Led Discussions for Better RAG in Medical QA},
  author={Dong, Xuanzhao and Zhu, Wenhui and Wang, Hao and Chen, Xiwen and Qiu, Peijie and Yin, Rui and Su, Yi and Wang, Yalin},
  journal={arXiv preprint arXiv:2504.21252},
  year={2025}
}
```




