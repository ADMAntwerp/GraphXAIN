# GraphXAIN: Narratives to Explain Graph Neural Networks

This repository contains the official implementation for the article **"GraphXAIN: Narratives to Explain Graph Neural Networks"**. Our method integrates Graph Neural Networks (GNNs), graph explainers, and Large Language Models (LLMs) to generate GraphXAINs — explainable AI (XAI) narratives that provide enhanced interpretability of GNN predictions.

# Workflow
![Workflow Diagram](images/XAIN_workflow.png)

![XAIN_57](images/XAIN_57.png)


## Abstract

Graph Neural Networks (GNNs) are a powerful technique for machine learning on graph-structured data, yet they pose interpretability challenges, especially for non-expert users. Existing GNN explanation methods often yield technical outputs such as subgraphs and feature importance scores, which are not easily understood. Building on recent insights from social science and other Explainable AI (XAI) methods, we propose *GraphXAIN*, a natural language narrative that explains individual predictions made by GNNs. We present a model-agnostic and explainer-agnostic XAI approach that complements graph explainers by generating GraphXAINs—coherent narratives explaining the GNN's prediction process—using Large Language Models (LLMs) and integrating graph data, individual predictions from GNNs, explanatory subgraphs, and feature importances. We define XAI Narratives and XAI Descriptions, highlighting their distinctions and emphasizing the importance of narrative principles in effective explanations. By incorporating natural language narratives, our approach supports graph practitioners and non-expert users, aligning with social science research on explainability and enhancing user understanding and trust in complex GNN models. We demonstrate GraphXAIN's capabilities on a real-world graph dataset, illustrating how its generated narratives can aid understanding compared to traditional graph explainer outputs or other descriptive explanation methods.

## Usage

To generate GraphXAINs for a given GNN model:

1. **Prepare Data**: Ensure that you have ready to use graph data or adjacency matrix with feature matrix ready for the input graph.
2. **Run the Graph Explainer**: Use ```notebooks/GraphXAIN_tutorial.ipynb``` notebook to extract subgraphs and feature importance values.
3. **Generate GraphXAINs**:  Use ```notebooks/GraphXAIN_tutorial.ipynb``` notebook to generate GraphXAINs based on the extracted data.


## Repository Structure

- `dataset/`: Contains sample datasets used in the paper.
- `explanations/`: Contains outputs from graph explainer.
- `images/`: Contains images used in publication 
- `notebooks/`: Jupyter notebook for generation GraphXAINs.
- `utils/`: Contains ```model.py``` with GNN model and ```utils.py``` with utility functions.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@mics{cedro2024graphxain,
  title={GraphXAIN: Narratives to Explain Graph Neural Networks},
  author={Mateusz Cedro and David Martens},
  year={2024},
  eprint={},
  primaryClass={},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/..}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or collaborations, feel free to contact:
- **Mateusz Cedro**: [mateusz.cedro@uantwerpen.be]
- **Affiliation**: University of Antwerp, Belgium

We appreciate any feedback or contributions to the project!
