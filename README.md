# Welcome to ToxiPep: Peptide toxicity prediction via fusion of context-aware representation and atomic-level graph
Peptide-based therapeutics have emerged as a promising avenue in drug development, offering high biocompatibility, specificity, and efficacy. However, the potential toxicity of peptides remains a significant challenge, necessitating the development of robust toxicity prediction methods. In this study, we introduce ToxiPep, a novel dual-model framework for peptide toxicity prediction that integrates sequence-based contextual information with atomic-level structural features. This framework combines BiGRU and Transformer to capture local and global sequence dependencies while leveraging multi-scale CNNs to extract refined structural features from molecular graphs derived from peptide SMILES representations. A cross-attention mechanism aligns and fuses these two feature modalities, enabling the model to capture intricate relationships between sequence and structural information. Evaluation results demonstrate the superior performance of ToxiPep, achieving an accuracy of 0.885 and an MCC score of 0.770 on independent test datasets, significantly outperforming existing methods. Additionally, interpretability analyses reveal that ToxiPep identifies key amino acids along with their structural features, providing insights into the molecular mechanisms of peptide toxicity. Overall, This framework has the potential to accelerate the identification of safer therapeutic peptides, offering new opportunities for peptide-based drug development in precision medicine.

This Peptide toxicity prediction tool developed by teams from the University of Hong Kong and the Chinese University of Hong Kong (Shenzhen)

![The workflow of this study](https://github.com/GGCL7/ToxiPep/blob/main/workflow.png)


# Dataset for this study
We provided our dataset and you can find them [Dataset](https://github.com/GGCL7/ToxiPep/tree/main/Dataset)
# Source code
We provide the source code and you can find them [Code](https://github.com/GGCL7/ToxiPep/tree/main/Code)
