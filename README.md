# SpaHGC:Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference
## 🔍 Overview
While spatial transcriptomics (ST) has advanced our understanding of gene expression within tissue context, its high experimental cost limits large-scale application. Predicting ST gene expression from pathology images offers a promising, cost-effective alternative, yet existing methods often struggle to capture the complex spatial relationships across slides. 
To address the challenge, we propose SpaHGC, a multi-modal heterogeneous graph-based model that captures both intra-slice and inter-slice spot-spot relationships from histology images. 
It integrates local spatial context within the target slide and cross-slide similarities computed from image embeddings extracted by a pathology foundation model. 
These embeddings enable inter-slice knowledge transfer across slides. Additionally, SpaHGC incorporates Masked Graph Contrastive Learning to enhance feature representation and transfer spatial gene expression knowledge from reference to target slides, enabling it to model complex spatial dependencies and significantly improve prediction accuracy.
We conducted comprehensive benchmarking on seven histology-ST datasets from different platforms, tissues, and cancer subtypes. The results demonstrate that SpaHGC significantly outperforms existing nine state-of-the-art methods across all evaluation metrics. The model’s predicted gene expression profiles closely align with the ground truth data and accurately correspond to tumor regions. Furthermore, the predictions are significantly enriched in multiple cancer-related pathways, highlighting its strong biological relevance and application potential.

![Overview.png](Overview.png)




