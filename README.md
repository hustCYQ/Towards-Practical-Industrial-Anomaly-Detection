# **Towards Practical Industrial Anomaly Detection** [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

As the field of anomaly detection rapidly evolves, this repository is dedicated to collecting and organizing cutting-edge methods across both 2D and 3D domains. Our goal is to provide a clear and structured overview of the latest techniques that push the boundaries of unsupervised, semi-supervised, and zero-shot anomaly detection. 🚀🚀🚀

This repository is continuously ***maintained and updated***. We warmly welcome contributions from the community. if you have ***relevant papers, datasets, codebases, or insightful resources*** that are not yet included, feel free to open a pull request or raise an issue for discussion. 🙋🙋🙋

Let’s ***work together*** to build a comprehensive, high-quality resource for the anomaly detection research and engineering community! 💡💡💡





---
## 🏅 Recent Featured Papers
- A Survey on Vision-Language-Action Models: An Action Tokenization Perspective [[paper](https://arxiv.org/pdf/2507.01925)]
- A Survey on Vision-Language-Action Models for Autonomous Driving [[paper](https://arxiv.org/pdf/2506.24044)] [[project](https://github.com/JohnsonJiang1996/Awesome-VLA4AD)]  
- Parallels Between VLA Model Post-Training and Human Motor Learning: Progress, Challenges, and Trends [[paper](https://arxiv.org/pdf/2506.20966)] [[project](https://github.com/AoqunJin/Awesome-VLA-Post-Training)]  
- Foundation Models in Robotics: Applications, Challenges, and the Future [[paper](https://arxiv.org/pdf/2312.07843)] [[project](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)]

---
## 🔗 Quick Navigation
- [Our-Publications](#Our-Publications) | [Survey](#Survey) | [BenchMark](#BenchMark)  | [NeurIPS-2025](#NeurIPS-2025) | [ICCV-2025](#ICCV-2025) | [IJCAI-2025](#IJCAI-2025)  | [ICML-2025](#ICML-2025) | [CVPR-2025](#CVPR-2025)  | [ICLR-2025](#ICLR-2025) | [AAAI-2025](#AAAI-2025) | [Journal-2025](#Journal-2025) | [Summary-before-2025](#Summary-before-2025)

---




## Survey

<details open>
<summary>📚 Show/Hide Survey Papers</summary>

- A Survey on Vision-Language-Action Models for Embodied AI [[paper](https://arxiv.org/abs/2405.14093)]
- A Survey of Embodied Learning for Object-Centric Robotic Manipulation [[paper](https://arxiv.org/pdf/2408.11537)]
- Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI [[paper](https://arxiv.org/pdf/2407.06886)]
- Vision-language navigation: a survey and taxonomy [[paper](https://arxiv.org/pdf/2108.11544)]

</details>

---


## BenchMark

|  Name  |   Venue  |   Date   |   Source   |   Modality   |
|:--------:|:----------:|:----------:|:------------:|:--------------:|
| MVTec | IJCV | 2021 | The MVTec anomaly detection dataset: A comprehensive real-world dataset for unsupervised anomaly detection [[paper](https://www.researchgate.net/publication/348278659_The_MVTec_Anomaly_Detection_Dataset_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_Detection	)]  [[code](https://github.com/liujiyuan13/MVTecAD	)] [[download](https://www.mvtec.com/company/research/datasets/mvtec-ad)]| Image |
| BTAD | ISIE | 2021 | A vision transformer network for image anomaly detection and localization [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9576231)]  [[code](https://github.com/pankajmishra000/VT-ADL)] [[download](https://datasetninja.com/btad)]| Image |
| VisA | ECCV | 2022 | Spot-thedifference self-supervised pre-training for anomaly detection and segmentation [[paper](https://link.springer.com/chapter/10.1007/978-3-031-20056-4_23)]  [[code](https://github.com/amazon-science/spot-diff)] [[download](https://github.com/amazon-science/spot-diff#data-download)] | Image |
| MVTec LOCO | IJCV | 2022 | Beyond dents and scratches: Logical constraints in unsupervised anomaly detection and localization [[paper](https://link.springer.com/article/10.1007/s11263-022-01578-9)]  [[code](https://github.com/areylng/SPADE-MVTEC-LOCO-AD-dataset)] [[download](https://www.mvtec.com/company/research/datasets/mvtec-loco)] | Image |
| PAD | NeurIPS | 2023 | Pad: A dataset and benchmark for pose-agnostic anomaly detection [[paper](https://papers.neurips.cc/paper_files/paper/2023/file/8bc5aef775aacc1650a9790f1428bcea-Paper-Datasets_and_Benchmarks.pdf)]  [[code](https://github.com/EricLee0224/PAD)] [[download](https://drive.google.com/file/d/1XlW5v_PCXMH49RSKICkskKjd2A5YDMQq/view)] | Image |
| MSC-AD | TII | 2024 | Msc-ad: A multiscene unsupervised anomaly detection dataset for small defect detection of casting surface [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10376373)]  [[code](https://msc-ad.github.io/MSC-AD2023/)] [[download](https://msc-ad.github.io/MSC-AD2023/)] | Image |
| Real-IAD | CVPR | 2024 | Real-iad: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Real-IAD_D3_A_Real-World_2DPseudo-3D3D_Dataset_for_Industrial_Anomaly_Detection_CVPR_2025_paper.pdf)]  [[code](https://github.com/Tencent/AnomalyDetection_Real-IAD)] [[download](https://realiad4ad.github.io/Real-IAD/)] | Image |
| VISION | Arxiv | 2024 | Vision datasets: A benchmark for vision-based industrial inspection [[paper](https://arxiv.org/pdf/2306.07890.pdf)]  [[code](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets)] [[download](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets)] | Image |
| WFDD | ECCV | 2024 | A unified anomaly synthesis strategy with gradient ascent for industrial anomaly detection and localization [[paper](https://link.springer.com/chapter/10.1007/978-3-031-72855-6_3)]  [[code](https://github.com/cqylunlun/GLASS)] [[download](https://drive.google.com/file/d/1P8yfNnfoFsb0Lv-HRzkPQ2nD9qsL--Vk/view)] | Image |
| VAD | CVPR | 2024 | Supervised anomaly detection for complex industrial images [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Baitieva_Supervised_Anomaly_Detection_for_Complex_Industrial_Images_CVPR_2024_paper.pdf)]  [[code](https://github.com/abc-125/vad)] [[download](https://drive.google.com/file/d/1sWNZCUnb2u4hcr6ak7gjpwsm6jtx6IF1/view)] | Image |
| RAD | Arxiv | 2024 | Rad: A dataset and benchmark for real-life anomaly detection with robotic observations [[paper](https://arxiv.org/pdf/2410.00713)]  [[code](https://github.com/kaichen-z/rad)] [[download](https://github.com/kaichen-z/rad)] | Image |
| 3CAD | AAAI | 2025 | 3cad: A large-scale real-world 3c product dataset for unsupervised anomaly detection [[paper](https://dl.acm.org/doi/10.1609/aaai.v39i9.32993)]  [[code](https://github.com/EnquanYang2022/3CAD)] [[download](https://github.com/EnquanYang2022/3CAD?tab=readme-ov-file#-our-benchmark)] | Image |
| MVTec2 | Arxiv | 2025 | Steger, The mvtec ad 2 dataset: Advanced scenarios for unsupervised anomaly detection [[paper](https://arxiv.org/abs/2503.21622)]  [[code](无可直接获取的强相关repo/code)] [[download](https://www.mvtec.com/company/research/datasets/mvtec-ad-2)] | Image |
| M2AD | Arxiv | 2025 | Visual anomaly detection under complex view-illumination interplay: A large-scale benchmark [[paper](https://arxiv.org/abs/2505.10996)]  [[code](https://github.com/hustCYQ/M2AD/)] [[download](https://huggingface.co/datasets/ChengYuQi99/M2AD)] | Image |
| MANTA | CVPR | 2025 | Manta: A large-scale multi-view and visual-text anomaly detection dataset for tiny objects [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zatsarynna_MANTA_Diffusion_Mamba_for_Efficient_and_Effective_Stochastic_Long-Term_Dense_CVPR_2025_paper.pdf)]  [[code](https://github.com/olga-zats/DIFF_MANTA?tab=readme-ov-file)] [[download](https://github.com/olga-zats/DIFF_MANTA?tab=readme-ov-file#datasets)] | Image&Text |
| Real3D-AD | NeurIPS | 2023 | Real3d-ad: A dataset of point cloud anomaly detection [[paper](https://arxiv.org/abs/2309.13226)]  [[code](https://github.com/M-3LAB/Real3D-AD?tab=readme-ov-file)] [[download](https://github.com/M-3LAB/Real3D-AD?tab=readme-ov-file#download)] | PC |
| Anomaly-ShapeNet | CVPR | 2024 | Towards scalable 3d anomaly detection and localization: A benchmark via 3d anomaly synthesis and a self-supervised learning network [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Towards_Scalable_3D_Anomaly_Detection_and_Localization_A_Benchmark_via_CVPR_2024_paper.pdf)]  [[code](https://github.com/Chopper-233/Anomaly-ShapeNet)] [[download](https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet)] | PC |
| MiniShift | Arxiv | 2025 | A scalable dataset and realtime framework for subtle industrial defects [[paper](https://arxiv.org/abs/2507.07435)]  [[code](https://github.com/hustCYQ/MiniShift-Simple3D)] [[download](https://huggingface.co/datasets/ChengYuQi99/MiniShift)] | PC |
| MVTec 3D-AD | VISAPP | 2021 | Towards highresolution 3d anomaly detection: A scalable dataset and realtime framework for subtle industrial defects [[paper](https://arxiv.org/abs/2112.09045)]  [[code](无可直接获取的强相关repo/code)] [[download](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)] | Image&PC |
| Eyecandies | ACCV | 2022 | The eyecandies dataset for unsupervised multimodal anomaly detection and localization [[paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Bonfiglioli_The_Eyecandies_Dataset_for_Unsupervised_Multimodal_Anomaly_Detection_and_Localization_ACCV_2022_paper.pdf)]  [[code](https://github.com/eyecan-ai/eyecandies)] [[download](https://github.com/eyecan-ai/eyecandies?tab=readme-ov-file#download-the-dataset)] | Image&PC |
| Real-IADD3 | CVPR | 2025 | Real-iad d3: A real-world 2d/pseudo-3d/3d dataset for industrial anomaly detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Real-IAD_D3_A_Real-World_2DPseudo-3D3D_Dataset_for_Industrial_Anomaly_Detection_CVPR_2025_paper.pdf)]  [[code](https://github.com/Tencent/AnomalyDetection_Real-IAD)] [[download](https://huggingface.co/datasets/Real-IAD/Real-IAD)] | Image&PC |
| MulSenAD | CVPR | 2025 | Multi-sensor object anomaly detection: Unifying appearance, geometry, and internal properties [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Multi-Sensor_Object_Anomaly_Detection_Unifying_Appearance_Geometry_and_Internal_Properties_CVPR_2025_paper.pdf)]  [[code](https://github.com/ZZZBBBZZZ/MulSen-AD)] [[download](https://huggingface.co/datasets/orgjy314159/MulSen_AD)] | Image&PC&IR |





---




## **Our-Publications**
<details open>
<summary>📚 Show/Hide</summary>


 - [Survey-Arxiv'25] A comprehensive survey for real-world industrial defect detection: Challenges, approaches, and prospects [[Paper](https://arxiv.org/abs/2507.13378)] [[Code](https://github.com/hustCYQ/Towards-Practical-Industrial-Anomaly-Detection)]
 - [2D-CVPR'25] Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection [[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Luo_Exploring_Intrinsic_Normal_Prototypes_within_a_Single_Image_for_Universal_CVPR_2025_paper.html)] [[Code](https://github.com/luow23/INP-Former)]
 - [2D-Arxiv'25] Visual Anomaly Detection under Complex View-Illumination Interplay: A Large-Scale Benchmark [[Paper](https://arxiv.org/abs/2505.10996)] [[Code](https://github.com/hustCYQ/M2AD)]
 - [2D-TCYB'25] Personalizing Vision-Language Models with Hybrid Prompts for Zero-Shot Anomaly Detection [[Paper](https://ieeexplore.ieee.org/abstract/document/10884560)] [[Code](https://github.com/caoyunkang/Segment-Any-Anomaly)]
 - [2D-ECCV'24] AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-72761-0_4)] [[Code](https://github.com/caoyunkang/AdaCLIP)]
 - [3D-Arxiv'25] Towards high-resolution 3d anomaly detection: A scalable dataset and real-time framework for subtle industrial defects [[Paper](https://arxiv.org/abs/2507.07435)] [[Code](https://github.com/hustCYQ/MiniShift-Simple3D)]
 - [3D-TASE'25] Boosting global-local feature matching via anomaly synthesis for multi-class point cloud anomaly detection [[Paper](https://ieeexplore.ieee.org/abstract/document/10898004)] [[Code](https://github.com/hustCYQ/GLFM-Multi-class-3DAD)]
 - [3D-Arxiv'24] Towards zero-shot point cloud anomaly detection: A multi-view projection framework [[Paper](https://arxiv.org/abs/2409.13162)] [[Code](https://github.com/hustCYQ/MVP-PCLIP)]
 - [3D-PR'24] Complementary pseudo multimodal feature for point cloud anomaly detection [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320324005120)] [[Code](https://github.com/caoyunkang/CPMF)]


</details>

---






## NeurIPS-2025

<details open>
<summary>📚 Show/Hide NeurIPS-2025 Papers</summary>

The official list of accepted papers will be updated here thereafter.

</details>

---



## ICCV-2025

<details open>
<summary>📚 Show/Hide ICCV-2025 Papers</summary>

- Towards Real Unsupervised Anomaly Detection Via Confident Meta-Learning [[paper](https://arxiv.org/abs/2508.02293)]
- SALAD -- Semantics-Aware Logical Anomaly Detection [[paper](https://arxiv.org/abs/2509.02101)]  [[code](https://github.com/MaticFuc/SALAD)]
- Salvaging the Overlooked: Leveraging Class-Aware Contrastive Learning for Multi–Class Anomaly Detection [[paper](https://arxiv.org/abs/2412.04769)]  [[code](https://github.com/LGC-AD/AD-LGC)]
- Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts [[paper](https://arxiv.org/abs/2507.16946)]
- Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation [[paper](https://arxiv.org/abs/2505.24431)]  [[code](https://github.com/ZZZBBBZZZ/PASDF)]
- DecAD: Decoupling Anomalies in Latent Space for Multi–Class Unsupervised Anomaly Detection [[code](https://github.com/wxl1122/DecAD)]
- RareCLIP: Rarity-aware Online Zero–shot Industrial Anomaly Detection [[code](https://github.com/hjf02/RareCLIP)]
- FIND: Few–Shot Anomaly Inspection with Normal-Only Multi–Modal Data
- FE-CLIP: Frequency Enhanced CLIP Model for Zero–Shot Anomaly Detection and Segmentation
- SeaS: Few–shot Industrial Anomaly Image Generation with Separation and Sharing Fine-tuning [[paper](https://arxiv.org/pdf/2410.14987)]  [[code](https://github.com/HUST-SLOW/SeaS)]
- DictAS: A Framework for Class-Generalizable Few–Shot Anomaly Segmentation via Dictionary Lookup [[paper](https://www.arxiv.org/abs/2508.13560)]  [[code](https://github.com/xiaozhen228/DictAS)]
- MultiADS: Defect-aware Supervision for Multi-type Anomaly Detection and Segmentation in Zero–Shot Learning [[paper](https://arxiv.org/abs/2504.06740)]  [[code](https://github.com/boschresearch/MultiADS)]
- Fine-grained Abnormality Prompt Learning for Zero–shot Anomaly Detection [[paper](https://arxiv.org/pdf/2410.10289)]  [[code](https://github.com/mala-lab/faprompt)]
- Anomaly Detection of Integrated Circuits Package Substrates Using the Large Vision Model SAIC: Dataset Construction, Methodology, and Application [[code](https://github.com/Bingyang0410/CPS2D-AD)]
- ReMP-AD: Retrieval-enhanced Multi-modal Prompt Fusion for Few–Shot Industrial Visual Anomaly Detection [[code](https://github.com/cshcma/ReMP-AD)]
- G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection [[paper](https://arxiv.org/abs/2503.10091)]  [[code](https://github.com/ctaoaa/G2SF)]
- SiM3D: Single-instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark [[paper](https://arxiv.org/abs/2506.21549)]
- Triad: Empowering LMM-based Anomaly Detection with Expert-guided Region-of-Interest Tokenizer and Manufacturing Process [[paper](https://arxiv.org/pdf/2503.13184v2)]  [[code](https://github.com/tzjtatata/Triad/)]
- DFM: Differentiable Feature Matching for Anomaly Detection [[code](https://github.com/wyattxuanyang/DFM)]
- Advancing Generalizable Tumor Segmentation with Anomaly-Aware Open-Vocabulary Attention Maps and Frozen Foundation Diffusion Models [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_Advancing_Generalizable_Tumor_Segmentation_with_Anomaly-Aware_Open-Vocabulary_Attention_Maps_and_CVPR_2025_paper.pdf)]  [[code](https://github.com/Yankai96/DiffuGTS)]

</details>

---

## IJCAI-2025

<details open>
<summary>📚 Show/Hide IJCAI-2025 Papers</summary>

- Rethinking Contrastive Learning in Graph Anomaly Detection: A Clean-View Perspective [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/80.pdf)]
- INFP: INdustrial Video Anomaly Detection via Frequency Prioritization [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/113.pdf)]  [[code](https://github.com/YuQianzi/INFP)]
- Template3D-AD: Point Cloud Template Matching Method Based on Center Points for 3D Anomaly Detection [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/240.pdf)]
- Free Lunch of Image-mask Alignment for Anomaly Image Generation and Segmentation [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/641.pdf)]
- HyperTrans: Efficient Hypergraph-Driven Cross-Domain Pattern Transfer in Image Anomaly Detection [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/1032.pdf)]  [[code](https://github.com/raRn0y/HyperTrans)]
- DriftRemover: Hybrid Energy Optimizations for Anomaly Images Synthesis and Segmentation [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/1044.pdf)]  [[code](https://github.com/JJessicaYao/DriftRemover)]
- Underground Diagnosis in 3D GPR Data by Learning in CuCoRes Model Space [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/1175.pdf)]
- Exploiting Self-Refining Normal Graph Structures for Robust Defense against Unsupervised Adversarial Attacks [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/2292.pdf)]
- Mitigating Message Imbalance in Fraud Detection with Dual-View Graph Representation Learning [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/2337.pdf)]
- Prototype-based Optimal Transport for Out-of-Distribution Detection [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/2661.pdf)]
- FedDLAD: A Federated Learning Dual-Layer Anomaly Detection Framework for Enhancing Resilience Against Backdoor Attacks [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/3425.pdf)]
- Dual Encoder Contrastive Learning with Augmented Views for Graph Anomaly Detection [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/3883.pdf)]
- GCTAM: Global and Contextual Truncated Affinity Combined Maximization Model For Unsupervised Graph Anomaly Detection [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/4461.pdf)]  [[code](https://github.com/kgccc/GCTAM)]
- NAAST-GNN: Neighborhood Adaptive Aggregation and Spectral Tuning for Graph Anomaly Detection [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/4545.pdf)]
- ABNet: Mitigating Sample Imbalance in Anomaly Detection Within Dynamic Graphs [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/5170.pdf)]
- ReplayCAD: Generative Diffusion Replay for Continual Anomaly Detection [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/5317.pdf)]  [[code](https://github.com/HULEI7/ReplayCAD)]
- MC3D-AD: A Unified Geometry-aware Reconstruction Model for Multi–category 3D Anomaly Detection [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/5901.pdf)]  [[code](https://github.com/jiayi-art/MC3D-AD)]
- Omni-Dimensional State Space Model-driven SAM for Pixel-level Anomaly Detection [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/6311.pdf)]
- A Novel Sparse Active Online Learning Framework for Fast and Accurate Streaming Anomaly Detection Over Data Streams [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/6819.pdf)]
- RTdetector: Deep Transformer Networks for Time Series Anomaly Detection Based on Reconstruction Trend [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/7398.pdf)]
- MEGAD: A Memory-Efficient Framework for Large-Scale Attributed Graph Anomaly Detection [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/7578.pdf)]
- Towards VLM-based Hybrid Explainable Prompt Enhancement for Zero–Shot Industrial Anomaly Detection [[paper](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/8054.pdf)]

</details>

---

## ICML-2025

<details open>
<summary>📚 Show/Hide ICML-2025 Papers</summary>

- CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering [[paper](https://arxiv.org/abs/2505.01476)]  [[code](https://github.com/ZHE-SAPI/CostFilter-AD)]
- Demeaned Sparse: Efficient Anomaly Detection by Residual Estimate [[paper](https://openreview.net/forum?id=F06FPb0Mtu)]
- OmiAD: One-Step Adaptive Masked Diffusion Model for Multi–class Anomaly Detection via Adversarial Distillation [[paper](https://openreview.net/forum?id=859NdHQv0Z)]
- An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination [[paper](https://openreview.net/forum?id=FT5eOdWSWN)]
- Demeaned Sparse: Efficient Anomaly Detection by Residual Estimate [[paper](https://openreview.net/pdf?id=F06FPb0Mtu)]
  
</details>

---


## CVPR-2025

<details open>
<summary>📚 Show/Hide CVPR-2025 Papers</summary>

- DFM: Differentiable Feature Matching for Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_DFM_Differentiable_Feature_Matching_for_Anomaly_Detection_CVPR_2025_paper.pdf)] [[code](https://github.com/wyattxuanyang/DFM)]  [![Info](https://img.shields.io/badge/Model-unsupervised_+_memorybank_+_MVTec–AD_and_VisA-blue)](#)
- Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Jin_Dual-Interrelated_Diffusion_Model_for_Few-Shot_Anomaly_Image_Generation_CVPR_2025_paper.pdf)] [[code](https://github.com/yinyjin/DualAnoDiff)] [![Info](https://img.shields.io/badge/Model-few–shot_+_diffusion_+_MVTec-blue)](#)
- AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_AnomalyNCD_Towards_Novel_Anomaly_Class_Discovery_in_Industrial_Scenarios_CVPR_2025_paper.pdf)] [[code](https://github.com/HUST-SLOW/AnomalyNCD)]  [![Info](https://img.shields.io/badge/Model-self_supervised_+_MVTec–AD_and_MTD-blue)](#)
- PIAD: Pose and Illumination agnostic Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_PIAD_Pose_and_Illumination_agnostic_Anomaly_Detection_CVPR_2025_paper.pdf)] [[code](https://github.com/Kaichen-Yang/piad_baseline)]  [![Info](https://img.shields.io/badge/Model-3DGS_representation_+_MAD_and_SYNT-blue)](#)
- Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Distribution_Prototype_Diffusion_Learning_for_Open-set_Supervised_Anomaly_Detection_CVPR_2025_paper.pdf)] [[code](https://github.com/fuyunwang/DPDL)]  [![Info](https://img.shields.io/badge/Model-supervised_+_diffusion_+_MVTec–AD_Optical_SDD_AITEX_ELPV_Mastcam-blue)](#)
- MANTA: A Large-Scale Multi-View and Visual-Text Anomaly Detection Dataset for Tiny Objects [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Fan_MANTA_A_Large-Scale_Multi-View_and_Visual-Text_Anomaly_Detection_Dataset_for_CVPR_2025_paper.pdf)] [[code](https://grainnet.github.io/MANTA)]  [![Info](https://img.shields.io/badge/Model-multiview_dataset-blue)](#)
- Spotting the Unexpected (STU): A 3D LiDAR Dataset for Anomaly Segmentation in Autonomous Driving [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Nekrasov_Spotting_the_Unexpected_STU_A_3D_LiDAR_Dataset_for_Anomaly_CVPR_2025_paper.pdf)] [[code](https://github.com/kumuji/stu_dataset)]  [![Info](https://img.shields.io/badge/Model-anomaly_segmentation_dataset-blue)](#)
- UniNet: A Contrastive Learning-guided Unified Framework with Feature Selection for Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wei_UniNet_A_Contrastive_Learning-guided_Unified_Framework_with_Feature_Selection_for_CVPR_2025_paper.pdf)] [[code](https://github.com/pangdaTangtt/UniNet)]  [![Info](https://img.shields.io/badge/Model-unsupervised_and_supervised_+_student–teacher_models_+_MVTec–AD_BTAD_MVTec–3D_VAD-blue)](#)
- Real-IAD D3: A Real-World 2D/Pseudo-3D Dataset for Industrial Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Real-IAD_D3_A_Real-World_2DPseudo-3D3D_Dataset_for_Industrial_Anomaly_Detection_CVPR_2025_paper.pdf)] [[code](https://github.com/Real-IAD-D3/main)]  [![Info](https://img.shields.io/badge/Model-unsupervised_+_Real–IAD_D³-blue)](#)
- Dinomaly: The Less is More Philosophy in Multi-Class Unsupervised Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.pdf)] [[code](https://github.com/guojiajeremy/Dinomaly)]  [![Info](https://img.shields.io/badge/Model-diffusion_+_MVTec–AD_and_Visa-blue)](#)
- Unseen Visual Anomaly Generation [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Unseen_Visual_Anomaly_Generation_CVPR_2025_paper.pdf)] [[code](https://github.com/EPFL-IMOS/AnomalyAny)]  [![Info](https://img.shields.io/badge/Model-unsupervised_+_memorybank_+_MVTec–AD_and_VisA-blue)](#)
- Odd-One-Out: Anomaly Detection by Comparing with Neighbors [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Bhunia_Odd-One-Out_Anomaly_Detection_by_Comparing_with_Neighbors_CVPR_2025_paper.pdf)]  [[code](https://github.com/VICO-UoE/OddOneOutAD)]
- Towards Zero–Shot Anomaly Detection and Reasoning with Multimodal Large Language Models [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Towards_Zero-Shot_Anomaly_Detection_and_Reasoning_with_Multimodal_Large_Language_CVPR_2025_paper.pdf)]  [[code](https://github.com/honda-research-institute/Anomaly-OneVision)]
- PO3AD: Predicting Point Offsets toward Better 3D Point Cloud Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_PO3AD_Predicting_Point_Offsets_toward_Better_3D_Point_Cloud_Anomaly_CVPR_2025_paper.pdf)]  [[code](https://github.com/yjnanan/PO3AD)]
- UniVAD: A Training-free Unified Model for Few–shot Visual Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.pdf)]  [[code](https://github.com/FantasticGNU/UniVAD)]
- Beyond Single-Modal Boundary: Cross-Modal Anomaly Detection through Visual Prototype and Harmonization [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Mao_Beyond_Single-Modal_Boundary_Cross-Modal_Anomaly_Detection_through_Visual_Prototype_and_CVPR_2025_paper.pdf)]
- Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_Exploring_Intrinsic_Normal_Prototypes_within_a_Single_Image_for_Universal_CVPR_2025_paper.pdf)]  [[code](https://github.com/luow23/INP-Former)]
- Track Any Anomalous Object: A Granular Video Anomaly Detection Pipeline [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Track_Any_Anomalous_ObjectA_Granular_Video_Anomaly_Detection_Pipeline_CVPR_2025_paper.pdf)]  [[code](https://github.com/TAO-25/tao-25.github.io)]
- PatchGuard: Adversarially Robust Anomaly Detection and Localization through Vision Transformers and Pseudo Anomalies [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Nafez_PatchGuard_Adversarially_Robust_Anomaly_Detection_and_Localization_through_Vision_Transformers_CVPR_2025_paper.pdf)]  [[code](https://github.com/rohban-lab/PatchGuard)]
- Multi–Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Multi-Sensor_Object_Anomaly_Detection_Unifying_Appearance_Geometry_and_Internal_Properties_CVPR_2025_paper.pdf)]  [[code](https://github.com/ZZZBBBZZZ/MulSen-AD/)]
- Towards Visual Discrimination and Reasoning of Real-World Physical Dynamics: Physics-Grounded Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Towards_Visual_Discrimination_and_Reasoning_of_Real-World_Physical_Dynamics_Physics-Grounded_CVPR_2025_paper.pdf)]  [[code](https://github.com/Chopper-233/Physics-AD)]
- One-for-More: Continual Diffusion Model for Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_One-for-More_Continual_Diffusion_Model_for_Anomaly_Detection_CVPR_2025_paper.pdf)]  [[code](https://github.com/FuNz-0/One-for-More)]
- Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Holmes-VAU_Towards_Long-term_Video_Anomaly_Understanding_at_Any_Granularity_CVPR_2025_paper.pdf)]  [[code](https://github.com/pipixin321/HolmesVAU)]
- Towards Training-free Anomaly Detection with Vision and Language Foundation Models [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Towards_Training-free_Anomaly_Detection_with_Vision_and_Language_Foundation_Models_CVPR_2025_paper.pdf)]  [[code](https://github.com/zhang0jhon/LogSAD)]
- Bayesian Prompt Flow Learning for Zero–Shot Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Qu_Bayesian_Prompt_Flow_Learning_for_Zero-Shot_Anomaly_Detection_CVPR_2025_paper.pdf)]  [[code](https://github.com/xiaozhen228/Bayes-PFL)]
- Correcting Deviations from Normality: A Reformulated Diffusion Model for Multi–Class Unsupervised Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Beizaee_Correcting_Deviations_from_Normality_A_Reformulated_Diffusion_Model_for_Multi-Class_CVPR_2025_paper.pdf)]  [[code](https://github.com/farzad-bz/DeCo-Diff)]
- TailedCore: Few–Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Jung_TailedCore_Few-Shot_Sampling_for_Unsupervised_Long-Tail_Noisy_Anomaly_Detection_CVPR_2025_paper.pdf)]  [[code](https://github.com/jungyg/TailedCore)]
- A Unified Latent Schrödinger Bridge Diffusion Model for Unsupervised Anomaly Detection and Localization1 [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Akshay_A_Unified_Latent_Schrodinger_Bridge_Diffusion_Model_for_Unsupervised_Anomaly_CVPR_2025_paper.pdf)]  [[code](https://github.com/ShilhoraAkshayPatel/LASB)]
- VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language-Models [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.pdf)]  [[code](https://github.com/vera-framework/VERA)]
- Just Dance with pi!: A Poly-modal Inductor for Weakly-supervised Video Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Majhi_Just_Dance_with_pi_A_Poly-modal_Inductor_for_Weakly-supervised_Video_CVPR_2025_paper.pdf)]
- Noise-Resistant Video Anomaly Detection via RGB Error-Guided Multiscale Predictive Coding and Dynamic Memory [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Noise-Resistant_Video_Anomaly_Detection_via_RGB_Error-Guided_Multiscale_Predictive_Coding_CVPR_2025_paper.pdf)]
- AA-CLIP: Enhancing Zero–Shot Anomaly Detection via Anomaly-Aware CLIP [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_AA-CLIP_Enhancing_Zero-Shot_Anomaly_Detection_via_Anomaly-Aware_CLIP_CVPR_2025_paper.pdf)]  [[code](https://github.com/Mwxinnn/AA-CLIP)]

</details>

---

## ICLR-2025

<details open>
<summary>📚 Show/Hide ICLR-2025 Papers</summary>

- One-for-All Few–Shot Anomaly Detection via Instance-Induced Prompt Learning [[paper](https://openreview.net/pdf?id=Zzs3JwknAY)]
- Language-Assisted Feature Transformation for Anomaly Detection [[paper](https://openreview.net/pdf?id=2p03KljxE9)]
- MMAD: A Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection [[paper](https://openreview.net/pdf?id=JDiER86r8v)]
- Adversarially Robust Anomaly Detection through Spurious Negative Pair Mitigation [[paper](https://openreview.net/pdf?id=t8fu5m8R5m)]

</details>

---

## AAAI-2025

<details open>
<summary>📚 Show/Hide AAAI-2025 Papers</summary>

- FiCo: Filter or Compensate: Towards Invariant Representation from Distribution Shift for Anomaly Detection [[paper](https://arxiv.org/abs/2412.10115)]  [[code](https://github.com/znchen666/FiCo)]
- Unlocking the Potential of Reverse Distillation for Anomaly Detection [[paper](https://arxiv.org/abs/2412.07579)]  [[code](https://github.com/hito2448/URD)]
- CNC: Cross-modal Normality Constraint for Unsupervised Multi–class Anomaly Detection [[paper](https://arxiv.org/abs/2501.00346)]  [[code](https://github.com/cvddl/CNC)]
- CKAAD: Boosting Fine-Grained Visual Anomaly Detection with Coarse-Knowledge-Aware Adversarial Learning [[paper](https://arxiv.org/abs/2412.12850)]  [[code](https://github.com/Faustinaqq/CKAAD)]
- Look Inside for More: Internal Spatial Modality Perception for 3D Anomaly Detection [[paper](https://arxiv.org/abs/2412.13461)]  [[code](https://github.com/M-3LAB/Look-Inside-for-More)]
- MVREC: A General Few–shot Defect Classification Model Using Multi-View Region-Context [[paper](https://arxiv.org/abs/2412.16897)]
- Revisiting Multimodal Fusion for 3D Anomaly Detection from an Architectural Perspective [[paper](https://arxiv.org/abs/2412.17297)]
- KAG-prompt: Kernel-Aware Graph Prompt Learning for Few–Shot Anomaly Detection [[paper](https://arxiv.org/abs/2412.17619)]  [[code](https://github.com/CVL-hub/KAG-prompt)]
- Promptable Anomaly Segmentation with SAM Through Self-Perception Tuning [[paper](https://arxiv.org/abs/2411.17217)]  [[code](https://github.com/THU-MIG/SAM-SPT)]
- 3CAD: A Large-Scale Real-World 3C Product Dataset for Unsupervised Anomaly [[paper](https://arxiv.org/abs/2502.05761)]  [[code](https://github.com/EnquanYang2022/3CAD)]

</details>

---




## Journal-2025

<details open>
<summary>📚 Show/Hide Journal-2025 Papers</summary>

- DiffusionAD: Norm-Guided One-Step Denoising Diffusion for Anomaly Detection [[TPAMI](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11005495)]  [[code](https://github.com/HuiZhang0812/DiffusionAD)]
- Personalizing Vision-Language Models With Hybrid Prompts for Zero–Shot Anomaly Detection [[TCYB](https://ieeexplore.ieee.org/document/10884560)]
- Global-Regularized Neighborhood Regression for Efficient Zero–Shot Texture Anomaly Detection [[TSMC](https://ieeexplore.ieee.org/document/11130672)]  [[code](https://github.com/hmyao22/GRNR)]
- VarAD: Lightweight High-Resolution Image Anomaly Detection via Visual Autoregressive Modeling [[TII](https://ieeexplore.ieee.org/abstract/document/10843956)]  [[code](https://github.com/caoyunkang/VarAD)]
- Center-Aware Residual Anomaly Synthesis for Multiclass Industrial Anomaly Detection [[TII](https://arxiv.org/pdf/2505.17551)]  [[code](https://github.com/cqylunlun/CRAS)]
- Multitask Hybrid Knowledge Distillation for Unsupervised Anomaly Detection [[TII](https://ieeexplore.ieee.org/abstract/document/10970077)]
- Cross-Scale Denoising Reverse Distillation for Anomaly Detection [[TII](https://ieeexplore.ieee.org/abstract/document/11075768)]
- Dynamic Expert Routing for Unsupervised Continual Anomaly Detection [[TII](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11046350)]
- tGARD: Text-Guided Adversarial Reconstruction for Industrial Anomaly Detection [[TII](https://ieeexplore.ieee.org/document/11146401)]
- TKDA: A Tensor-Based Knowledge Distillation Approach of Anomaly Detection for Industrial Cyber-Physical Intelligence [[TII](https://ieeexplore.ieee.org/document/11131479)]
- Knowledge Distillation-Based Anomaly Detection via Adaptive Discrepancy Optimization [[TII](https://ieeexplore.ieee.org/abstract/document/11031083)]
- Multimodal Industrial Anomaly Detection via Uni-Modal and Cross-Modal Fusion [[TII](https://ieeexplore.ieee.org/abstract/document/10948502)]
- Feature Consistency Learning for Anomaly Detection [[TIM](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10816187)]
- Dual-Branch Knowledge Distillation via Residual Features Aggregation Module for Anomaly Segmentation [[TIM](https://ieeexplore.ieee.org/document/10770827)]
- REDAD: A Reliable Distillation for Image Anomaly Detection [[TIM](https://ieeexplore.ieee.org/document/10926908)]
- LScAD: A Large-Small Model Collaboration Framework for Unsupervised Industrial Anomaly Detection [[TIM](https://ieeexplore.ieee.org/document/11080008)]
- FC2P: Feature Cross-Channel Projection for Unsupervised Anomaly Segmentation [[TIM](https://ieeexplore.ieee.org/document/11155879)]
- CFFDist: Cross-Scale Feature Fusion Distillation Network for Industrial Anomaly Localization [[TIM](https://ieeexplore.ieee.org/document/10758791)]
- Manifold-Constrained Dynamic Decoupling Learning for Unsupervised Multiclass Anomaly Detection [[TIM](https://ieeexplore.ieee.org/document/11151817)]
- Industrial Anomaly Detection System: A Multicase Algorithm Leveraging Feature Information and Memory Bank [[TIM](https://ieeexplore.ieee.org/document/10929676)]
- ToCoAD: Two-Stage Contrastive Learning for Industrial Anomaly Detection [[TIM](https://ieeexplore.ieee.org/document/10904852)]
- ADFormer: Generalizable Few–Shot Anomaly Detection With Dual CNN-Transformer Architecture [[TIM](https://ieeexplore.ieee.org/document/10816355)]
- LECLIP: Boosting Zero–Shot Anomaly Detection With Local Enhanced CLIP [[TIM](https://ieeexplore.ieee.org/document/11007048)]
- VLDFNet: Views-Graph and Latent Feature Disentangled Fusion Network for Multimodal Industrial Anomaly Detection [[TIM](https://ieeexplore.ieee.org/document/11004494)]
- DualFlow: Dual-Branch Flow for Unsupervised Anomaly Detection and Localization [[TIM](https://ieeexplore.ieee.org/document/10979555)]
- PADiff: Reconstruction From Patch to Pixel With Normality-Guided Diffusion Model for Unsupervised Anomaly Localization [[TNNLS](https://ieeexplore.ieee.org/document/11038825)]  [[code](https://github.com/PaddlePaddle/PaDiff)]
- Boosting Global-Local Feature Matching via Anomaly Synthesis for Multi–Class Point Cloud Anomaly Detection [[TASE](https://ieeexplore.ieee.org/document/10898004)]  [[code](https://github.com/hustCYQ/GLFM-Multi-class-3DAD/)]
- A Hierarchical Patch Feature Distribution Network for Industrial Multiscale Defect Detection [[TASE](https://ieeexplore.ieee.org/document/11123455)]
- IIIM-SAM: Zero–Shot Texture Anomaly Detection Without External Prompts [[TASE](https://ieeexplore.ieee.org/abstract/document/10967541)]
- PFEAL: A Novel Framework for Image Anomaly Detection Using Pre-Trained Feature Extraction and Activation Learning [[TETCI](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11034765)]
- Progressive Boundary Guided Anomaly Synthesis for Industrial Anomaly Detection [[TETCI](https://ieeexplore.ieee.org/abstract/document/10716437)]  [[code](https://github.com/cqylunlun/PBAS)]
- Sensitivity Analysis-Based Explainable Diffusion Transformers for Anomaly Detection in Consumer Electronics Manufacturing [[TCE](https://ieeexplore.ieee.org/abstract/document/10835408)]
- Deep feature clustering for multi–class industrial image anomaly detection [[KBS](https://www.sciencedirect.com/science/article/pii/S0950705125001819)]
- SAM-LAD: Segment Anything Model meets zero–shot logic anomaly detection [[KBS](https://www.sciencedirect.com/science/article/pii/S0950705125002230)]
- Integrating local and global correlations with Mamba-Transformer for multi–class anomaly detection [[KBS](https://www.sciencedirect.com/science/article/pii/S0950705125007865)]  [[code](https://github.com/Mazeqi/MTAD-KBS)]
- DC-AD: A Divide-and-Conquer Method for Few–Shot Anomaly Detection [[PR](https://www.sciencedirect.com/science/article/pii/S0031320325000202)]
- SoftPatch+: Fully unsupervised anomaly classification and segmentation [[PR](https://www.sciencedirect.com/science/article/pii/S003132032401046X#fig6)]  [[code](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch)]
- MSAttnFlow: Normalizing flow for unsupervised anomaly detection with multi–scale attention [[PR](https://www.sciencedirect.com/science/article/pii/S0031320324009713)]
- Hard-Normal Example-Aware Template Mutual Matching for Industrial Anomaly Detection [[IJCV](https://link.springer.com/article/10.1007/s11263-024-02323-0)]  [[code](https://github.com/NarcissusEx/HETMM)]
- A novel FuseDecode Autoencoder for industrial visual inspection: Incremental anomaly detection improvement with gradual transition from unsupervised to mixed-supervision learning with reduced human effort [[CII](https://www.sciencedirect.com/science/article/pii/S016636152400126X)]

</details>

---










## Summary-before-2025

### 2D ANOMALY DETECTION
#### 📊 Unsupervised 
- TEXEMS: Texture Exemplars for Defect Detection on Random Textured Surfaces [[paper](https://ieeexplore.ieee.org/document/4250469)]  [![Info](https://img.shields.io/badge/Model-handcrafted_features-blue)](#)
- CutPaste: Self-Supervised Learning for Anomaly Detection and Localization [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.pdf)]  [[code](https://github.com/LilitYolyan/CutPaste)]  [![Info](https://img.shields.io/badge/Model-self–supervised_representations-blue)](#)
- Natural synthetic anomalies for self-supervised anomaly detection and localization [[paper](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_27)]  [[code](https://github.com/hmsch/natural-synthetic-anomalies)]  [![Info](https://img.shields.io/badge/Model-self–supervised_representations-blue)](#)
- Deep residual learning for image recognition [[paper](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)]  [[code](https://github.com/KaimingHe/deep-residual-networks)]  [![Info](https://img.shields.io/badge/Model-leveraging_pre--trained_features-blue)](#)
- Dinov2: Learning robust visual features without supervision [[paper](https://arxiv.org/abs/2304.07193)]  [[code](https://github.com/facebookresearch/dinov2)]  [![Info](https://img.shields.io/badge/Model-large--scale_foundational_models-blue)](#)
- Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.pdf)]  [[code](https://github.com/guojiajeremy/Dinomaly)]  [![Info](https://img.shields.io/badge/Model-large--scale_foundational_models-blue)](#)
- Dinov2: Learning robust visual features without supervision [[paper](https://arxiv.org/abs/2304.07193)]  [[code](https://github.com/facebookresearch/dinov2)]  [![Info](https://img.shields.io/badge/Model-large--scale_foundational_models-blue)](#)
- EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies [[paper](https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf)]  [[code](https://github.com/nelson1425/EfficientAD)]  [![Info](https://img.shields.io/badge/Model-ImageNet--pretrained_networks-blue)](#)
- CFA: Coupled-Hypersphere-Based Feature Adaptation for Target-Oriented Anomaly Localization [[paper](https://ieeexplore.ieee.org/document/9839549)]  [[code](https://github.com/sungwool/CFA_for_anomaly_localization)]  [![Info](https://img.shields.io/badge/Model-introduces_a_learnable_patch_descriptor-blue)](#)
- REB: Reducing biases in representation for industrial anomaly detection [[paper](https://www.sciencedirect.com/science/article/pii/S0950705124001989)]  [[code](https://github.com/ShuaiLYU/REB)]  [![Info](https://img.shields.io/badge/Model-self–supervised_tasks-blue)](#)
- ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction [[paper](https://neurips.cc/virtual/2023/poster/72002)]  [[code](https://github.com/guojiajeremy/ReContrast)]  [![Info](https://img.shields.io/badge/Model-end--to--end_network-blue)](#)
- UniNet: A Contrastive Learning-guided Unified Framework with Feature Selection for Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wei_UniNet_A_Contrastive_Learning-guided_Unified_Framework_with_Feature_Selection_for_CVPR_2025_paper.pdf)]  [[code](https://github.com/pangdatangtt/UniNet)]  [![Info](https://img.shields.io/badge/Model-end--to--end_network-blue)](#)
- Informative knowledge distillation for image anomaly segmentation [[paper](https://www.sciencedirect.com/science/article/pii/S0950705122004038)]  [[code](https://github.com/caoyunkang/IKD)]  [![Info](https://img.shields.io/badge/Model-regression_based-blue)](#)
- Anomaly Detection via Reverse Distillation from One-Class Embedding [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.pdf)]  [[code](https://github.com/hq-deng/RD4AD)]  [![Info](https://img.shields.io/badge/Model-regression_based-blue)](#)
- Towards Total Recall in Industrial Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)]  [[code](https://github.com/amazon-science/patchcore-inspection)]  [![Info](https://img.shields.io/badge/Model-memorybank--based-blue)](#)
- Industrial Image Anomaly Localization Based on Gaussian Clustering of Pretrained Feature [[paper](https://ieeexplore.ieee.org/document/9479740)]  [![Info](https://img.shields.io/badge/Model-memorybank--based-blue)](#)
- CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows [[paper](https://openaccess.thecvf.com/content/WACV2022/papers/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.pdf)]  [[code](https://github.com/gudovskiy/cflow-ad)]  [![Info](https://img.shields.io/badge/Model-flow--based-blue)](#)
- MSFlow: Multi-scale Flow-Based Framework for Unsupervised Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/10384766)]  [[code](https://github.com/cool-xuan/msflow)]  [![Info](https://img.shields.io/badge/Model-flow--based-blue)](#)
- Local-global normality learning and discrepancy normalizing flow for unsupervised image anomaly detection [[paper](https://www.sciencedirect.com/science/article/pii/S0952197624013939)]  [![Info](https://img.shields.io/badge/Model-flow--based-blue)](#)
- SimpleNet: A Simple Network for Image Anomaly Detection and Localization [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)]  [[code](https://github.com/DonaldRR/SimpleNet)]  [![Info](https://img.shields.io/badge/Model-discrimination--based-blue)](#)
- A unified anomaly synthesis strategy with gradient ascent for industrial anomaly detection and localization [[paper](https://link.springer.com/chapter/10.1007/978-3-031-72855-6_3)]  [[code](https://github.com/cqylunlun/GLASS)]  [![Info](https://img.shields.io/badge/Model-discrimination--based-blue)](#)
- Student-Teacher Feature Pyramid Matching for Anomaly Detection [[paper](https://www.bmvc2021-virtualconference.com/assets/papers/1273.pdf)]  [[code](https://github.com/gdwang08/STFPM)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Multiresolution Knowledge Distillation for Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Salehi_Multiresolution_Knowledge_Distillation_for_Anomaly_Detection_CVPR_2021_paper.pdf)]  [[code](https://github.com/rohban-lab/Knowledge_Distillation_AD)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Uninformed Students: Student-Teacher Anomaly Detection With Discriminative Latent Embeddings [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.pdf)]  [[code](https://github.com/taikiinoue45/STAD)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Unsupervised Image Anomaly Detection and Segmentation Based on Pretrained Feature Mapping [[paper](https://ieeexplore.ieee.org/document/9795121)]  [[code](https://github.com/smiler96/PFM-and-PEFM-for-Image-Anomaly-Detection-and-Segmentation)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Position Encoding Enhanced Feature Mapping for Image Anomaly Detection [[paper](https://ieeexplore.ieee.org/abstract/document/9926547)]  [[code](https://github.com/smiler96/PFM-and-PEFM-for-Image-Anomaly-Detection-and-Segmentation)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Unsupervised anomaly segmentation via deep feature reconstruction [[paper](https://www.sciencedirect.com/science/article/pii/S0925231220317951)]  [[code](https://github.com/YoungGod/DFR)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Anomaly Detection via Reverse Distillation from One-Class Embedding [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.pdf)]  [[code](https://github.com/hq-deng/RD4AD)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Asymmetric student-teacher networks for industrial anomaly detection [[paper](https://openaccess.thecvf.com/content/WACV2023/papers/Rudolph_Asymmetric_Student-Teacher_Networks_for_Industrial_Anomaly_Detection_WACV_2023_paper.pdf)]  [[code](https://github.com/marco-rudolph/ast)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- UTRAD: Anomaly detection and localization with U-Transformer [[paper](https://www.sciencedirect.com/science/article/pii/S0893608021004810)]  [[code](https://github.com/gordon-chenmo/UTRAD)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- A Unified Model for Multi-class Anomaly Detection [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/1d774c112926348c3e25ea47d87c835b-Paper-Conference.pdf)]  [[code](https://github.com/zhiyuanyou/UniAD)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- A Diffusion-Based Framework for Multi-Class Anomaly Detection [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28690)]  [[code](https://github.com/lewandofskee/DiAD)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- MTDiff: Visual anomaly detection with multi-scale diffusion models [[paper](https://www.sciencedirect.com/science/article/pii/S0950705124009985)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Correcting Deviations from Normality: A Reformulated Diffusion Model for Multi-Class Unsupervised Anomaly Detection [[paper](https://cvpr.thecvf.com/virtual/2025/poster/33146)]  [[code](https://github.com/farzad-bz/DeCo-Diff)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/833b21da1956c6b92f6df253bf655cf5-Paper-Conference.pdf)]  [[code](https://github.com/lewandofskee/MambaAD)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Unsupervised Anomaly Segmentation Via Multilevel Image Reconstruction and Adaptive Attention-Level Transition [[paper](https://ieeexplore.ieee.org/document/9521893)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Omni-Frequency Channel-Selection Representations for Unsupervised Anomaly Detection [[paper](https://ieeexplore.ieee.org/abstract/document/10192551)]  [[code](https://github.com/zhangzjn/OCR-GAN)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- AFSC: Adaptive Fourier Space Compression for Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/10609854)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization [[paper](https://ieeexplore.ieee.org/document/10034849)]  [[code](https://github.com/caoyunkang/CDO)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf)]  [[code](https://github.com/donggong1/memae-anomaly-detection)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Diversity-Measurable Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Diversity-Measurable_Anomaly_Detection_CVPR_2023_paper.pdf)]  [[code](https://github.com/FlappyPeggy/DMAD)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Visual Anomaly Detection via Partition Memory Bank Module and Error Estimation [[paper](https://ieeexplore.ieee.org/document/10018373)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Template-guided Hierarchical Feature Restoration for Anomaly Detection [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Template-guided_Hierarchical_Feature_Restoration_for_Anomaly_Detection_ICCV_2023_paper.pdf)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Template-based feature aggregation network for industrial anomaly detection [[paper](https://www.sciencedirect.com/science/article/pii/S0952197623019942)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Hierarchical vector quantized transformer for multi-class unsupervised anomaly detection [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/1abc87c67cc400a67b869358e627fe37-Paper-Conference.pdf)]  [[code](https://github.com/RuiyingLu/HVQ-Trans)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Remembering Normality: Memory-guided Knowledge Distillation for Unsupervised Anomaly Detection [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Gu_Remembering_Normality_Memory-guided_Knowledge_Distillation_for_Unsupervised_Anomaly_Detection_ICCV_2023_paper.pdf)]  [[code](https://github.com/SimonThomine/RememberingNormality)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Normal Reference Attention and Defective Feature Perception Network for Surface Defect Detection [[paper](https://ieeexplore.ieee.org/document/10106037)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Focus the Discrepancy: Intra- and Inter-Correlation Learning for Image Anomaly Detection [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yao_Focus_the_Discrepancy_Intra-_and_Inter-Correlation_Learning_for_Image_Anomaly_ICCV_2023_paper.pdf)]  [[code](https://github.com/xcyao00/FOD)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Prior normality prompt transformer for multiclass industrial image anomaly detection [[paper](https://ieeexplore.ieee.org/document/10574313)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Learning Unified Reference Representation for Unsupervised Multi-class Anomaly Detection [[paper](https://eccv.ecva.net/virtual/2024/poster/1912)]  [[code](https://github.com/hlr7999/RLR)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_Exploring_Intrinsic_Normal_Prototypes_within_a_Single_Image_for_Universal_CVPR_2025_paper.pdf)]  [[code](https://github.com/luow23/INP-Former)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Unsupervised Anomaly Detection for Surface Defects With Dual-Siamese Network [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9681338)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Revisiting Reverse Distillation for Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf)]  [[code](https://github.com/tientrandinh/Revisiting-Reverse-Distillation)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- OmniAL: A Unified CNN Framework for Unsupervised Anomaly Localization [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_OmniAL_A_Unified_CNN_Framework_for_Unsupervised_Anomaly_Localization_CVPR_2023_paper.pdf)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_DeSTSeg_Segmentation_Guided_Denoising_Student-Teacher_for_Anomaly_Detection_CVPR_2023_paper.pdf)]  [[code](https://github.com/apple/ml-destseg)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- A masked reverse knowledge distillation method incorporating global and local information for image anomaly detection [[paper](https://www.sciencedirect.com/science/article/pii/S0950705123007323)]  [[code](https://github.com/yuxin-jiang/MRKD)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization [[paper](https://ieeexplore.ieee.org/document/10034849)]  [[code](https://github.com/caoyunkang/CDO)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Pull & Push: Leveraging Differential Knowledge Distillation for Efficient Unsupervised Anomaly Detection and Localization [[paper](https://ieeexplore.ieee.org/document/9940966)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Generalad: Anomaly detection across domains by attending to distorted features [[paper](https://link.springer.com/chapter/10.1007/978-3-031-72913-3_25)]  [[code](https://github.com/LucStrater/GeneralAD)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- ADPS: Asymmetric Distillation Postsegmentation for Image Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/10509807)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_RealNet_A_Feature_Selection_Network_with_Realistic_Synthetic_Anomaly_for_CVPR_2024_paper.pdf)]  [[code](https://github.com/cnulab/RealNet)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Self-Supervised Masked Convolutional Transformer Block for Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/10273635)]  [[code](https://github.com/ristea/ssmctb)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Reconstruction by inpainting for visual anomaly detection [[paper](https://www.sciencedirect.com/science/article/pii/S0031320320305094)]  [[code](https://github.com/plutoyuxie/Reconstruction-by-inpainting-for-visual-anomaly-detection)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Self-Supervised Masking for Unsupervised Anomaly Detection and Localization [[paper](https://ieeexplore.ieee.org/document/9779083)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Masked Swin Transformer Unet for Industrial Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/9858596)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- MSTAD: A masked subspace-like transformer for multi-class anomaly detection [[paper](https://www.sciencedirect.com/science/article/pii/S095070512300936X)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- MLDFR: A Multilevel Features Restoration Method Based on Damaged Images for Anomaly Detection and Localization [[paper](https://ieeexplore.ieee.org/document/10186002)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- AMI-Net: Adaptive Mask Inpainting Network for Industrial Anomaly Detection and Localization [[paper](https://ieeexplore.ieee.org/document/10445116)]  [[code](https://github.com/luow23/AMI-Net)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- One-for-All: Proposal Masked Cross-Class Anomaly Detection [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25604)]  [[code](https://github.com/xcyao00/PMAD)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Scalable Industrial Visual Anomaly Detection With Partial Semantics Aggregation Vision Transformer [[paper](https://ieeexplore.ieee.org/document/10363401)]  [![Info](https://img.shields.io/badge/Model-Regression--based_Methods-blue)](#)
- Sub-Image Anomaly Detection with Deep Pyramid Correspondences [[paper](https://arxiv.org/abs/2005.02357)]  [[code](https://github.com/byungjae89/SPADE-pytorch)]  [![Info](https://img.shields.io/badge/Model-Memory_Bank--Based_Methods-blue)](#)
- Industrial Image Anomaly Localization Based on Gaussian Clustering of Pretrained Feature [[paper](https://ieeexplore.ieee.org/document/9479740)]  [![Info](https://img.shields.io/badge/Model-Memory_Bank--Based_Methods-blue)](#)
- Towards Total Recall in Industrial Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)]  [[code](https://github.com/amazon-science/patchcore-inspection)]  [![Info](https://img.shields.io/badge/Model-Memory_Bank--Based_Methods-blue)](#)
- ReConPatch: Contrastive Patch Representation Learning for Industrial Anomaly Detection [[paper](https://openaccess.thecvf.com/content/WACV2024/papers/Hyun_ReConPatch_Contrastive_Patch_Representation_Learning_for_Industrial_Anomaly_Detection_WACV_2024_paper.pdf)]  [[code](https://github.com/travishsu/ReConPatch-TF)]  [![Info](https://img.shields.io/badge/Model-Memory_Bank--Based_Methods-blue)](#)
- REB: Reducing biases in representation for industrial anomaly detection [[paper](https://www.sciencedirect.com/science/article/pii/S0950705124001989)]  [[code](https://github.com/ShuaiLYU/REB)]  [![Info](https://img.shields.io/badge/Model-Memory_Bank--Based_Methods-blue)](#)
- CFA: Coupled-Hypersphere-Based Feature Adaptation for Target-Oriented Anomaly Localization [[paper](https://ieeexplore.ieee.org/document/9839549)]  [[code](https://github.com/sungwool/CFA_for_anomaly_localization)]  [![Info](https://img.shields.io/badge/Model-Memory_Bank--Based_Methods-blue)](#)
- PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization [[paper](https://link.springer.com/chapter/10.1007/978-3-030-68799-1_35)]  [[code](https://github.com/taikiinoue45/PaDiM)]  [![Info](https://img.shields.io/badge/Model-Memory_Bank--Based_Methods-blue)](#)
- PNI: Industrial Anomaly Detection using Position and Neighborhood Information [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.pdf)]  [[code](https://github.com/wogur110/PNI_anomaly_detection)]  [![Info](https://img.shields.io/badge/Model-Memory_Bank--Based_Methods-blue)](#)
- FastRecon: Few shot Industrial Anomaly Detection via Fast Feature Reconstruction [[paper](https://openaccess.thecvf.com/content/ICCV2023/html/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.html)]  [![Info](https://img.shields.io/badge/Model-Memory_Bank--Based_Methods-blue)](#)
- Variational inference with normalizing flows [[paper](https://proceedings.mlr.press/v37/rezende15.html)]  [[code](https://github.com/federicobergamin/Variational-Inference-with-Normalizing-Flows)]  [![Info](https://img.shields.io/badge/Model-Flow--based_Methods-blue)](#)
- Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows [[paper](https://openaccess.thecvf.com/content/WACV2021/papers/Rudolph_Same_Same_but_DifferNet_Semi-Supervised_Defect_Detection_With_Normalizing_Flows_WACV_2021_paper.pdf)]  [[code](https://github.com/marco-rudolph/differnet)]  [![Info](https://img.shields.io/badge/Model-Flow--based_Methods-blue)](#)
- CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows [[paper](https://openaccess.thecvf.com/content/WACV2022/papers/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.pdf)]  [[code](https://github.com/gudovskiy/cflow-ad)]  [![Info](https://img.shields.io/badge/Model-Flow--based_Methods-blue)](#)
- Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection [[paper](https://openaccess.thecvf.com/content/WACV2022/papers/Rudolph_Fully_Convolutional_Cross-Scale-Flows_for_Image-Based_Defect_Detection_WACV_2022_paper.pdf)]  [[code](https://github.com/marco-rudolph/cs-flow)]  [![Info](https://img.shields.io/badge/Model-Flow--based_Methods-blue)](#)
- MSFlow: Multi-scale Flow-Based Framework for Unsupervised Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/10384766)]  [[code](https://github.com/cool-xuan/msflow)]  [![Info](https://img.shields.io/badge/Model-Flow--based_Methods-blue)](#)
- Cross-Attention Regression Flow for Defect Detection [[paper](https://ieeexplore.ieee.org/document/10681003)]  [![Info](https://img.shields.io/badge/Model-Flow--based_Methods-blue)](#)
- PyramidFlow: High-Resolution Defect Contrastive Localization Using Pyramid Normalizing Flow [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lei_PyramidFlow_High-Resolution_Defect_Contrastive_Localization_Using_Pyramid_Normalizing_Flow_CVPR_2023_paper.pdf)]  [![Info](https://img.shields.io/badge/Model-Flow--based_Methods-blue)](#)
- Unsupervised Anomaly Detection and Localization based on Two-Hierarchy Normalizing Flow [[paper](https://ieeexplore.ieee.org/document/10677524)]  [![Info](https://img.shields.io/badge/Model-Flow--based_Methods-blue)](#)
- Hierarchical gaussian mixture normalizing flow modeling for unified anomaly detection [[paper](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_6)]  [[code](https://github.com/xcyao00/HGAD)]  [![Info](https://img.shields.io/badge/Model-Flow--based_Methods-blue)](#)
- Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yao_Explicit_Boundary_Guided_Semi-Push-Pull_Contrastive_Learning_for_Supervised_Anomaly_Detection_CVPR_2023_paper.pdf)]  [[code](https://github.com/xcyao00/BGAD)]  [![Info](https://img.shields.io/badge/Model-Flow--based_Methods-blue)](#)
- Dual Attention Transformer and Discriminative Flow for Industrial Visual Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/10283467)]  [[code](https://github.com/hmyao22/DADF)]  [![Info](https://img.shields.io/badge/Model-Flow--based_Methods-blue)](#)
- CutPaste: Self-Supervised Learning for Anomaly Detection and Localization [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.pdf)]  [[code](https://github.com/LilitYolyan/CutPaste)]  [![Info](https://img.shields.io/badge/Model-Discrimination--based_Methods-blue)](#)
- Natural synthetic anomalies for self-supervised anomaly detection and localization [[paper](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_27)]  [[code](https://github.com/hmsch/natural-synthetic-anomalies)]  [![Info](https://img.shields.io/badge/Model-Discrimination--based_Methods-blue)](#)
- SimpleNet: A Simple Network for Image Anomaly Detection and Localization [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)]  [[code](https://github.com/DonaldRR/SimpleNet)]  [![Info](https://img.shields.io/badge/Model-Discrimination--based_Methods-blue)](#)
- Progressive Boundary Guided Anomaly Synthesis for Industrial Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/10716437)]  [[code](https://github.com/cqylunlun/PBAS)]  [![Info](https://img.shields.io/badge/Model-Discrimination--based_Methods-blue)](#)
- A unified anomaly synthesis strategy with gradient ascent for industrial anomaly detection and localization [[paper](https://link.springer.com/chapter/10.1007/978-3-031-72855-6_3)]  [[code](https://github.com/cqylunlun/GLASS)]  [![Info](https://img.shields.io/badge/Model-Discrimination--based_Methods-blue)](#)
- DRÃĘM: A discriminatively trained reconstruction embedding for surface anomaly detection [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf)]  [[code](https://github.com/VitjanZ/DRAEM)]  [![Info](https://img.shields.io/badge/Model-Discrimination--based_Methods-blue)](#)
- Produce Once, Utilize Twice for Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/10577185)]  [![Info](https://img.shields.io/badge/Model-Discrimination--based_Methods-blue)](#)
- Normal Image Guided Segmentation Framework for Unsupervised Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/10295508)]  [![Info](https://img.shields.io/badge/Model-Discrimination--based_Methods-blue)](#)
- Unsupervised Surface Anomaly Detection with Diffusion Probabilistic Model [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.pdf)]  [[code](https://github.com/xzhang-t/DiffAD)]  [![Info](https://img.shields.io/badge/Model-Discrimination--based_Methods-blue)](#)


#### 📊 Semi-supervised
- Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Catching_Both_Gray_and_Black_Swans_Open-Set_Supervised_Anomaly_Detection_CVPR_2022_paper.pdf)]  [[code](https://github.com/Choubo/DRA)]  [![Info](https://img.shields.io/badge/Model-open–set_anomaly_detection-blue)](#)
- Explainable Deep Few shot Anomaly Detection with Deviation Networks [[paper](https://arxiv.org/abs/2108.00462)]  [[code](https://github.com/mala-lab/deviation-network-image)]  [![Info](https://img.shields.io/badge/Model-open–set_anomaly_detection-blue)](#)
- Deep one-class classification [[paper](https://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf)]  [[code](https://github.com/lukasruff/Deep-SVDD)]  [![Info](https://img.shields.io/badge/Model-open–set_anomaly_detection-blue)](#)
- Explainable Deep Few shot Anomaly Detection with Deviation Networks [[paper](https://arxiv.org/abs/2108.00462)]  [[code](https://github.com/mala-lab/deviation-network-image)]  [![Info](https://img.shields.io/badge/Model-open–set_anomaly_detection-blue)](#)
- Supervised Anomaly Detection for Complex Industrial Images [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Baitieva_Supervised_Anomaly_Detection_for_Complex_Industrial_Images_CVPR_2024_paper.pdf)]  [[code](https://github.com/abc-125/segad)]  [![Info](https://img.shields.io/badge/Model-unsupervised_detectors-blue)](#)
- SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection [[paper](https://link.springer.com/chapter/10.1007/978-3-031-78192-6_4)]  [[code](https://github.com/blaz-r/SuperSimpleNet)]  [![Info](https://img.shields.io/badge/Model-extends_SimpleNet_by_replacing_synthetic_pseudo–anomalies_with_real_labeled_anomalies-blue)](#)
- Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Catching_Both_Gray_and_Black_Swans_Open-Set_Supervised_Anomaly_Detection_CVPR_2022_paper.pdf)]  [[code](https://github.com/Choubo/DRA)]  [![Info](https://img.shields.io/badge/Model-open–set_anomaly_detection-blue)](#)
- BiaS: Incorporating Biased Knowledge to Boost Unsupervised Image Anomaly Localization [[paper](https://ieeexplore.ieee.org/abstract/document/10402554)]  [![Info](https://img.shields.io/badge/Model-semi–supervised_anomaly_detection-blue)](#)
- Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization [[paper](https://ieeexplore.ieee.org/document/10034849)]  [[code](https://github.com/caoyunkang/CDO)]  [![Info](https://img.shields.io/badge/Model-Regression–based_Methods-blue)](#)
- Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yao_Explicit_Boundary_Guided_Semi-Push-Pull_Contrastive_Learning_for_Supervised_Anomaly_Detection_CVPR_2023_paper.pdf)]  [[code](https://github.com/xcyao00/BGAD)]  [![Info](https://img.shields.io/badge/Model-push–and–pull_learning_objective-blue)](#)
- Prototypical Residual Networks for Anomaly Detection and Localization [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Prototypical_Residual_Networks_for_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)]  [[code](https://github.com/xcyao00/PRNet)]  [![Info](https://img.shields.io/badge/Model-feature_residuals_to_reconstruct_anomalysegmentationmaps-blue)](#)
- Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection [[paper](https://cvpr.thecvf.com/virtual/2025/poster/35073)]  [![Info](https://img.shields.io/badge/Model-Gaussian_prototypes_and_diffusion_learning-blue)](#)
- Anomaly heterogeneity learning for open-set supervised anomaly detection [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Anomaly_Heterogeneity_Learning_for_Open-set_Supervised_Anomaly_Detection_CVPR_2024_paper.pdf)]  [[code](https://github.com/mala-lab/AHL)]  [![Info](https://img.shields.io/badge/Model-adaptively_enhance_abnormality_modeling_in_existing_frameworks-blue)](#)
- Semi-supervised noise-resilient anomaly detection with feature autoencoder [[paper](https://www.sciencedirect.com/science/article/pii/S0950705124010797)]  [![Info](https://img.shields.io/badge/Model-a_self–cross_scoring_mechanism_and_feature_AutoEncoder-blue)](#)


#### 📊 Zero/Few-shot 
- Registration based few–shot anomaly detection [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840300.pdf)]  [[code](https://github.com/MediaBrain-SJTU/RegAD)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Few–Shot_Anomaly_Detection-blue)](#)
- Few–Shot Anomaly Detection via Category-Agnostic Registration Learning [[paper](https://ieeexplore.ieee.org/document/10704738)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Few–Shot_Anomaly_Detection-blue)](#)
- Pushing the Limits of Fewshot Anomaly Detection in Industry Vision: Graphcore [[paper](https://openreview.net/pdf?id=xzmqxHdZAwO)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Few–Shot_Anomaly_Detection-blue)](#)
- UniVAD: A Training-free Unified Model for Few–shot Visual Anomaly Detection [[paper](https://arxiv.org/abs/2412.03342)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Few–Shot_Anomaly_Detection-blue)](#)
- One-to-Normal: Anomaly Personalization for Few–shot Anomaly Detection [[paper](https://openreview.net/pdf?id=tIzW3l2uaN)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Few–Shot_Anomaly_Detection-blue)](#)
- Dual-path frequency discriminators for few–shot anomaly detection [[paper](https://www.sciencedirect.com/science/article/pii/S0950705124010311)]  [[code](https://github.com/ybai111/DFD)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Few–Shot_Anomaly_Detection-blue)](#)
- Toward generalist anomaly detection via in-context residual learning with few–shot sample prompts [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Toward_Generalist_Anomaly_Detection_via_In-context_Residual_Learning_with_Few-shot_CVPR_2024_paper.pdf)]  [[code](https://github.com/mala-lab/inctrl)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Few–Shot_Anomaly_Detection-blue)](#)
- ResAD: A Simple Framework for Class Generalizable Anomaly Detection [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/e2ba11d128034f0c544693374c59e49a-Paper-Conference.pdf)]  [[code](https://github.com/xcyao00/ResAD)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Few–Shot_Anomaly_Detection-blue)](#)
- Prototypical Learning Guided ContextAware Segmentation Network for Few–shot Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/10702559)]  [[code](https://github.com/yuxin-jiang/PCSNet)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Few–Shot_Anomaly_Detection-blue)](#)
- MetaUAS: Universal Anomaly Segmentation with OnePrompt Meta-Learning [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/463a91da3c832bd28912cd0d1b8d9974-Paper-Conference.pdf)]  [[code](https://github.com/gaobb/MetaUAS)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Few–Shot_Anomaly_Detection-blue)](#)
- Text–Guided variational image generation for industrial anomaly detection and segmentation [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_Text-Guided_Variational_Image_Generation_for_Industrial_Anomaly_Detection_and_Segmentation_CVPR_2024_paper.pdf)]  [[code](https://github.com/MingyuLee82/TGI_AD_v1)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Few–Shot_Anomaly_Detection-blue)](#)
- Zero–Shot versus Many-shot: Unsupervised Texture Anomaly Detection [[paper](https://openaccess.thecvf.com/content/WACV2023/papers/Aota_Zero-Shot_Versus_Many-Shot_Unsupervised_Texture_Anomaly_Detection_WACV_2023_paper.pdf)]  [[code](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Zero–Shot_Anomaly_Detection-blue)](#)
- High-Fidelity Zero–Shot Texture Anomaly Localization Using Feature Correspondence Analysis [[paper](https://openaccess.thecvf.com/content/WACV2024/papers/Ardelean_High-Fidelity_Zero-Shot_Texture_Anomaly_Localization_Using_Feature_Correspondence_Analysis_WACV_2024_paper.pdf)]  [[code](https://github.com/TArdelean/AnomalyLocalizationFCA)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Zero–Shot_Anomaly_Detection-blue)](#)
- GlobalRegularized Neighborhood Regression for Efficient Zero–Shot Texture Anomaly Detection [[paper](https://arxiv.org/abs/2406.07333)]  [[code](https://github.com/hmyao22/GRNR)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Zero–Shot_Anomaly_Detection-blue)](#)
- MuSc: Zero–Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images [[paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/096b1019463f34eb241e87cfce8dfe16-Abstract-Conference.html)]  [[code](https://github.com/xrli-U/MuSc)]  [![Info](https://img.shields.io/badge/Model-Vanilla_Zero–Shot_Anomaly_Detection-blue)](#)
- WinCLIP: Zero–/Few–Shot Anomaly Classification and Segmentation [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf)]  [[code](https://github.com/caoyunkang/WinClip)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- Random Word Data Augmentation with CLIP for ZeroShot Anomaly Detection [[paper](https://papers.bmvc2023.org/0018.pdf)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- MAEDAY: MAE for few– and zero–shot AnomalY-Detection [[paper](https://arxiv.org/abs/2211.14307)]  [[code](https://github.com/EliSchwartz/MAEDAY)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- Personalizing Vision-Language Models With Hybrid Prompts for Zero–Shot Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/10884560)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- Grounding dino: Marrying dino with grounded pre-training for open-set object detection [[paper](https://arxiv.org/abs/2303.05499v4)]  [[code](https://github.com/IDEA-Research/GroundingDINO)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- Segment anything [[paper](https://ai.meta.com/research/publications/segment-anything/)]  [[code](https://segment-anything.com/)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- A zero–/few–shot anomaly classification and segmentation method for cvpr 2023 vand workshop challenge tracks 1&2: 1st place on zero–shot ad and 4th place on few–shot ad [[paper](https://arxiv.org/abs/2305.17382)]  [[code](https://github.com/ByChelsea/VAND-APRIL-GAN)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- Clip-ad: A language-guided staged dual-path model for zero–shot anomaly detection [[paper](https://arxiv.org/abs/2311.00453)]  [[code](https://github.com/ByChelsea/CLIP-AD)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- A closer look at the explainability of contrastive language-image pre-training [[paper](https://arxiv.org/abs/2304.05653)]  [[code](https://github.com/xmed-lab/CLIP_Surgery)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/27963)]  [[code](https://github.com/CASIA-IVA-Lab/AnomalyGPT)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- Adapting visual-language models for generalizable anomaly detection in medical images [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Adapting_Visual-Language_Models_for_Generalizable_Anomaly_Detection_in_Medical_Images_CVPR_2024_paper.pdf)]  [[code](https://github.com/MediaBrain-SJTU/MVFA-AD)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- Beyond single-modal boundary: Cross-modal anomaly detection through visual prototype and harmonization [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Mao_Beyond_Single-Modal_Boundary_Cross-Modal_Anomaly_Detection_through_Visual_Prototype_and_CVPR_2025_paper.pdf)]  [[code](https://github.com/Kerio99/CMAD)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- AnomalyCLIP: Objectagnostic Prompt Learning for Zero–shot Anomaly Detection [[paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d7b50b8ac2c781a12f26155f48310d8d-Abstract-Conference.html)]  [[code](https://github.com/zqhang/AnomalyCLIP)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- Long-Tailed Anomaly Detection with Learnable Class Names [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Ho_Long-Tailed_Anomaly_Detection_with_Learnable_Class_Names_CVPR_2024_paper.pdf)]  [[code](https://zenodo.org/records/10854201)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- VCP-CLIP: A Visual Context Prompting Model for Zero–Shot Anomaly Segmentation [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08761.pdf)]  [[code](https://github.com/xiaozhen228/VCP-CLIP)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- Adaclip: Adapting clip with hybrid learnable prompts for zero–shot anomaly detection [[paper](https://arxiv.org/abs/2407.15795)]  [[code](https://github.com/caoyunkang/AdaCLIP)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- Bayesian Prompt Flow Learning for Zero–Shot Anomaly Detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Qu_Bayesian_Prompt_Flow_Learning_for_Zero-Shot_Anomaly_Detection_CVPR_2025_paper.pdf)]  [[code](https://github.com/xiaozhen228/Bayes-PFL)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- One-for-All Few–Shot Anomaly Detection via Instance-Induced Prompt Learning [[paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9f7f2f57d8eaf44b2f09020f64ff6d96-Abstract-Conference.html)]  [[code](https://github.com/Vanssssry/One-For-All-Few-Shot-Anomaly-Detection)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- Kernel-Aware Graph Prompt Learning for Few–Shot Anomaly Detection [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/32790)]  [[code](https://github.com/CVL-hub/KAG-prompt)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- Promptad: Learning prompts with only normal samples for few–shot anomaly detection [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_PromptAD_Learning_Prompts_with_only_Normal_Samples_for_Few-Shot_Anomaly_CVPR_2024_paper.pdf)]  [[code](https://github.com/FuNz-0/PromptAD)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- AA-CLIP: Enhancing Zero–Shot Anomaly Detection via Anomaly-Aware CLIP [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_AA-CLIP_Enhancing_Zero-Shot_Anomaly_Detection_via_Anomaly-Aware_CLIP_CVPR_2025_paper.pdf)]  [[code](https://github.com/Mwxinnn/AA-CLIP)]  [![Info](https://img.shields.io/badge/Model-VLMs–based_Methods-blue)](#)
- FADE: Few–shot/zero–shot Anomaly Detection Engine using Large Vision-Language Model [[paper](https://arxiv.org/abs/2409.00556)]  [[code](https://github.com/BMVC-FADE/BMVC-FADE)]  [![Info](https://img.shields.io/badge/Model-MLLMs–based_Methods-blue)](#)
- Towards Training-free Anomaly Detection with Vision and Language Foundation Models [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Towards_Training-free_Anomaly_Detection_with_Vision_and_Language_Foundation_Models_CVPR_2025_paper.pdf)]  [[code](https://github.com/zhang0jhon/LogSAD)]  [![Info](https://img.shields.io/badge/Model-MLLMs–based_Methods-blue)](#)
- LogiCode: An LLM-Driven Framework for Logical Anomaly Detection [[paper](https://ieeexplore.ieee.org/document/10710633)]  [![Info](https://img.shields.io/badge/Model-MLLMs–based_Methods-blue)](#)
- FiLo: ZeroShot Anomaly Detection by Fine-Grained Description and HighQuality Localization [[paper](https://arxiv.org/abs/2404.13671)]  [[code](https://github.com/CASIA-IVA-Lab/FiLo)]  [![Info](https://img.shields.io/badge/Model-MLLMs–based_Methods-blue)](#)
- Do LLMs Understand Visual Anomalies? Uncovering LLM’s Capabilities in Zero–shot Anomaly Detection [[paper](https://openreview.net/forum?id=JyOGUqYrbV)]  [![Info](https://img.shields.io/badge/Model-MLLMs–based_Methods-blue)](#)
- MMAD: The First-Ever Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection [[paper](https://arxiv.org/abs/2410.09453)]  [[code](https://github.com/jam-cc/MMAD)]  [![Info](https://img.shields.io/badge/Model-MLLMs–based_Methods-blue)](#)
- Customizing visuallanguage foundation models for multi-modal anomaly detection and reasoning [[paper](https://ieeexplore.ieee.org/document/11033177)]  [[code](https://github.com/Xiaohao-Xu/Customizable-VLM)]  [![Info](https://img.shields.io/badge/Model-MLLMs–based_Methods-blue)](#)
- Towards Zero–Shot Anomaly Detection and Reasoning with Multimodal Large Language Models [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Towards_Zero-Shot_Anomaly_Detection_and_Reasoning_with_Multimodal_Large_Language_CVPR_2025_paper.pdf)]  [[code](https://github.com/honda-research-institute/Anomaly-OneVision)]  [![Info](https://img.shields.io/badge/Model-MLLMs–based_Methods-blue)](#)


---
### 3D ANOMALY DETECTION

#### 📊 Point Cloud
- PointNet: Deep learning on point sets for 3D classification and segmentation [[paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html)]  [[code](https://github.com/charlesq34/pointnet)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Anomaly detection in 3D point clouds using deep geometric descriptors [[paper](https://openaccess.thecvf.com/content/WACV2023/papers/Bergmann_Anomaly_Detection_in_3D_Point_Clouds_Using_Deep_Geometric_Descriptors_WACV_2023_paper.pdf)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Toward unsupervised 3D point cloud anomaly detection using variational autoencoder [[paper](http://hvrl.ics.keio.ac.jp/paper/pdf/international_Conference/2021/ICIP2021_masuda.pdf)]  [[code](https://github.com/llien30/point_cloud_anomaly_detection)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Towards scalable 3D anomaly detection and localization: A benchmark via 3D anomaly synthesis and a self–supervised learning network [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Towards_Scalable_3D_Anomaly_Detection_and_Localization_A_Benchmark_via_CVPR_2024_paper.pdf)]  [[code](https://github.com/Chopper-233/Anomaly-ShapeNet)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- R3D-AD: Reconstruction via diffusion for 3D anomaly detection [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05272.pdf)]  [[code](https://github.com/zhouzheyuan/r3d-ad)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Uni-3DAD: GAN-inversion aided universal 3D anomaly detection on model-free products [[paper](https://arxiv.org/abs/2408.16201)]  [[code](https://github.com/JiayuLiu666/Uni3DAD)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- MC3D-AD: A unified geometry-aware reconstruction model for multi–category 3D anomaly detection [[paper](https://arxiv.org/pdf/2505.01969)]  [[code](https://github.com/iCAN-SZU/MC3D-AD)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Back to the feature: classical 3D features are (almost) all you need for 3D anomaly detection [[paper](https://openaccess.thecvf.com/content/CVPR2023W/VAND/papers/Horwitz_Back_to_the_Feature_Classical_3D_Features_Are_Almost_All_CVPRW_2023_paper.pdf)]  [[code](https://github.com/eliahuhorwitz/3D-ADS)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Complementary pseudo multimodal feature for point cloud anomaly detection [[paper](https://arxiv.org/abs/2303.13194)]  [[code](https://github.com/caoyunkang/CPMF)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Real3D-AD: A dataset of point cloud anomaly detection [[paper](https://papers.nips.cc/paper/2023/file/611b896d447df43c898062358df4c114-Paper-Datasets_and_Benchmarks.pdf)]  [[code](https://github.com/M-3LAB/Real3D-AD)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Towards high-resolution 3D anomaly detection via group-level feature contrastive learning [[paper](https://arxiv.org/pdf/2408.04604)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Boosting global-local feature matching via anomaly synthesis for multi–class point cloud anomaly detection [[paper](https://export.arxiv.org/abs/2409.13162)]  [[code](https://github.com/hustCYQ/GLFM-Multi-class-3DAD)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Fence theorem: Towards dual-objective semantic-structure isolation in preprocessing phase for 3D anomaly detection [[paper](https://arxiv.org/pdf/2503.01100)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Masked autoencoders for 3D point cloud self–supervised learning [[paper](https://arxiv.org/abs/2203.06604)]  [[code](https://github.com/Pang-Yatian/Point-MAE)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Look inside for more: Internal spatial modality perception for 3D anomaly detection [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/32546)]  [[code](https://github.com/M-3LAB/Look-Inside-for-More?tab=readme-ov-file)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Towards high-resolution 3D anomaly detection: A scalable dataset and realtime framework for subtle industrial defects [[paper](https://arxiv.org/pdf/2507.07435)]  [[code](https://github.com/hustCYQ/MiniShift-Simple3D)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Complementary pseudo multimodal feature for point cloud anomaly detection [[paper](https://arxiv.org/abs/2303.13194)]  [[code](https://github.com/caoyunkang/CPMF)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Towards zero–shot point cloud anomaly detection: A multi-view projection framework [[paper](https://export.arxiv.org/abs/2409.13162)]  [[code](https://github.com/hustCYQ/MVP-PCLIP)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- PointAD: Comprehending 3D anomalies from points and pixels for zero–shot 3D anomaly detection [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9a263e235f6d1521d13a8531c7974951-Abstract-Conference.html)]  [[code](https://github.com/zqhang/PointAD)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Towards zero–shot point cloud anomaly detection: A multi-view projection framework [[paper](https://export.arxiv.org/abs/2409.13162)]  [[code](https://github.com/hustCYQ/MVP-PCLIP)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- PointAD: Comprehending 3D anomalies from points and pixels for zero–shot 3D anomaly detection [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9a263e235f6d1521d13a8531c7974951-Abstract-Conference.html)]  [[code](https://github.com/zqhang/PointAD)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Exploiting point-language models with dual-prompts for 3D anomaly detection [[paper](https://arxiv.org/abs/2502.11307)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- PO3AD: Predicting point offsets toward better 3D point cloud anomaly detection [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_PO3AD_Predicting_Point_Offsets_toward_Better_3D_Point_Cloud_Anomaly_CVPR_2025_paper.pdf)]  [[code](https://github.com/yjnanan/PO3AD)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- A new approach to detect surface defects from 3D point cloud data with surface normal gabor filter (SNGF) [[paper](https://www.sciencedirect.com/science/article/abs/pii/S152661252300172X)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)
- Automatic point cloud clustering method for surface defect diagnosis [[paper](https://ieeexplore.ieee.org/document/10887345)]  [![Info](https://img.shields.io/badge/Model-3D_Point_Cloud_Anomaly_Detection-blue)](#)


#### 📊 Multi-modal-based
- Multimodal industrial anomaly detection via hybrid fusion [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Multimodal_Industrial_Anomaly_Detection_via_Hybrid_Fusion_CVPR_2023_paper.pdf)]  [[code](https://github.com/nomewang/M3DM)]  [![Info](https://img.shields.io/badge/Model-Multimodal_Feature_Fusion-blue)](#)
- Shape–guided dual-memory learning for 3D anomaly detection [[paper](https://openreview.net/pdf?id=IkSGn9fcPz)]  [[code](https://github.com/jayliu0313/Shape-Guided)]  [![Info](https://img.shields.io/badge/Model-Multimodal_Feature_Fusion-blue)](#)
- Asymmetric student–teacher networks for industrial anomaly detection [[paper](https://openaccess.thecvf.com/content/WACV2023/papers/Rudolph_Asymmetric_Student-Teacher_Networks_for_Industrial_Anomaly_Detection_WACV_2023_paper.pdf)]  [[code](https://github.com/marco-rudolph/ast)]  [![Info](https://img.shields.io/badge/Model-Multimodal_Feature_Fusion-blue)](#)
- Easynet: An easy network for 3D industrial anomaly detection [[paper](https://arxiv.org/abs/2307.13925)]  [[code](https://github.com/TaoTao9/EasyNet)]  [![Info](https://img.shields.io/badge/Model-Multimodal_Feature_Fusion-blue)](#)
- Unsupervised visual–to–geometric feature reconstruction for visionbased industrial anomaly detection [[paper](https://ieeexplore.ieee.org/abstract/document/10820339)]  [[code](https://github.com/MaiAnhTruong/Unsupervised-Visual-to-Geometric-Feature-Reconstruction-for-Vision-Based-Anomaly-Detection)]  [![Info](https://img.shields.io/badge/Model-Multimodal_Feature_Fusion-blue)](#)
- Multimodal industrial anomaly detection via uni–modal and cross–modal fusion [[paper](https://ieeexplore.ieee.org/document/10948502)]  [![Info](https://img.shields.io/badge/Model-Multimodal_Feature_Fusion-blue)](#)
- Easynet: An easy network for 3D industrial anomaly detection [[paper](https://arxiv.org/abs/2307.13925)]  [[code](https://github.com/TaoTao9/EasyNet)]  [![Info](https://img.shields.io/badge/Model-Multimodal_Feature_Fusion-blue)](#)
- Cheating depth: Enhancing 3D surface anomaly detection via depth simulation [[paper](https://arxiv.org/abs/2311.01117)]  [[code](https://github.com/VitjanZ/3DSR)]  [![Info](https://img.shields.io/badge/Model-Multimodal_Feature_Fusion-blue)](#)
- Keep dræming: discriminative 3D anomaly detection through anomaly simulation [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865524000862)]  [![Info](https://img.shields.io/badge/Model-Multimodal_Feature_Fusion-blue)](#)
- Multimodal industrial anomaly detection by crossmodal feature mapping [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Costanzino_Multimodal_Industrial_Anomaly_Detection_by_Crossmodal_Feature_Mapping_CVPR_2024_paper.pdf)]  [[code](https://github.com/CVLAB-Unibo/crossmodal-feature-mapping)]  [![Info](https://img.shields.io/badge/Model-Crossmodal_Reconstruction-blue)](#)
- Incomplete multimodal industrial anomaly detection via cross–modal distillation [[paper](https://www.sciencedirect.com/science/article/pii/S156625352500644X?via%3Dihub)]  [[code](https://github.com/evenrose/CMDIAD)]  [![Info](https://img.shields.io/badge/Model-Crossmodal_Reconstruction-blue)](#)
- Cpir: Multimodal industrial anomaly detection via latent bridged cross–modal prediction and intra–modal reconstruction [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1474034625001338)]  [[code](https://github.com/CVLAB-Unibo/crossmodal-feature-mapping)]  [![Info](https://img.shields.io/badge/Model-Crossmodal_Reconstruction-blue)](#)
- Multimodal industrial anomaly detection by crossmodal reverse distillation [[paper](https://arxiv.org/html/2412.08949v2)]  [![Info](https://img.shields.io/badge/Model-Teacher–Student_Framework-blue)](#)
- Lpfstnet: A lightweight and parameter-free head attention-based student–teacher network for fast 3D industrial anomaly detection [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231225000803)]  [[code](https://github.com/YahuiCheng/LPFSTNet)]  [![Info](https://img.shields.io/badge/Model-Teacher–Student_Framework-blue)](#)
- Rethinking reverse distillation for multi–modal anomaly detection [[paper](https://dl.acm.org/doi/10.1609/aaai.v38i8.28687)]  [![Info](https://img.shields.io/badge/Model-Teacher–Student_Framework-blue)](#)
- Revisiting multimodal fusion for 3D anomaly detection from an architectural perspective [[paper](https://arxiv.org/abs/2412.17297)]  [[code](https://github.com/longkaifang/3D-ADNAS)]  [![Info](https://img.shields.io/badge/Model-Teacher–Student_Framework-blue)](#)
- 2m3df: Advancing 3D industrial defect detection with multi–perspective multimodal fusion network [[paper](https://ieeexplore.ieee.org/document/10858074)]  [![Info](https://img.shields.io/badge/Model-Enriching_Multi–Modal_Spaces-blue)](#)
- 3d-mmfn: Multi–level multimodal fusion network for 3D industrial image anomaly detection [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1474034625001776)]  [![Info](https://img.shields.io/badge/Model-Enriching_Multi–Modal_Spaces-blue)](#)
- Multi–sensor object anomaly detection: Unifying appearance, geometry, and internal properties [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Multi-Sensor_Object_Anomaly_Detection_Unifying_Appearance_Geometry_and_Internal_Properties_CVPR_2025_paper.pdf)]  [[code](https://github.com/ZZZBBBZZZ/MulSen-AD)]  [![Info](https://img.shields.io/badge/Model-Enriching_Multi–Modal_Spaces-blue)](#)
- Self–supervised feature adaptation for 3D industrial anomaly detection [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00180.pdf)]  [[code](https://github.com/yuanpengtu/LSFA)]  [![Info](https://img.shields.io/badge/Model-Others-blue)](#)
- Incremental template neighborhood matching for 3D anomaly detection [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231224002546?via%3Dihub)]  [![Info](https://img.shields.io/badge/Model-Others-blue)](#)
- Clip3d-ad: Extending clip for 3D few–shot anomaly detection with multi–view images generation [[paper](https://arxiv.org/abs/2406.18941)]  [![Info](https://img.shields.io/badge/Model-Others-blue)](#)
- Towards zero–shot 3D anomaly localization [[paper](https://openaccess.thecvf.com/content/WACV2025/papers/Wang_Towards_Zero-Shot_3D_Anomaly_Localization_WACV_2025_paper.pdf)]  [[code](https://github.com/wyzjack/3DzAL)]  [![Info](https://img.shields.io/badge/Model-Others-blue)](#)
- Zero–shot 3D anomaly detection via online voter mechanism [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608025002771)]  [![Info](https://img.shields.io/badge/Model-Others-blue)](#)
- M3dm-nr: Rgb-3D noisy–resistant industrial anomaly detection via multimodal denoising [[paper](https://arxiv.org/abs/2406.02263)]  [![Info](https://img.shields.io/badge/Model-Others-blue)](#)

---






## BibTex Citation

If you find this paper and repository useful, please cite our paper☺️.

```
@article{ADSurvey,
  title={A comprehensive survey for real-world industrial defect detection: Challenges, approaches, and prospects},
  author={Cheng, Yuqi and Cao, Yunkang and Yao, Haiming and Luo, Wei and Jiang, Cheng and Zhang, Hui and Shen, Weiming},
  journal={arXiv preprint arXiv:2507.13378},
  year={2025}
}
```
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hustCYQ/Towards-Practical-Industrial-Anomaly-Detection&type=Date)](https://www.star-history.com/#hustCYQ/Towards-Practical-Industrial-Anomaly-Detection&Date)
























