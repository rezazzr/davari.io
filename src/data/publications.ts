export interface Publication {
  name: string;
  descr: string;
  paper: string;
  published: string;
  award?: string;
  cite: string;
  visual: string;
}

export const publications: Publication[] = [
  {
    name: "Model breadcrumbs: Scaling multi-task model merging with sparse masks",
    descr: "The rapid development of AI systems has been greatly influenced by the emergence of foundation models. A common approach for targeted problems involves fine-tuning these pre-trained foundation models for specific target tasks, resulting in a rapid spread of models fine-tuned across a diverse array of tasks. This work focuses on the problem of merging multiple fine-tunings of the same foundation model derived from a spectrum of auxiliary tasks. We introduce a new simple method, Model Breadcrumbs, which consists of a sparsely defined weight set that guides model adaptation within the weight space of a pre-trained model. These breadcrumbs are constructed by subtracting the weights from a pre-trained model before and after fine-tuning, followed by a sparsification process that eliminates weight outliers and negligible perturbations. Our experiments demonstrate the effectiveness of Model Breadcrumbs to simultaneously improve performance across multiple tasks. This contribution aligns with the evolving paradigm of updatable machine learning, reminiscent of the collaborative principles underlying open-source software development, fostering a community-driven effort to reliably update machine learning models. Our method is shown to be more efficient and unlike previous proposals does not require hyperparameter tuning for each new task added. Through extensive experimentation involving various models, tasks, and modalities we establish that integrating Model Breadcrumbs offers a simple, efficient, and highly effective approach for constructing multi-task models and facilitating updates to foundation models.",
    paper: "https://link.springer.com/chapter/10.1007/978-3-031-73226-3_16",
    published:
      'MohammadReza Davari and Eugene Belilovsky (<a href="https://eccv.ecva.net/Conferences/2024" target="_blank">ECCV 2024</a>)',
    cite: `@inproceedings{davari2024model,
  title      = {Model breadcrumbs: Scaling multi-task model merging with sparse masks},
  author     = {Davari, MohammadReza and Belilovsky, Eugene},
  booktitle  = {European Conference on Computer Vision},
  pages      = {270--287},
  year       = {2024},
  organization = {Springer}
}`,
    visual: "model_breadcrumbs.png",
  },
  {
    name: "CLaC at SemEval-2024 Task 2: Faithful Clinical Trial Inference",
    descr: "This paper presents the methodology used for our participation in SemEval 2024 Task 2 (Jullien et al., 2024) – Safe Biomedical Natural Language Inference for Clinical Trials. The task involved Natural Language Inference (NLI) on clinical trial data, where statements were provided regarding information within Clinical Trial Reports (CTRs). These statements could pertain to a single CTR or compare two CTRs, requiring the identification of the inference relation (entailment vs contradiction) between CTR-statement pairs. Evaluation was based on F1, Faithfulness, and Consistency metrics, with priority given to the latter two by the organizers. Our approach aims to maximize Faithfulness and Consistency, guided by intuitive definitions provided by the organizers, without detailed metric calculations. Experimentally, our approach yielded models achieving maximal Faithfulness (top rank) and average Consistency (mid rank) at the expense of F1 (low rank). Future work will focus on refining our approach to achieve a balance among all three metrics.",
    paper: "https://aclanthology.org/2024.semeval-1.239/",
    published: 'Jennifer Marks, MohammadReza Davari, and Leila Kosseim (<a href="https://semeval.github.io/SemEval2024/" target="_blank">SemEval 2024</a>)',
    cite: `@inproceedings{marks2024clac,
  title={Clac at semeval-2024 task 2: Faithful clinical trial inference},
  author={Marks, Jennifer and Davari, MohammadReza and Kosseim, Leila},
  booktitle={Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024)},
  pages={1673--1677},
  year={2024}
}`,
    visual: "clinical_trial.png",
  },
  {
    name: "Scalable Upcycling of Finetuned Foundation Models via Sparse Task Vectors Merging",
    descr: "The rapid development of AI systems has been greatly influenced by foundation models. Typically, these models are fine-tuned for specific tasks, leading to numerous task-specific versions. This paper addresses the challenge of merging and upcycling these fine-tuned models. We introduce Model Breadcrumbs, a simple method using sparse weight trajectories to guide model adaptation within a pre-trained model's weight space. Our approach improves performance across multiple tasks without the need for hyperparameter tuning for each new task. Extensive experiments, involving various models, tasks, and modalities, demonstrate that Model Breadcrumbs provides an efficient and effective solution for creating and updating multi-task models, promoting a community-driven effort for updatable machine learning.",
    paper: "https://openreview.net/pdf?id=vuyP3tupig",
    published:
      'MohammadReza Davari and Eugene Belilovsky (<a href="https://fm-wild-community.github.io/index_2024.html#introduction" target="_blank">ICML 2024 Workshop FM-Wild</a>)',
    cite: `@inproceedings{davari2024scalable,
  title      = {Scalable Upcycling of Finetuned Foundation Models via Sparse Task Vectors Merging},
  author     = {Davari, MohammadReza and Belilovsky, Eugene},
  booktitle  = {ICML 2024 Workshop on Foundation Models in the Wild},
  year       = {2024}
}`,
    visual: "scale_models.png",
  },
  {
    name: "Prototype-Sample Relation Distillation: Towards Replay-Free Continual Learning",
    descr: "In Continual learning (CL) balancing effective adaptation while combating catastrophic forgetting is a central challenge. Many of the recent best-performing methods utilize various forms of prior task data, eg a replay buffer, to tackle the catastrophic forgetting problem. Having access to previous task data can be restrictive in many real-world scenarios, for example when task data is sensitive or proprietary. To overcome the necessity of using previous tasks' data, in this work, we start with strong representation learning methods that have been shown to be less prone to forgetting. We propose a holistic approach to jointly learn the representation and class prototypes while maintaining the relevance of old class prototypes and their embedded similarities. Specifically, samples are mapped to an embedding space where the representations are learned using a supervised contrastive loss. Class prototypes are evolved continually in the same latent space, enabling learning and prediction at any point. To continually adapt the prototypes without keeping any prior task data, we propose a novel distillation loss that constrains class prototypes to maintain relative similarities as compared to new task data. This method yields state-of-the-art performance in the task-incremental setting, outperforming methods relying on large amounts of data, and provides strong performance in the class-incremental setting without using any stored data points.",
    paper: "https://proceedings.mlr.press/v202/asadi23a.html",
    published: 'Nader Asadi, MohammadReza Davari, Sudhir Mudur, Rahaf Aljundi, and Eugene Belilovsky (<a href="https://icml.cc/Conferences/2023" target="_blank">ICML 2023</a>)',
    cite: `@inproceedings{asadi2023prototype,
  title={Prototype-sample relation distillation: towards replay-free continual learning},
  author={Asadi, Nader and Davari, MohammadReza and Mudur, Sudhir and Aljundi, Rahaf and Belilovsky, Eugene},
  booktitle={International conference on machine learning},
  pages={1093--1106},
  year={2023},
  organization={PMLR}
}`,
    visual: "prototype.png",
  },
  {
    name: "Reliability of CKA as a Similarity Measure in Deep Learning",
    descr: "Comparing learned neural representations in neural networks is a challenging but important problem, which has been approached in different ways. The Centered Kernel Alignment (CKA) similarity metric, particularly its linear variant, has recently become a popular approach and has been widely used to compare representations of a network's different layers, of architecturally similar networks trained differently, or of models with different architectures trained on the same data. A wide variety of conclusions about similarity and dissimilarity of these various representations have been made using CKA. In this work we present analysis that formally characterizes CKA sensitivity to a large class of simple transformations, which can naturally occur in the context of modern machine learning. This provides a concrete explanation of CKA sensitivity to outliers, which has been observed in past works, and to transformations that preserve the linear separability of the data, an important generalization attribute. We empirically investigate several weaknesses of the CKA similarity metric, demonstrating situations in which it gives unexpected or counter-intuitive results. Finally we study approaches for modifying representations to maintain functional behaviour while changing the CKA value. Our results illustrate that, in many cases, the CKA value can be easily manipulated without substantial changes to the functional behaviour of the models, and call for caution when leveraging activation alignment metrics.",
    paper: "https://arxiv.org/abs/2210.16156",
    published: 'MohammadReza Davari<sup>*</sup>, Stefan Horoi<sup>*</sup>, Amine Natik, Guillaume Lajoie, Guy Wolf, and Eugene Belilovsky (<a href="https://iclr.cc/virtual/2023/poster/12037" target="_blank">ICLR 2023</a>)',
    cite: `@inproceedings{
  davari2023reliability,
  title={Reliability of {CKA} as a Similarity Measure in Deep Learning},
  author={MohammadReza Davari and Stefan Horoi and Amine Natik and Guillaume Lajoie and Guy Wolf and Eugene Belilovsky},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=8HRvyxc606}
}`,
    visual: "reliability.png",
  },
  {
    name: "Deceiving the CKA Similarity Measure in Deep Learning",
    descr: "Understanding the behaviour of trained deep neural networks is a critical step in allowing reliable deployment of these networks in critical applications. One direction for obtaining insights on neural networks is through comparison of their internal representations. Comparing neural representations in neural networks is thus a challenging but important problem, which has been approached in different ways. The Centered Kernel Alignment (CKA) similarity metric, particularly its linear variant, has recently become a popular approach and has been widely used to compare representations of a network's different layers, of architecturally similar networks trained differently, or of models with different architectures trained on the same data. A wide variety of conclusions about similarity and dissimilarity of these various representations have been made using CKA. In this work we present an analysis that formally characterizes CKA sensitivity to a large class of simple transformations, which can naturally occur in the context of modern machine learning. This provides a concrete explanation of CKA sensitivity to outliers and to transformations that preserve the linear separability of the data, an important generalization attribute. Finally we propose an optimization-based approach for modifying representations to maintain functional behaviour while changing the CKA value. Our results illustrate that, in many cases, the CKA value can be easily manipulated without substantial changes to the functional behaviour of the models, and call for caution when leveraging activation alignment metrics.",
    paper: "https://openreview.net/forum?id=hITONWhDIIJ",
    published:
      'MohammadReza Davari<sup>*</sup>, Stefan Horoi<sup>*</sup>, Amine Natik, Guillaume Lajoie, Guy Wolf and Eugene Belilovsky (<a href="https://neurips2022.mlsafety.org/" target="_blank">NeurIPS ML Safety Workshop 2022</a>)',
    cite: `@inproceedings{davari2022deceiving,
  title      = {Deceiving the {CKA} Similarity Measure in Deep Learning},
  author     = {Davari, MohammadReza and Horoi, Stefan and Natik, Amine and Lajoie, Guillaume and Wolf, Guy and Belilovsky, Eugene},
  booktitle  = {NeurIPS 2022 Workshop on Machine Learning Safety (NeurIPS MLSW 2022)},
  year       = {2022},
  month      = {December},
}`,
    visual: "cka_comical_op.gif",
  },
  {
    name: "Probing Representation Forgetting in Supervised and Unsupervised Continual Learning",
    descr: "Continual Learning (CL) research typically focuses on tackling the phenomenon of catastrophic forgetting in neural networks. Catastrophic forgetting is associated with an abrupt loss of knowledge previously learned by a model when the task, or more broadly the data distribution, being trained on changes. In supervised learning problems this forgetting, resulting from a change in the model's representation, is typically measured or observed by evaluating the decrease in old task performance. However, a model's representation can change without losing knowledge about prior tasks. In this work we consider the concept of representation forgetting, observed by using the difference in performance of an optimal linear classifier before and after a new task is introduced. Using this tool we revisit a number of standard continual learning benchmarks and observe that, through this lens, model representations trained without any explicit control for forgetting often experience small representation forgetting and can sometimes be comparable to methods which explicitly control for forgetting, especially in longer task sequences. We also show that representation forgetting can lead to new insights on the effect of model capacity and loss function used in continual learning. Based on our results, we show that a simple yet competitive approach is to learn representations continually with standard supervised contrastive learning while constructing prototypes of class samples when queried on old samples.",
    paper: "https://openaccess.thecvf.com/content/CVPR2022/html/Davari_Probing_Representation_Forgetting_in_Supervised_and_Unsupervised_Continual_Learning_CVPR_2022_paper.html",
    published:
      'MohammadReza Davari<sup>*</sup>, Nader Asadi<sup>*</sup>, Sudhir Mudur, Rahaf Aljundi and Eugene Belilovsky (<a href="https://cvpr2022.thecvf.com/">CVPR 2022</a>)',
    cite: `@inproceedings{davari2022probing,
  title      = {Probing Representation Forgetting in Supervised and Unsupervised Continual Learning},
  author     = {Davari, MohammadReza and Asadi, Nader and Mudur, Sudhir and Aljundi, Rahaf and Belilovsky, Eugene},
  booktitle  = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022)},
  year       = {2022},
  month      = {June},
}`,
    visual: "imagenet_200_tasks.gif",
  },
  {
    name: "On the Inadequacy of CKA as a Measure of Similarity in Deep Learning",
    descr: "Comparing learned representations is a challenging problem which has been approached in different ways. The CKA similarity metric, particularly it's linear variant, has recently become a popular approach and has been widely used to compare representations of a network's different layers, of similar networks trained differently, or of models with different architectures trained on the same data. CKA results have been used to make a wide variety of claims about similarity and dissimilarity of these various representations. In this work we investigate several weaknesses of the CKA similarity metric, demonstrating situations in which it gives unexpected or counterintuitive results. We then study approaches for modifying representations to maintain functional behaviour while changing the CKA value. Indeed we illustrate in some cases the CKA value can be heavily manipulated without substantial changes to the functional behaviour.",
    paper: "https://openreview.net/forum?id=rK841rby6xc",
    published:
      'MohammadReza Davari<sup>*</sup>, Stefan Horoi<sup>*</sup>, Amine Natik, Guillaume Lajoie, Guy Wolf and Eugene Belilovsky (<a href="https://gt-rl.github.io/">ICLR GTRL Workshop 2022</a>)',
    cite: `@inproceedings{davari2022inadequacy,
  title      = {On the Inadequacy of CKA as a Measure of Similarity in Deep Learning},
  author     = {Davari, MohammadReza and Horoi, Stefan and Natik, Amine and Lajoie, Guillaume and Wolf, Guy and Belilovsky, Eugene},
  booktitle  = {ICLR 2022 Workshop on Geometrical and Topological Representation Learning (ICLR GTRL Workshop 2022)},
  year       = {2022},
  month      = {April},
}`,
    visual: "cka_analytical_op.gif",
  },
  {
    name: "Probing Representation Forgetting in Continual Learning",
    descr: "Continual Learning methods typically focus on tackling the phenomenon of catastrophic forgetting in the context of neural networks. Catastrophic forgetting is associated with an abrupt loss of knowledge previously learned by a model. In supervised learning problems this forgetting is typically measured or observed by evaluating decrease in task performance. However, a model's representations can change without losing knowledge. In this work we consider the concept of representation forgetting, which relies on using the difference in performance of an optimal linear classifier before and after a new task is introduced. Using this tool we revisit a number of standard continual learning benchmarks and observe that through this lens, model representations trained without any special control for forgetting often experience minimal representation forgetting. Furthermore we find that many approaches to continual learning that aim to resolve the catastrophic forgetting problem do not improve the representation forgetting upon the usefulness of the representation.",
    paper: "https://openreview.net/forum?id=YzAZmjJ-4fL",
    published:
      'MohammadReza Davari and Eugene Belilovsky (<a href="https://sites.google.com/view/distshift2021">NeurIPS DistShift Workshop 2021</a>)',
    cite: `@inproceedings{davari2021probing,
  title      = {Probing Representation Forgetting in Continual Learning},
  author     = {Davari, MohammadReza and Belilovsky, Eugene},
  booktitle  = {NeurIPS 2021 Workshop on Distribution Shifts: Connecting Methods and Applications (NeurIPS DistShift Workshop 2021)},
  year       = {2021},
  month      = {December},
}`,
    visual: "depth_forgetting.gif",
  },
  {
    name: "Semantic Similarity Matching Using Contextualized Representations",
    descr: "Different approaches to address semantic similarity matching generally fall into one of the two categories of interaction-based and representation-based models. While each approach offers its own benefits and can be used in certain scenarios, using a transformer-based model with a completely interaction-based approach may not be practical in many real-life use cases. In this work, we compare the performance and inference time of interaction-based and representation-based models using contextualized representations. We also propose a novel approach which is based on the late interaction of textual representations, thus benefiting from the advantages of both model types.",
    paper: "https://caiac.pubpub.org/pub/6tk0unzp/release/2",
    published:
      'Farhood Farahnak, Elham Mohammadi<sup>*</sup>, MohammadReza Davari<sup>*</sup>, and Leila Kosseim (<a href="https://www.caiac.ca/en/conferences/canadianai-2021/home">CanAI 2021</a>)',
    cite: `@inproceedings{Farahnak2021Semantic,
  title      = {Semantic Similarity Matching Using Contextualized Representations},
  author     = {Farahnak, Farhood and Mohammadi, Elham and Davari, MohammadReza and Kosseim, Leila},
  booktitle  = {Proceedings of the 34th Canadian Conference on Artificial Intelligence (CAIAC 2021)},
  address    = {Vancouver, Canada (Online)},
  year       = {2021},
  month      = {June},
}`,
    visual: "semantic_matching_op.gif",
  },
  {
    name: "TIMBERT: Toponym Identifier For The Medical Domain Based on BERT",
    descr: "In this paper, we propose an approach to automate the process of place name detection in the medical domain to enable epidemiologists to better study and model the spread of viruses. We created a family of Toponym Identification Models based on BERT (TIMBERT), in order to learn in an end-to-end fashion the mapping from an input sentence to the associated sentence labeled with toponyms. When evaluated with the SemEval 2019 task 12 test set (Weissenbacher et al., 2019), our best TIMBERT model achieves an F1 score of 90.85%, a significant improvement compared to the state-of-the-art of 89.10% (Wang et al., 2019).",
    paper: "https://www.aclweb.org/anthology/2020.coling-main.58/",
    published:
      'MohammdReza Davari, Leila Kosseim and Tien Bui (<a href="https://coling2020.org/">COLING 2020</a>)',
    cite: `@inproceedings{davari2019toponym,
  title      = {TIMBERT: Toponym Identifier For The Medical Domain Based on BERT},
  author     = {Davari, MohammadReza and Kosseim, Leila and Bui, Tien D},
  booktitle  = {Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020)},
  address    = {Barcelona, Spain (Online)},
  year       = {2020},
  month      = {December},
}`,
    visual: "timbert.gif",
  },
  {
    name: "Toponym Identification in Epidemiology Articles - A Deep Learning Approach",
    descr: "When analyzing the spread of viruses, epidemiologists often need to identify the location of infected hosts. This information can be found in public databases, such as GenBank, however, information provided in these databases are usually limited to the country or state level. More fine-grained localization information requires phylogeographers to manually read relevant scientific articles. In this work we propose an approach to automate the process of place name identification from medical (epidemiology) articles. The focus of this paper is to propose a deep learning based model for toponym detection and experiment with the use of external linguistic features and domain specific information. The model was evaluated using a collection of 105 epidemiology articles from PubMed Central provided by the recent SemEval task 12. Our best detection model achieves an F1 score of 80.13%, a significant improvement compared to the state of the art of 69.84%. These results underline the importance of domain specific embedding as well as specific linguistic features in toponym detection in medical journals.",
    paper: "https://arxiv.org/abs/1904.11018",
    published:
      'MohammdReza Davari, Leila Kosseim and Tien Bui (<a href="http://www.cicling.org/2019/">CICLing 2019</a>)',
    award: "Best Poster Award",
    cite: `@inproceedings{davari2019toponym,
  title      = {Toponym Identification in Epidemiology Articles - A Deep Learning Approach},
  author     = {Davari, MohammadReza and Kosseim, Leila and Bui, Tien D},
  booktitle  = {Proceedings of The 20th International Conference on Computational Linguistics and Intelligent Text Processing (CICLing 2019)},
  address    = {La Rochelle, France},
  year       = {2019},
  month      = {April},
}`,
    visual: "toponym.gif",
  },
];
