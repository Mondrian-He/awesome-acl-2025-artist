# ACL 2025 Long Papers


**Summary:** 1599 papers with extracted content:
- 📊 Total images: 13877
- 📋 Total tables: 16805
- 📄 Total files: 3068
---

# ACL 2025 Long Papers - Part 17 of 50

## 目录 (Table of Contents)

1. [Praetor: A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria](#Praetor-A-Fine-Grained-GenerativeLLMEvaluator-with-Instance-Level-Customizable-Evaluation-Criteria)
2. [Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking](#Mitigating-Confounding-in-Speech-Based-Dementia-Detection-through-Weight-Masking)
3. [MCS-Bench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Classical Studies](#MCS-Bench-A-Comprehensive-Benchmark-for-Evaluating-Multimodal-Large-Language-Models-inChinese-Classical-Studies)
4. [The Knowledge Microscope: Features as Better Analytical Lenses than Neurons](#The-Knowledge-Microscope-Features-as-Better-Analytical-Lenses-than-Neurons)
5. [From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding](#From-Real-to-Synthetic-Synthesizing-Millions-of-Diversified-and-Complicated-User-Instructions-with-Attributed-Grounding)
6. [PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance](#PrivaCI-Bench-Evaluating-Privacy-with-Contextual-Integrity-and-Legal-Compliance)
7. [Unveiling Environmental Impacts of Large Language Model Serving: A Functional Unit View](#Unveiling-Environmental-Impacts-of-Large-Language-Model-Serving-A-Functional-Unit-View)
8. [ExpeTrans:LLMs Are Experiential Transfer Learners](#ExpeTransLLMs-Are-Experiential-Transfer-Learners)
9. [Cool-Fusion: Fuse Large Language Models without Training](#Cool-Fusion-Fuse-Large-Language-Models-without-Training)
10. [DAPEV2: Process Attention Score as Feature Map for Length Extrapolation](#DAPEV2-Process-Attention-Score-as-Feature-Map-for-Length-Extrapolation)
11. [MuSC: Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training](#MuSC-Improving-Complex-Instruction-Following-with-Multi-granularity-Self-Contrastive-Training)
12. [LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](#LongReD-Mitigating-Short-Text-Degradation-of-Long-Context-Large-Language-Models-via-Restoration-Distillation)
13. [APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs](#APB-Accelerating-Distributed-Long-Context-Inference-by-Passing-Compressed-Context-Blocks-acrossGPUs)
14. [PPT: A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer](#PPT-A-Minor-Language-News-Recommendation-Model-via-Cross-Lingual-Preference-Pattern-Transfer)
15. [GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](#GainRAG-Preference-Alignment-in-Retrieval-Augmented-Generation-through-Gain-Signal-Synthesis)
16. [Top-n𝜎: Eliminating Noise in Logit Space for Robust Token Sampling ofLLM](#Top-n𝜎-Eliminating-Noise-in-Logit-Space-for-Robust-Token-Sampling-ofLLM)
17. [SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation](#SCOPE-Optimizing-Key-Value-Cache-Compression-in-Long-context-Generation)
18. [Mitigating Non-Representative Prototypes and Representation Bias in Few-Shot Continual Relation Extraction](#Mitigating-Non-Representative-Prototypes-and-Representation-Bias-in-Few-Shot-Continual-Relation-Extraction)
19. [MoQAE: Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware Experts](#MoQAE-Mixed-Precision-Quantization-for-Long-ContextLLMInference-via-Mixture-of-Quantization-Aware-Experts)
20. [PrivacyRestore: Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restoration](#PrivacyRestore-Privacy-Preserving-Inference-in-Large-Language-Models-via-Privacy-Removal-and-Restoration)
21. [Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](#Meta-rater-A-Multi-dimensional-Data-Selection-Method-for-Pre-training-Language-Models)
22. [GuessArena: Guess WhoIAm? A Self-Adaptive Framework for EvaluatingLLMs in Domain-Specific Knowledge and Reasoning](#GuessArena-Guess-WhoIAm-A-Self-Adaptive-Framework-for-EvaluatingLLMs-in-Domain-Specific-Knowledge-and-Reasoning)
23. [Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition](#Sample-Efficient-Human-Evaluation-of-Large-Language-Models-via-Maximum-Discrepancy-Competition)
24. [DTCRS: Dynamic Tree Construction for Recursive Summarization](#DTCRS-Dynamic-Tree-Construction-for-Recursive-Summarization)
25. [A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning](#A-Generative-Adaptive-Replay-Continual-Learning-Model-for-Temporal-Knowledge-Graph-Reasoning)
26. [ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](#ARise-Towards-Knowledge-Augmented-Reasoning-via-Risk-Adaptive-Search)
27. [PKAG-DDI: Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generation](#PKAG-DDI-Pairwise-Knowledge-Augmented-Language-Model-for-Drug-Drug-Interaction-Event-Text-Generation)
28. [Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Language Models](#Knowledge-Augmented-Multimodal-Clinical-Rationale-Generation-for-Disease-Diagnosis-with-Small-Language-Models)
29. [TWIST: Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models](#TWIST-Text-encoder-Weight-editing-for-Inserting-Secret-Trojans-in-Text-to-Image-Models)
30. [Frictional Agent Alignment Framework: Slow Down and Don’t Break Things](#Frictional-Agent-Alignment-Framework-Slow-Down-and-Dont-Break-Things)
31. [Powerformer: Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encryption](#Powerformer-Efficient-and-High-Accuracy-Privacy-Preserving-Language-Model-with-Homomorphic-Encryption)
32. [Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs](#Beware-of-Your-Po-Measuring-and-MitigatingAISafety-Risks-in-Role-Play-Fine-Tuning-ofLLMs)

---


## Praetor: A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria

### Images

![47d06712fd0c7c7ca28f043e5f968edbbd5ea164d9ce44706a7c959112ad3310.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/images/47d06712fd0c7c7ca28f043e5f968edbbd5ea164d9ce44706a7c959112ad3310.jpg)

![6664c56156c23617b98ad7bf5b4e315b2390b37bb55cfd4757344a50e3f496ed.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/images/6664c56156c23617b98ad7bf5b4e315b2390b37bb55cfd4757344a50e3f496ed.jpg)

![6b94bdb882e1646bbf6580ad2a63271947ffadb44aa9f8c8fa50090bbbe77095.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/images/6b94bdb882e1646bbf6580ad2a63271947ffadb44aa9f8c8fa50090bbbe77095.jpg)

![97dc6f5a54056c6fe21bde4186677f66b05d3ed5b96470725e9e986be1f0942e.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/images/97dc6f5a54056c6fe21bde4186677f66b05d3ed5b96470725e9e986be1f0942e.jpg)

### Tables

![03be25d55aaac7044ea1a9edcc75a7f5f195752ef0ab3e49db5963d8c11e5d77.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/03be25d55aaac7044ea1a9edcc75a7f5f195752ef0ab3e49db5963d8c11e5d77.jpg)

![1f98f493ca5de32348b96a7a5a9e19fea54f83049c4ec2b0a35e610ecdd06dfa.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/1f98f493ca5de32348b96a7a5a9e19fea54f83049c4ec2b0a35e610ecdd06dfa.jpg)

![230ce2ac3e881fe5cdb76ecbe747d93e5a59e124077396b48840b3eb90501fe5.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/230ce2ac3e881fe5cdb76ecbe747d93e5a59e124077396b48840b3eb90501fe5.jpg)

![36fda34c714871bb9c03da1543e5d3eb9fc4257cdb1bda05f27bb7f23758a824.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/36fda34c714871bb9c03da1543e5d3eb9fc4257cdb1bda05f27bb7f23758a824.jpg)

![37ed3d9de27648a4509e217c932ea189d905b9b4858acaf6c9213268caab12c0.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/37ed3d9de27648a4509e217c932ea189d905b9b4858acaf6c9213268caab12c0.jpg)

![3a7040e6c1009223ec62f01a3e95de0ee997a2d7eac5ee2ac2de0034e291ec17.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/3a7040e6c1009223ec62f01a3e95de0ee997a2d7eac5ee2ac2de0034e291ec17.jpg)

![3c15870545ea7782cb2ecf6eccd66830ceda1117bb0b7192fca6f4adfdd236dc.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/3c15870545ea7782cb2ecf6eccd66830ceda1117bb0b7192fca6f4adfdd236dc.jpg)

![413eea65c9a7f872a3b800f5a07ea98a1173607b5381c9223bc9443046c61ecd.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/413eea65c9a7f872a3b800f5a07ea98a1173607b5381c9223bc9443046c61ecd.jpg)

![42ad0a6e7b7d0117fb12bdbd17228eb8391e0636755abd732d0f7ced662339ec.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/42ad0a6e7b7d0117fb12bdbd17228eb8391e0636755abd732d0f7ced662339ec.jpg)

![472f01a80e95bcd99b4077eb43e7c811d5d289dbc4d6db93b1b679fcf85d7500.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/472f01a80e95bcd99b4077eb43e7c811d5d289dbc4d6db93b1b679fcf85d7500.jpg)

![4a38e4fa7ec6ca07c4aaddf69255fca396a6f98e80b32eaf11cc61d762a1ff36.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/4a38e4fa7ec6ca07c4aaddf69255fca396a6f98e80b32eaf11cc61d762a1ff36.jpg)

![528fcb73e2494e784b9e4bae34940b6b2ea05cf36bcaba07709f18ac3058927b.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/528fcb73e2494e784b9e4bae34940b6b2ea05cf36bcaba07709f18ac3058927b.jpg)

![552f0d658af3d7d5c862c266b282188caf0a57899d998e9d78d719e1ca9245ae.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/552f0d658af3d7d5c862c266b282188caf0a57899d998e9d78d719e1ca9245ae.jpg)

![5e50637e8bc328be9266aadd637c4049419c2a47acba6f20099e87344fb3fed1.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/5e50637e8bc328be9266aadd637c4049419c2a47acba6f20099e87344fb3fed1.jpg)

![6322110fda53bb9fc6c8d981c6a89535344674c7ddb7a08f7224bbd5ac5b1b4f.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/6322110fda53bb9fc6c8d981c6a89535344674c7ddb7a08f7224bbd5ac5b1b4f.jpg)

![63bab39cc75c4c3c829f131c50bce258a07828b351bd65a7807dd4413d8bedbe.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/63bab39cc75c4c3c829f131c50bce258a07828b351bd65a7807dd4413d8bedbe.jpg)

![71fd4363232e6766c86157a399474f6b1bb6acdc6e8f87522894de53a6311c36.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/71fd4363232e6766c86157a399474f6b1bb6acdc6e8f87522894de53a6311c36.jpg)

![9df1a46c0b1d10643ac9d2155983bb5745119803332997de99c24d0e60f31565.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/9df1a46c0b1d10643ac9d2155983bb5745119803332997de99c24d0e60f31565.jpg)

![a18e8ea366b5dd96c828505a45452e897b831af47ec2d3b94df50ba4b739d942.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/a18e8ea366b5dd96c828505a45452e897b831af47ec2d3b94df50ba4b739d942.jpg)

![a4551643ba3336aaabb1362539ff73ecd4635ea4895c03ccfda1fad63be6d4e7.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/a4551643ba3336aaabb1362539ff73ecd4635ea4895c03ccfda1fad63be6d4e7.jpg)

![b8a43cf89fe4e621f63dd9288501f4ecce76f28d77017f20afd359a797fd5f02.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/b8a43cf89fe4e621f63dd9288501f4ecce76f28d77017f20afd359a797fd5f02.jpg)

![d8a61a1990848e2df225d78cb583776488996662ee0480c3a04a3259a5f13b07.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/d8a61a1990848e2df225d78cb583776488996662ee0480c3a04a3259a5f13b07.jpg)

![f2485083a8140d4dc13b6ec6ec8f57c0d94fe56ff338a9211209c792b5d33fae.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/f2485083a8140d4dc13b6ec6ec8f57c0d94fe56ff338a9211209c792b5d33fae.jpg)

![fabc0b1ea0643bae864972c222925e91274de47392c02e5dfbb130b034b376cc.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/fabc0b1ea0643bae864972c222925e91274de47392c02e5dfbb130b034b376cc.jpg)

![fbb184964e2ae2e8bab792ec79bea07218c67c4733a97d4e6dc8f9634b37318b.jpg](acl_results/513_Praetor_ A Fine-Grained GenerativeLLMEvaluator with Instance-Level Customizable Evaluation Criteria/tables/fbb184964e2ae2e8bab792ec79bea07218c67c4733a97d4e6dc8f9634b37318b.jpg)

## Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking

### Images

![06effb9b0c0ac9de918b813607eeda87e296f51be5798e698eaf8356d7735303.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/06effb9b0c0ac9de918b813607eeda87e296f51be5798e698eaf8356d7735303.jpg)

![12d071b133885d84726b0d2298c61d0f6ea08918516ed652d09a06bd2cdd6845.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/12d071b133885d84726b0d2298c61d0f6ea08918516ed652d09a06bd2cdd6845.jpg)

![134770f8f2a072e8599865f6a4549a96ca4f227eba44e52d31b9dc7d93399135.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/134770f8f2a072e8599865f6a4549a96ca4f227eba44e52d31b9dc7d93399135.jpg)

![171f6898fbfa098bcb747e3ac8f1da6bdf62cc56dc7b7ebde874b4a9b68fa4e8.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/171f6898fbfa098bcb747e3ac8f1da6bdf62cc56dc7b7ebde874b4a9b68fa4e8.jpg)

![36ff81fc79177ad8c47128c06ac524583015cc0d17709c5dcc839c4f1d5e3320.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/36ff81fc79177ad8c47128c06ac524583015cc0d17709c5dcc839c4f1d5e3320.jpg)

![3f0287ac81a7472e61b910c12dc23c7ca422d823352ed0c641070c9ac861dc0f.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/3f0287ac81a7472e61b910c12dc23c7ca422d823352ed0c641070c9ac861dc0f.jpg)

![4c785678c8a026f27bfc99dc858c506e31349b931ce577c0bb5f854ee1ce0899.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/4c785678c8a026f27bfc99dc858c506e31349b931ce577c0bb5f854ee1ce0899.jpg)

![624a19dede4f22efc56fcfebed25e606cc9f6d043986ad3c75460c29d642eacd.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/624a19dede4f22efc56fcfebed25e606cc9f6d043986ad3c75460c29d642eacd.jpg)

![763dfe165003afddc3d703a81ca4ed782f52d0f34f941af79bc03d96072b26d8.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/763dfe165003afddc3d703a81ca4ed782f52d0f34f941af79bc03d96072b26d8.jpg)

![7e84bae3e6db7194c8ee4d5d555b049c19398bee13899e7b8ae45c4982b75de9.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/7e84bae3e6db7194c8ee4d5d555b049c19398bee13899e7b8ae45c4982b75de9.jpg)

![86a1de7a252b44e3fd2565dc6f53f8ed1ba5afb6d306188ca2f4ccaba12823cb.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/86a1de7a252b44e3fd2565dc6f53f8ed1ba5afb6d306188ca2f4ccaba12823cb.jpg)

![8b3f9887d8f4d52e61792a83f28174babde994cea7046a5515c22d5307e4bcfe.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/8b3f9887d8f4d52e61792a83f28174babde994cea7046a5515c22d5307e4bcfe.jpg)

![a1500b94734bcda29df174f1814df26d1d8af4d874aaa7e6c5b62be6082648c7.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/a1500b94734bcda29df174f1814df26d1d8af4d874aaa7e6c5b62be6082648c7.jpg)

![adef4ca29742e631741e7a676846ea39d8965516118a73544aa002e48aa4398f.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/adef4ca29742e631741e7a676846ea39d8965516118a73544aa002e48aa4398f.jpg)

![ae2e8c4b5a8130528920b1b3026c4b7e2b3919aed5cca5dff1b7ee3f28431bf3.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/ae2e8c4b5a8130528920b1b3026c4b7e2b3919aed5cca5dff1b7ee3f28431bf3.jpg)

![b222b86f6fee95a27ee36776ed5650d5ae1aba1fe5040fe0f05f278717f010dd.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/b222b86f6fee95a27ee36776ed5650d5ae1aba1fe5040fe0f05f278717f010dd.jpg)

![d96e148d4aae887906e3f5097f6a67537901818208743e5b9e15f79926fd3315.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/d96e148d4aae887906e3f5097f6a67537901818208743e5b9e15f79926fd3315.jpg)

![dc8dc4baa498e805db9da8741ebbc97e5c875d83dcb74d021942af29fab8755e.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/dc8dc4baa498e805db9da8741ebbc97e5c875d83dcb74d021942af29fab8755e.jpg)

![e8e327d5cd6757021034be13f52deca7671d166c53dd90fa37a91c0db180ecb5.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/e8e327d5cd6757021034be13f52deca7671d166c53dd90fa37a91c0db180ecb5.jpg)

![ee4933d7cac7f0217269730e0649fe941d83ae826cfdebabf6e80d7f069e4cef.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/images/ee4933d7cac7f0217269730e0649fe941d83ae826cfdebabf6e80d7f069e4cef.jpg)

### Tables

![01ae1af59ef1c685543485c4657f1bd214321a01057854b9427854889f08ccc9.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/tables/01ae1af59ef1c685543485c4657f1bd214321a01057854b9427854889f08ccc9.jpg)

![b9904dd7b33f9e54dd9cf63d7671a312fc6d4faa7a1f5cee9eafdfef3854a6f1.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/tables/b9904dd7b33f9e54dd9cf63d7671a312fc6d4faa7a1f5cee9eafdfef3854a6f1.jpg)

![f7a1623875da5f1584cbefbd54984aec98f806248241189149fa62251109307f.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/tables/f7a1623875da5f1584cbefbd54984aec98f806248241189149fa62251109307f.jpg)

![fff0fbad91758c8c7e9011ccfa9292dd122c80ddd323539efd91f05e962964b4.jpg](acl_results/514_Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking/tables/fff0fbad91758c8c7e9011ccfa9292dd122c80ddd323539efd91f05e962964b4.jpg)

## MCS-Bench: A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Classical Studies

### Images

![02b743726146f5ddb8a355f46694935d03e09ed8cade4ce4e856fefd14e7d50e.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/02b743726146f5ddb8a355f46694935d03e09ed8cade4ce4e856fefd14e7d50e.jpg)

![0a53f630b6947dfc3854cb31d7eea7434dcb243a0742b20dedf0e030535e0de4.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/0a53f630b6947dfc3854cb31d7eea7434dcb243a0742b20dedf0e030535e0de4.jpg)

![189ff9f307f18d5ffe95277e0531e8f85d95ed5de4f5fbc4ebf5388eddad1760.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/189ff9f307f18d5ffe95277e0531e8f85d95ed5de4f5fbc4ebf5388eddad1760.jpg)

![286636df55f4c6fcd43c40240fff67506cf2401cab76163134dbd29b873acae8.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/286636df55f4c6fcd43c40240fff67506cf2401cab76163134dbd29b873acae8.jpg)

![2defb2904e9cc3bd2a40c3be4042088baf8041430052bfc24c1badab79f569fb.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/2defb2904e9cc3bd2a40c3be4042088baf8041430052bfc24c1badab79f569fb.jpg)

![32fd1212eb895db4afeef4a464ff919ac8d96d4ace3baefaf233e3a8c798e0b8.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/32fd1212eb895db4afeef4a464ff919ac8d96d4ace3baefaf233e3a8c798e0b8.jpg)

![3a78139307dbcae2651353f068d86d4c520f8dca4f790c6c688bb67791a57780.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/3a78139307dbcae2651353f068d86d4c520f8dca4f790c6c688bb67791a57780.jpg)

![40af8cd01184c8300fcf32b8bb94dba98087846077d5bab7438603f437a9e1b2.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/40af8cd01184c8300fcf32b8bb94dba98087846077d5bab7438603f437a9e1b2.jpg)

![54e0cc4f6403b8edfc28c853261788be53d31a072809bc2f3387442f2dc6415a.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/54e0cc4f6403b8edfc28c853261788be53d31a072809bc2f3387442f2dc6415a.jpg)

![6242e71f0173294b6b847716c873d80fd9713eb433fe88a8c266ecd6d87d698e.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/6242e71f0173294b6b847716c873d80fd9713eb433fe88a8c266ecd6d87d698e.jpg)

![65abffc64121ee12e455a49cba18a2d11350fd3c966ef95e466a56682830bfdb.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/65abffc64121ee12e455a49cba18a2d11350fd3c966ef95e466a56682830bfdb.jpg)

![6708d135b9fc073d5e865bfafebf7fbebd577ac004c968b09e63332f077bb7e9.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/6708d135b9fc073d5e865bfafebf7fbebd577ac004c968b09e63332f077bb7e9.jpg)

![67223f76ad986b8d77321259b66ec98975bf855198286b3df098fb96cbe79283.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/67223f76ad986b8d77321259b66ec98975bf855198286b3df098fb96cbe79283.jpg)

![6c45e5d4ff54ad87d741af24467a9cc5e5a4bf1082bb703ca8dbd988e870cf39.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/6c45e5d4ff54ad87d741af24467a9cc5e5a4bf1082bb703ca8dbd988e870cf39.jpg)

![6ceee466600e1048992affe427de014e6338fd9d088d1faf40175774904d6949.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/6ceee466600e1048992affe427de014e6338fd9d088d1faf40175774904d6949.jpg)

![6e3fdb6f7d6050d082175986f20dd12ef481219bb1d7ea368b8f2826a481c4f5.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/6e3fdb6f7d6050d082175986f20dd12ef481219bb1d7ea368b8f2826a481c4f5.jpg)

![6e66a2e0dd3cdffd1151fd30375acd3b84c27c54d16c228c5b94e3ca6d200d5d.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/6e66a2e0dd3cdffd1151fd30375acd3b84c27c54d16c228c5b94e3ca6d200d5d.jpg)

![7014ad78a3f1916ac9d35f426c94d35120f355504924e7381ad1844257abf2d7.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/7014ad78a3f1916ac9d35f426c94d35120f355504924e7381ad1844257abf2d7.jpg)

![768639d5e68a12e80e22d3a172052a38c06835b4da16bb063baa56db9a18aa6d.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/768639d5e68a12e80e22d3a172052a38c06835b4da16bb063baa56db9a18aa6d.jpg)

![874aa24e555a138133cab9f6a22b0744ee80c06156a8964a36bfe4aadc96160c.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/874aa24e555a138133cab9f6a22b0744ee80c06156a8964a36bfe4aadc96160c.jpg)

![8d0efa5f7e1098bab93854ca0d8ab9e689a0d0fac712a1ebcd69df032278f7f3.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/8d0efa5f7e1098bab93854ca0d8ab9e689a0d0fac712a1ebcd69df032278f7f3.jpg)

![91e0fb387c30bb887a3513acf5b34a29b955fa9935e51d6ba0b8e9674cac4dee.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/91e0fb387c30bb887a3513acf5b34a29b955fa9935e51d6ba0b8e9674cac4dee.jpg)

![96581b4f652e5012963fc91482f1e00543a4b8608b15f9c7ade81c35b4363d1d.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/96581b4f652e5012963fc91482f1e00543a4b8608b15f9c7ade81c35b4363d1d.jpg)

![9b97dc7cb6e65d0b517c48dff033f1c2f668e9782c97bb669efb3382e4079ed2.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/9b97dc7cb6e65d0b517c48dff033f1c2f668e9782c97bb669efb3382e4079ed2.jpg)

![ab93738eb0ab9bc57b54d5a72d28b86875103103b5c3f4508aa9cad44e87d33a.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/ab93738eb0ab9bc57b54d5a72d28b86875103103b5c3f4508aa9cad44e87d33a.jpg)

![af9d27b04d857c3cceda13380f51d1e5703e102f2a87449eeafba4c814cf7df9.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/af9d27b04d857c3cceda13380f51d1e5703e102f2a87449eeafba4c814cf7df9.jpg)

![b33505e84c2f8494a7f3dc1fb730ff91a090b5b945ff6b58632a51a572d12a64.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/b33505e84c2f8494a7f3dc1fb730ff91a090b5b945ff6b58632a51a572d12a64.jpg)

![b96cd7dc304ef564bcc8f7e38983628f49415e00f9ab484c7db0cdaa53428dd0.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/b96cd7dc304ef564bcc8f7e38983628f49415e00f9ab484c7db0cdaa53428dd0.jpg)

![b99fc5bde1cbb56cb3695071398352ac9e5598bd5dafd467598bb479b540a04f.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/b99fc5bde1cbb56cb3695071398352ac9e5598bd5dafd467598bb479b540a04f.jpg)

![bc584b1adc082df166233d98faab61bec6530cd7b4913937abe517e4291ab210.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/bc584b1adc082df166233d98faab61bec6530cd7b4913937abe517e4291ab210.jpg)

![d187906a1024cfa4c810f3ac370e9a99bdf5cd1b349bb4488250e08e6d51106c.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/d187906a1024cfa4c810f3ac370e9a99bdf5cd1b349bb4488250e08e6d51106c.jpg)

![eade05bd1fb1e1ee0cb770a033dfb29a222efcf5d93a0e2ba15d5c942aa61cac.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/eade05bd1fb1e1ee0cb770a033dfb29a222efcf5d93a0e2ba15d5c942aa61cac.jpg)

![ec3379dc5668a7d7970a95007ac46ca0468620cb13a7a314bd46dc145eec524e.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/ec3379dc5668a7d7970a95007ac46ca0468620cb13a7a314bd46dc145eec524e.jpg)

![f4f11f135f5d6d079aed1b174a3031d1c561ebf29ebf7f20b80e20d60dc754ba.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/f4f11f135f5d6d079aed1b174a3031d1c561ebf29ebf7f20b80e20d60dc754ba.jpg)

![fa70a3f601ab8c5bd5275c5b844a6f43c7db85820cf3ef60b54667c8e23610d2.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/fa70a3f601ab8c5bd5275c5b844a6f43c7db85820cf3ef60b54667c8e23610d2.jpg)

![fc93edd0ca3f8ce5d692da7ca2153abb51761721614e4a8919f2961ad7be4b40.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/fc93edd0ca3f8ce5d692da7ca2153abb51761721614e4a8919f2961ad7be4b40.jpg)

![feb64ace0efcb1c7ff722c77ee683dc8b4de9d6dd56bae09e5f0efb530da79fe.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/images/feb64ace0efcb1c7ff722c77ee683dc8b4de9d6dd56bae09e5f0efb530da79fe.jpg)

### Tables

![0188cbefac6123b043a41e00a2c7856d8a7c4ee350121882df16c634af5d3b29.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/0188cbefac6123b043a41e00a2c7856d8a7c4ee350121882df16c634af5d3b29.jpg)

![0633b88da09eaa6b873d866749cf67a142fe9ae7613a3bb2f2f907534a97c2bd.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/0633b88da09eaa6b873d866749cf67a142fe9ae7613a3bb2f2f907534a97c2bd.jpg)

![08b34b917f0e8b846f33425ef1f3f80c5221f1bdda5dab2725ae3d73c0d56f6e.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/08b34b917f0e8b846f33425ef1f3f80c5221f1bdda5dab2725ae3d73c0d56f6e.jpg)

![0ea6a8abc2c99d7ebb475707d0ba9165eaaca1065114d24cb9534c024178ef7d.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/0ea6a8abc2c99d7ebb475707d0ba9165eaaca1065114d24cb9534c024178ef7d.jpg)

![16d9de62fa40820b5c596638d751f2c3722c806afbffc5cb2123598ecba64e9f.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/16d9de62fa40820b5c596638d751f2c3722c806afbffc5cb2123598ecba64e9f.jpg)

![20d2bd53b58bce682920d8b86bd5cff6629632c0dbd7f052637862370074881c.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/20d2bd53b58bce682920d8b86bd5cff6629632c0dbd7f052637862370074881c.jpg)

![2d3262e8fc8da7ca3a5136da6ef20ff9197efbf4983455dda946629ed1126e05.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/2d3262e8fc8da7ca3a5136da6ef20ff9197efbf4983455dda946629ed1126e05.jpg)

![2ef3d6c2c29c2cbfb8d3753238eca91810b40354f3fb69ed459ae7128ed73474.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/2ef3d6c2c29c2cbfb8d3753238eca91810b40354f3fb69ed459ae7128ed73474.jpg)

![4338339cefad56c34e1eb0c8ee5bde26b5c410315836cbd781d5a2406cb81f5e.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/4338339cefad56c34e1eb0c8ee5bde26b5c410315836cbd781d5a2406cb81f5e.jpg)

![4523ea9b2ca21300839ea3570ef1e3beb99e6524cf188e732bced3818016df49.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/4523ea9b2ca21300839ea3570ef1e3beb99e6524cf188e732bced3818016df49.jpg)

![568e316f4e74316b241babb38c46465b30b7646f0b241381baf522450a281de0.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/568e316f4e74316b241babb38c46465b30b7646f0b241381baf522450a281de0.jpg)

![7dee5586aa95d13b3db862e746e0f9af4e66636d2f975bc5d91fe0ae2eafd69d.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/7dee5586aa95d13b3db862e746e0f9af4e66636d2f975bc5d91fe0ae2eafd69d.jpg)

![7ee692f87f9fedc9115961b700eebc457495aeca614e7d97929d45d5eaf63a7c.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/7ee692f87f9fedc9115961b700eebc457495aeca614e7d97929d45d5eaf63a7c.jpg)

![807a6bba1c5ea898c223afc4ee601cceca9b791fae2db69cb83f3a3786ca405b.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/807a6bba1c5ea898c223afc4ee601cceca9b791fae2db69cb83f3a3786ca405b.jpg)

![89707ab4a2de9270592ed427c8085854c05e1e9c23c917ce8ca5c6199ffcc3cd.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/89707ab4a2de9270592ed427c8085854c05e1e9c23c917ce8ca5c6199ffcc3cd.jpg)

![91e1b65b39057f4859009ac83b5de1f62bd0e835639013919427daababd235b3.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/91e1b65b39057f4859009ac83b5de1f62bd0e835639013919427daababd235b3.jpg)

![a7e360919848a9c64f24bc37a36a33e00e6fb4824e9e3e93a2d0ce4d87e91a76.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/a7e360919848a9c64f24bc37a36a33e00e6fb4824e9e3e93a2d0ce4d87e91a76.jpg)

![ad32c18821f1d17368503162771683f23c6043a9ce547d61e907a266c2393663.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/ad32c18821f1d17368503162771683f23c6043a9ce547d61e907a266c2393663.jpg)

![bd7f4cc3e71bbbea0d027342413cf1408485c1d4cadc2c4fe497bc8830cb30f7.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/bd7f4cc3e71bbbea0d027342413cf1408485c1d4cadc2c4fe497bc8830cb30f7.jpg)

![c1d5407af8b17f3a3cb79dfec74787a00545455301ca3af9e9a26ffa37c3399d.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/c1d5407af8b17f3a3cb79dfec74787a00545455301ca3af9e9a26ffa37c3399d.jpg)

![c4f1dda865f848b252c177c95c306d9f91ceba6e9610153c3e4ef854237bb341.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/c4f1dda865f848b252c177c95c306d9f91ceba6e9610153c3e4ef854237bb341.jpg)

![dd1317028b494f701de6fd76520e441eb7ea48d038bc52147104e2e20b0da55e.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/dd1317028b494f701de6fd76520e441eb7ea48d038bc52147104e2e20b0da55e.jpg)

![e2fe049e5672a6f4232e992d7d0ade198f5ae1569c8862f2c169b19a10005010.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/e2fe049e5672a6f4232e992d7d0ade198f5ae1569c8862f2c169b19a10005010.jpg)

![f0b7b43db13e6be20e793d567ebfe6767bedd40e9b36a019d66713ea27eb52eb.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/f0b7b43db13e6be20e793d567ebfe6767bedd40e9b36a019d66713ea27eb52eb.jpg)

![fc4f977a504d04f2ce5cab6c359d88e4771df6b079fe91c67588db7f527918e3.jpg](acl_results/515_MCS-Bench_ A Comprehensive Benchmark for Evaluating Multimodal Large Language Models inChinese Class/tables/fc4f977a504d04f2ce5cab6c359d88e4771df6b079fe91c67588db7f527918e3.jpg)

## The Knowledge Microscope: Features as Better Analytical Lenses than Neurons

### Images

![116bf69eff7bd9ec99059dc232e42d3ae415ae3acf8cfc0136957953e51050f9.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/images/116bf69eff7bd9ec99059dc232e42d3ae415ae3acf8cfc0136957953e51050f9.jpg)

![1769b29ca5738bd7019a1e332e96ded9224f06863797573d550abb3c68674906.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/images/1769b29ca5738bd7019a1e332e96ded9224f06863797573d550abb3c68674906.jpg)

![24cf590b6a7f5a22a9353042aa41df2a833f9c304f389a68bbab49f6415cddd1.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/images/24cf590b6a7f5a22a9353042aa41df2a833f9c304f389a68bbab49f6415cddd1.jpg)

![562239e58afb578efb929f8f6790cad4dd0086823e6229069cdc811e080fcaa2.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/images/562239e58afb578efb929f8f6790cad4dd0086823e6229069cdc811e080fcaa2.jpg)

![7f9766d78bd0be94ff7d2fd242b5c1f8211e7e02cc8fef49e94cd0a5b3f51bf5.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/images/7f9766d78bd0be94ff7d2fd242b5c1f8211e7e02cc8fef49e94cd0a5b3f51bf5.jpg)

![884af1da7a995f417d3ad7867339e7847ada3f2c6879afddb55cb52305c5393a.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/images/884af1da7a995f417d3ad7867339e7847ada3f2c6879afddb55cb52305c5393a.jpg)

![a9652868cbb344b9d47fe5a57b6f24bfa5b14b668a2b5c1342025be850e487ab.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/images/a9652868cbb344b9d47fe5a57b6f24bfa5b14b668a2b5c1342025be850e487ab.jpg)

![b8931682e1d5e30fa616f7f5bda26bfa58bb134a2816f089717fed7377f0acae.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/images/b8931682e1d5e30fa616f7f5bda26bfa58bb134a2816f089717fed7377f0acae.jpg)

![be05a441df63bce1d560fee779fb05a626bfeee08b508ef93eb3dda841dcd141.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/images/be05a441df63bce1d560fee779fb05a626bfeee08b508ef93eb3dda841dcd141.jpg)

![cd6b83e0713b604a37de729916d63c13535654ae424c236edfb77086ca604bf1.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/images/cd6b83e0713b604a37de729916d63c13535654ae424c236edfb77086ca604bf1.jpg)

![d1ad764be996391f189cbaf4a22e74bfd475c862e54f967e95ba5bca54e7bd2d.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/images/d1ad764be996391f189cbaf4a22e74bfd475c862e54f967e95ba5bca54e7bd2d.jpg)

### Tables

![14700a2c8bc608f0ca5ad8886d9e86dde93edd562ce61a8a984aa181610bb8d3.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/tables/14700a2c8bc608f0ca5ad8886d9e86dde93edd562ce61a8a984aa181610bb8d3.jpg)

![4f8f9e1424b6b4a8084154621cd4f84ecb6b913a87d347faef75d58053e24337.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/tables/4f8f9e1424b6b4a8084154621cd4f84ecb6b913a87d347faef75d58053e24337.jpg)

![67c8032b488ff3b4bd630a0daa56e6e3606c9388cdbca6ea955ac5de6adbe21c.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/tables/67c8032b488ff3b4bd630a0daa56e6e3606c9388cdbca6ea955ac5de6adbe21c.jpg)

![6b41c69abf1b4edb6a3a787841de66cce7cf0eaf40e3c7f888b38e9d2fc652a6.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/tables/6b41c69abf1b4edb6a3a787841de66cce7cf0eaf40e3c7f888b38e9d2fc652a6.jpg)

![a3aa178118cb78f26c4cbb36c44b2de9ccf7fae5687886712b888c75c84a1656.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/tables/a3aa178118cb78f26c4cbb36c44b2de9ccf7fae5687886712b888c75c84a1656.jpg)

![a6f3a9e79fc1e03b83b642fcf216e0cac55fba9283dbee5d30a6e7fbe59994cd.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/tables/a6f3a9e79fc1e03b83b642fcf216e0cac55fba9283dbee5d30a6e7fbe59994cd.jpg)

![c635aec83158a0ce050f2629ee75321b8c3912544798bad018b3c6681d18f514.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/tables/c635aec83158a0ce050f2629ee75321b8c3912544798bad018b3c6681d18f514.jpg)

![c83b42c1365efa110ba08995131cdd22c0022f89276d546db26f2b10c515d25d.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/tables/c83b42c1365efa110ba08995131cdd22c0022f89276d546db26f2b10c515d25d.jpg)

![f449246759a2af210dadbbee6e85b550ed7b15d8083ef12d407e09e09c83f80b.jpg](acl_results/516_The Knowledge Microscope_ Features as Better Analytical Lenses than Neurons/tables/f449246759a2af210dadbbee6e85b550ed7b15d8083ef12d407e09e09c83f80b.jpg)

## From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding

### Images

![0fc576203e46d675183e9eb3b2c9f88a3f3d700d28184ef09cba79ace16cde4d.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /images/0fc576203e46d675183e9eb3b2c9f88a3f3d700d28184ef09cba79ace16cde4d.jpg)

![2982797989400fffc3a9bacf2b68a1f732ac4fd816519b3f43dcdcc12eab0213.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /images/2982797989400fffc3a9bacf2b68a1f732ac4fd816519b3f43dcdcc12eab0213.jpg)

![5fb024c1ed77ba1f74ab4a341bcfd7d11e4c3da0bd1b6638ecdcae8037aa0bf6.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /images/5fb024c1ed77ba1f74ab4a341bcfd7d11e4c3da0bd1b6638ecdcae8037aa0bf6.jpg)

![6f9eee2793eeba24a27e377e9b472f358ea4a8095819b492c097a285a7767228.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /images/6f9eee2793eeba24a27e377e9b472f358ea4a8095819b492c097a285a7767228.jpg)

![7d3963d3b5abee435897fbb116f8017a4d460e50a1c0e5397ecd00e25d79ed1c.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /images/7d3963d3b5abee435897fbb116f8017a4d460e50a1c0e5397ecd00e25d79ed1c.jpg)

![aaefa6bfc5d1e9903fbdea71c0967c45676a82f31d95b25df3181e3a4494fb98.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /images/aaefa6bfc5d1e9903fbdea71c0967c45676a82f31d95b25df3181e3a4494fb98.jpg)

![c4db3809725c602f718d77ddcf9a538769083fe6d3713886911be74851b53d9c.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /images/c4db3809725c602f718d77ddcf9a538769083fe6d3713886911be74851b53d9c.jpg)

![c72be9e2e422ee2a493602ab4fce4d4020de29fbe93cc9fcdeccd7190f8ff8a1.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /images/c72be9e2e422ee2a493602ab4fce4d4020de29fbe93cc9fcdeccd7190f8ff8a1.jpg)

![ce7231280c92d9dfc89cf8c0820780710a5c20425fffc607407e063fda6720c4.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /images/ce7231280c92d9dfc89cf8c0820780710a5c20425fffc607407e063fda6720c4.jpg)

### Tables

![02297877a4013a4c5f64cb14dd68f870b04e0364336dd481451dcd479e673fc9.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/02297877a4013a4c5f64cb14dd68f870b04e0364336dd481451dcd479e673fc9.jpg)

![02925e97721bda79348e55965c65f4bd6b3cbeab8265af7faf7900b1a4a92bd8.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/02925e97721bda79348e55965c65f4bd6b3cbeab8265af7faf7900b1a4a92bd8.jpg)

![1cf41b0113beaa8c293ba1247be9b3e774efc489b536f63426cfb1cb7be7b6a2.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/1cf41b0113beaa8c293ba1247be9b3e774efc489b536f63426cfb1cb7be7b6a2.jpg)

![1d77100cd03fb12e1e192c27f7d51d7c4005ef8d2b913b1ced5ae2c451f79172.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/1d77100cd03fb12e1e192c27f7d51d7c4005ef8d2b913b1ced5ae2c451f79172.jpg)

![2dc845202377528cc8c82cacd3cdd6ffbe0006b1be0f16ea9cede83e1447de03.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/2dc845202377528cc8c82cacd3cdd6ffbe0006b1be0f16ea9cede83e1447de03.jpg)

![47a9bd071530cbc87607d2bc0eaf4b6272300ee303c8f89c7f2208c2b3a90062.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/47a9bd071530cbc87607d2bc0eaf4b6272300ee303c8f89c7f2208c2b3a90062.jpg)

![6c11fcd6eb1ad612a0f4fe90feff18d6a4298fcd7369e0c859e5437799d96fc7.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/6c11fcd6eb1ad612a0f4fe90feff18d6a4298fcd7369e0c859e5437799d96fc7.jpg)

![6dfc81dbba37d6ab6f2e32c01defc8f3f6b953f61336b3bf654f56a46ceb9c7e.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/6dfc81dbba37d6ab6f2e32c01defc8f3f6b953f61336b3bf654f56a46ceb9c7e.jpg)

![7dd2ea7aea19acb953fb5fe121ce866d0feb684bcb35ebd7c76e9fccae17e15a.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/7dd2ea7aea19acb953fb5fe121ce866d0feb684bcb35ebd7c76e9fccae17e15a.jpg)

![8c7031d41e1324970584d41a5103f6dffa240a0cc3204fc6ff8be17b30ee6fd1.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/8c7031d41e1324970584d41a5103f6dffa240a0cc3204fc6ff8be17b30ee6fd1.jpg)

![af45f0e55a5eb721ce09a7dbc07a0dd32031fcbf6cd36b8a7dc59e900a3d0679.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/af45f0e55a5eb721ce09a7dbc07a0dd32031fcbf6cd36b8a7dc59e900a3d0679.jpg)

![bbe7159fb32c04a815b0bb353ca7fdd972beadda9f09e013ec7f75cb66032048.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/bbe7159fb32c04a815b0bb353ca7fdd972beadda9f09e013ec7f75cb66032048.jpg)

![ca5c845f2d631dd884f7a4cebba403d76ecebbcaba9805aec6692c018a7a7c2a.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/ca5c845f2d631dd884f7a4cebba403d76ecebbcaba9805aec6692c018a7a7c2a.jpg)

![d657e91e259939e4318dfcecbbda97ef7eecf91438f6a281902f47583a7fc755.jpg](acl_results/517_From Real to Synthetic_ Synthesizing Millions of Diversified and Complicated User Instructions with /tables/d657e91e259939e4318dfcecbbda97ef7eecf91438f6a281902f47583a7fc755.jpg)

## PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance

### Images

![3c5c826cbcb898f9ab21fdec65ba23dbd3fcc5500f28d1854e627cba19653ced.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/images/3c5c826cbcb898f9ab21fdec65ba23dbd3fcc5500f28d1854e627cba19653ced.jpg)

![ba97e2bc44bb28b459e74f2a51f9d7a7b0292266b0adf9da48a9d0142f83e8be.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/images/ba97e2bc44bb28b459e74f2a51f9d7a7b0292266b0adf9da48a9d0142f83e8be.jpg)

### Tables

![016e50ec8a8779df7b91051d5fed18684dd480b7fa5de0fc188ff728d2d4de91.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/016e50ec8a8779df7b91051d5fed18684dd480b7fa5de0fc188ff728d2d4de91.jpg)

![0ad1ce0c343be60cb4b9827d14a76495b6d0c5d522d4a62ea7a99f000f29a45b.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/0ad1ce0c343be60cb4b9827d14a76495b6d0c5d522d4a62ea7a99f000f29a45b.jpg)

![1b44f22fab185a13ff4f239ab289d4e47ee08603bb18f2823082600366cf8096.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/1b44f22fab185a13ff4f239ab289d4e47ee08603bb18f2823082600366cf8096.jpg)

![2edcd5fa09323b751513efcec2b9793cd7beb3c8246e4a22bf614b7dd6a3becc.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/2edcd5fa09323b751513efcec2b9793cd7beb3c8246e4a22bf614b7dd6a3becc.jpg)

![51953299b183d389e93da3be6278534d9b1dd9892c1bbad6c55d790c7767077f.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/51953299b183d389e93da3be6278534d9b1dd9892c1bbad6c55d790c7767077f.jpg)

![66f3ebb5e6883b1f6beaa1a3d079b64900185a3ac424baaf4e10b35c29770e37.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/66f3ebb5e6883b1f6beaa1a3d079b64900185a3ac424baaf4e10b35c29770e37.jpg)

![6b3f3d7fa2a148769bfe114e926e0f04f6c3fc9868b0abf5692f83c6c96b4c06.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/6b3f3d7fa2a148769bfe114e926e0f04f6c3fc9868b0abf5692f83c6c96b4c06.jpg)

![822eba2e5e974b3b03d7f70564b5aeda877894bfe5830ab47edbc97bc58d9334.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/822eba2e5e974b3b03d7f70564b5aeda877894bfe5830ab47edbc97bc58d9334.jpg)

![946045974e04e01886633b783dcf51355e91df5abe9aa11e66dbe8b5bd006cf4.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/946045974e04e01886633b783dcf51355e91df5abe9aa11e66dbe8b5bd006cf4.jpg)

![ad5a802a09477d1b9e2a61041aab2615388d31ed8685a640903ddc1ffce70d4a.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/ad5a802a09477d1b9e2a61041aab2615388d31ed8685a640903ddc1ffce70d4a.jpg)

![c3d2a1c19c9d647dead2a5c12d4e15a606e062e61f9d032b06653a1c3486e490.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/c3d2a1c19c9d647dead2a5c12d4e15a606e062e61f9d032b06653a1c3486e490.jpg)

![cfca56b2ab3ad1d5f16580de250e8c40bdd34a7246ef221a24f2aea1156155e7.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/cfca56b2ab3ad1d5f16580de250e8c40bdd34a7246ef221a24f2aea1156155e7.jpg)

![f8c66432629e6b6d3015db5c50e62db48123bd6406856eaf49e3f9d2d0194d61.jpg](acl_results/518_PrivaCI-Bench_ Evaluating Privacy with Contextual Integrity and Legal Compliance/tables/f8c66432629e6b6d3015db5c50e62db48123bd6406856eaf49e3f9d2d0194d61.jpg)

## Unveiling Environmental Impacts of Large Language Model Serving: A Functional Unit View

### Images

![0834712d71dc2614ffed7d03ff78104b58d776fc8bd4156e478350399af652d8.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/0834712d71dc2614ffed7d03ff78104b58d776fc8bd4156e478350399af652d8.jpg)

![160bc537b5f6a2ed0f0540692c1ff4bbfe51cc97b1e5657a205cc19695050637.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/160bc537b5f6a2ed0f0540692c1ff4bbfe51cc97b1e5657a205cc19695050637.jpg)

![18c804a0ac0471ce7dc733673ff353c5efb39c628f050e6270b55822311d34d0.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/18c804a0ac0471ce7dc733673ff353c5efb39c628f050e6270b55822311d34d0.jpg)

![19092f34943cace23fc1786dbbf02fb4e6e170415230e35b82a073d2b55778cf.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/19092f34943cace23fc1786dbbf02fb4e6e170415230e35b82a073d2b55778cf.jpg)

![240e391932d2b97769d2d2137c91858d783af47e5d2c982c6c59c27481e54f36.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/240e391932d2b97769d2d2137c91858d783af47e5d2c982c6c59c27481e54f36.jpg)

![356f7bb5d755c2b5003d4b8079946566af1c0e9141c87a3c4698f5befd81f7a4.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/356f7bb5d755c2b5003d4b8079946566af1c0e9141c87a3c4698f5befd81f7a4.jpg)

![359d244f8a36a8c4cfec99dd4d119fbf2dfb01be1f80277ed768040b98f132f6.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/359d244f8a36a8c4cfec99dd4d119fbf2dfb01be1f80277ed768040b98f132f6.jpg)

![35fa9398a0c7c5e3d1577d0ed27bf8237008bbe441ef53c630df78fd50c8b7dd.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/35fa9398a0c7c5e3d1577d0ed27bf8237008bbe441ef53c630df78fd50c8b7dd.jpg)

![3cea58b0f404ee4f3a8a495be182562df33a321127520d28c8fdc0099ce1dc13.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/3cea58b0f404ee4f3a8a495be182562df33a321127520d28c8fdc0099ce1dc13.jpg)

![488937fa4e5e5b7fd913efa4e089b6a58eda0daf1d78664f0c394d81fbced1dc.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/488937fa4e5e5b7fd913efa4e089b6a58eda0daf1d78664f0c394d81fbced1dc.jpg)

![4e4f3c37e8ae02c52360e51c3cbaa9a8121c712b2bc1600e486b2306957faf74.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/4e4f3c37e8ae02c52360e51c3cbaa9a8121c712b2bc1600e486b2306957faf74.jpg)

![529e3922151e13dd6dd940abaca928e86cb52d2cfc8730d2686fe44e07e53259.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/529e3922151e13dd6dd940abaca928e86cb52d2cfc8730d2686fe44e07e53259.jpg)

![594de23b8da6d9c39e16dce9327dd3d1226c49d839de550cbaccc6135b6aa8f1.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/594de23b8da6d9c39e16dce9327dd3d1226c49d839de550cbaccc6135b6aa8f1.jpg)

![5dd771801b7592ae92a9b9bb0dafb78a552be4c321f2423dcb00e41a0ecc17a3.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/5dd771801b7592ae92a9b9bb0dafb78a552be4c321f2423dcb00e41a0ecc17a3.jpg)

![6421119ec62bd02c36beebc1ac3f2fc29e30593df2175bca4ab4e92cb5668ec9.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/6421119ec62bd02c36beebc1ac3f2fc29e30593df2175bca4ab4e92cb5668ec9.jpg)

![6f0785250e22f56744a4528ceebeab7de58b29a5b69520dadacef6306023cc77.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/6f0785250e22f56744a4528ceebeab7de58b29a5b69520dadacef6306023cc77.jpg)

![6fc043f0193f83bc9aed01f517d4c1e9a29919613227c4605e145d4cf5d025a6.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/6fc043f0193f83bc9aed01f517d4c1e9a29919613227c4605e145d4cf5d025a6.jpg)

![72466191145b5f92692b46001480db0f90b8764317d6e31a3e2312697a9bbb0f.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/72466191145b5f92692b46001480db0f90b8764317d6e31a3e2312697a9bbb0f.jpg)

![753420e431aff5b55bec7240a4d41b78ffad7fd21d66af143e7dfab397731a8c.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/753420e431aff5b55bec7240a4d41b78ffad7fd21d66af143e7dfab397731a8c.jpg)

![7643d105351b3d0b8bdebed1dc35745d630a5d7ff174c79d809933f72bc802af.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/7643d105351b3d0b8bdebed1dc35745d630a5d7ff174c79d809933f72bc802af.jpg)

![82dc4b0fdd6ac77bdcaca583a7ee8543089543f7796a8b38f4a2e01e460d22b3.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/82dc4b0fdd6ac77bdcaca583a7ee8543089543f7796a8b38f4a2e01e460d22b3.jpg)

![881e0d351761f4fce60aad34bb2b8c26b88e626bf30b4a99381960c80236106d.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/881e0d351761f4fce60aad34bb2b8c26b88e626bf30b4a99381960c80236106d.jpg)

![8d897eb0a4831c28a59a19305ea8c502e5a4566f43d54ee9da50373b53e68041.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/8d897eb0a4831c28a59a19305ea8c502e5a4566f43d54ee9da50373b53e68041.jpg)

![8e8111fef33f68c9328beaa55958a7d567e0cfb1a1f8ea98f8bf1f8ebdb7e121.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/8e8111fef33f68c9328beaa55958a7d567e0cfb1a1f8ea98f8bf1f8ebdb7e121.jpg)

![8fb788882a4bbe736dabb4724dfa7f66b5cd352de03cc0ef85497af2d61a19cc.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/8fb788882a4bbe736dabb4724dfa7f66b5cd352de03cc0ef85497af2d61a19cc.jpg)

![9b5015a50344b3031d76946e0de0ae6d90c9f50b63c1333e6c71582a1190a27e.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/9b5015a50344b3031d76946e0de0ae6d90c9f50b63c1333e6c71582a1190a27e.jpg)

![a72ea94ddf99f1d597a4257027541e294adfc89ce2bf07faf465a7483202d4d2.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/a72ea94ddf99f1d597a4257027541e294adfc89ce2bf07faf465a7483202d4d2.jpg)

![a9b9e190fe0d00c697a787f8ab008a704ec83c40380119baf06dbcb80ab8da97.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/a9b9e190fe0d00c697a787f8ab008a704ec83c40380119baf06dbcb80ab8da97.jpg)

![ac16c1699c02489387d93d2cac41b32399a1eae2d7fd34a0d78b19009f0278dd.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/ac16c1699c02489387d93d2cac41b32399a1eae2d7fd34a0d78b19009f0278dd.jpg)

![bdde36c2557d74e7d71253f6b3b4187001d708b08f7452566772f6cb30ef19a1.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/bdde36c2557d74e7d71253f6b3b4187001d708b08f7452566772f6cb30ef19a1.jpg)

![cee576eb67deb18c533ec045e8f37d59e41318376f1dabc8ba6b02d2bbf70eba.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/cee576eb67deb18c533ec045e8f37d59e41318376f1dabc8ba6b02d2bbf70eba.jpg)

![e36ec1515a18e685f97a16f779d1cb59a829e874a872d65c1607ab8686f5f772.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/e36ec1515a18e685f97a16f779d1cb59a829e874a872d65c1607ab8686f5f772.jpg)

![e92ef07fc3bb2d7f5a18d553f43f0d54c8f529affd7f94e66951d90efb0c0c41.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/e92ef07fc3bb2d7f5a18d553f43f0d54c8f529affd7f94e66951d90efb0c0c41.jpg)

![eed53ec8820464fa0953bfabfc984649a3fb87e249d132a35b0ad90ec1d97941.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/eed53ec8820464fa0953bfabfc984649a3fb87e249d132a35b0ad90ec1d97941.jpg)

![f187dbc6e2e4e14ed28ddab43c58d5d4844e2cb8410adda27f260108ffae8da2.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/f187dbc6e2e4e14ed28ddab43c58d5d4844e2cb8410adda27f260108ffae8da2.jpg)

![f5a7fb61f1ec4e9a7a17a20bb7a47c6c3a61f5ae3e8077ffaca857c8edbae8d5.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/f5a7fb61f1ec4e9a7a17a20bb7a47c6c3a61f5ae3e8077ffaca857c8edbae8d5.jpg)

![fa0e64beb011912a43594cce1ad89553feebb4f7f00854dca55d2199179561ce.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/fa0e64beb011912a43594cce1ad89553feebb4f7f00854dca55d2199179561ce.jpg)

![fae79f081a467aba59040b624a8673208b6af9aa10979a31bd75071ee0a9d3a1.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/images/fae79f081a467aba59040b624a8673208b6af9aa10979a31bd75071ee0a9d3a1.jpg)

### Tables

![964ae330f958f07fac7f9023473a19bd80518a9abe97712cbf7c17f5ded554b4.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/tables/964ae330f958f07fac7f9023473a19bd80518a9abe97712cbf7c17f5ded554b4.jpg)

![ecc07dcf447a4b38a2f9d7973379863bb99b603db849500a3e1060406a94755d.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/tables/ecc07dcf447a4b38a2f9d7973379863bb99b603db849500a3e1060406a94755d.jpg)

![fd6b86980b86b592e185f6cf54b2a5f61990d42e735c91cb041af27ba44c1135.jpg](acl_results/519_Unveiling Environmental Impacts of Large Language Model Serving_ A Functional Unit View/tables/fd6b86980b86b592e185f6cf54b2a5f61990d42e735c91cb041af27ba44c1135.jpg)

## ExpeTrans:LLMs Are Experiential Transfer Learners

### Images

![0287fd628f14cec571d3161708cac0027e2e51dcf999245e2d9d28315b11309a.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/images/0287fd628f14cec571d3161708cac0027e2e51dcf999245e2d9d28315b11309a.jpg)

![4c522b516043c4782b197f19dbc5706dbb0a47d216577e2b8096e396efc6402f.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/images/4c522b516043c4782b197f19dbc5706dbb0a47d216577e2b8096e396efc6402f.jpg)

![6134f274f86551d8660b06df847bbc12f0ae02266c14f7a44a6ef40dd0c5cb6f.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/images/6134f274f86551d8660b06df847bbc12f0ae02266c14f7a44a6ef40dd0c5cb6f.jpg)

![814c7f63dd7d3fba8e36cd278cd47bf312dd120c3d2f458e9ddb5dd5897b9179.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/images/814c7f63dd7d3fba8e36cd278cd47bf312dd120c3d2f458e9ddb5dd5897b9179.jpg)

![8cd5d1688b3e183589264f8b4411e70b9d2579b9e384daf501190eb63e091e9e.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/images/8cd5d1688b3e183589264f8b4411e70b9d2579b9e384daf501190eb63e091e9e.jpg)

![a5452e78f57fdd2e108ae37f62f5c79a252865823ff0270de36e4768c7988786.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/images/a5452e78f57fdd2e108ae37f62f5c79a252865823ff0270de36e4768c7988786.jpg)

![b1fdc9332d8c2ca80c84eebf6c98853c8eedc469ba54e3095e8037a095bec772.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/images/b1fdc9332d8c2ca80c84eebf6c98853c8eedc469ba54e3095e8037a095bec772.jpg)

![d285e1bbc9e520d4eb48c373170b2205b033726e6037ae99e4cd252144f68748.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/images/d285e1bbc9e520d4eb48c373170b2205b033726e6037ae99e4cd252144f68748.jpg)

### Tables

![117bbed1169fa7cef57d8b9795f2d295896d50e3bcde8a7cc31d2e217924d810.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/tables/117bbed1169fa7cef57d8b9795f2d295896d50e3bcde8a7cc31d2e217924d810.jpg)

![3c2c5685a467e1936b460150645aa529e0c310cc345ec3068068a4934bcd5db2.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/tables/3c2c5685a467e1936b460150645aa529e0c310cc345ec3068068a4934bcd5db2.jpg)

![40a17ff7a09e03d31e7f3b1b41b52379f29bf67593807c8acd8bde5abed1df68.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/tables/40a17ff7a09e03d31e7f3b1b41b52379f29bf67593807c8acd8bde5abed1df68.jpg)

![40c95454d855cd5b8f56d6538c0ea593651685d4c9c1709c333a269bbeaf5132.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/tables/40c95454d855cd5b8f56d6538c0ea593651685d4c9c1709c333a269bbeaf5132.jpg)

![42e50a7ba18a2c1695e011a699faa2de65618e1fe7bf5da5354953654b8acf0e.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/tables/42e50a7ba18a2c1695e011a699faa2de65618e1fe7bf5da5354953654b8acf0e.jpg)

![531f110a2d71e45c9553b7f9640114b2b9344055ce2707c6c886294983810a17.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/tables/531f110a2d71e45c9553b7f9640114b2b9344055ce2707c6c886294983810a17.jpg)

![5423264b0f5792229c889e11de6d6be7ab77c406a08aa42caafdfcddad13799f.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/tables/5423264b0f5792229c889e11de6d6be7ab77c406a08aa42caafdfcddad13799f.jpg)

![64ef0ee7aa1a91704176b73d9a3b77a03db4656be04a611cf0cb88a856e8fb0f.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/tables/64ef0ee7aa1a91704176b73d9a3b77a03db4656be04a611cf0cb88a856e8fb0f.jpg)

![698e796f31144bb1ffc5961a893cae5cbfbe1a175d6e66d87e87844b4e76bb36.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/tables/698e796f31144bb1ffc5961a893cae5cbfbe1a175d6e66d87e87844b4e76bb36.jpg)

![6e8d2326eb80bfa5ffbb3bd28cd8249ec658bffde6effe66ac146aaf56436e5b.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/tables/6e8d2326eb80bfa5ffbb3bd28cd8249ec658bffde6effe66ac146aaf56436e5b.jpg)

![eb5e7fabd99eaa616bdc96d85b1048462c069b9fef849e77ec07d1fc54fb85b3.jpg](acl_results/520_ExpeTrans_LLMs Are Experiential Transfer Learners/tables/eb5e7fabd99eaa616bdc96d85b1048462c069b9fef849e77ec07d1fc54fb85b3.jpg)

## Cool-Fusion: Fuse Large Language Models without Training

### Images

![0213d0660810fafaee2256fd12c2fdc5a1b07dac214e099e8655bf217ba77abf.jpg](acl_results/521_Cool-Fusion_ Fuse Large Language Models without Training/images/0213d0660810fafaee2256fd12c2fdc5a1b07dac214e099e8655bf217ba77abf.jpg)

![0e36cd85cd5ef19eb839108d38c5b0349e2e9ad24259afef0e17643380222fb6.jpg](acl_results/521_Cool-Fusion_ Fuse Large Language Models without Training/images/0e36cd85cd5ef19eb839108d38c5b0349e2e9ad24259afef0e17643380222fb6.jpg)

![61bb4c17c9adb9361124d5cd3587c91d53c42166baa37ebf057ac90f0b54cf58.jpg](acl_results/521_Cool-Fusion_ Fuse Large Language Models without Training/images/61bb4c17c9adb9361124d5cd3587c91d53c42166baa37ebf057ac90f0b54cf58.jpg)

![9a3ffe5c2373a33b1fc07468d99472d9644ff9f49fce58d9ccf0d6e20dbdcf1a.jpg](acl_results/521_Cool-Fusion_ Fuse Large Language Models without Training/images/9a3ffe5c2373a33b1fc07468d99472d9644ff9f49fce58d9ccf0d6e20dbdcf1a.jpg)

![a456acc8bc7209cc3031d14ca35a25f8044ba384c687ca04a7c55c9b4e55c48e.jpg](acl_results/521_Cool-Fusion_ Fuse Large Language Models without Training/images/a456acc8bc7209cc3031d14ca35a25f8044ba384c687ca04a7c55c9b4e55c48e.jpg)

### Tables

![421ca1dc3a93d139c5c8bde336892d4faadc248351ebad8a6b816f01943c4f44.jpg](acl_results/521_Cool-Fusion_ Fuse Large Language Models without Training/tables/421ca1dc3a93d139c5c8bde336892d4faadc248351ebad8a6b816f01943c4f44.jpg)

![8685ed1f0d436d028951f864410ec77691646256c11e2fa7885d035e943b5801.jpg](acl_results/521_Cool-Fusion_ Fuse Large Language Models without Training/tables/8685ed1f0d436d028951f864410ec77691646256c11e2fa7885d035e943b5801.jpg)

![89d360ab89b14efde487cbbd3ec08dead687b0427f15a54c71992fdf6aa96e94.jpg](acl_results/521_Cool-Fusion_ Fuse Large Language Models without Training/tables/89d360ab89b14efde487cbbd3ec08dead687b0427f15a54c71992fdf6aa96e94.jpg)

![95b92f7100f701f46e624bb6d271708c12a2d102720346176f918632619ef8d2.jpg](acl_results/521_Cool-Fusion_ Fuse Large Language Models without Training/tables/95b92f7100f701f46e624bb6d271708c12a2d102720346176f918632619ef8d2.jpg)

![a22e330cb0abd66883ec516e83bad6150ec32dcec12b37dbcd44cd07f9943535.jpg](acl_results/521_Cool-Fusion_ Fuse Large Language Models without Training/tables/a22e330cb0abd66883ec516e83bad6150ec32dcec12b37dbcd44cd07f9943535.jpg)

## DAPEV2: Process Attention Score as Feature Map for Length Extrapolation

### Images

![05375948e60c6cc62260127dd713997dee64f69f29fe2e85f8428441165381b9.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/05375948e60c6cc62260127dd713997dee64f69f29fe2e85f8428441165381b9.jpg)

![0b52d98500474391368fa2865f687cc3d9f5cd31a37953a37349136510dcf962.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/0b52d98500474391368fa2865f687cc3d9f5cd31a37953a37349136510dcf962.jpg)

![0e0f3f441e6a263a6eb0dc3c91bac267dbfead67fa70b10049055bf4e70d80b2.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/0e0f3f441e6a263a6eb0dc3c91bac267dbfead67fa70b10049055bf4e70d80b2.jpg)

![0f344ef009157c67508029cd2f55cc92ad8fc6a44635f5046e9355d1c40776b6.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/0f344ef009157c67508029cd2f55cc92ad8fc6a44635f5046e9355d1c40776b6.jpg)

![1ff6296b08ba608ba558e71a9b70f8a3b377acde0b688fb48eb63255ba7a7f5a.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/1ff6296b08ba608ba558e71a9b70f8a3b377acde0b688fb48eb63255ba7a7f5a.jpg)

![2132c927b8f335653976230887a4618286f0134b641247e41fb542fbe075b52e.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/2132c927b8f335653976230887a4618286f0134b641247e41fb542fbe075b52e.jpg)

![355a72574f427cc487afaa9a4fab62406be206da7afb0598bc61d5cd84f7303e.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/355a72574f427cc487afaa9a4fab62406be206da7afb0598bc61d5cd84f7303e.jpg)

![40b6e42d7069794384542b288d6d0ef367c9f0ba390f21db5e3bfb29d7b7567a.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/40b6e42d7069794384542b288d6d0ef367c9f0ba390f21db5e3bfb29d7b7567a.jpg)

![429e96337fbc05deafca1c92ad4c781936c366230ce65bdb0b91a12e336e06ab.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/429e96337fbc05deafca1c92ad4c781936c366230ce65bdb0b91a12e336e06ab.jpg)

![4b69025cd5cc3244154b259682a2491fd9c167fcfd56c49e1a353438defa77d2.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/4b69025cd5cc3244154b259682a2491fd9c167fcfd56c49e1a353438defa77d2.jpg)

![5b0bea207ef755b46d48183a25b6ba15aaa672386ebb0d48536639f3b0c17271.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/5b0bea207ef755b46d48183a25b6ba15aaa672386ebb0d48536639f3b0c17271.jpg)

![6796da569f3808f0c7b7a3d491c8da977f6ab86f9880cdd7a39acefa51a7874e.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/6796da569f3808f0c7b7a3d491c8da977f6ab86f9880cdd7a39acefa51a7874e.jpg)

![69a124e4878f5018ce9b1e17741a3e3f2533b9c5c60e946dfe01e19a9176b2ae.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/69a124e4878f5018ce9b1e17741a3e3f2533b9c5c60e946dfe01e19a9176b2ae.jpg)

![72682f052e1d8422eb96cb0af7a8bb7a7d02e508ce883dea2b82f10f2f627b0a.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/72682f052e1d8422eb96cb0af7a8bb7a7d02e508ce883dea2b82f10f2f627b0a.jpg)

![763fbdf23071fbffffe3d3b0d9c1d56e753390143685ef2a75c9f8ca31dd3740.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/763fbdf23071fbffffe3d3b0d9c1d56e753390143685ef2a75c9f8ca31dd3740.jpg)

![8beaae08a76b007310b83f86bb3cb3931160c64ddd9680e33d2d403c57ed2213.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/8beaae08a76b007310b83f86bb3cb3931160c64ddd9680e33d2d403c57ed2213.jpg)

![93f0ab375b386430da1de2fbf442408b1d5069d251e696fba64d9a4cf85e7fab.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/93f0ab375b386430da1de2fbf442408b1d5069d251e696fba64d9a4cf85e7fab.jpg)

![c6071a799bdb4d788fa2fda60218697c7527db3aea1250de4bfb088904419d3b.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/c6071a799bdb4d788fa2fda60218697c7527db3aea1250de4bfb088904419d3b.jpg)

![c88e6c5395b0f82bb8707d84651fe9c54fb64090c45de1434df9235087b26fa5.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/c88e6c5395b0f82bb8707d84651fe9c54fb64090c45de1434df9235087b26fa5.jpg)

![c8c8d4c799ae2d0342620df48579ca97f3827f51209267ba8109a8c08559757f.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/c8c8d4c799ae2d0342620df48579ca97f3827f51209267ba8109a8c08559757f.jpg)

![c8cc4aac9730fe59c6459dbaeb4483482dddd0683f4762ac6b319595d36c50da.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/c8cc4aac9730fe59c6459dbaeb4483482dddd0683f4762ac6b319595d36c50da.jpg)

![ca3394ec6a65a4477aabf7011b7c1816e1e04fd87f36db74ddf3e6cd74ffc22a.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/ca3394ec6a65a4477aabf7011b7c1816e1e04fd87f36db74ddf3e6cd74ffc22a.jpg)

![eff5d29ebec8930dfd420ac75a3de7cda1827cda8b80e2d13041b167719f1d18.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/eff5d29ebec8930dfd420ac75a3de7cda1827cda8b80e2d13041b167719f1d18.jpg)

![f22e33e9912920640e4716a2b7e94bc4c31e736411bf4340a827686e6f3d38cc.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/f22e33e9912920640e4716a2b7e94bc4c31e736411bf4340a827686e6f3d38cc.jpg)

![fbc6b26d69760ad736544ab3f72fed82efed452f88d7ca2d512f88859233b170.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/images/fbc6b26d69760ad736544ab3f72fed82efed452f88d7ca2d512f88859233b170.jpg)

### Tables

![2dd50838f663fbd5d53b5587689adb523cdefe0bcdab9d8dbd3dd5c830ca2b88.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/tables/2dd50838f663fbd5d53b5587689adb523cdefe0bcdab9d8dbd3dd5c830ca2b88.jpg)

![2f4d7740979bbabab134dff7e2b81bd7bf9cbbde2276aff7234b0f8f9a6051af.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/tables/2f4d7740979bbabab134dff7e2b81bd7bf9cbbde2276aff7234b0f8f9a6051af.jpg)

![545f39609984124c805deed1d0867e5e588784da93ef98d9d7dd5309b91c2b22.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/tables/545f39609984124c805deed1d0867e5e588784da93ef98d9d7dd5309b91c2b22.jpg)

![5cc0d1d43f3a6a8cfb9e2f5d869a2f63184a9d65d29bb73b3648d2cdbd24d79a.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/tables/5cc0d1d43f3a6a8cfb9e2f5d869a2f63184a9d65d29bb73b3648d2cdbd24d79a.jpg)

![62bff0bc09296ca11dca838276af0d4b92bf3392901ffff94896b6ca0b54cf30.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/tables/62bff0bc09296ca11dca838276af0d4b92bf3392901ffff94896b6ca0b54cf30.jpg)

![82078a0ac6d63a2f212ec947427dbc6487696b15e4b387a92e0d3dde1e0e81c8.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/tables/82078a0ac6d63a2f212ec947427dbc6487696b15e4b387a92e0d3dde1e0e81c8.jpg)

![88548ed0772e7898d28e396c61251b9c26361ed3fb7aa27fa3b2562beab9c0ad.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/tables/88548ed0772e7898d28e396c61251b9c26361ed3fb7aa27fa3b2562beab9c0ad.jpg)

![99bd90f51a952298199ef3d33764d6059e764225489663c869d5fae7bb80386d.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/tables/99bd90f51a952298199ef3d33764d6059e764225489663c869d5fae7bb80386d.jpg)

![9f47b37bbe66d2b81c5786af94cba0faf8a7911728206118fc3af0eba3ccb1ff.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/tables/9f47b37bbe66d2b81c5786af94cba0faf8a7911728206118fc3af0eba3ccb1ff.jpg)

![aaaad9fca4d57f922e76fdabe7251b40f36f0e73c0c1246edf8eda8963a94de7.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/tables/aaaad9fca4d57f922e76fdabe7251b40f36f0e73c0c1246edf8eda8963a94de7.jpg)

![ca5130e3e356c0cb89c27f52640b9c1e5cf702088a5712c8518a8546db72f3f7.jpg](acl_results/522_DAPEV2_ Process Attention Score as Feature Map for Length Extrapolation/tables/ca5130e3e356c0cb89c27f52640b9c1e5cf702088a5712c8518a8546db72f3f7.jpg)

## MuSC: Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training

### Images

![059ff29dc9eaa6be81b83dfaa0ff254c4bd51760b6e2e408d4975c5c16af4648.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/images/059ff29dc9eaa6be81b83dfaa0ff254c4bd51760b6e2e408d4975c5c16af4648.jpg)

![147d72b852956039b620558f5b0782f3aadbbd6596b3b5cd0d79a64189bf11e6.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/images/147d72b852956039b620558f5b0782f3aadbbd6596b3b5cd0d79a64189bf11e6.jpg)

![39ee2111d0df86133d35b2595e66acf3c09e24129f74301b8e6bf582d73794c4.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/images/39ee2111d0df86133d35b2595e66acf3c09e24129f74301b8e6bf582d73794c4.jpg)

![96ed4b42dc18d31db084247d1a8a28f4072589e8e23f2c5345a64996c5717297.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/images/96ed4b42dc18d31db084247d1a8a28f4072589e8e23f2c5345a64996c5717297.jpg)

![a7646ddc5a8072bb281b8f7b3f6f49481543ec7153a07f37fedb0b1dc1857523.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/images/a7646ddc5a8072bb281b8f7b3f6f49481543ec7153a07f37fedb0b1dc1857523.jpg)

![ac58fb51ed98f5c555572ccf0d7c0d663718f263e6dc96197f4c850b0ceed9b6.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/images/ac58fb51ed98f5c555572ccf0d7c0d663718f263e6dc96197f4c850b0ceed9b6.jpg)

![b0e6dd89ae400c5657e82b581c15dcb4211b1bee45dfa48d8b23f323d2dadb85.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/images/b0e6dd89ae400c5657e82b581c15dcb4211b1bee45dfa48d8b23f323d2dadb85.jpg)

![f54458b540c528dfe87aaa901fc815631294fbf920592f007395608a4071dd79.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/images/f54458b540c528dfe87aaa901fc815631294fbf920592f007395608a4071dd79.jpg)

### Tables

![00e3c88ed02868ea969f6bdc51542e246108792cb397a04e972172a49588bc0f.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/tables/00e3c88ed02868ea969f6bdc51542e246108792cb397a04e972172a49588bc0f.jpg)

![06cc024e43c684c6aa5db8459c438356d523e192642d6c7a178d0a293eaa693e.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/tables/06cc024e43c684c6aa5db8459c438356d523e192642d6c7a178d0a293eaa693e.jpg)

![2a76abb4047c8e7eca7b82d65d7bcef5ad4c62a68bb452edeb915a6e4044bb1d.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/tables/2a76abb4047c8e7eca7b82d65d7bcef5ad4c62a68bb452edeb915a6e4044bb1d.jpg)

![3d3ddbd89ab768e047bd2361ea99bfd0a9f58d669e022f3be55c660a82ece0c9.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/tables/3d3ddbd89ab768e047bd2361ea99bfd0a9f58d669e022f3be55c660a82ece0c9.jpg)

![48ee862b107e1f3010a63d9c5625556fc240c44c267e5307204670232075b52a.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/tables/48ee862b107e1f3010a63d9c5625556fc240c44c267e5307204670232075b52a.jpg)

![4c518c3c47dcf0f9380776986534ea41b526d20973498ef47c0278dc6bf49184.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/tables/4c518c3c47dcf0f9380776986534ea41b526d20973498ef47c0278dc6bf49184.jpg)

![4e16a573645f03f167487156b06a06729d220be8fe6a42e9fc7afcdd05d2882a.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/tables/4e16a573645f03f167487156b06a06729d220be8fe6a42e9fc7afcdd05d2882a.jpg)

![636b56e372b1058a3989b2d66f9fc278729977819d68fba26ae6b85b97974461.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/tables/636b56e372b1058a3989b2d66f9fc278729977819d68fba26ae6b85b97974461.jpg)

![8374e58c34f0d83a747a96e9ae3dbd948584afc6bcb7deda9e02bbaf1d2db20b.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/tables/8374e58c34f0d83a747a96e9ae3dbd948584afc6bcb7deda9e02bbaf1d2db20b.jpg)

![850f6927a935da3b685de67b5fcd24d7d7321604e13f577b1f54663cd2c6f098.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/tables/850f6927a935da3b685de67b5fcd24d7d7321604e13f577b1f54663cd2c6f098.jpg)

![c95f755d32c0ff34d4abab4f0cccaf9c558123677700924883c28184a5832b15.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/tables/c95f755d32c0ff34d4abab4f0cccaf9c558123677700924883c28184a5832b15.jpg)

![e66889595d861166c9c3d3ed34a48ed35afddc0a705184528160eeefbaeef3d6.jpg](acl_results/523_MuSC_ Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training/tables/e66889595d861166c9c3d3ed34a48ed35afddc0a705184528160eeefbaeef3d6.jpg)

## LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation

### Images

![199b01986c49492e9047f0e7454b76e433c86833bfacc15a3cfd0681dfcb07f4.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/images/199b01986c49492e9047f0e7454b76e433c86833bfacc15a3cfd0681dfcb07f4.jpg)

![4dbbc7abc82a020d41eac2827a520dbc33472f04dafcec3616189a8df5ef58cb.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/images/4dbbc7abc82a020d41eac2827a520dbc33472f04dafcec3616189a8df5ef58cb.jpg)

![8a80463a3271e5ec234df8862e32cdfc932e173731b585c45069204a63f6103d.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/images/8a80463a3271e5ec234df8862e32cdfc932e173731b585c45069204a63f6103d.jpg)

![913e09b71e070d222e752c0601fb8175d2a595df940598b3feeb26a62a88749a.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/images/913e09b71e070d222e752c0601fb8175d2a595df940598b3feeb26a62a88749a.jpg)

![b6d7e353960d32ee0b06b58f71353a5e4e082b10684d76bad9eb095438ceab99.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/images/b6d7e353960d32ee0b06b58f71353a5e4e082b10684d76bad9eb095438ceab99.jpg)

### Tables

![0e5ca6bdc70098306858b03da250c239b81fb805ffbc161d3de1651d305dd621.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/0e5ca6bdc70098306858b03da250c239b81fb805ffbc161d3de1651d305dd621.jpg)

![11e10319a45e0423a8373c5b8d5a3f766523f3d2780442704c2bd2909a08675a.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/11e10319a45e0423a8373c5b8d5a3f766523f3d2780442704c2bd2909a08675a.jpg)

![1aa4c4178ea112acc93b48686274e974531652754420cba20c558e74f2668711.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/1aa4c4178ea112acc93b48686274e974531652754420cba20c558e74f2668711.jpg)

![1dd0e15dc11e13a439f11b9c27abdae7a5e97a61c6e696fef4ecbfbc57e88b04.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/1dd0e15dc11e13a439f11b9c27abdae7a5e97a61c6e696fef4ecbfbc57e88b04.jpg)

![3b62dc56af3238aba51925f2b077a10c396dc6aeb6db62ed7d797e1c785ba753.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/3b62dc56af3238aba51925f2b077a10c396dc6aeb6db62ed7d797e1c785ba753.jpg)

![58cddc744b07a5cbfbd4a874b825120aa4cae2bbedd99a8f461f1124ebd7ef4b.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/58cddc744b07a5cbfbd4a874b825120aa4cae2bbedd99a8f461f1124ebd7ef4b.jpg)

![6af973d5ae9770bf5405c0b8ef0fc1193c41ce3939421a1d520923011849dd9e.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/6af973d5ae9770bf5405c0b8ef0fc1193c41ce3939421a1d520923011849dd9e.jpg)

![78e521ebe313e1ab61abdf25b4a994321e052314d5beb99d430cac73d5da2231.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/78e521ebe313e1ab61abdf25b4a994321e052314d5beb99d430cac73d5da2231.jpg)

![7a18008258ebb48efac30cf6efbef6f3711624e3864a4754a00387d955e3ebb7.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/7a18008258ebb48efac30cf6efbef6f3711624e3864a4754a00387d955e3ebb7.jpg)

![97f901e7b493039d01d8fca294f06227d84f8a618ef009b5a04f320686604e90.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/97f901e7b493039d01d8fca294f06227d84f8a618ef009b5a04f320686604e90.jpg)

![9e84312b934eb50aeffa5ca25def1e18379b972b8b4e19df809509447fdceaac.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/9e84312b934eb50aeffa5ca25def1e18379b972b8b4e19df809509447fdceaac.jpg)

![ad27c3483a39f455d47b0bde40bc5123cfb95e86b0d70864680199c6b69a87a6.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/ad27c3483a39f455d47b0bde40bc5123cfb95e86b0d70864680199c6b69a87a6.jpg)

![bbc6fa3f6026752efe198f31d86d61e994aad90669d5373af2058ab6e50b24d9.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/bbc6fa3f6026752efe198f31d86d61e994aad90669d5373af2058ab6e50b24d9.jpg)

![bc00aff8721736893514384001ebea5149c7d87c27771f40ea96fe6ec989e647.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/bc00aff8721736893514384001ebea5149c7d87c27771f40ea96fe6ec989e647.jpg)

![e069d6afeb6f1b4e68f1b9d6a1409ae1c75ad3a2c517291fb7b8bddbed2966b2.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/e069d6afeb6f1b4e68f1b9d6a1409ae1c75ad3a2c517291fb7b8bddbed2966b2.jpg)

![ff47c9b5513f464f388cfe8bba45928b364fd290ee040b41d51a89bb59e79282.jpg](acl_results/524_LongReD_ Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Dis/tables/ff47c9b5513f464f388cfe8bba45928b364fd290ee040b41d51a89bb59e79282.jpg)

## APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs

### Images

![65d5a38cb199ff12f5fed6d25e1867be31bbf8311ccda88a27f8a6c76b2429ab.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/images/65d5a38cb199ff12f5fed6d25e1867be31bbf8311ccda88a27f8a6c76b2429ab.jpg)

![7e4fbeb69ad66c21a60e24b1eaa5e756ac6d5f93e56970ba161d09af975a7cf7.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/images/7e4fbeb69ad66c21a60e24b1eaa5e756ac6d5f93e56970ba161d09af975a7cf7.jpg)

![ad2ac01331533d3b3016879fd4aedfa9a4e9dc65e18268d03d7f57b271db6eac.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/images/ad2ac01331533d3b3016879fd4aedfa9a4e9dc65e18268d03d7f57b271db6eac.jpg)

![cd8b55e5f9e85905c0d66d9af4b31c0fecb75a9fe480fb42d5721ee2b7690e80.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/images/cd8b55e5f9e85905c0d66d9af4b31c0fecb75a9fe480fb42d5721ee2b7690e80.jpg)

![e1df9657dcc8a8a912addb28bf864e3673b64c4c51efa8ada773c19ba78a9966.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/images/e1df9657dcc8a8a912addb28bf864e3673b64c4c51efa8ada773c19ba78a9966.jpg)

![f9e59b1c240bad08f79d391451ba021a9d2dd346cce33081adb360223361715c.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/images/f9e59b1c240bad08f79d391451ba021a9d2dd346cce33081adb360223361715c.jpg)

![fa220422d3865270fa38316877f36078a205e572c52b90d396a47ea15a5020d0.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/images/fa220422d3865270fa38316877f36078a205e572c52b90d396a47ea15a5020d0.jpg)

### Tables

![06883f019483f64ff832da0d7f0309b219b06a6006b8e8e92990a0afbbb402e3.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/06883f019483f64ff832da0d7f0309b219b06a6006b8e8e92990a0afbbb402e3.jpg)

![2311a7d46de18c0cfd080e2f51024d58780c264fd0e57aebf4dc778abde3f4cc.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/2311a7d46de18c0cfd080e2f51024d58780c264fd0e57aebf4dc778abde3f4cc.jpg)

![280745682ce01c018a6c6ceaf40c80f0c801e9bf8c74a26405fb4df0e7e7975c.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/280745682ce01c018a6c6ceaf40c80f0c801e9bf8c74a26405fb4df0e7e7975c.jpg)

![287ce67b5126b9dca56f6140b6ad6f06f706749ec742c687d9dca238b5e47b6a.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/287ce67b5126b9dca56f6140b6ad6f06f706749ec742c687d9dca238b5e47b6a.jpg)

![28b17ec35f72ff9f1da259641bf7ef24f4293726ac3cdf6213fa6f3799bd2ffd.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/28b17ec35f72ff9f1da259641bf7ef24f4293726ac3cdf6213fa6f3799bd2ffd.jpg)

![2d7426d81b67a91df0da583759a0b1c30b98de937f3a85ff26d629a6c7ae3ac6.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/2d7426d81b67a91df0da583759a0b1c30b98de937f3a85ff26d629a6c7ae3ac6.jpg)

![3686aba67a8c7d401249b410b1ea8002ac712527a92106e9f59b3ae4f6aad85d.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/3686aba67a8c7d401249b410b1ea8002ac712527a92106e9f59b3ae4f6aad85d.jpg)

![3ac2620f5a040b574dac8c48b746329ccd41a355f1ada4a15f2882f9558dda9b.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/3ac2620f5a040b574dac8c48b746329ccd41a355f1ada4a15f2882f9558dda9b.jpg)

![638d21e8261d0594b272f68ac567a9f5ad0823dbf95cfaeb4c83b9ad94f6509f.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/638d21e8261d0594b272f68ac567a9f5ad0823dbf95cfaeb4c83b9ad94f6509f.jpg)

![67d669dbd1cd52a3f0f653ac54e29f996a83cae4c039f7393f0f660dd326dedd.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/67d669dbd1cd52a3f0f653ac54e29f996a83cae4c039f7393f0f660dd326dedd.jpg)

![7ecb7c759a42ab836fd5b0a4c7f8eb85834baf421a0a0c37661589a8258b4bda.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/7ecb7c759a42ab836fd5b0a4c7f8eb85834baf421a0a0c37661589a8258b4bda.jpg)

![8ff579179a6e21c78b5a6d3bceeb865fd76ccffff2ef4cd8c438348555a80941.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/8ff579179a6e21c78b5a6d3bceeb865fd76ccffff2ef4cd8c438348555a80941.jpg)

![9773ed6319d76f919e4b6af2e9188c09f397c4837c457901a96d649dc33c5a37.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/9773ed6319d76f919e4b6af2e9188c09f397c4837c457901a96d649dc33c5a37.jpg)

![b157d678dcb29f487edfccc801c1f661b877a6b218db0d6b2a0dbf272495f01f.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/b157d678dcb29f487edfccc801c1f661b877a6b218db0d6b2a0dbf272495f01f.jpg)

![bf968e1d3de52e90fe888059f2e34b3f04a3ce40fd0736cf35ce569ec2abefb1.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/bf968e1d3de52e90fe888059f2e34b3f04a3ce40fd0736cf35ce569ec2abefb1.jpg)

![c2b5592be25a47725df454bf56b06ca4e493bc16578ef3f02d199da718b75f55.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/c2b5592be25a47725df454bf56b06ca4e493bc16578ef3f02d199da718b75f55.jpg)

![cf7dfced3f3f976cc5ea1e7f11defef113b0406bf57bb3895c8f7b7987e69c53.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/cf7dfced3f3f976cc5ea1e7f11defef113b0406bf57bb3895c8f7b7987e69c53.jpg)

![db87a6d0a4b4ec0f5405a1b8d9e4310230f046edde02111746fa696e615600bd.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/db87a6d0a4b4ec0f5405a1b8d9e4310230f046edde02111746fa696e615600bd.jpg)

![dc4c7d90fc744e3f9f2e6504dc287187b1e99a7409f30a1c20bbde9a1a5e3de3.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/dc4c7d90fc744e3f9f2e6504dc287187b1e99a7409f30a1c20bbde9a1a5e3de3.jpg)

![ddfa7a4fe2245039b49ccb9786fc8cbae15f05e2389785123849f61fc50e8390.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/ddfa7a4fe2245039b49ccb9786fc8cbae15f05e2389785123849f61fc50e8390.jpg)

![f3cc254d047018e56344bc6133c0b55b1fd00991f185ec5ab8e87af486e3f4a4.jpg](acl_results/525_APB_ Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks acrossGPUs/tables/f3cc254d047018e56344bc6133c0b55b1fd00991f185ec5ab8e87af486e3f4a4.jpg)

## PPT: A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer

### Images

![05593f9fdcadeaecb08634f3a2df3bf829437f74287b744a866c596751f2cd43.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/images/05593f9fdcadeaecb08634f3a2df3bf829437f74287b744a866c596751f2cd43.jpg)

![3cbbb81e449cf6c321a944fe6a1cc03c0659d07588b7d55cc52a69884c719c33.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/images/3cbbb81e449cf6c321a944fe6a1cc03c0659d07588b7d55cc52a69884c719c33.jpg)

![40507c249f5846402648bd49a51cd92d2dca6cc87ed08d332e547a57cc5952a6.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/images/40507c249f5846402648bd49a51cd92d2dca6cc87ed08d332e547a57cc5952a6.jpg)

![69976fa5814ac6dad93b1cd17eeec11331ecd541482e78c922172efba8b0e00f.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/images/69976fa5814ac6dad93b1cd17eeec11331ecd541482e78c922172efba8b0e00f.jpg)

![a1968c01cfeeaf0974c96e3d1d2affea7910b6e68d0dd9012fdfe63879d0cae7.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/images/a1968c01cfeeaf0974c96e3d1d2affea7910b6e68d0dd9012fdfe63879d0cae7.jpg)

![e96e4fcdd50f5c7bf26969e80c17a53ea7837ef34aed719b7895c2e9f2ff9b10.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/images/e96e4fcdd50f5c7bf26969e80c17a53ea7837ef34aed719b7895c2e9f2ff9b10.jpg)

![f8255228267a9ab1b67eb6b5260c8d4d11fb8a0f09ebd78c6639078b7f164465.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/images/f8255228267a9ab1b67eb6b5260c8d4d11fb8a0f09ebd78c6639078b7f164465.jpg)

### Tables

![24625532fba50929b0e9af62235b5295537a0029b22c4e7bf5e205eaaf57a606.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/24625532fba50929b0e9af62235b5295537a0029b22c4e7bf5e205eaaf57a606.jpg)

![2770e1eec338b9e2c17908cf1cb4bb3b72c6ae6a296f78961c922b4f162191ae.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/2770e1eec338b9e2c17908cf1cb4bb3b72c6ae6a296f78961c922b4f162191ae.jpg)

![503deb687dd6d3b730f890f9b63bde6e4e10253bdfeedd2f4f5609ff18e27255.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/503deb687dd6d3b730f890f9b63bde6e4e10253bdfeedd2f4f5609ff18e27255.jpg)

![57de0b8187ba03c1a4232c29c289bec7fcb4786b17d32fea509e70c99ce6c8fe.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/57de0b8187ba03c1a4232c29c289bec7fcb4786b17d32fea509e70c99ce6c8fe.jpg)

![6673af0b9660c9673376d05ce2caa8da04f355516b6d6a5492863e3606c2a904.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/6673af0b9660c9673376d05ce2caa8da04f355516b6d6a5492863e3606c2a904.jpg)

![6bf8130f3394152687df60890fe8aabb4b6218332e3aee4223e7cd8c55f4e6d4.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/6bf8130f3394152687df60890fe8aabb4b6218332e3aee4223e7cd8c55f4e6d4.jpg)

![7c2f1a8193a3e1e727cc97b389cc5af837fbf3f6a97685f5563686dfc4a28380.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/7c2f1a8193a3e1e727cc97b389cc5af837fbf3f6a97685f5563686dfc4a28380.jpg)

![84ef2adb52d9a5864084fa7e1d2ce2984964a022398f8524e6c301b58e06ae30.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/84ef2adb52d9a5864084fa7e1d2ce2984964a022398f8524e6c301b58e06ae30.jpg)

![8713dd0edb427157f8a8eef9303b52784067bc566627ddb6f55104affbee27c2.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/8713dd0edb427157f8a8eef9303b52784067bc566627ddb6f55104affbee27c2.jpg)

![952855d9a780bc55fef74706b22d115413821a6cadb49a1aa0e908fb8170adab.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/952855d9a780bc55fef74706b22d115413821a6cadb49a1aa0e908fb8170adab.jpg)

![b4c717d335ba1a38f42262856be7eff4bc6234afb82ae1dc6f5fa60e4ac613db.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/b4c717d335ba1a38f42262856be7eff4bc6234afb82ae1dc6f5fa60e4ac613db.jpg)

![b6824d6ea64cd7a597853b68111e1a53cd72c59f89eba4d8accf69ddb9b97a06.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/b6824d6ea64cd7a597853b68111e1a53cd72c59f89eba4d8accf69ddb9b97a06.jpg)

![c3e3969171c911421095fa15f2df67af5771c7381396bbf2472383179fe5fffc.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/c3e3969171c911421095fa15f2df67af5771c7381396bbf2472383179fe5fffc.jpg)

![d8ab50756cf9c2d482a2154cc1a793c06d226075daa7effea51022644440d9e3.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/d8ab50756cf9c2d482a2154cc1a793c06d226075daa7effea51022644440d9e3.jpg)

![ec25fcb2c6568eb1cc34da33ed140306515ca4f3bd43fadcb10483f0e9929342.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/ec25fcb2c6568eb1cc34da33ed140306515ca4f3bd43fadcb10483f0e9929342.jpg)

![ec7425ce3f655b5ad1e00cd84c045295b33570924d943e1bed97c7246b22b9fb.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/ec7425ce3f655b5ad1e00cd84c045295b33570924d943e1bed97c7246b22b9fb.jpg)

![eededa151d8897de26eb0650d2591cd8b4604b4ccc7190fa203894b5ce9387e8.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/eededa151d8897de26eb0650d2591cd8b4604b4ccc7190fa203894b5ce9387e8.jpg)

![f6444fe274c2042f2f4c36200df1d719038866f3deabb2ca76852322951b2263.jpg](acl_results/526_PPT_ A Minor Language News Recommendation Model via Cross-Lingual Preference Pattern Transfer/tables/f6444fe274c2042f2f4c36200df1d719038866f3deabb2ca76852322951b2263.jpg)

## GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis

### Images

![0ca04eaf40f7254adca4a6c076e808ed02f54a667f22b6cbf6ee4eafd1bb5bf4.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/images/0ca04eaf40f7254adca4a6c076e808ed02f54a667f22b6cbf6ee4eafd1bb5bf4.jpg)

![36818fc9336664f0c9f2a08610b27fb1da4da8a7043c86d669a0905d682aacdb.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/images/36818fc9336664f0c9f2a08610b27fb1da4da8a7043c86d669a0905d682aacdb.jpg)

![457e30f7aeaab095d8456139e29823ad0fe2dac73ee6cda812d5329433275888.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/images/457e30f7aeaab095d8456139e29823ad0fe2dac73ee6cda812d5329433275888.jpg)

![558e155e4dc58c8eacab8a6448550caea9b8d91c17bb0e44483d992370cc6c98.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/images/558e155e4dc58c8eacab8a6448550caea9b8d91c17bb0e44483d992370cc6c98.jpg)

![b7e4f4b7615e125bd7a4275d30893680105f4a0d8d08478c15bf7a6f49ba90fe.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/images/b7e4f4b7615e125bd7a4275d30893680105f4a0d8d08478c15bf7a6f49ba90fe.jpg)

![bca5046be2c8294d24fa0ccb9105b71f7e6762daff875d8025fe34f50cbbb72b.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/images/bca5046be2c8294d24fa0ccb9105b71f7e6762daff875d8025fe34f50cbbb72b.jpg)

![be890a128ef0c7f8496f8d278dbd82d309b9395252f6a3f72b718f9bca6df8de.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/images/be890a128ef0c7f8496f8d278dbd82d309b9395252f6a3f72b718f9bca6df8de.jpg)

![db5961f155cb1c766917010b458c590043f296b2b9def7dd03897a7c8f0ecf7c.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/images/db5961f155cb1c766917010b458c590043f296b2b9def7dd03897a7c8f0ecf7c.jpg)

### Tables

![0076b7e54143a3ca1a1d8307358116e7776698a61c0f9ec45f19ac49d9124882.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/tables/0076b7e54143a3ca1a1d8307358116e7776698a61c0f9ec45f19ac49d9124882.jpg)

![6233542f2570a84505b2228278ecba8b6e4a78e784c2da9111eeeba1daea764d.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/tables/6233542f2570a84505b2228278ecba8b6e4a78e784c2da9111eeeba1daea764d.jpg)

![71d440f07f943c9b8c4f8ac973d4e30b1d947cf0bcf6aba36c1da6e3907479f3.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/tables/71d440f07f943c9b8c4f8ac973d4e30b1d947cf0bcf6aba36c1da6e3907479f3.jpg)

![8d3b5d943e38014c45e4e54ee8901c0ab3215a21e9fb6039a7bc7a78b2021e31.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/tables/8d3b5d943e38014c45e4e54ee8901c0ab3215a21e9fb6039a7bc7a78b2021e31.jpg)

![91f7edd54607f73b27f25bd1bf1f31ef837267b382838e7de4a16f87eedf5faa.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/tables/91f7edd54607f73b27f25bd1bf1f31ef837267b382838e7de4a16f87eedf5faa.jpg)

![9481739f8d4541bcce24d03a56ec082995df672fd775d6a906869bcbaf9109ef.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/tables/9481739f8d4541bcce24d03a56ec082995df672fd775d6a906869bcbaf9109ef.jpg)

![e5d169621d6b641a5d2a2a95cc86fb0e38346bf85e8b37a0d86d56d378272d65.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/tables/e5d169621d6b641a5d2a2a95cc86fb0e38346bf85e8b37a0d86d56d378272d65.jpg)

![f4527ec018cd05996562a13fdaf14e5345c2b7b4890d7923da9245a537d0f28e.jpg](acl_results/527_GainRAG_ Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis/tables/f4527ec018cd05996562a13fdaf14e5345c2b7b4890d7923da9245a537d0f28e.jpg)

## Top-n𝜎: Eliminating Noise in Logit Space for Robust Token Sampling ofLLM

### Images

![0f6f8114d73a9dce42ed8774a3ef99a79d887a03ef547de378ae4331a9f304fc.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/images/0f6f8114d73a9dce42ed8774a3ef99a79d887a03ef547de378ae4331a9f304fc.jpg)

![14dabf34eb67ecc04fd79399dab30781cbfa6c32107009ad7dc729c441dc3e76.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/images/14dabf34eb67ecc04fd79399dab30781cbfa6c32107009ad7dc729c441dc3e76.jpg)

![24fe42077f7720c8bf9fdc1f29843e8665977447ba1f46f32df868232a10f466.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/images/24fe42077f7720c8bf9fdc1f29843e8665977447ba1f46f32df868232a10f466.jpg)

![a8ae3b0b5c6b5f565f723b6968b847dbc42e0aff047b581ee206ff14b192a4bd.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/images/a8ae3b0b5c6b5f565f723b6968b847dbc42e0aff047b581ee206ff14b192a4bd.jpg)

![b068cfa61573102702c8970c7631c932fc8214309987d93fc8e32b40217508a0.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/images/b068cfa61573102702c8970c7631c932fc8214309987d93fc8e32b40217508a0.jpg)

![d933b2281c9901e637111fbe077409699cc74a8a5d23688202fcb4492eaabb37.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/images/d933b2281c9901e637111fbe077409699cc74a8a5d23688202fcb4492eaabb37.jpg)

![df301e288c794d385e18fb2da517d74df28cce9a9ebb323d1f0e17ae31ae7431.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/images/df301e288c794d385e18fb2da517d74df28cce9a9ebb323d1f0e17ae31ae7431.jpg)

![e4c62ce7a91aca74c5cb37145ab084aab9b84959e918ae44dea9dda95ae12e48.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/images/e4c62ce7a91aca74c5cb37145ab084aab9b84959e918ae44dea9dda95ae12e48.jpg)

### Tables

![13d86d44f3176ad0f543662c0dccdaf3d5a132694dff663411b8a9d7f837d1fd.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/tables/13d86d44f3176ad0f543662c0dccdaf3d5a132694dff663411b8a9d7f837d1fd.jpg)

![285a8c9b433f1ed2425f3854d1006b9cb5a37a84efa5ba0c21ec389014a54008.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/tables/285a8c9b433f1ed2425f3854d1006b9cb5a37a84efa5ba0c21ec389014a54008.jpg)

![2ff999930d28d11a3fb0aafb79b46a8cebe352bc3ba09ce701b6ce6db5e2120e.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/tables/2ff999930d28d11a3fb0aafb79b46a8cebe352bc3ba09ce701b6ce6db5e2120e.jpg)

![8128c40ae6ac8c2940e08263f2c4b33ee6239dccc1f9eeb7a5a6160b1cdf6427.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/tables/8128c40ae6ac8c2940e08263f2c4b33ee6239dccc1f9eeb7a5a6160b1cdf6427.jpg)

![a6eb5744ca1ebbe33dac8d4c3cd1ed8f08a6c053a235d2107aa9a5fe258db942.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/tables/a6eb5744ca1ebbe33dac8d4c3cd1ed8f08a6c053a235d2107aa9a5fe258db942.jpg)

![b435b761acdc5c643b8de20ff3c401510b7828dacec8e14b4a46299a35408c07.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/tables/b435b761acdc5c643b8de20ff3c401510b7828dacec8e14b4a46299a35408c07.jpg)

![d5096b1877b98f0974a2939086cc9c0b457a6dc4a6c2843ff9deb554e6335562.jpg](acl_results/528_Top-n𝜎_ Eliminating Noise in Logit Space for Robust Token Sampling ofLLM/tables/d5096b1877b98f0974a2939086cc9c0b457a6dc4a6c2843ff9deb554e6335562.jpg)

## SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation

### Images

![09ddd5106176e38fc2b4931ce726a2e8d45858baa8a2c91446d72b0224b5a09c.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/images/09ddd5106176e38fc2b4931ce726a2e8d45858baa8a2c91446d72b0224b5a09c.jpg)

![2e044884bf53b80ee5e0f8f3dc63246313b2bf524773399c65d73181d16cda86.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/images/2e044884bf53b80ee5e0f8f3dc63246313b2bf524773399c65d73181d16cda86.jpg)

![5c6a5df2ee3bc3233a88e337f0be71b35a5aae9f85a6ce28c5809906420c3692.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/images/5c6a5df2ee3bc3233a88e337f0be71b35a5aae9f85a6ce28c5809906420c3692.jpg)

![688bc0d19beead31be40805ee7e6fb7dfd26a263a8348d18cb0150152ff79847.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/images/688bc0d19beead31be40805ee7e6fb7dfd26a263a8348d18cb0150152ff79847.jpg)

![8d33017c293d188c6e856513f142158adcd3b84025eec5da9308ffa38bda765c.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/images/8d33017c293d188c6e856513f142158adcd3b84025eec5da9308ffa38bda765c.jpg)

![bd8aa2cb0bbf222e7c456edce9a2ed5be7920ce68c4b9a41e5d26226a5df8381.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/images/bd8aa2cb0bbf222e7c456edce9a2ed5be7920ce68c4b9a41e5d26226a5df8381.jpg)

![d1b64005aac029047664fdd3a13655a970ccc9a4eff285524fe5e41bce406be4.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/images/d1b64005aac029047664fdd3a13655a970ccc9a4eff285524fe5e41bce406be4.jpg)

![ffcc40cec447663715913b0241ba9adb91fc185286fa9406fdef26c2c7ea689c.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/images/ffcc40cec447663715913b0241ba9adb91fc185286fa9406fdef26c2c7ea689c.jpg)

### Tables

![560e1c80f2debb8064e816342d11fde7119253d2e0174ac9cf7759a2420381d7.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/tables/560e1c80f2debb8064e816342d11fde7119253d2e0174ac9cf7759a2420381d7.jpg)

![8c4ffbc6aabef4fd9639f7979a5243396a3c9b9300b0e47536a8d0f63c1f307d.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/tables/8c4ffbc6aabef4fd9639f7979a5243396a3c9b9300b0e47536a8d0f63c1f307d.jpg)

![c06c8ccc866a821071ffc228afc58620823918ea614360401ea274028ddd53f5.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/tables/c06c8ccc866a821071ffc228afc58620823918ea614360401ea274028ddd53f5.jpg)

![cc5aba52abc4e379787ef4b1b11ccc1585afe7360f5dc9715e5fbc8614e97e12.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/tables/cc5aba52abc4e379787ef4b1b11ccc1585afe7360f5dc9715e5fbc8614e97e12.jpg)

![e2103ae2ee1a68fdb134b1e35606f3b63a143d61844f841c59952c53878501e9.jpg](acl_results/529_SCOPE_ Optimizing Key-Value Cache Compression in Long-context Generation/tables/e2103ae2ee1a68fdb134b1e35606f3b63a143d61844f841c59952c53878501e9.jpg)

## Mitigating Non-Representative Prototypes and Representation Bias in Few-Shot Continual Relation Extraction

### Images

![40e3620b927c13266a96c2018c4469689c0dd8ff98f330701d523839deee7c8b.jpg](acl_results/530_Mitigating Non-Representative Prototypes and Representation Bias in Few-Shot Continual Relation Extr/images/40e3620b927c13266a96c2018c4469689c0dd8ff98f330701d523839deee7c8b.jpg)

![dd3e7831ae30681eec3ebe557b7e2f15d1a5bba34d6ad6787f2de0af45b51ad5.jpg](acl_results/530_Mitigating Non-Representative Prototypes and Representation Bias in Few-Shot Continual Relation Extr/images/dd3e7831ae30681eec3ebe557b7e2f15d1a5bba34d6ad6787f2de0af45b51ad5.jpg)

### Tables

![5a91d5445c5827ec222c6fe914d938efa011c98fd346ddfedba7b6766ca2a5ad.jpg](acl_results/530_Mitigating Non-Representative Prototypes and Representation Bias in Few-Shot Continual Relation Extr/tables/5a91d5445c5827ec222c6fe914d938efa011c98fd346ddfedba7b6766ca2a5ad.jpg)

![5af4fbed40eed3f7a4bfbe222ce6db4442abdade7579cec75f0be6e94978d82e.jpg](acl_results/530_Mitigating Non-Representative Prototypes and Representation Bias in Few-Shot Continual Relation Extr/tables/5af4fbed40eed3f7a4bfbe222ce6db4442abdade7579cec75f0be6e94978d82e.jpg)

![70d40f1382e21714ab47bad1bb4afd97d27a047a10bfad1cc2de995db6836dda.jpg](acl_results/530_Mitigating Non-Representative Prototypes and Representation Bias in Few-Shot Continual Relation Extr/tables/70d40f1382e21714ab47bad1bb4afd97d27a047a10bfad1cc2de995db6836dda.jpg)

![7b8536a35327a27ae82ea85013bc0968efd9d5a2a72221ba6ca8af6826f73652.jpg](acl_results/530_Mitigating Non-Representative Prototypes and Representation Bias in Few-Shot Continual Relation Extr/tables/7b8536a35327a27ae82ea85013bc0968efd9d5a2a72221ba6ca8af6826f73652.jpg)

![84fa05c95d39d2ec7694f8e081559bbdfdee56abb960278df9c7ef67b88b5947.jpg](acl_results/530_Mitigating Non-Representative Prototypes and Representation Bias in Few-Shot Continual Relation Extr/tables/84fa05c95d39d2ec7694f8e081559bbdfdee56abb960278df9c7ef67b88b5947.jpg)

![9d2d2646af8823e8fc4ee40bbd84283514fddc22f422d843f6aec98c9a216357.jpg](acl_results/530_Mitigating Non-Representative Prototypes and Representation Bias in Few-Shot Continual Relation Extr/tables/9d2d2646af8823e8fc4ee40bbd84283514fddc22f422d843f6aec98c9a216357.jpg)

![da03fbb47b6c7ce002ce11bb25fe920b56018f476afa12375f9758bddc965e36.jpg](acl_results/530_Mitigating Non-Representative Prototypes and Representation Bias in Few-Shot Continual Relation Extr/tables/da03fbb47b6c7ce002ce11bb25fe920b56018f476afa12375f9758bddc965e36.jpg)

## MoQAE: Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware Experts

### Images

![17c1409c2f901af9d663b91513630d6d688e0e437bed8df2a71d8ddaccafa93e.jpg](acl_results/531_MoQAE_ Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware E/images/17c1409c2f901af9d663b91513630d6d688e0e437bed8df2a71d8ddaccafa93e.jpg)

![6597e400dd262734956c0d609f1ec31dbe583f0aff9c34b761fedbea56ef6da1.jpg](acl_results/531_MoQAE_ Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware E/images/6597e400dd262734956c0d609f1ec31dbe583f0aff9c34b761fedbea56ef6da1.jpg)

![68318ae33ce04db254e49f41d75a632620431fd88170a0c0d0c92b2aed8dcb23.jpg](acl_results/531_MoQAE_ Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware E/images/68318ae33ce04db254e49f41d75a632620431fd88170a0c0d0c92b2aed8dcb23.jpg)

![6a94b29d4c1c12f9b2ad9b2bf1a0cdaf4985d8589604e307a621dd49c244d1bc.jpg](acl_results/531_MoQAE_ Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware E/images/6a94b29d4c1c12f9b2ad9b2bf1a0cdaf4985d8589604e307a621dd49c244d1bc.jpg)

![a669e1c6592948f776c377c3d9db21097f7c000139dc792a98ad5ee815c00332.jpg](acl_results/531_MoQAE_ Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware E/images/a669e1c6592948f776c377c3d9db21097f7c000139dc792a98ad5ee815c00332.jpg)

![cf5b18ecdcc6047030e07f1bd3a40af15f5f1508a3e17ead52e8d958d2f5d25a.jpg](acl_results/531_MoQAE_ Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware E/images/cf5b18ecdcc6047030e07f1bd3a40af15f5f1508a3e17ead52e8d958d2f5d25a.jpg)

![e2384b2028d6271ae8227b8b3ddcc881157482f6e568b6cf955f09ac1eb2325b.jpg](acl_results/531_MoQAE_ Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware E/images/e2384b2028d6271ae8227b8b3ddcc881157482f6e568b6cf955f09ac1eb2325b.jpg)

### Tables

![0a990bcd758ead65ce161da3881f8b66ceaa45826311296adcde7edbb7e98395.jpg](acl_results/531_MoQAE_ Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware E/tables/0a990bcd758ead65ce161da3881f8b66ceaa45826311296adcde7edbb7e98395.jpg)

![217eb6989b13700bf3269c6a1e5db1459331cf2ec02495497814fdbef172d8d6.jpg](acl_results/531_MoQAE_ Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware E/tables/217eb6989b13700bf3269c6a1e5db1459331cf2ec02495497814fdbef172d8d6.jpg)

![7a07761d64b05bee75b0ec95ca2bb44e084d57349693e9673391f92873d7b38a.jpg](acl_results/531_MoQAE_ Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware E/tables/7a07761d64b05bee75b0ec95ca2bb44e084d57349693e9673391f92873d7b38a.jpg)

![ac572eab0439b394f2f8bf2bb05be3d834a78b6094607c7e9f18623cb818c1d7.jpg](acl_results/531_MoQAE_ Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware E/tables/ac572eab0439b394f2f8bf2bb05be3d834a78b6094607c7e9f18623cb818c1d7.jpg)

![d78476912046fb5ab0a7e3cf5f4b201ca6f404918b46722811af45f6a43f3a07.jpg](acl_results/531_MoQAE_ Mixed-Precision Quantization for Long-ContextLLMInference via Mixture of Quantization-Aware E/tables/d78476912046fb5ab0a7e3cf5f4b201ca6f404918b46722811af45f6a43f3a07.jpg)

## PrivacyRestore: Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restoration

### Images

![0fc7a87af58c67ed1e05f12a485f49ccae6f973852ddcac5fe3f2b807ee2c4a3.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/0fc7a87af58c67ed1e05f12a485f49ccae6f973852ddcac5fe3f2b807ee2c4a3.jpg)

![1822d34d239e0033ce9b13ecf8afa30ffb2e9a0707b217914cc8776cb50ea1dd.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/1822d34d239e0033ce9b13ecf8afa30ffb2e9a0707b217914cc8776cb50ea1dd.jpg)

![2cdd5eefcb85e615cc4c19c14cf6cb17748a672cb4b93107c3ca9da9fe2104aa.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/2cdd5eefcb85e615cc4c19c14cf6cb17748a672cb4b93107c3ca9da9fe2104aa.jpg)

![57e6993c106769b1230a27c8fc21980717ed21bdbd0f72f7a0de06ffd942484e.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/57e6993c106769b1230a27c8fc21980717ed21bdbd0f72f7a0de06ffd942484e.jpg)

![66fb9a6f4004fe1b7e7fb7f3a64eb57310c5a7260afc8773bcdad8aa8b1ff45b.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/66fb9a6f4004fe1b7e7fb7f3a64eb57310c5a7260afc8773bcdad8aa8b1ff45b.jpg)

![760ea2836e3165ed044e042d8971d48f8821b663787d64ea7dc3da2c455a8953.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/760ea2836e3165ed044e042d8971d48f8821b663787d64ea7dc3da2c455a8953.jpg)

![7ceeb9ab9c11b62ba7e9f01d46f104d53c8c002d92398fc0e071fb6041acb70b.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/7ceeb9ab9c11b62ba7e9f01d46f104d53c8c002d92398fc0e071fb6041acb70b.jpg)

![9766d8b70b885f0d116ba91960d0eeb50de093c20793b8b9636d74335b7fde15.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/9766d8b70b885f0d116ba91960d0eeb50de093c20793b8b9636d74335b7fde15.jpg)

![9892ebccebd3af18bca643850051e28f075b40ec1b407f437d6f0b865c38486c.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/9892ebccebd3af18bca643850051e28f075b40ec1b407f437d6f0b865c38486c.jpg)

![9c58c9fe05359d2625cc59f2239b941108b64635ddf336d1e49de2b330c2184a.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/9c58c9fe05359d2625cc59f2239b941108b64635ddf336d1e49de2b330c2184a.jpg)

![b046e477085dccf1f9c2a864d3ca06cd65ededf2a1117a0a2cc9f026e83f6c8c.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/b046e477085dccf1f9c2a864d3ca06cd65ededf2a1117a0a2cc9f026e83f6c8c.jpg)

![d936584929fc8659ca5ac4ea2030f3f9e8ff41d39aa0ed2134cabe1323563f1b.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/d936584929fc8659ca5ac4ea2030f3f9e8ff41d39aa0ed2134cabe1323563f1b.jpg)

![dd1691eddf24cc168daed5b54eb9178603260514dbd4ded68e018be998b84ead.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/dd1691eddf24cc168daed5b54eb9178603260514dbd4ded68e018be998b84ead.jpg)

![dfc2434b25bc689bca0d07b7f308494625f3d5602d6cada038d59ceacd79cba8.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/dfc2434b25bc689bca0d07b7f308494625f3d5602d6cada038d59ceacd79cba8.jpg)

![ea85c9e4b251d4cd7f801c6c9222bc1408c8e83159d6c4474448f19207dfbe05.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/ea85c9e4b251d4cd7f801c6c9222bc1408c8e83159d6c4474448f19207dfbe05.jpg)

![fffcfe4d1b915c6b0b3437f93e9f1666fe45d24ab963ca092d565253bbdde973.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/images/fffcfe4d1b915c6b0b3437f93e9f1666fe45d24ab963ca092d565253bbdde973.jpg)

### Tables

![06c33a2b65d8774c7e8b34e6386b1d799535f4de903bb4ac9976fcb3c7ce9572.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/06c33a2b65d8774c7e8b34e6386b1d799535f4de903bb4ac9976fcb3c7ce9572.jpg)

![22397d66cd9e3a93058ef99389525f936e6d41878f48ae7d3c52ca825953b2f6.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/22397d66cd9e3a93058ef99389525f936e6d41878f48ae7d3c52ca825953b2f6.jpg)

![24ea15c77e94e8569b7aa326e25fc52302c4a123fc112829106e6e750889fd16.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/24ea15c77e94e8569b7aa326e25fc52302c4a123fc112829106e6e750889fd16.jpg)

![308329015bd8a3a09dbe0e196fada416d0307fed285df36fc9cb671e14c30121.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/308329015bd8a3a09dbe0e196fada416d0307fed285df36fc9cb671e14c30121.jpg)

![3f8db1bee3919ed6827e8235cd4ef3f703f70587eb4468ef3913918d4957ef34.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/3f8db1bee3919ed6827e8235cd4ef3f703f70587eb4468ef3913918d4957ef34.jpg)

![5366e1d07a74472abca464eb3084d7cf025ab5eb3520797313cda797b707751e.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/5366e1d07a74472abca464eb3084d7cf025ab5eb3520797313cda797b707751e.jpg)

![5a1e3b2711607da2de5180402c8f1459ca9ea1540a8d79ddc8d028dad4a0ee03.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/5a1e3b2711607da2de5180402c8f1459ca9ea1540a8d79ddc8d028dad4a0ee03.jpg)

![5c5891f54e09a116ed68ede770f19e7bd3903eaeadbc2ea5fb50e939032fcf5e.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/5c5891f54e09a116ed68ede770f19e7bd3903eaeadbc2ea5fb50e939032fcf5e.jpg)

![63fce737ed3d0712c85bf82b0944e76587d9fd66b11fe3b7a5eed0bb61daf673.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/63fce737ed3d0712c85bf82b0944e76587d9fd66b11fe3b7a5eed0bb61daf673.jpg)

![6a224b193ad6071b2cea152869bfec2b38d39652bced8e0c2d34b3e113d9c80d.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/6a224b193ad6071b2cea152869bfec2b38d39652bced8e0c2d34b3e113d9c80d.jpg)

![70c9aacffc8f5caf0db0792a900d547a6cafaac3098b6646c0f47fed54f7951b.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/70c9aacffc8f5caf0db0792a900d547a6cafaac3098b6646c0f47fed54f7951b.jpg)

![81c602ce376a2bee30145cac821a7c7e4ffc6844b9fd9a91c8e47bedcfd07223.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/81c602ce376a2bee30145cac821a7c7e4ffc6844b9fd9a91c8e47bedcfd07223.jpg)

![9b1d762fc76fbab48bb2fee2173d8912c2942f7179f1a02d765b7ee2e348057d.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/9b1d762fc76fbab48bb2fee2173d8912c2942f7179f1a02d765b7ee2e348057d.jpg)

![ba52de20b9e38626c37fb8e2a4fd507f4657f26c0a09671a11db5a6efdca9e3f.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/ba52de20b9e38626c37fb8e2a4fd507f4657f26c0a09671a11db5a6efdca9e3f.jpg)

![be4d91b47e68a1d37b32a0d7bf48cb344f7f8b131c8a6508e433d80dc7e57b6a.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/be4d91b47e68a1d37b32a0d7bf48cb344f7f8b131c8a6508e433d80dc7e57b6a.jpg)

![d4b69b99fc5fcb61ef06a1d38ec1323db127c39965eb0172b58efb5246cd5111.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/d4b69b99fc5fcb61ef06a1d38ec1323db127c39965eb0172b58efb5246cd5111.jpg)

![d6080830468023a12e89e12e8f65283baf9e2502f535ddad64d30322259d0798.jpg](acl_results/532_PrivacyRestore_ Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restor/tables/d6080830468023a12e89e12e8f65283baf9e2502f535ddad64d30322259d0798.jpg)

## Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models

### Images

![05b8cb1332c244fe2a7db1352cf9b172fce84630a05a147a79bb1bc912487185.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/images/05b8cb1332c244fe2a7db1352cf9b172fce84630a05a147a79bb1bc912487185.jpg)

![2556e8869ad5b7c3120600194f2774e8b51d9652b43c9ef0cdd21a95360ed58b.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/images/2556e8869ad5b7c3120600194f2774e8b51d9652b43c9ef0cdd21a95360ed58b.jpg)

![2c534913ee215beff95d9c7b20104cee89d0f53321f8c45d8b9b1bf7ef64db2b.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/images/2c534913ee215beff95d9c7b20104cee89d0f53321f8c45d8b9b1bf7ef64db2b.jpg)

![2d03b84346e544cb9c59a0cfda35a2517e2f0f7b3a9cd744fd9c0965321fd02d.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/images/2d03b84346e544cb9c59a0cfda35a2517e2f0f7b3a9cd744fd9c0965321fd02d.jpg)

![3f33ee4559b5db8d559fba6ea4cd26d043d7c7ddd6f91014984e716e21d93165.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/images/3f33ee4559b5db8d559fba6ea4cd26d043d7c7ddd6f91014984e716e21d93165.jpg)

![4db47f8cc0d48ea969aec4981060af995da689c45b522c6110dbca27334cee00.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/images/4db47f8cc0d48ea969aec4981060af995da689c45b522c6110dbca27334cee00.jpg)

![66551841bce58ca5cd86d3ad77df0d9bddfd39076da02f7804bf3232cd15a229.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/images/66551841bce58ca5cd86d3ad77df0d9bddfd39076da02f7804bf3232cd15a229.jpg)

![73df77b95b7543771ab374a82e09bca7aa98858334925528a4d7e95a0fa00fa6.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/images/73df77b95b7543771ab374a82e09bca7aa98858334925528a4d7e95a0fa00fa6.jpg)

![c59e4521038adfb13571ad8465efe008dd8789d23be996de5c7b780742e34cde.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/images/c59e4521038adfb13571ad8465efe008dd8789d23be996de5c7b780742e34cde.jpg)

![cc119aa32052b2f94e8e032bc652e1b8566c153ced19bda1c30fc9f28244654c.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/images/cc119aa32052b2f94e8e032bc652e1b8566c153ced19bda1c30fc9f28244654c.jpg)

![dbb273abbd7edb7db96ac40d5c9d787d5d5a412d57fc2b2ae08ce0ab17b476d2.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/images/dbb273abbd7edb7db96ac40d5c9d787d5d5a412d57fc2b2ae08ce0ab17b476d2.jpg)

![febff0674d7c179d90889baefc340ceca90432ee791c690f347ee8416b9e6e23.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/images/febff0674d7c179d90889baefc340ceca90432ee791c690f347ee8416b9e6e23.jpg)

### Tables

![0be6c761f26e5215f6d78758d780023d04737ac4758ef73d11f1efb63041480e.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/0be6c761f26e5215f6d78758d780023d04737ac4758ef73d11f1efb63041480e.jpg)

![14de205c8075ede23e928800b044fa9b17868a909c3d425ba172d125c3ebcb9a.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/14de205c8075ede23e928800b044fa9b17868a909c3d425ba172d125c3ebcb9a.jpg)

![1cdabf3365ba62a5df26f0f2837994a19ba707762a2da6591a8d8dc0f8d18779.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/1cdabf3365ba62a5df26f0f2837994a19ba707762a2da6591a8d8dc0f8d18779.jpg)

![37aa781f5268fbb6f14aef113c3df06791b2204059dd0954b72ffd750077ce0f.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/37aa781f5268fbb6f14aef113c3df06791b2204059dd0954b72ffd750077ce0f.jpg)

![40dc64933bab952cfa0f659992011e7a82b012a635c5fddf19e4cf9521dd1315.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/40dc64933bab952cfa0f659992011e7a82b012a635c5fddf19e4cf9521dd1315.jpg)

![5ab03740856c77e152e47ff457d626396a2759d00d295fd9a2645aec18aace06.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/5ab03740856c77e152e47ff457d626396a2759d00d295fd9a2645aec18aace06.jpg)

![6665edb547fca0a8fec516bac43de4004a5d6eba1125282f898bb68886b2ccdb.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/6665edb547fca0a8fec516bac43de4004a5d6eba1125282f898bb68886b2ccdb.jpg)

![6896448031b07ce5d171d90a2be47f52cb278877b1a5db5512bde78ee26978fa.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/6896448031b07ce5d171d90a2be47f52cb278877b1a5db5512bde78ee26978fa.jpg)

![6e7594334742f8ed12aec4726a9de3b3135bea1d474f6f2f2a46cfbf82d3a87f.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/6e7594334742f8ed12aec4726a9de3b3135bea1d474f6f2f2a46cfbf82d3a87f.jpg)

![77762f3e22635449264935f09ea542c946485577542eb1b17cc01405839fe32d.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/77762f3e22635449264935f09ea542c946485577542eb1b17cc01405839fe32d.jpg)

![7fc5e83aedac102c6a1f3d18eb0fe554fcd559c386cebfb723ce55abccc4a4aa.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/7fc5e83aedac102c6a1f3d18eb0fe554fcd559c386cebfb723ce55abccc4a4aa.jpg)

![90189baf21fac2eefb1788fa50721859fa093925f1f248191fce31e2c25e4e29.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/90189baf21fac2eefb1788fa50721859fa093925f1f248191fce31e2c25e4e29.jpg)

![a6c91bc08fb61147040160d05e9d749f11c4139e6e877582e5e528ec041ddf5e.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/a6c91bc08fb61147040160d05e9d749f11c4139e6e877582e5e528ec041ddf5e.jpg)

![b2e1fefc62d9765a7683df4f4ad735b6a0454bd8cd9db90c816f45083d1b5bf9.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/b2e1fefc62d9765a7683df4f4ad735b6a0454bd8cd9db90c816f45083d1b5bf9.jpg)

![caa173a6f8c982d538e68773108ec60c2cd11551a436adcbdd0e77ecaf88723d.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/caa173a6f8c982d538e68773108ec60c2cd11551a436adcbdd0e77ecaf88723d.jpg)

![d9ab04733f6367885ec9702284b688b309fb87938b50051529ce8302bccbe524.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/d9ab04733f6367885ec9702284b688b309fb87938b50051529ce8302bccbe524.jpg)

![d9ef8364cc35fb8655cb027ea3ea282d6f2b651e8bc473017b34b446fc6374bd.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/d9ef8364cc35fb8655cb027ea3ea282d6f2b651e8bc473017b34b446fc6374bd.jpg)

![ef33dc8912fbc80c434570d80b0ab3d7bfa613e1ee3038e0acb5c4a089f2c3c1.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/ef33dc8912fbc80c434570d80b0ab3d7bfa613e1ee3038e0acb5c4a089f2c3c1.jpg)

![f6978f4d93f03c6f370ebd5928ad18d64d29fda4cb33b21528b1c8c2bc228b87.jpg](acl_results/533_Meta-rater_ A Multi-dimensional Data Selection Method for Pre-training Language Models/tables/f6978f4d93f03c6f370ebd5928ad18d64d29fda4cb33b21528b1c8c2bc228b87.jpg)

## GuessArena: Guess WhoIAm? A Self-Adaptive Framework for EvaluatingLLMs in Domain-Specific Knowledge and Reasoning

### Images

![5dd9c7fb7bacf62e5766982852cc3080611af540732833b03bb7eeefd0407422.jpg](acl_results/534_GuessArena_ Guess WhoIAm_ A Self-Adaptive Framework for EvaluatingLLMs in Domain-Specific Knowledge /images/5dd9c7fb7bacf62e5766982852cc3080611af540732833b03bb7eeefd0407422.jpg)

![9b2b63565b0076d00cd48e7416860366678895168ad73c50b04692c2080901f5.jpg](acl_results/534_GuessArena_ Guess WhoIAm_ A Self-Adaptive Framework for EvaluatingLLMs in Domain-Specific Knowledge /images/9b2b63565b0076d00cd48e7416860366678895168ad73c50b04692c2080901f5.jpg)

![a5cbb86f3b1c1953729d5639754d27c430981f4d2d6374dc53eb9cdd73eec92c.jpg](acl_results/534_GuessArena_ Guess WhoIAm_ A Self-Adaptive Framework for EvaluatingLLMs in Domain-Specific Knowledge /images/a5cbb86f3b1c1953729d5639754d27c430981f4d2d6374dc53eb9cdd73eec92c.jpg)

![fc40372acd4dbd1d8d227590fbb67dee9e8a472de91e6b836295c4652573efa1.jpg](acl_results/534_GuessArena_ Guess WhoIAm_ A Self-Adaptive Framework for EvaluatingLLMs in Domain-Specific Knowledge /images/fc40372acd4dbd1d8d227590fbb67dee9e8a472de91e6b836295c4652573efa1.jpg)

### Tables

![03dbcc4ff0e7d75b514253c72bc4658ef994bb75fd006185b0034635ebb48e61.jpg](acl_results/534_GuessArena_ Guess WhoIAm_ A Self-Adaptive Framework for EvaluatingLLMs in Domain-Specific Knowledge /tables/03dbcc4ff0e7d75b514253c72bc4658ef994bb75fd006185b0034635ebb48e61.jpg)

![124dc8fc5fcdfc6e867222dbc51992508758eaca8e01eed8bb5a53c7d1683d57.jpg](acl_results/534_GuessArena_ Guess WhoIAm_ A Self-Adaptive Framework for EvaluatingLLMs in Domain-Specific Knowledge /tables/124dc8fc5fcdfc6e867222dbc51992508758eaca8e01eed8bb5a53c7d1683d57.jpg)

![64e0ea9137ad3d18ffdf252e4e06781ccdcd8189e46bb540f0a299f889f354b3.jpg](acl_results/534_GuessArena_ Guess WhoIAm_ A Self-Adaptive Framework for EvaluatingLLMs in Domain-Specific Knowledge /tables/64e0ea9137ad3d18ffdf252e4e06781ccdcd8189e46bb540f0a299f889f354b3.jpg)

![814f3a89ec1b006a06fbee60c30eeddd05f787980ae07dbf726a9fae521e818e.jpg](acl_results/534_GuessArena_ Guess WhoIAm_ A Self-Adaptive Framework for EvaluatingLLMs in Domain-Specific Knowledge /tables/814f3a89ec1b006a06fbee60c30eeddd05f787980ae07dbf726a9fae521e818e.jpg)

![8a87e2995c9aa65427ccb6f8f4500b70ddc7254887ee225ec0b5d4c4d3316826.jpg](acl_results/534_GuessArena_ Guess WhoIAm_ A Self-Adaptive Framework for EvaluatingLLMs in Domain-Specific Knowledge /tables/8a87e2995c9aa65427ccb6f8f4500b70ddc7254887ee225ec0b5d4c4d3316826.jpg)

![b75da9f13f4b79fa0f7d5a871e91e6e50cf1b610eb754f7f62a02977b3d3eda5.jpg](acl_results/534_GuessArena_ Guess WhoIAm_ A Self-Adaptive Framework for EvaluatingLLMs in Domain-Specific Knowledge /tables/b75da9f13f4b79fa0f7d5a871e91e6e50cf1b610eb754f7f62a02977b3d3eda5.jpg)

## Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition

### Images

![080c2db8eb418141671151a2ad53f8349029f4746008791a6335d13121fc684a.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/images/080c2db8eb418141671151a2ad53f8349029f4746008791a6335d13121fc684a.jpg)

![0e6ddb2d28d89ed889400c721438dafefbcae55556c2204fa0a78f70c32cffef.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/images/0e6ddb2d28d89ed889400c721438dafefbcae55556c2204fa0a78f70c32cffef.jpg)

![690191ee0e6d44346352f35290988f0f4dcac3520e25874a4c33619d9556f658.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/images/690191ee0e6d44346352f35290988f0f4dcac3520e25874a4c33619d9556f658.jpg)

![87934450056798cd947df094c9d82cc8fa63f38d35e709c32479b079c87575e0.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/images/87934450056798cd947df094c9d82cc8fa63f38d35e709c32479b079c87575e0.jpg)

![c24fb4f5423e8644b7d85d51ac62398f404af37bdd233e7c64487e5fe444a824.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/images/c24fb4f5423e8644b7d85d51ac62398f404af37bdd233e7c64487e5fe444a824.jpg)

![ef40d537d7468b521a83585eae3ba1aaa5fc8544bc4b247df91fb73ab1ff9ee0.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/images/ef40d537d7468b521a83585eae3ba1aaa5fc8544bc4b247df91fb73ab1ff9ee0.jpg)

### Tables

![01bdaee1800b739e18b854119b20156106da51263de4de26f70a1b5feaecca39.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/01bdaee1800b739e18b854119b20156106da51263de4de26f70a1b5feaecca39.jpg)

![1b0537ed1c54d4ad54bd01bf94c4edde1ac3aa74b8e47d027fec4adbf159e097.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/1b0537ed1c54d4ad54bd01bf94c4edde1ac3aa74b8e47d027fec4adbf159e097.jpg)

![1bf144c2ad536ec2030626ab67eedf40cc728773b990ee44f2ca42e20fc79a34.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/1bf144c2ad536ec2030626ab67eedf40cc728773b990ee44f2ca42e20fc79a34.jpg)

![1d013dc20c456a14955f1755ffe3b032ae117e192f390937a5258f8812c5238e.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/1d013dc20c456a14955f1755ffe3b032ae117e192f390937a5258f8812c5238e.jpg)

![27d3453949b1cbfb4ecc99d5cebe299d0841bd95d2f9da304e9b643afb0c8f39.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/27d3453949b1cbfb4ecc99d5cebe299d0841bd95d2f9da304e9b643afb0c8f39.jpg)

![3ff7d5826aaf07b6c28dddda769bb35675b195a89065b4d7d7202389fcb7ba85.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/3ff7d5826aaf07b6c28dddda769bb35675b195a89065b4d7d7202389fcb7ba85.jpg)

![41f6f5be75bc07789f4d27f801e441a348ae7463d7c4f68916e0da8fc84ef2fb.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/41f6f5be75bc07789f4d27f801e441a348ae7463d7c4f68916e0da8fc84ef2fb.jpg)

![558a129d586af547e88bff8da1962eaabddb15787049b707dc1e054bf287196a.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/558a129d586af547e88bff8da1962eaabddb15787049b707dc1e054bf287196a.jpg)

![714255c31c89da7d3e6a3b9f13ffbf142d021a948417e476ea52ee2306b0769b.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/714255c31c89da7d3e6a3b9f13ffbf142d021a948417e476ea52ee2306b0769b.jpg)

![71dcf4faa40d86e2cb169e6cbfc24a837de3605b86a2cbb97adb7581270b44ed.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/71dcf4faa40d86e2cb169e6cbfc24a837de3605b86a2cbb97adb7581270b44ed.jpg)

![7643ae496557d47ccdee50203b1eb99176081e09e3a69ce404838f7d3f087e94.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/7643ae496557d47ccdee50203b1eb99176081e09e3a69ce404838f7d3f087e94.jpg)

![8827864df534553c47aef3640680eea37d0e010c7c88c3ed298aa6ba79b6a908.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/8827864df534553c47aef3640680eea37d0e010c7c88c3ed298aa6ba79b6a908.jpg)

![965da70ea9e0718e1b6e6e9d1db9466c6e2334cf40d542002cafba80535e6d3f.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/965da70ea9e0718e1b6e6e9d1db9466c6e2334cf40d542002cafba80535e6d3f.jpg)

![a1823e8e46ece620c0725cbfd89691f40229776a97383b93622b40ce7ed9f21d.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/a1823e8e46ece620c0725cbfd89691f40229776a97383b93622b40ce7ed9f21d.jpg)

![a533c96c8c4ffacbb7cbdc274ccb4125d4c17e1420dbdfb2365d2272ebc15469.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/a533c96c8c4ffacbb7cbdc274ccb4125d4c17e1420dbdfb2365d2272ebc15469.jpg)

![aafc692c3b6994e215cc276bd8621058c0ccfbb86ebead8b82f88986c1ff3007.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/aafc692c3b6994e215cc276bd8621058c0ccfbb86ebead8b82f88986c1ff3007.jpg)

![ad805ea7d81425ccdf65ee7b3983e4bfb67b40d361c93dd6624515a700caaa93.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/ad805ea7d81425ccdf65ee7b3983e4bfb67b40d361c93dd6624515a700caaa93.jpg)

![c521f52239de7a28dd264a572dce425d010132bf8baff5b588d47eb524006bd5.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/c521f52239de7a28dd264a572dce425d010132bf8baff5b588d47eb524006bd5.jpg)

![c5f78fb403f8a43a8c8bb0a77c1e9304664ef0305a35a2302653a90ce532505e.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/c5f78fb403f8a43a8c8bb0a77c1e9304664ef0305a35a2302653a90ce532505e.jpg)

![c92924a99f985a557ef19f413005b379a7e0742bffe6de7816d882400e608ee8.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/c92924a99f985a557ef19f413005b379a7e0742bffe6de7816d882400e608ee8.jpg)

![cd5a53a7269692b53d7f686540e94e218b7190d60b5b7ec9d0161ddc378bbb27.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/cd5a53a7269692b53d7f686540e94e218b7190d60b5b7ec9d0161ddc378bbb27.jpg)

![e5550d64a9f48318f95fa8789725a85d296252716dec67b6f95a4d777090cc56.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/e5550d64a9f48318f95fa8789725a85d296252716dec67b6f95a4d777090cc56.jpg)

![e829b0e2b6dc81d7f7a589e0414f52ba94e9d15c54b22bf6e1cbd29586632703.jpg](acl_results/535_Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition/tables/e829b0e2b6dc81d7f7a589e0414f52ba94e9d15c54b22bf6e1cbd29586632703.jpg)

## DTCRS: Dynamic Tree Construction for Recursive Summarization

### Images

![478997212c09b1fee004c784a1c775805ad3c26c0d14a6e64760924d7732a0be.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/images/478997212c09b1fee004c784a1c775805ad3c26c0d14a6e64760924d7732a0be.jpg)

![6a2c662b7b15a1b70ed58bc99a694ec82df901830692ca5284eacd1260c4685b.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/images/6a2c662b7b15a1b70ed58bc99a694ec82df901830692ca5284eacd1260c4685b.jpg)

![6e6c6c15b80490de8bb9e23f48e9f05027373768a861aa3e47fd4fa1e858f8b3.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/images/6e6c6c15b80490de8bb9e23f48e9f05027373768a861aa3e47fd4fa1e858f8b3.jpg)

![6f73524806043e61f5dcdba439ad1cd6d89743c68f88cac47b2b93315efd1277.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/images/6f73524806043e61f5dcdba439ad1cd6d89743c68f88cac47b2b93315efd1277.jpg)

![8866c8f170c7d2ad951716ac1c38aa35a15655a777c30c028c605f03b04c3a1b.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/images/8866c8f170c7d2ad951716ac1c38aa35a15655a777c30c028c605f03b04c3a1b.jpg)

![8c0c91a9cd26c06739fdf1e15aaf33ae6ddafd49c17331b1deac86c212044ce2.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/images/8c0c91a9cd26c06739fdf1e15aaf33ae6ddafd49c17331b1deac86c212044ce2.jpg)

![9b61a6835b9eac9ed1d30c8915084b34603099a255f6414a3ab33e782210f76c.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/images/9b61a6835b9eac9ed1d30c8915084b34603099a255f6414a3ab33e782210f76c.jpg)

![b7ca0300c7fad2c161a065ce35c2fbc3919c4b9877e3ece1da7a472c3a546d01.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/images/b7ca0300c7fad2c161a065ce35c2fbc3919c4b9877e3ece1da7a472c3a546d01.jpg)

![c1328533be84edcc3a263f576ef9420cc581ac1b7adb0f222f713c7241f63eb7.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/images/c1328533be84edcc3a263f576ef9420cc581ac1b7adb0f222f713c7241f63eb7.jpg)

![ddde17d36e45de95ed101bb1c1266e079ecde7f60b342136e7d376264d01e018.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/images/ddde17d36e45de95ed101bb1c1266e079ecde7f60b342136e7d376264d01e018.jpg)

![eff2044e668c06d98c1a462d33802133981ab533688caae5709ca32bd0fab1df.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/images/eff2044e668c06d98c1a462d33802133981ab533688caae5709ca32bd0fab1df.jpg)

![f6bf2829df4763a180b854f8d7ccd4b6ed942bc217e55aa4f1d8bbf7a4e52094.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/images/f6bf2829df4763a180b854f8d7ccd4b6ed942bc217e55aa4f1d8bbf7a4e52094.jpg)

### Tables

![03984b9572a753a3da5a5a467369d2443641ce165e59636a0deab850aef47816.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/03984b9572a753a3da5a5a467369d2443641ce165e59636a0deab850aef47816.jpg)

![0b3170fb1f6a49348deb9954942d5827d07ba4fda5a8a763c6bc5d8de3866fd1.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/0b3170fb1f6a49348deb9954942d5827d07ba4fda5a8a763c6bc5d8de3866fd1.jpg)

![2643486ecb42dcd99417ec672c18f7e2db777330f960bbe05cf42cd96af8f752.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/2643486ecb42dcd99417ec672c18f7e2db777330f960bbe05cf42cd96af8f752.jpg)

![34cdeb8d1fc16f4af840f2cabe04af85f75aae261469957f144b77b36998dc60.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/34cdeb8d1fc16f4af840f2cabe04af85f75aae261469957f144b77b36998dc60.jpg)

![3c87e6e8e0ad6315be7f853f90897de3cf2e6950829d66a94e5c754f4762bca2.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/3c87e6e8e0ad6315be7f853f90897de3cf2e6950829d66a94e5c754f4762bca2.jpg)

![4b6deeeab51247b57f8d4becea91ce3f2327882889d30349cca8a35214d7a8b5.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/4b6deeeab51247b57f8d4becea91ce3f2327882889d30349cca8a35214d7a8b5.jpg)

![4cbce636f4037a62d1bc9cd7c43eec10960efa28f6d9c66793d08a842b6c5fcd.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/4cbce636f4037a62d1bc9cd7c43eec10960efa28f6d9c66793d08a842b6c5fcd.jpg)

![6890d247b67f6ce892ecc3810824f8a6ba9173852335890d7466aa730ac46858.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/6890d247b67f6ce892ecc3810824f8a6ba9173852335890d7466aa730ac46858.jpg)

![6aacc75d7ac2028fd3380d222ca0cdaa90d7b22202d7dcf3f11465abbe4bc1bb.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/6aacc75d7ac2028fd3380d222ca0cdaa90d7b22202d7dcf3f11465abbe4bc1bb.jpg)

![7a20070c7738ee0973a64a8678a578e62b7ab65ea52e1161f1d8512add2fab26.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/7a20070c7738ee0973a64a8678a578e62b7ab65ea52e1161f1d8512add2fab26.jpg)

![84a8fe9881c40441adcc430de8c8b201bc66b15ac8672908bc0b319a4a627a5c.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/84a8fe9881c40441adcc430de8c8b201bc66b15ac8672908bc0b319a4a627a5c.jpg)

![8ff6b7f032b4763586975e5e75d256c698fa372ddf6f2f448f9e14b59189ff7a.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/8ff6b7f032b4763586975e5e75d256c698fa372ddf6f2f448f9e14b59189ff7a.jpg)

![b47bbd34ab916f856036e24c355b10a2054cdbbb1a8d4bca60e8888a19cc7c4c.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/b47bbd34ab916f856036e24c355b10a2054cdbbb1a8d4bca60e8888a19cc7c4c.jpg)

![cef124ad6d57db1e98e8e8e428cc082967730ae11643657f8ba86856f1773400.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/cef124ad6d57db1e98e8e8e428cc082967730ae11643657f8ba86856f1773400.jpg)

![f20afb5717447e72518688bfe071813492edfec8c7d15442903e7b82966719f8.jpg](acl_results/536_DTCRS_ Dynamic Tree Construction for Recursive Summarization/tables/f20afb5717447e72518688bfe071813492edfec8c7d15442903e7b82966719f8.jpg)

## A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning

### Images

![404d539b40e4ce86a1903ba20fd4d39da0f5f125eae34b2a37de6bc3715fe774.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/images/404d539b40e4ce86a1903ba20fd4d39da0f5f125eae34b2a37de6bc3715fe774.jpg)

![6ecc26cecc774ba26300d99e681dbfae876d07872b1c18084f386ff6612f022a.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/images/6ecc26cecc774ba26300d99e681dbfae876d07872b1c18084f386ff6612f022a.jpg)

![72e9039ee203a3db1915477bb7014c54075ce0f3c268c1b5c582361e42c1ecdb.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/images/72e9039ee203a3db1915477bb7014c54075ce0f3c268c1b5c582361e42c1ecdb.jpg)

![95284094d0cc457ade6d2cac65bbd3f54a2e05e4fed5da8a3ec660ed85b1cfea.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/images/95284094d0cc457ade6d2cac65bbd3f54a2e05e4fed5da8a3ec660ed85b1cfea.jpg)

![c58c9058b461ad6354c16341cfc4f23a805a67731e0025e4e64b45542071cc04.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/images/c58c9058b461ad6354c16341cfc4f23a805a67731e0025e4e64b45542071cc04.jpg)

![ed065caf23374550963317248cdb8b8bb9ff492fadd2d76105495653ad771553.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/images/ed065caf23374550963317248cdb8b8bb9ff492fadd2d76105495653ad771553.jpg)

### Tables

![540c190d02cc5bea2e5d3da4822ff9378e853548d038623a6cd3355e057c23e1.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/tables/540c190d02cc5bea2e5d3da4822ff9378e853548d038623a6cd3355e057c23e1.jpg)

![5975f12ae20eb600c3a6805b6d216b8e06a0f767cb4a379cb71936e80ff962cc.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/tables/5975f12ae20eb600c3a6805b6d216b8e06a0f767cb4a379cb71936e80ff962cc.jpg)

![c86cca0a0249e8848d7555f13d9592828c16e4bf00148caa621cd400855eaa03.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/tables/c86cca0a0249e8848d7555f13d9592828c16e4bf00148caa621cd400855eaa03.jpg)

![d465816c769e34885f2e45deca478e38ed924dee9b2a683201fa602d90a60567.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/tables/d465816c769e34885f2e45deca478e38ed924dee9b2a683201fa602d90a60567.jpg)

![f0a2cfdd50cd8ade095f67a49ec6e16e2e010704a7e87cf0e5265d874560ce00.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/tables/f0a2cfdd50cd8ade095f67a49ec6e16e2e010704a7e87cf0e5265d874560ce00.jpg)

![f37001c23913b989e8a0040ebb02af137f1f2baa23a91a722fe0b2ed8a89ee90.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/tables/f37001c23913b989e8a0040ebb02af137f1f2baa23a91a722fe0b2ed8a89ee90.jpg)

![ff24bd26f3c1c30c6ceea9cd3181e6a4dd5a1034b7758309711fec3fa687fd71.jpg](acl_results/537_A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning/tables/ff24bd26f3c1c30c6ceea9cd3181e6a4dd5a1034b7758309711fec3fa687fd71.jpg)

## ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search

### Images

![21785fc4c381c1696a3f2bc6f305c0df634e82bdc7c3e8d04c7c143b35dea261.jpg](acl_results/538_ARise_ Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search/images/21785fc4c381c1696a3f2bc6f305c0df634e82bdc7c3e8d04c7c143b35dea261.jpg)

![4c5fedf6a3604a851dcf3488f0ec8382b36d66de09ef9b627d4e1a8200af1f32.jpg](acl_results/538_ARise_ Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search/images/4c5fedf6a3604a851dcf3488f0ec8382b36d66de09ef9b627d4e1a8200af1f32.jpg)

![5bbda93977cf45f96098ed0fd31cb2bc0e2fff5c2d108faa2a6ffb6e7826c617.jpg](acl_results/538_ARise_ Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search/images/5bbda93977cf45f96098ed0fd31cb2bc0e2fff5c2d108faa2a6ffb6e7826c617.jpg)

![85bd11c4034a43a51badd42e905fcf4f4e2f4606b969e4abe1f9ea8089539412.jpg](acl_results/538_ARise_ Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search/images/85bd11c4034a43a51badd42e905fcf4f4e2f4606b969e4abe1f9ea8089539412.jpg)

![9d3772f4931fac6cc51b589c60288de485cbecb13977a77019b470b8ea7d7b95.jpg](acl_results/538_ARise_ Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search/images/9d3772f4931fac6cc51b589c60288de485cbecb13977a77019b470b8ea7d7b95.jpg)

![a5663b627aaf6ff05f6ab96887da53b3dfc0219189b1c58e82036cf33e8d5c19.jpg](acl_results/538_ARise_ Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search/images/a5663b627aaf6ff05f6ab96887da53b3dfc0219189b1c58e82036cf33e8d5c19.jpg)

### Tables

![031efae9af090844fbd867312a264192f4e6d860c765fc25c956628d35469cd6.jpg](acl_results/538_ARise_ Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search/tables/031efae9af090844fbd867312a264192f4e6d860c765fc25c956628d35469cd6.jpg)

![15c87a4361d08e289970c95ccb8499af3325b160713e475d58628b41fbdb5990.jpg](acl_results/538_ARise_ Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search/tables/15c87a4361d08e289970c95ccb8499af3325b160713e475d58628b41fbdb5990.jpg)

![315b4880810e7322324e2dc02ba19a7d429f1f66b613916c49e3984be4425187.jpg](acl_results/538_ARise_ Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search/tables/315b4880810e7322324e2dc02ba19a7d429f1f66b613916c49e3984be4425187.jpg)

![3d60f952242471875560a47d7fa6709434f6b08d0611d87ce716055eaf4693a9.jpg](acl_results/538_ARise_ Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search/tables/3d60f952242471875560a47d7fa6709434f6b08d0611d87ce716055eaf4693a9.jpg)

![8951821b8adf4be2dc6f6d884cd87f1fb8c8ff3d358dd3ff2fae9a85cafd7591.jpg](acl_results/538_ARise_ Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search/tables/8951821b8adf4be2dc6f6d884cd87f1fb8c8ff3d358dd3ff2fae9a85cafd7591.jpg)

## PKAG-DDI: Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generation

### Images

![0a13d622545d1b5bfed2165d8390cd9a38bbe5f96f6ccea6c02a966e75a18182.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/images/0a13d622545d1b5bfed2165d8390cd9a38bbe5f96f6ccea6c02a966e75a18182.jpg)

![237a1979c0ae8599a46b597be5342eda788fd1fe74207c7a4a5a2e6313d8e0fe.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/images/237a1979c0ae8599a46b597be5342eda788fd1fe74207c7a4a5a2e6313d8e0fe.jpg)

![35f3828b7cb5faa6890ad798f63cb252fc1234cc39ad91c8384dd9356769a239.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/images/35f3828b7cb5faa6890ad798f63cb252fc1234cc39ad91c8384dd9356769a239.jpg)

![6e5acaed921bb08035ea70f262af43931763245592a9a3b26c7bd081ffda1419.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/images/6e5acaed921bb08035ea70f262af43931763245592a9a3b26c7bd081ffda1419.jpg)

![8ecaff382fb26f31c37d65e75678cd844be0f4b30d8b7f11d456d837139a372b.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/images/8ecaff382fb26f31c37d65e75678cd844be0f4b30d8b7f11d456d837139a372b.jpg)

![a3abc163bb5501fc2065ce31b494c9cb176bb12bcfdcbe78383708be323c7906.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/images/a3abc163bb5501fc2065ce31b494c9cb176bb12bcfdcbe78383708be323c7906.jpg)

![c718ab0ec4189ff5c29ea5ecc47e269e769f87cb4a808567ea72c3f92bb7c26f.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/images/c718ab0ec4189ff5c29ea5ecc47e269e769f87cb4a808567ea72c3f92bb7c26f.jpg)

### Tables

![23ce0d9be540c99daccd49ee69e7609fdaffa0668b9a72471d3d5591f5662775.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/tables/23ce0d9be540c99daccd49ee69e7609fdaffa0668b9a72471d3d5591f5662775.jpg)

![3573f2d41264695fcba6ddd572fa4a6189ea1ed6d41b9338ec44b26c19b99282.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/tables/3573f2d41264695fcba6ddd572fa4a6189ea1ed6d41b9338ec44b26c19b99282.jpg)

![51027cbba3ba3ed975abb3b0e8bfcd051ce855307098ec2579a3281172cd42e5.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/tables/51027cbba3ba3ed975abb3b0e8bfcd051ce855307098ec2579a3281172cd42e5.jpg)

![544b383d33bda7976e1d46e34a56dab87477a83c201afc96b2ce31cb117cae2d.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/tables/544b383d33bda7976e1d46e34a56dab87477a83c201afc96b2ce31cb117cae2d.jpg)

![a1cc02a4cc136fdc74d7f9ac010b450e895e81d301accf044a4dd0f4d7f0a84f.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/tables/a1cc02a4cc136fdc74d7f9ac010b450e895e81d301accf044a4dd0f4d7f0a84f.jpg)

![b9969947e46c437199399083fd3c4cc6af535be4f3cf865abbf88d6e1a32d13b.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/tables/b9969947e46c437199399083fd3c4cc6af535be4f3cf865abbf88d6e1a32d13b.jpg)

![ef8cf8979d4034303d0e27986a9b8f426a7cc58d018e65054fb48639bc361e3e.jpg](acl_results/539_PKAG-DDI_ Pairwise Knowledge-Augmented Language Model for Drug-Drug Interaction Event Text Generatio/tables/ef8cf8979d4034303d0e27986a9b8f426a7cc58d018e65054fb48639bc361e3e.jpg)

## Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Language Models

### Images

![176ddf29b3447db22c152136234f369445c43bf8c92371f93074b8d8941ba988.jpg](acl_results/540_Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Langua/images/176ddf29b3447db22c152136234f369445c43bf8c92371f93074b8d8941ba988.jpg)

![36367c5c3bf5293b715ca870813515b40cfe7c9243741a34fd9c8798d952708a.jpg](acl_results/540_Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Langua/images/36367c5c3bf5293b715ca870813515b40cfe7c9243741a34fd9c8798d952708a.jpg)

![5e7f54a619e172ff4edb0cde422c188b2edddea12fa5d10d09d236ea16f23990.jpg](acl_results/540_Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Langua/images/5e7f54a619e172ff4edb0cde422c188b2edddea12fa5d10d09d236ea16f23990.jpg)

![68368237dcba95232903a0ca46f904acc6f1bf2de847635bef49b9c94d1fabba.jpg](acl_results/540_Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Langua/images/68368237dcba95232903a0ca46f904acc6f1bf2de847635bef49b9c94d1fabba.jpg)

![8dd7eb23bc9b556cc807e19d13e22edba11e79077ced2edbb96a93f960fc6dfb.jpg](acl_results/540_Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Langua/images/8dd7eb23bc9b556cc807e19d13e22edba11e79077ced2edbb96a93f960fc6dfb.jpg)

![a8d405e5f327f85bff92866d8b1f5bc34c63771b9b6ef650a18eda47db099ecc.jpg](acl_results/540_Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Langua/images/a8d405e5f327f85bff92866d8b1f5bc34c63771b9b6ef650a18eda47db099ecc.jpg)

### Tables

![1ade282f6ebcabb4321e9fc77902a0ec986d86d296509bf95d09734ce6fc2210.jpg](acl_results/540_Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Langua/tables/1ade282f6ebcabb4321e9fc77902a0ec986d86d296509bf95d09734ce6fc2210.jpg)

![2b3cdd4779cd908c1840af005d75369640f9c70a764e7c3bd2b8dd09f5f23980.jpg](acl_results/540_Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Langua/tables/2b3cdd4779cd908c1840af005d75369640f9c70a764e7c3bd2b8dd09f5f23980.jpg)

![5fa54dbfb7907b0ca99d13c14a696d6d877d0c03b525f62f37fa04b8df422fce.jpg](acl_results/540_Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Langua/tables/5fa54dbfb7907b0ca99d13c14a696d6d877d0c03b525f62f37fa04b8df422fce.jpg)

![653440b0953638a7d0089cf3daa70856cf3e9b1af3633bd5210cc3bbf0b266a9.jpg](acl_results/540_Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Langua/tables/653440b0953638a7d0089cf3daa70856cf3e9b1af3633bd5210cc3bbf0b266a9.jpg)

![b511467fee16f6fb171f548f098343b9a82609c997e76e8c6f01e7b96c673408.jpg](acl_results/540_Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Langua/tables/b511467fee16f6fb171f548f098343b9a82609c997e76e8c6f01e7b96c673408.jpg)

![f1816d83569e077b2d51fb2ac2ba8082ffa6286e921f9490ed6de01ce42a70b0.jpg](acl_results/540_Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Langua/tables/f1816d83569e077b2d51fb2ac2ba8082ffa6286e921f9490ed6de01ce42a70b0.jpg)

## TWIST: Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models

### Images

![1042b1ec8ab991dc889831b77cb52c1fa4e46ab28a4123cf011c802a31c6b4c5.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/1042b1ec8ab991dc889831b77cb52c1fa4e46ab28a4123cf011c802a31c6b4c5.jpg)

![26e7a15b4e5eafb72a5dadc69f739d3f281a7bc74a9bfba6ff390ebfded20457.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/26e7a15b4e5eafb72a5dadc69f739d3f281a7bc74a9bfba6ff390ebfded20457.jpg)

![32c414d032411383490a98de3e8c0d337974ec7633d06c5a04a527605e995f0f.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/32c414d032411383490a98de3e8c0d337974ec7633d06c5a04a527605e995f0f.jpg)

![3bb9673b93c9394836e0eddf8160e3ae85ace82665e1d1fe2fa653f194deb013.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/3bb9673b93c9394836e0eddf8160e3ae85ace82665e1d1fe2fa653f194deb013.jpg)

![3da2420dd3399c7740179851320f47ac57e8d3cf195c3eab69a64e3cd07dcc0f.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/3da2420dd3399c7740179851320f47ac57e8d3cf195c3eab69a64e3cd07dcc0f.jpg)

![3de00732f78baf9886723a6aa8d81224836fa0e9e59eecfd75bebe5f93967eee.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/3de00732f78baf9886723a6aa8d81224836fa0e9e59eecfd75bebe5f93967eee.jpg)

![48ca6a530bb3375f872389701375daa1bc95745ce1a1cb65813f1eb084de033e.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/48ca6a530bb3375f872389701375daa1bc95745ce1a1cb65813f1eb084de033e.jpg)

![7b016be47205077391c1ab452f3f84153fec6989e69ecf8323fe42d4d4b108de.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/7b016be47205077391c1ab452f3f84153fec6989e69ecf8323fe42d4d4b108de.jpg)

![bc213eab98fce5c8a0faa1d203c0ec53d59a07fa42a1fcd82bce73ebef8c913f.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/bc213eab98fce5c8a0faa1d203c0ec53d59a07fa42a1fcd82bce73ebef8c913f.jpg)

![c3d615ecde444b592e6d2c0c3f8281457948314ffacf4a1a30d144cb261c02bd.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/c3d615ecde444b592e6d2c0c3f8281457948314ffacf4a1a30d144cb261c02bd.jpg)

![e3cfd29f0df60fa818429d1693f1e1c667872479c189c5cfc6af8af2db67ef3f.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/e3cfd29f0df60fa818429d1693f1e1c667872479c189c5cfc6af8af2db67ef3f.jpg)

![e84df2edfb15dc8bc2bb9a3ea0b90a88a8d38c11bc179dcc6037bf1c652991b8.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/e84df2edfb15dc8bc2bb9a3ea0b90a88a8d38c11bc179dcc6037bf1c652991b8.jpg)

![fead23c2e3a6a53aca63b2a077653a8ab5cb0339adbd4b24be19250696b6fd75.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/images/fead23c2e3a6a53aca63b2a077653a8ab5cb0339adbd4b24be19250696b6fd75.jpg)

### Tables

![3caf13484285a507dbbe671878be7f1290a21ab589cd79890d2b1e8dd32e28b9.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/tables/3caf13484285a507dbbe671878be7f1290a21ab589cd79890d2b1e8dd32e28b9.jpg)

![3d505a15a12919718c8a3ea3a3c91cc35cd407264680536b2718a7519d15f766.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/tables/3d505a15a12919718c8a3ea3a3c91cc35cd407264680536b2718a7519d15f766.jpg)

![4245fd75feb382881598eb3f381fb3e46c640e92c76b52283fb3a9c80d0bd391.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/tables/4245fd75feb382881598eb3f381fb3e46c640e92c76b52283fb3a9c80d0bd391.jpg)

![6de6444a04749af39256c187af40330740f305620a2d16e0384e8bed8c20a55e.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/tables/6de6444a04749af39256c187af40330740f305620a2d16e0384e8bed8c20a55e.jpg)

![8875dc33a3ed8ee5236c21424d11c5680ac8726a9f8793326dec12e7522826a0.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/tables/8875dc33a3ed8ee5236c21424d11c5680ac8726a9f8793326dec12e7522826a0.jpg)

![adbddb790cdbb645835fe3eaf46ff86a91d62952391026b6c5920f9d01f8ef53.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/tables/adbddb790cdbb645835fe3eaf46ff86a91d62952391026b6c5920f9d01f8ef53.jpg)

![aec8f8b07432638811ec64cb71bd4dbb5a8dab04b3df1015b4945da5a9b8bbc3.jpg](acl_results/541_TWIST_ Text-encoder Weight-editing for Inserting Secret Trojans in Text-to-Image Models/tables/aec8f8b07432638811ec64cb71bd4dbb5a8dab04b3df1015b4945da5a9b8bbc3.jpg)

## Frictional Agent Alignment Framework: Slow Down and Don’t Break Things

### Images

![34513f628161349434dfe1757d7c3667090c91843452c7f3cc06d9cbda44a9fb.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/images/34513f628161349434dfe1757d7c3667090c91843452c7f3cc06d9cbda44a9fb.jpg)

![7bf7e300b060d308a4bec94cedb8f1b7041558e46ca8ea2858b76405d187b157.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/images/7bf7e300b060d308a4bec94cedb8f1b7041558e46ca8ea2858b76405d187b157.jpg)

![80ba140541a27f5c812ae7c24cb6d021e6d463f85770ff3c3f42d61473e0aaee.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/images/80ba140541a27f5c812ae7c24cb6d021e6d463f85770ff3c3f42d61473e0aaee.jpg)

![ad66381953af1b59d4de08a2872b134b0e4b0d23364cbbf92630bc758cbe7494.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/images/ad66381953af1b59d4de08a2872b134b0e4b0d23364cbbf92630bc758cbe7494.jpg)

![ddb1166aea9dd7a582f0be32354518fd7841d4a841b5ba849f2e738b83837824.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/images/ddb1166aea9dd7a582f0be32354518fd7841d4a841b5ba849f2e738b83837824.jpg)

### Tables

![0cb225aebcd169dd726acf47d1ce11d2538728fce701184afe04488fdd8f3ebf.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/0cb225aebcd169dd726acf47d1ce11d2538728fce701184afe04488fdd8f3ebf.jpg)

![17b3a00fcddea488110d1be49761eb2f086729a98f3f1d4a648c3a72820be0aa.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/17b3a00fcddea488110d1be49761eb2f086729a98f3f1d4a648c3a72820be0aa.jpg)

![1bc543524372dca4746a0d7e8d6ef3a3230b9300e47db7272e9e7561c7a98291.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/1bc543524372dca4746a0d7e8d6ef3a3230b9300e47db7272e9e7561c7a98291.jpg)

![2073f16275b9a82d4e2cc7c1506ac513b89ee046df90979d75d9ad22421d92a6.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/2073f16275b9a82d4e2cc7c1506ac513b89ee046df90979d75d9ad22421d92a6.jpg)

![2a8026935ca66be1303d3481d0494d0033994c3107e24f6cb0ec63c89744ca70.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/2a8026935ca66be1303d3481d0494d0033994c3107e24f6cb0ec63c89744ca70.jpg)

![56bcc18080ad4ef3c4deb5a7b3830a5e0f4c7922d16778d553e14fe38ffb1096.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/56bcc18080ad4ef3c4deb5a7b3830a5e0f4c7922d16778d553e14fe38ffb1096.jpg)

![6dec99d4f07241de03c5af1ead72e814b6c4cf22493d08bf48033be16a1111d0.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/6dec99d4f07241de03c5af1ead72e814b6c4cf22493d08bf48033be16a1111d0.jpg)

![b28c08a53f328c2ecdf8cc4b0911b87878d2944c5563f4259b3f8c3d67b1e3c3.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/b28c08a53f328c2ecdf8cc4b0911b87878d2944c5563f4259b3f8c3d67b1e3c3.jpg)

![c571bfc33a7d17e5d609b36791daa47d9c9b087d350e2fbae94a7ff5276b9e55.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/c571bfc33a7d17e5d609b36791daa47d9c9b087d350e2fbae94a7ff5276b9e55.jpg)

![e3cd7e4417b5f47c286c0eaebca0c30facad644120ea4206dbece26884e22012.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/e3cd7e4417b5f47c286c0eaebca0c30facad644120ea4206dbece26884e22012.jpg)

![e86337dcec61fa08e177cbd3ce3cc29d92b5c25c0327ca7c5bcc41fb0c21c5cd.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/e86337dcec61fa08e177cbd3ce3cc29d92b5c25c0327ca7c5bcc41fb0c21c5cd.jpg)

![f1367ef2743223327f481e972239c7f34064d1f098a43b347671e9e189209b9a.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/f1367ef2743223327f481e972239c7f34064d1f098a43b347671e9e189209b9a.jpg)

![fd5045f8a065007b30f19087f4dcf22472831e224bad2504700c637414f60611.jpg](acl_results/542_Frictional Agent Alignment Framework_ Slow Down and Don’t Break Things/tables/fd5045f8a065007b30f19087f4dcf22472831e224bad2504700c637414f60611.jpg)

## Powerformer: Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encryption

### Images

![6ebda65d265de36339c1a876daf0a97e4a7d46bfe59aa7e61850c89f8bf46f53.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/images/6ebda65d265de36339c1a876daf0a97e4a7d46bfe59aa7e61850c89f8bf46f53.jpg)

![c513008778f19922b32e573383e31ce6b921a023726a5ae8ec3b3be5d0c94ca1.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/images/c513008778f19922b32e573383e31ce6b921a023726a5ae8ec3b3be5d0c94ca1.jpg)

![d93ac4216f2f33fbda0e0d8356e2cdbe5a0ec9cda8d64ca9dfb9a4e8239e5ad1.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/images/d93ac4216f2f33fbda0e0d8356e2cdbe5a0ec9cda8d64ca9dfb9a4e8239e5ad1.jpg)

![f447b73c21638b61660bd3ae3821a15f2e29f2466dc184ea50eac673084a6430.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/images/f447b73c21638b61660bd3ae3821a15f2e29f2466dc184ea50eac673084a6430.jpg)

![f4e88fb9b54bbf9e252bc0fc985842442eaa9949ada9979c7f78732fa9214203.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/images/f4e88fb9b54bbf9e252bc0fc985842442eaa9949ada9979c7f78732fa9214203.jpg)

### Tables

![0966ac42cf6ae9044b44a31f94860dd00347ba541190682fbfd682fead7371e8.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/0966ac42cf6ae9044b44a31f94860dd00347ba541190682fbfd682fead7371e8.jpg)

![0c1f05c3136eed38f2a0edcea45134381e1d6d1124c3af5de28ee50e546bd116.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/0c1f05c3136eed38f2a0edcea45134381e1d6d1124c3af5de28ee50e546bd116.jpg)

![166a0d826e3bac505af6a33cc86978e91e994622dbbda88b68832db563fb367d.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/166a0d826e3bac505af6a33cc86978e91e994622dbbda88b68832db563fb367d.jpg)

![1b4a37824d76d1c7cc95f43f32072d6a32d0e1b40b8c2bd626bc044f16e4c4b7.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/1b4a37824d76d1c7cc95f43f32072d6a32d0e1b40b8c2bd626bc044f16e4c4b7.jpg)

![202dc7457756fc0827df44b9031b0d4edb05810b6207db6df55d715de246d531.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/202dc7457756fc0827df44b9031b0d4edb05810b6207db6df55d715de246d531.jpg)

![24e009392f6beb64e901a79e827a82a624b0147d53d658c91358d142362fbf7a.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/24e009392f6beb64e901a79e827a82a624b0147d53d658c91358d142362fbf7a.jpg)

![2d9fae58ff5df0c8ce3fead0bdc60140b1ab1926cf3d702788cd9e5a49f44c63.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/2d9fae58ff5df0c8ce3fead0bdc60140b1ab1926cf3d702788cd9e5a49f44c63.jpg)

![2e4523e214a47ced72f5711149d465e8415a804647b0f21271666f0dd6f1ead7.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/2e4523e214a47ced72f5711149d465e8415a804647b0f21271666f0dd6f1ead7.jpg)

![307a40f5700e15bb5b2c9f80befd13b2207764bab9892c2e189594c638bec3bc.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/307a40f5700e15bb5b2c9f80befd13b2207764bab9892c2e189594c638bec3bc.jpg)

![4066e19ad3149328495d6acef023021af47db1deaf757dd32ee6b3fb1b357129.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/4066e19ad3149328495d6acef023021af47db1deaf757dd32ee6b3fb1b357129.jpg)

![46634266e4b5e67ac973b5d5c0c0449c3b6335eb2da91c1efec40e8bdc560e82.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/46634266e4b5e67ac973b5d5c0c0449c3b6335eb2da91c1efec40e8bdc560e82.jpg)

![803634b3f3ef08e30f45cdcb6e09b41d91534595b2efd45bb69657418f2866f3.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/803634b3f3ef08e30f45cdcb6e09b41d91534595b2efd45bb69657418f2866f3.jpg)

![895583261571d447cf039d87b92d5386f2ccbb1cbf5ea173d607785b3c2efaf8.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/895583261571d447cf039d87b92d5386f2ccbb1cbf5ea173d607785b3c2efaf8.jpg)

![8a3ad33465bcfb5b50fa0de74acdffd5440ae8a357d32810592675eed457c17e.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/8a3ad33465bcfb5b50fa0de74acdffd5440ae8a357d32810592675eed457c17e.jpg)

![927af29c28f8e4d0943be6c0443c9ce453e5ce6c61f0d1a03d17acde89674e6e.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/927af29c28f8e4d0943be6c0443c9ce453e5ce6c61f0d1a03d17acde89674e6e.jpg)

![b565b0919b0f57a22f16d8e8a9bbf47790da789605c95784b131d535f68da31c.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/b565b0919b0f57a22f16d8e8a9bbf47790da789605c95784b131d535f68da31c.jpg)

![bd0723800dc3e6c8752f281efbfb926754ba5232c5500a0fbe8f13c49300a71b.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/bd0723800dc3e6c8752f281efbfb926754ba5232c5500a0fbe8f13c49300a71b.jpg)

![c494ad2c2e4b7116c5ab25f323e824c704ecea0368575fc097612103e4cf4608.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/c494ad2c2e4b7116c5ab25f323e824c704ecea0368575fc097612103e4cf4608.jpg)

![c584808e94554302bfc90574c6358261c0e73b3fffd917abd34865cf1689c8a2.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/c584808e94554302bfc90574c6358261c0e73b3fffd917abd34865cf1689c8a2.jpg)

![da66b93ec7478b0bb6997ad21fad4ffd9d9922a366225926f6cc70c8e811dce2.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/da66b93ec7478b0bb6997ad21fad4ffd9d9922a366225926f6cc70c8e811dce2.jpg)

![deecf8664d4a2081d17793706dc497a02dad2c3bf22a324a5bebf88982df7544.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/deecf8664d4a2081d17793706dc497a02dad2c3bf22a324a5bebf88982df7544.jpg)

![e727c5348e56277f745efe6bac337f8a630b847dabd9ee93c18e3d022b3841e4.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/e727c5348e56277f745efe6bac337f8a630b847dabd9ee93c18e3d022b3841e4.jpg)

![e911cf10dd79dee884f6a2face6f3acee6946d1df4fe6d1c64e56c55c9b71cdc.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/e911cf10dd79dee884f6a2face6f3acee6946d1df4fe6d1c64e56c55c9b71cdc.jpg)

![eba49d9fb267b1f1b7a2c435b90de183a42c23d323f26e4baf920308e80be273.jpg](acl_results/543_Powerformer_ Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encrypti/tables/eba49d9fb267b1f1b7a2c435b90de183a42c23d323f26e4baf920308e80be273.jpg)

## Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs

### Images

![04c0da4e876baf9aa76e7f3323c5328678caaf8830dc84a75092161c40c5680d.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/images/04c0da4e876baf9aa76e7f3323c5328678caaf8830dc84a75092161c40c5680d.jpg)

![1214e10a18f3227d25ccb9ad155ad80b2151c6a379e5410c940867038689285d.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/images/1214e10a18f3227d25ccb9ad155ad80b2151c6a379e5410c940867038689285d.jpg)

![40639e48f3702976696e6408034c3ccac6e2de114a45f5d1f98fc84c65017657.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/images/40639e48f3702976696e6408034c3ccac6e2de114a45f5d1f98fc84c65017657.jpg)

![6876779bad5388712beaefbe81238f719de0ca4f5fa3dd45c9b3598afa914715.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/images/6876779bad5388712beaefbe81238f719de0ca4f5fa3dd45c9b3598afa914715.jpg)

![6b4ace594d86c877af97d502cbf57516d346a63f392bfe22d3bbf07b344e4144.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/images/6b4ace594d86c877af97d502cbf57516d346a63f392bfe22d3bbf07b344e4144.jpg)

![70ce3641d1dcc64c7538b8fce27048eebaafd0babebd4906633b8971f5db3aa1.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/images/70ce3641d1dcc64c7538b8fce27048eebaafd0babebd4906633b8971f5db3aa1.jpg)

![a9ba5c5b7bdfd60e11e5393d55354276b6c28761942962b489d8f9bc0aad68e6.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/images/a9ba5c5b7bdfd60e11e5393d55354276b6c28761942962b489d8f9bc0aad68e6.jpg)

![bec83dd60d32efae453ac468942cd340f1773998508b6c41984926a3047469e1.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/images/bec83dd60d32efae453ac468942cd340f1773998508b6c41984926a3047469e1.jpg)

![d2d12a0f9339d23f135b15b872bb326dd22ba52b2f26581b78281433dac5ea5a.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/images/d2d12a0f9339d23f135b15b872bb326dd22ba52b2f26581b78281433dac5ea5a.jpg)

![e9f89aa4e1208a9ed7d1e0f6240448e425dfb2c4f4e47cb350937059683a8184.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/images/e9f89aa4e1208a9ed7d1e0f6240448e425dfb2c4f4e47cb350937059683a8184.jpg)

### Tables

![009586f136031a639ee03980cf7b7798d88ed4231e2b98f593bb49c8bccf253a.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/009586f136031a639ee03980cf7b7798d88ed4231e2b98f593bb49c8bccf253a.jpg)

![1156d49f8b52fb609db811988ce5761702669ef7e7746a91d7b30f809f9aa36b.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/1156d49f8b52fb609db811988ce5761702669ef7e7746a91d7b30f809f9aa36b.jpg)

![132ee4f270180e51d0b31197519bff9a940d9a4bdbcbc9fb8960b25ff89262b6.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/132ee4f270180e51d0b31197519bff9a940d9a4bdbcbc9fb8960b25ff89262b6.jpg)

![140b6c42c994e8460c9a22cb9341e6d930574c71954d1c1ad4d035aa17e12068.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/140b6c42c994e8460c9a22cb9341e6d930574c71954d1c1ad4d035aa17e12068.jpg)

![2e3372924321788cda165a913418edd3417a3b423f1bea1ea5a857c49ba1c312.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/2e3372924321788cda165a913418edd3417a3b423f1bea1ea5a857c49ba1c312.jpg)

![413d96d2b74405fbcf724fd2ceeb9362b44325c02c8ed55bc49a216101482eb4.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/413d96d2b74405fbcf724fd2ceeb9362b44325c02c8ed55bc49a216101482eb4.jpg)

![6cee432b4c2723dc16441c9c19c285e47797cc64df48152fdb99153fcccdcd86.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/6cee432b4c2723dc16441c9c19c285e47797cc64df48152fdb99153fcccdcd86.jpg)

![7effad6e59437245ed18a2696697ffb84bf89127c8ba8833307dca3b23cd4a7e.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/7effad6e59437245ed18a2696697ffb84bf89127c8ba8833307dca3b23cd4a7e.jpg)

![7f6b380b05e04b8c4314491818ffcfa23f8509e11103c281138ab0654f01ebc5.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/7f6b380b05e04b8c4314491818ffcfa23f8509e11103c281138ab0654f01ebc5.jpg)

![880cdec5b25f1a10e218744323386dcb7b41af20067c3b61a28776c9da5869a3.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/880cdec5b25f1a10e218744323386dcb7b41af20067c3b61a28776c9da5869a3.jpg)

![bdce9423e16707683ecb93e7b3670b7a0c91be5fd5c8153124946739e54c03dd.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/bdce9423e16707683ecb93e7b3670b7a0c91be5fd5c8153124946739e54c03dd.jpg)

![ca9d377131bd1c8cef9c64d8a6e61971c4dcf58c8815635716f3ba620ba83f7d.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/ca9d377131bd1c8cef9c64d8a6e61971c4dcf58c8815635716f3ba620ba83f7d.jpg)

![fc44a622d0b1af8c52c2bad50b4681c716e97b3ee83f89e8d4432ec0115d301b.jpg](acl_results/544_Beware of Your Po! Measuring and MitigatingAISafety Risks in Role-Play Fine-Tuning ofLLMs/tables/fc44a622d0b1af8c52c2bad50b4681c716e97b3ee83f89e8d4432ec0115d301b.jpg)