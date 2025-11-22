# ACL 2025 Long Papers


**Summary:** 1599 papers with extracted content:
- 📊 Total images: 13877
- 📋 Total tables: 16805
- 📄 Total files: 3068
---

# ACL 2025 Long Papers - Part 19 of 50

## 目录 (Table of Contents)

1. [TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](#TROVE-A-Challenge-for-Fine-Grained-Text-Provenance-via-Source-Sentence-Tracing-and-Relationship-Classification)
2. [CaLMQA: Exploring culturally specific long-form question answering across 23 languages](#CaLMQA-Exploring-culturally-specific-long-form-question-answering-across-23-languages)
3. [Croppable Knowledge Graph Embedding](#Croppable-Knowledge-Graph-Embedding)
4. [HyKGE: A Hypothesis Knowledge Graph EnhancedRAGFramework for Accurate and Reliable MedicalLLMs Responses](#HyKGE-A-Hypothesis-Knowledge-Graph-EnhancedRAGFramework-for-Accurate-and-Reliable-MedicalLLMs-Responses)
5. [LongRecipe: Recipe for Efficient Long Context Generalization in Large Language Models](#LongRecipe-Recipe-for-Efficient-Long-Context-Generalization-in-Large-Language-Models)
6. [BeamLoRA: Beam-Constraint Low-Rank Adaptation](#BeamLoRA-Beam-Constraint-Low-Rank-Adaptation)
7. [GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](#GODBench-A-Benchmark-for-Multimodal-Large-Language-Models-in-Video-Comment-Art)
8. [UniLR: Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever](#UniLR-Unleashing-the-Power-ofLLMs-on-Multiple-Legal-Tasks-with-a-Unified-Legal-Retriever)
9. [Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models](#Generative-Psycho-Lexical-Approach-for-Constructing-Value-Systems-in-Large-Language-Models)
10. [Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model](#Beyond-Dialogue-A-Profile-Dialogue-Alignment-Framework-Towards-General-Role-Playing-Language-Model)
11. [ACECODER: Acing CoderRLvia Automated Test-Case Synthesis](#ACECODER-Acing-CoderRLvia-Automated-Test-Case-Synthesis)
12. [Quantifying Semantic Emergence in Language Models](#Quantifying-Semantic-Emergence-in-Language-Models)
13. [DebateCoder: Towards Collective Intelligence ofLLMs via Test Case DrivenLLMDebate for Code Generation](#DebateCoder-Towards-Collective-Intelligence-ofLLMs-via-Test-Case-DrivenLLMDebate-for-Code-Generation)
14. [The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](#The-Tug-of-War-Within-Mitigating-the-Fairness-Privacy-Conflicts-in-Large-Language-Models)
15. [GraphInsight: Unlocking Insights in Large Language Models for Graph Structure Understanding](#GraphInsight-Unlocking-Insights-in-Large-Language-Models-for-Graph-Structure-Understanding)
16. [Phonotomizer: A Compact, Unsupervised, Online Training Approach to Real-Time, Multilingual Phonetic Segmentation](#Phonotomizer-A-Compact-Unsupervised-Online-Training-Approach-to-Real-Time-Multilingual-Phonetic-Segmentation)
17. [A Multi-persona Framework for Argument Quality Assessment](#A-Multi-persona-Framework-for-Argument-Quality-Assessment)
18. [Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](#Safe-Enhancing-Mathematical-Reasoning-in-Large-Language-Models-via-Retrospective-Step-aware-Formal-Verification)
19. [SAMDecoding: Speculative Decoding via Suffix Automaton](#SAMDecoding-Speculative-Decoding-via-Suffix-Automaton)
20. [PsyAdvisor: A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conversations](#PsyAdvisor-A-Plug-and-Play-Strategy-Advice-Planner-with-Proactive-Questioning-in-Psychological-Conversations)
21. [HomeBench: EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multiple Devices](#HomeBench-EvaluatingLLMs-in-Smart-Homes-with-Valid-and-Invalid-Instructions-Across-Single-and-Multiple-Devices)
22. [Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment](#Advancing-Zero-shot-Text-to-Speech-Intelligibility-across-Diverse-Domains-via-Preference-Alignment)
23. [GiFT:Gibbs Fine-Tuning for Code Generation](#GiFTGibbs-Fine-Tuning-for-Code-Generation)
24. [Enhancing Interpretable Image Classification ThroughLLMAgents and Conditional Concept Bottleneck Models](#Enhancing-Interpretable-Image-Classification-ThroughLLMAgents-and-Conditional-Concept-Bottleneck-Models)
25. [Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction](#Reliably-Bounding-False-Positives-A-Zero-Shot-Machine-Generated-Text-Detection-Framework-via-Multiscaled-Conformal-Prediction)
26. [RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph](#RSCF-Relation-Semantics-Consistent-Filter-for-Entity-Embedding-of-Knowledge-Graph)
27. [RolePlot: A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of Role-Playing Agents](#RolePlot-A-Systematic-Framework-for-Evaluating-and-Enhancing-the-Plot-Progression-Capabilities-of-Role-Playing-Agents)
28. [TreeRL:LLMReinforcement Learning with On-Policy Tree Search](#TreeRLLLMReinforcement-Learning-with-On-Policy-Tree-Search)
29. [Can a Single Model Master Both Multi-turn Conversations and Tool Use?CoALM: A Unified Conversational Agentic Language Model](#Can-a-Single-Model-Master-Both-Multi-turn-Conversations-and-Tool-UseCoALM-A-Unified-Conversational-Agentic-Language-Model)
30. [Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation](#Single-to-mix-Modality-Alignment-with-Multimodal-Large-Language-Model-for-Document-Image-Machine-Translation)
31. [SDPO: Segment-Level Direct Preference Optimization for Social Agents](#SDPO-Segment-Level-Direct-Preference-Optimization-for-Social-Agents)
32. [KokoroChat: AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors](#KokoroChat-AJapanese-Psychological-Counseling-Dialogue-Dataset-Collected-via-Role-Playing-by-Trained-Counselors)

---


## TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification

### Images

![085dfc14169e400b80d186a5ceb4a66f01cae65c531c547be9e5f9694dc99c3a.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/images/085dfc14169e400b80d186a5ceb4a66f01cae65c531c547be9e5f9694dc99c3a.jpg)

![26df301dfc03f0d222d94d6cbc4968384f7eb36e1f3f569086e7dd751a09f65d.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/images/26df301dfc03f0d222d94d6cbc4968384f7eb36e1f3f569086e7dd751a09f65d.jpg)

![39b4adde1d4cce64d226596279e447d2455620c07b3a954ba4547e727b2dc940.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/images/39b4adde1d4cce64d226596279e447d2455620c07b3a954ba4547e727b2dc940.jpg)

![45b5c7ea3c6fad94adbd26da9c95f7ef877931b12835005b6ae9c8818b867b55.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/images/45b5c7ea3c6fad94adbd26da9c95f7ef877931b12835005b6ae9c8818b867b55.jpg)

![4e37e4cfd8a8f43309266071c184b6b9ce8d79840f319ea974013d81d9b8d893.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/images/4e37e4cfd8a8f43309266071c184b6b9ce8d79840f319ea974013d81d9b8d893.jpg)

![53f071812db033db722647ab55bedf8b12e02810e73be2ffdac3772a537173ff.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/images/53f071812db033db722647ab55bedf8b12e02810e73be2ffdac3772a537173ff.jpg)

![b99e4b8d1a9cfdf077eef53911ae55bdf9b1de93c672440742daad56f7fa3d02.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/images/b99e4b8d1a9cfdf077eef53911ae55bdf9b1de93c672440742daad56f7fa3d02.jpg)

![bd5de7d514c9fe3cd04c58ef200d3371fab80df635fe39dbc594c0e5dece04a6.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/images/bd5de7d514c9fe3cd04c58ef200d3371fab80df635fe39dbc594c0e5dece04a6.jpg)

![c9bcbdc127e16661593af4207a35f02b4da52ce26085fa9beabdaf94625b01f7.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/images/c9bcbdc127e16661593af4207a35f02b4da52ce26085fa9beabdaf94625b01f7.jpg)

![db950f0424dfbaaeffd46108afdcbc8ab7d6900010835241ca10b06945579fba.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/images/db950f0424dfbaaeffd46108afdcbc8ab7d6900010835241ca10b06945579fba.jpg)

![fe3604f531e9f9c2242094f5944b9f4818e258cb6e27f6400a659e8d6baf6a91.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/images/fe3604f531e9f9c2242094f5944b9f4818e258cb6e27f6400a659e8d6baf6a91.jpg)

### Tables

![18230ad40484377dc37337b68e609e1907e1d491190034887ab1f640a3bd53b6.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/tables/18230ad40484377dc37337b68e609e1907e1d491190034887ab1f640a3bd53b6.jpg)

![67e9416fb62e57f35640eb20f56b14af5cb00141ebd8b5788f089424a751244b.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/tables/67e9416fb62e57f35640eb20f56b14af5cb00141ebd8b5788f089424a751244b.jpg)

![742fa65e83a08e508e03d2d922ecc0806e849795269d781d1d93ba5f0de41084.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/tables/742fa65e83a08e508e03d2d922ecc0806e849795269d781d1d93ba5f0de41084.jpg)

![a7ab93aa2821ea7239fface735544a869d95965e02b8a736ba9edd394b7120d8.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/tables/a7ab93aa2821ea7239fface735544a869d95965e02b8a736ba9edd394b7120d8.jpg)

![b6d9f860ef827b1643e8f999509069d75d2bd58228ccc35d2221182ae08a353b.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/tables/b6d9f860ef827b1643e8f999509069d75d2bd58228ccc35d2221182ae08a353b.jpg)

![ec3efe977ab38c0d1939003371376c5da6e139a5db77fdc4ffae678033e79a23.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/tables/ec3efe977ab38c0d1939003371376c5da6e139a5db77fdc4ffae678033e79a23.jpg)

![f96ae057c9b2d24a6c1748192cf8124681a8cb0170ae1b6444bb53bdec76bcba.jpg](acl_results/577_TROVE_ A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Cla/tables/f96ae057c9b2d24a6c1748192cf8124681a8cb0170ae1b6444bb53bdec76bcba.jpg)

## CaLMQA: Exploring culturally specific long-form question answering across 23 languages

### Images

![0dd28bf1ae121d39fe447206ab4cc4dff7326587f14a388555f18894e7c771da.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/0dd28bf1ae121d39fe447206ab4cc4dff7326587f14a388555f18894e7c771da.jpg)

![1a3fb6303b4541d604bc57573cc4951082be78351a77872068c6d16a47d360ae.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/1a3fb6303b4541d604bc57573cc4951082be78351a77872068c6d16a47d360ae.jpg)

![20966e18d52976592de7311ccc89e6d65ee13732dcf16460292bb14a1cc4de5d.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/20966e18d52976592de7311ccc89e6d65ee13732dcf16460292bb14a1cc4de5d.jpg)

![5db6dd4f98191b21dfe923f39a0f980273e10fe7e4273df531db471adf3ff5b6.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/5db6dd4f98191b21dfe923f39a0f980273e10fe7e4273df531db471adf3ff5b6.jpg)

![69974a7a059136723c5d4294b62495759d2e3b4e3f89630b7a44246120409404.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/69974a7a059136723c5d4294b62495759d2e3b4e3f89630b7a44246120409404.jpg)

![76c651101a064f90df364ddd3f814d72b8117071c8742546de1b76100444059b.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/76c651101a064f90df364ddd3f814d72b8117071c8742546de1b76100444059b.jpg)

![88960773d9b54c9ea08aef600f79d9460ad702abb61ab5f5419b529c02947811.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/88960773d9b54c9ea08aef600f79d9460ad702abb61ab5f5419b529c02947811.jpg)

![922cc1123822b833511823fe3c953c246ebf2d730cb5c63611fdd333947f2564.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/922cc1123822b833511823fe3c953c246ebf2d730cb5c63611fdd333947f2564.jpg)

![97c84330c0e6f36b68f643909e698e43e2658b51c6876fc89403df2b7e78935b.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/97c84330c0e6f36b68f643909e698e43e2658b51c6876fc89403df2b7e78935b.jpg)

![9992dd116d1ef37932173046dd2c4045f1114ad610a172627e6303e94523a7fb.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/9992dd116d1ef37932173046dd2c4045f1114ad610a172627e6303e94523a7fb.jpg)

![a39ad71725fea0fbff89cebbe94965ed0b5477101488a19ecfb4c388bb9bf3bb.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/a39ad71725fea0fbff89cebbe94965ed0b5477101488a19ecfb4c388bb9bf3bb.jpg)

![aa4723c7054a799290a894094fdea32fa981527f9e853ea1d56b01c1c72b2ca6.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/aa4723c7054a799290a894094fdea32fa981527f9e853ea1d56b01c1c72b2ca6.jpg)

![b577bfef8d96ddb20363410fe32e3333d3c52551f9096dc1d22fbfeca8cd43a5.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/b577bfef8d96ddb20363410fe32e3333d3c52551f9096dc1d22fbfeca8cd43a5.jpg)

![b6715ef8d61fb01686baf1e71ca672f8b8964a14d5d857d8fa3c67257364e110.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/b6715ef8d61fb01686baf1e71ca672f8b8964a14d5d857d8fa3c67257364e110.jpg)

![c1eec5d2facb928155a8740416d884a7af442c920c48b69ec230be2e68e72918.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/c1eec5d2facb928155a8740416d884a7af442c920c48b69ec230be2e68e72918.jpg)

![db71c1aa4bb692d06c634aa9b9ee39267ad839edccad0a5fc162b2128be751a2.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/db71c1aa4bb692d06c634aa9b9ee39267ad839edccad0a5fc162b2128be751a2.jpg)

![e1befd4dc52628b20bd0357a28d49c448bbd7a160feec63f210ea3985567db2e.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/e1befd4dc52628b20bd0357a28d49c448bbd7a160feec63f210ea3985567db2e.jpg)

![eccde16c8c4d3610f926ef5bee2ec8eff3273e60f73d585ffa14051bac1bed94.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/eccde16c8c4d3610f926ef5bee2ec8eff3273e60f73d585ffa14051bac1bed94.jpg)

![f20d93fc1f8ef26ed30439001747ec05f2e86336b05747f0d310e09fdb2ea1f3.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/f20d93fc1f8ef26ed30439001747ec05f2e86336b05747f0d310e09fdb2ea1f3.jpg)

![f44677541877525fa25b1cb1d813b406d87726dde5e485761030340d57984d62.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/f44677541877525fa25b1cb1d813b406d87726dde5e485761030340d57984d62.jpg)

![faee6ba3c13a65ded9de0224a7f136eae51697e25bc8dba5aa41904ab84bb88f.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/faee6ba3c13a65ded9de0224a7f136eae51697e25bc8dba5aa41904ab84bb88f.jpg)

![fd28a49af54347327607ef3dc0bf66f50dfbcd63680b7928092a5e98eb61dd19.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/fd28a49af54347327607ef3dc0bf66f50dfbcd63680b7928092a5e98eb61dd19.jpg)

![ff7386bb3256bcb9e49cec3ed7ff1db5118342489986e95382bf0d3486eae9e2.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/images/ff7386bb3256bcb9e49cec3ed7ff1db5118342489986e95382bf0d3486eae9e2.jpg)

### Tables

![02aabfd6ebc305605f5b869bc7bfe6350bda9b598e27bd9694c9c5fbcb57a5c7.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/02aabfd6ebc305605f5b869bc7bfe6350bda9b598e27bd9694c9c5fbcb57a5c7.jpg)

![04c0140a01356e2a15683f29d8a30a733c54b55974f5fd1000664a2804cb034f.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/04c0140a01356e2a15683f29d8a30a733c54b55974f5fd1000664a2804cb034f.jpg)

![087be93dc4d45f30ddd36e4368a58cdc30e88b98a66d33cc32243ceb57789c56.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/087be93dc4d45f30ddd36e4368a58cdc30e88b98a66d33cc32243ceb57789c56.jpg)

![1407eb03460e0c9f8ba48088604b351e98d9a8b0ea17ed12729320aea5b2e44a.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/1407eb03460e0c9f8ba48088604b351e98d9a8b0ea17ed12729320aea5b2e44a.jpg)

![17de327eb07a0a2fd6d4d1a2c62bbfff3041782ba412bd3e6b882a5360febfde.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/17de327eb07a0a2fd6d4d1a2c62bbfff3041782ba412bd3e6b882a5360febfde.jpg)

![1ff822022295392a075ca84ac6eae3f24a4b2ae93b7a64a0d0a80f2dad0ece52.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/1ff822022295392a075ca84ac6eae3f24a4b2ae93b7a64a0d0a80f2dad0ece52.jpg)

![23c6b1af24d155972e09e2524353ffff38a7576e55b8593a8c0893929a3e9b26.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/23c6b1af24d155972e09e2524353ffff38a7576e55b8593a8c0893929a3e9b26.jpg)

![34b89cf83af2fc366bea84fb7fff5df8c463de499ef41e1b6170d457d0407edb.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/34b89cf83af2fc366bea84fb7fff5df8c463de499ef41e1b6170d457d0407edb.jpg)

![491e4813dbb12c3283e95b8600de3ec807c6e7752f0e8637b13bb871acaf4a2f.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/491e4813dbb12c3283e95b8600de3ec807c6e7752f0e8637b13bb871acaf4a2f.jpg)

![618254de068181d737a4d267aa42ffa685984bddb732f66104b61208e1f25acf.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/618254de068181d737a4d267aa42ffa685984bddb732f66104b61208e1f25acf.jpg)

![66742827a4d9b78cc6f1cd0caf016bf8607a27a53818bd60ebbdd6ed46719149.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/66742827a4d9b78cc6f1cd0caf016bf8607a27a53818bd60ebbdd6ed46719149.jpg)

![6b0bdb83fd9ba243e4129a1d6a1e5d2fccb1503bb2e831df4d8b158bb483987c.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/6b0bdb83fd9ba243e4129a1d6a1e5d2fccb1503bb2e831df4d8b158bb483987c.jpg)

![6fa757ba5855ad922533e91312b0e8cf46e82a4f6870b0771d59d35e4d2337af.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/6fa757ba5855ad922533e91312b0e8cf46e82a4f6870b0771d59d35e4d2337af.jpg)

![756550011190925c06515ac16b64b695bd7ba12495a8045f5a4a5afbc2263a3a.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/756550011190925c06515ac16b64b695bd7ba12495a8045f5a4a5afbc2263a3a.jpg)

![78203067eb0f23e6034934e411b0623b3e96bbb6fa35c52f375c995ec3118de5.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/78203067eb0f23e6034934e411b0623b3e96bbb6fa35c52f375c995ec3118de5.jpg)

![808514940a476f15034bad2207170c1cbd2084bfe6f2ffd9fb35f6709c6332f2.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/808514940a476f15034bad2207170c1cbd2084bfe6f2ffd9fb35f6709c6332f2.jpg)

![96ee5f69477bdb3437b9f21928c6d8712851a40633336d08ca2e6ab8d69c9388.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/96ee5f69477bdb3437b9f21928c6d8712851a40633336d08ca2e6ab8d69c9388.jpg)

![9d079f4ecdcf2a47cc30344d73930614216f314f57ab419eb62461e680906fb9.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/9d079f4ecdcf2a47cc30344d73930614216f314f57ab419eb62461e680906fb9.jpg)

![a3b402ddbcd32b7b162f6de5e24aa6d1f0afb9ff4d08ec66823db74b4211f024.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/a3b402ddbcd32b7b162f6de5e24aa6d1f0afb9ff4d08ec66823db74b4211f024.jpg)

![a898653df8ebe318ac57dbf4288e6fd834a31357a20574f7aeae094448e80061.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/a898653df8ebe318ac57dbf4288e6fd834a31357a20574f7aeae094448e80061.jpg)

![ac6ab71303018070565d503bed7594ca5c6751b5d186fd7235b83fcd630a150e.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/ac6ab71303018070565d503bed7594ca5c6751b5d186fd7235b83fcd630a150e.jpg)

![aef902e08785d503f8b62d18237608ab36e67261dd545e10dbc6108c8d5c1e24.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/aef902e08785d503f8b62d18237608ab36e67261dd545e10dbc6108c8d5c1e24.jpg)

![b16388daf73976297a39f1ff4130b92303764ca59ec97a5b940ec122acc70e0c.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/b16388daf73976297a39f1ff4130b92303764ca59ec97a5b940ec122acc70e0c.jpg)

![bd11e490d4365c1afc78775366f9bf0f614385843f2f8a4fed7627159abc5456.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/bd11e490d4365c1afc78775366f9bf0f614385843f2f8a4fed7627159abc5456.jpg)

![c6147f9a520557323729fa8ddc4a15689b7e77c3f1cd68d554afb86ce0c6cfcc.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/c6147f9a520557323729fa8ddc4a15689b7e77c3f1cd68d554afb86ce0c6cfcc.jpg)

![ce042921291fbf75cfa53ab4b53f8c85e65c5fa04d8989ec80955d36781a0264.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/ce042921291fbf75cfa53ab4b53f8c85e65c5fa04d8989ec80955d36781a0264.jpg)

![d3c96894e85b2444279ead0827e6cb3ccae09529133743600845189f178bd0d7.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/d3c96894e85b2444279ead0827e6cb3ccae09529133743600845189f178bd0d7.jpg)

![e38674138f9dc0d21b4d99f5c714bfff71fb66d077b95d7d32a7951de51db1a3.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/e38674138f9dc0d21b4d99f5c714bfff71fb66d077b95d7d32a7951de51db1a3.jpg)

![ef12f57e2a5c9c4248dec9694ea5f6a229948d4e99addbaf24dc6d934fa13c5a.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/ef12f57e2a5c9c4248dec9694ea5f6a229948d4e99addbaf24dc6d934fa13c5a.jpg)

![f15921918b25ce883153b0322c7925f8dd532839a26fb1b613e047a3a1e4a563.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/f15921918b25ce883153b0322c7925f8dd532839a26fb1b613e047a3a1e4a563.jpg)

![f1de620774bfe098049ca435c961ee7c948742e683a299e4828f7b9efa832e5f.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/f1de620774bfe098049ca435c961ee7c948742e683a299e4828f7b9efa832e5f.jpg)

![f476d62a75b19becc0049db345d8c080e5b817850ca4e58e3f2548efbf58c1a7.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/f476d62a75b19becc0049db345d8c080e5b817850ca4e58e3f2548efbf58c1a7.jpg)

![f8a6ebb08ea0e51bb543cd9624e3b521ed68493722c8a6e84f36924d13799e1a.jpg](acl_results/578_CaLMQA_ Exploring culturally specific long-form question answering across 23 languages/tables/f8a6ebb08ea0e51bb543cd9624e3b521ed68493722c8a6e84f36924d13799e1a.jpg)

## Croppable Knowledge Graph Embedding

### Images

![05cdde18ab8f8121d6454e1b857b1dc616dbdd9a493c752c6c2d5cdc3979eaa3.jpg](acl_results/579_Croppable Knowledge Graph Embedding/images/05cdde18ab8f8121d6454e1b857b1dc616dbdd9a493c752c6c2d5cdc3979eaa3.jpg)

![5da5a0bfc6dac162aed32679dd6338478be1d4681cbf1a0dbc84c5e6ca8b4afb.jpg](acl_results/579_Croppable Knowledge Graph Embedding/images/5da5a0bfc6dac162aed32679dd6338478be1d4681cbf1a0dbc84c5e6ca8b4afb.jpg)

![6aac61af5c87b8d9139902d144f3a7a8ac6b687fdb61cc18421ab3be12fa6852.jpg](acl_results/579_Croppable Knowledge Graph Embedding/images/6aac61af5c87b8d9139902d144f3a7a8ac6b687fdb61cc18421ab3be12fa6852.jpg)

![b34f0c28bd83f3c26d3a2a79fd5243ce23ba286c60604541b2c6ce0448802887.jpg](acl_results/579_Croppable Knowledge Graph Embedding/images/b34f0c28bd83f3c26d3a2a79fd5243ce23ba286c60604541b2c6ce0448802887.jpg)

![d78b45f767a902ab7c6d945e0da1c2cccd613baa681414dcd70b271f7b0a8fca.jpg](acl_results/579_Croppable Knowledge Graph Embedding/images/d78b45f767a902ab7c6d945e0da1c2cccd613baa681414dcd70b271f7b0a8fca.jpg)

![dd10718653d3255af22c9bae856f928430e60073ab1d457e194815f44d6e4f71.jpg](acl_results/579_Croppable Knowledge Graph Embedding/images/dd10718653d3255af22c9bae856f928430e60073ab1d457e194815f44d6e4f71.jpg)

![f8661e7a5d0ccb2756e92e67b207e6a94077aefc282bad18e7eb64d7c7a3cf57.jpg](acl_results/579_Croppable Knowledge Graph Embedding/images/f8661e7a5d0ccb2756e92e67b207e6a94077aefc282bad18e7eb64d7c7a3cf57.jpg)

### Tables

![1d6ad398a649bb35b652d450b1feb1385eb3d7626ad7f91b77c01bc38a9ecf24.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/1d6ad398a649bb35b652d450b1feb1385eb3d7626ad7f91b77c01bc38a9ecf24.jpg)

![207bf2add0bf273ec72c93b11ed78d0fdd7d521ce6b5fc4a75492f10bf96cbe5.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/207bf2add0bf273ec72c93b11ed78d0fdd7d521ce6b5fc4a75492f10bf96cbe5.jpg)

![4532fd2ee73c7f6ee80f630dc8f56fc25e5696b6d8c64e53c4a7a033ec07cd2a.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/4532fd2ee73c7f6ee80f630dc8f56fc25e5696b6d8c64e53c4a7a033ec07cd2a.jpg)

![478daf120d37c0f4753eb297ff09eaedbfd5179ef35c50a5438e9ff712281af4.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/478daf120d37c0f4753eb297ff09eaedbfd5179ef35c50a5438e9ff712281af4.jpg)

![4808627f6a4250a89ba0d5dd117f8e40e8db0a4eb2675e6bfd6a6e3e4f12023b.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/4808627f6a4250a89ba0d5dd117f8e40e8db0a4eb2675e6bfd6a6e3e4f12023b.jpg)

![53fc8c20fbc019c275df5371e3f1071a540263dc746b5a6cf045e7d51c72f37f.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/53fc8c20fbc019c275df5371e3f1071a540263dc746b5a6cf045e7d51c72f37f.jpg)

![8619a41ba82d5520532dd6be17b3f1061591b437970ca918be16a44b85da93a2.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/8619a41ba82d5520532dd6be17b3f1061591b437970ca918be16a44b85da93a2.jpg)

![8650c136c274a276ea48f7be319d956d6b3f6369517b1dbe618e3ecfa3695472.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/8650c136c274a276ea48f7be319d956d6b3f6369517b1dbe618e3ecfa3695472.jpg)

![aac5cd42961a9d178e576a07a58dc5b7ff2b8fbc4692ed1eca865a84a58856d6.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/aac5cd42961a9d178e576a07a58dc5b7ff2b8fbc4692ed1eca865a84a58856d6.jpg)

![c727688ad08db181374be8641853ba3c946bb7c9704c323444e08f33af155e84.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/c727688ad08db181374be8641853ba3c946bb7c9704c323444e08f33af155e84.jpg)

![c87605d262ac0fa591640c73bbaab2f3064d7ccea65841925da25f11d3e87cc2.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/c87605d262ac0fa591640c73bbaab2f3064d7ccea65841925da25f11d3e87cc2.jpg)

![d099788bb8923e5642d0c252469cc19a8ed818879e5e60ab51bb22ba9f8bad1d.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/d099788bb8923e5642d0c252469cc19a8ed818879e5e60ab51bb22ba9f8bad1d.jpg)

![e5e9544bf6aca2d9bb40d35a1425cfdef15beaea5c55d859e659a3f33e0a27e3.jpg](acl_results/579_Croppable Knowledge Graph Embedding/tables/e5e9544bf6aca2d9bb40d35a1425cfdef15beaea5c55d859e659a3f33e0a27e3.jpg)

## HyKGE: A Hypothesis Knowledge Graph EnhancedRAGFramework for Accurate and Reliable MedicalLLMs Responses

### Images

![058cf7f874b285496c5b8008c42eaff58cb85c928dcc4f58e697215d6631837e.jpg](acl_results/580_HyKGE_ A Hypothesis Knowledge Graph EnhancedRAGFramework for Accurate and Reliable MedicalLLMs Respo/images/058cf7f874b285496c5b8008c42eaff58cb85c928dcc4f58e697215d6631837e.jpg)

![15d2bec50ea3114f4e3c33ee5580f2460405bcd9bb25bf1d4fb1645a759d38db.jpg](acl_results/580_HyKGE_ A Hypothesis Knowledge Graph EnhancedRAGFramework for Accurate and Reliable MedicalLLMs Respo/images/15d2bec50ea3114f4e3c33ee5580f2460405bcd9bb25bf1d4fb1645a759d38db.jpg)

![ad1a236f8ae7c67435f46730abfa0fc0b48c8f8e91ab263d07e7111af844fa25.jpg](acl_results/580_HyKGE_ A Hypothesis Knowledge Graph EnhancedRAGFramework for Accurate and Reliable MedicalLLMs Respo/images/ad1a236f8ae7c67435f46730abfa0fc0b48c8f8e91ab263d07e7111af844fa25.jpg)

### Tables

![06de2348d0eab0dd111b94a1732c4237d4b6455bc6b30b9ad05002c981fdf8ab.jpg](acl_results/580_HyKGE_ A Hypothesis Knowledge Graph EnhancedRAGFramework for Accurate and Reliable MedicalLLMs Respo/tables/06de2348d0eab0dd111b94a1732c4237d4b6455bc6b30b9ad05002c981fdf8ab.jpg)

![308e30b4778d2a651683eb7b75b21f3a33cdefe87439650938f5a5558541dbae.jpg](acl_results/580_HyKGE_ A Hypothesis Knowledge Graph EnhancedRAGFramework for Accurate and Reliable MedicalLLMs Respo/tables/308e30b4778d2a651683eb7b75b21f3a33cdefe87439650938f5a5558541dbae.jpg)

![52531314ef44063a9ee25c8751c96e17228b3934f1e9b6a311807f8c835b9fbc.jpg](acl_results/580_HyKGE_ A Hypothesis Knowledge Graph EnhancedRAGFramework for Accurate and Reliable MedicalLLMs Respo/tables/52531314ef44063a9ee25c8751c96e17228b3934f1e9b6a311807f8c835b9fbc.jpg)

![8be0d870c385fb83358253ddff1272983d2c5e014b084bd410afa48be1b2e2a9.jpg](acl_results/580_HyKGE_ A Hypothesis Knowledge Graph EnhancedRAGFramework for Accurate and Reliable MedicalLLMs Respo/tables/8be0d870c385fb83358253ddff1272983d2c5e014b084bd410afa48be1b2e2a9.jpg)

## LongRecipe: Recipe for Efficient Long Context Generalization in Large Language Models

### Images

![19f797fa9206978be89474eb0439f042716573b4f1f44b84f39f068bd5add62b.jpg](acl_results/581_LongRecipe_ Recipe for Efficient Long Context Generalization in Large Language Models/images/19f797fa9206978be89474eb0439f042716573b4f1f44b84f39f068bd5add62b.jpg)

![af0cf7374dfdc3ca2441e6a68a5aeb2369b7f112b2f03c3572d7a66317b8c981.jpg](acl_results/581_LongRecipe_ Recipe for Efficient Long Context Generalization in Large Language Models/images/af0cf7374dfdc3ca2441e6a68a5aeb2369b7f112b2f03c3572d7a66317b8c981.jpg)

![e3fa82b9cba3f238f71a78db1e49d421d5f67a559d767859adef68e4bb0b9f62.jpg](acl_results/581_LongRecipe_ Recipe for Efficient Long Context Generalization in Large Language Models/images/e3fa82b9cba3f238f71a78db1e49d421d5f67a559d767859adef68e4bb0b9f62.jpg)

### Tables

![1d9fef09cd7e66db972c701bd78adc0072f57b5943760bba06ecc13abdb2c2b9.jpg](acl_results/581_LongRecipe_ Recipe for Efficient Long Context Generalization in Large Language Models/tables/1d9fef09cd7e66db972c701bd78adc0072f57b5943760bba06ecc13abdb2c2b9.jpg)

![739d7aef64cbc18273f6012d7e5f7099b236a079d571ce2950285f8d6f29a705.jpg](acl_results/581_LongRecipe_ Recipe for Efficient Long Context Generalization in Large Language Models/tables/739d7aef64cbc18273f6012d7e5f7099b236a079d571ce2950285f8d6f29a705.jpg)

![7812d360bf5614cce2158f4e2621e41307f2b5a7b8c909cebb1bdb107fa56762.jpg](acl_results/581_LongRecipe_ Recipe for Efficient Long Context Generalization in Large Language Models/tables/7812d360bf5614cce2158f4e2621e41307f2b5a7b8c909cebb1bdb107fa56762.jpg)

![87570dc88ece87b179193a239af701513f0e02056d7605cce022c00dfcf6df6c.jpg](acl_results/581_LongRecipe_ Recipe for Efficient Long Context Generalization in Large Language Models/tables/87570dc88ece87b179193a239af701513f0e02056d7605cce022c00dfcf6df6c.jpg)

![9394ab3fafe35fbf8d5b4c822fb6d5fa3b0433bf78b81247cd505ffe6640706c.jpg](acl_results/581_LongRecipe_ Recipe for Efficient Long Context Generalization in Large Language Models/tables/9394ab3fafe35fbf8d5b4c822fb6d5fa3b0433bf78b81247cd505ffe6640706c.jpg)

![98110a8d18bafe9ffea21509de3d90ce729e426a55273472d540112b20bb7646.jpg](acl_results/581_LongRecipe_ Recipe for Efficient Long Context Generalization in Large Language Models/tables/98110a8d18bafe9ffea21509de3d90ce729e426a55273472d540112b20bb7646.jpg)

![cd5f0519bdbb79bfd2cb4f9df2e535e67b69122d2dface14b18223a157220941.jpg](acl_results/581_LongRecipe_ Recipe for Efficient Long Context Generalization in Large Language Models/tables/cd5f0519bdbb79bfd2cb4f9df2e535e67b69122d2dface14b18223a157220941.jpg)

## BeamLoRA: Beam-Constraint Low-Rank Adaptation

### Images

![5ebdd27e4103fe30b6341de01da1d313a80e32cf33875b73107abd7444158285.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/images/5ebdd27e4103fe30b6341de01da1d313a80e32cf33875b73107abd7444158285.jpg)

![694ebb0a47a666229350d86fcc9ba54fb4112f33d8dc1b8de1ff0baa31800569.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/images/694ebb0a47a666229350d86fcc9ba54fb4112f33d8dc1b8de1ff0baa31800569.jpg)

![701308d420729f4594e1e75db8ae589d88855c51a7a285cf98259e8f3ff215af.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/images/701308d420729f4594e1e75db8ae589d88855c51a7a285cf98259e8f3ff215af.jpg)

![7bda6cad98272158de4357806d721200fc57beb16ab9c6e695686246a1824a5d.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/images/7bda6cad98272158de4357806d721200fc57beb16ab9c6e695686246a1824a5d.jpg)

![83e73088475ada16294d44feb594dfe70eb29ce636af2638a729124d92c1ff59.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/images/83e73088475ada16294d44feb594dfe70eb29ce636af2638a729124d92c1ff59.jpg)

![9461ecf8c340532d6540eda7007d7bd83d10755010371290154138ed70e25f8a.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/images/9461ecf8c340532d6540eda7007d7bd83d10755010371290154138ed70e25f8a.jpg)

![c71ee05a8cde38a81d2d4c6727976674760f9b324269a1693788dfc3d6823b8b.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/images/c71ee05a8cde38a81d2d4c6727976674760f9b324269a1693788dfc3d6823b8b.jpg)

![e847d0b61e3cf6d670ea660789d30c6274abbdcdaa195a9f2b104292d3dbe746.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/images/e847d0b61e3cf6d670ea660789d30c6274abbdcdaa195a9f2b104292d3dbe746.jpg)

![ea69bd6cb4c7a38049c56244fce3f6ccc89de427790ff1e170adc9f6db32cd77.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/images/ea69bd6cb4c7a38049c56244fce3f6ccc89de427790ff1e170adc9f6db32cd77.jpg)

### Tables

![3f49993d41d694e2aeb68a8b5bd10bd225924f2b9c9ac4dd1af6d06f4608a26b.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/tables/3f49993d41d694e2aeb68a8b5bd10bd225924f2b9c9ac4dd1af6d06f4608a26b.jpg)

![6c3b61c3efe2db1ac76d864a6419176a391168648d01a56226def3cfc559fb1a.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/tables/6c3b61c3efe2db1ac76d864a6419176a391168648d01a56226def3cfc559fb1a.jpg)

![a6b88882faf7300c0791e3df1fdb5f2700c2373a79455a22d850fcb5a3e078c2.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/tables/a6b88882faf7300c0791e3df1fdb5f2700c2373a79455a22d850fcb5a3e078c2.jpg)

![d6943ccabc239d21b00878bffe60fb548cd24c923a014f9a7e1e1ec366e00702.jpg](acl_results/582_BeamLoRA_ Beam-Constraint Low-Rank Adaptation/tables/d6943ccabc239d21b00878bffe60fb548cd24c923a014f9a7e1e1ec366e00702.jpg)

## GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art

### Images

![00758d9063d3bdefa311b198a0b258c4e5c79fdbb2a829df74181e9b259f772a.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/00758d9063d3bdefa311b198a0b258c4e5c79fdbb2a829df74181e9b259f772a.jpg)

![0097a20cfa4b86a752b895eb74f65a8ea1bdab84f29732fbb520422af9b1a687.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/0097a20cfa4b86a752b895eb74f65a8ea1bdab84f29732fbb520422af9b1a687.jpg)

![026ce3c5ccfee24eb1bc558fefe2866892fe91880ade67c948d59971f799bbda.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/026ce3c5ccfee24eb1bc558fefe2866892fe91880ade67c948d59971f799bbda.jpg)

![053fa18bc1d59308d1b21aa627bfff14d41be4d5d8bb995898aa4984575c8614.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/053fa18bc1d59308d1b21aa627bfff14d41be4d5d8bb995898aa4984575c8614.jpg)

![05a63a19cb595ec06df822efe97a63a51056e10b360073713e58dd4ed503f392.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/05a63a19cb595ec06df822efe97a63a51056e10b360073713e58dd4ed503f392.jpg)

![093d030595a8a5c62119c32cfe89f2b147fe98e47ad467b50d610399187f358c.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/093d030595a8a5c62119c32cfe89f2b147fe98e47ad467b50d610399187f358c.jpg)

![09b230f2e0ee91149bff1441ab46ab60e7c9fb22de4652124942ea772c92405f.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/09b230f2e0ee91149bff1441ab46ab60e7c9fb22de4652124942ea772c92405f.jpg)

![1662739fa18a4d60d5d1daae95d18cce23de9f57e6d65735d45eb10e42ea4105.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/1662739fa18a4d60d5d1daae95d18cce23de9f57e6d65735d45eb10e42ea4105.jpg)

![18c5f52aa6f91a916254e1d5dd392f6842de677be7a4e3d2ac8b123685581853.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/18c5f52aa6f91a916254e1d5dd392f6842de677be7a4e3d2ac8b123685581853.jpg)

![1ff06e611bdbd9a19a5940821210bbf7428fafb4050c6abe5a59cd6965f54dbd.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/1ff06e611bdbd9a19a5940821210bbf7428fafb4050c6abe5a59cd6965f54dbd.jpg)

![2414ffa72ac7179f5439a3e57e176a234de347fe55bc8e382f5d4e1e6f6f4cf8.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/2414ffa72ac7179f5439a3e57e176a234de347fe55bc8e382f5d4e1e6f6f4cf8.jpg)

![25c83f82790b70517ddf3e27147a2148d95462373bf1aeac1e0d1672bfecf9a7.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/25c83f82790b70517ddf3e27147a2148d95462373bf1aeac1e0d1672bfecf9a7.jpg)

![29e74a3e0bf1957f8d588ab89ba1756934012dc2e3afe296f8661aab228acf37.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/29e74a3e0bf1957f8d588ab89ba1756934012dc2e3afe296f8661aab228acf37.jpg)

![2d859678ec2384d6d9f1e5836624a471cdf5cca071726fe56323a68356934443.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/2d859678ec2384d6d9f1e5836624a471cdf5cca071726fe56323a68356934443.jpg)

![30adbc856be1b5ba35a6aafe8209ce6436962ce1668d749ed65922d892b1a602.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/30adbc856be1b5ba35a6aafe8209ce6436962ce1668d749ed65922d892b1a602.jpg)

![347172a22972d569b37d71e688e37a95567f1bdd19cf762ea3fc442bed8ac520.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/347172a22972d569b37d71e688e37a95567f1bdd19cf762ea3fc442bed8ac520.jpg)

![36c7fbdb591a808b29f73a4ecb706ae8f3165e3ce5aee696d4c5993830a3de9f.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/36c7fbdb591a808b29f73a4ecb706ae8f3165e3ce5aee696d4c5993830a3de9f.jpg)

![46157b84957c58d6b0d40602400841f0daae788f061775f4eda4ac5595e85ac3.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/46157b84957c58d6b0d40602400841f0daae788f061775f4eda4ac5595e85ac3.jpg)

![498eb41a88727fd5ee48f997ce84fc6df8edad2f2cfc5052ba374e53f694fd49.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/498eb41a88727fd5ee48f997ce84fc6df8edad2f2cfc5052ba374e53f694fd49.jpg)

![4ab8ab94d02c8f469558db60c710bc4e955974f1215f87efa8639b1a59c17d49.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/4ab8ab94d02c8f469558db60c710bc4e955974f1215f87efa8639b1a59c17d49.jpg)

![510eb3f235bb47edd0370d6bf12a11a4f5f545daedbf84c4a772a8fbd218319f.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/510eb3f235bb47edd0370d6bf12a11a4f5f545daedbf84c4a772a8fbd218319f.jpg)

![591306fa06f283e89aed4df54480e9f45e7756e2e0964df34a5a44e669c0b35c.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/591306fa06f283e89aed4df54480e9f45e7756e2e0964df34a5a44e669c0b35c.jpg)

![5f8878bb6324e515f4266cf3560f12ce0a892da5b1fff18165399cee00243c65.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/5f8878bb6324e515f4266cf3560f12ce0a892da5b1fff18165399cee00243c65.jpg)

![6907d520657cdef7ec441db123e8d13e8bf68eabad0d4db3609ee77bcf87a6d2.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/6907d520657cdef7ec441db123e8d13e8bf68eabad0d4db3609ee77bcf87a6d2.jpg)

![6d7d43b5e3557e212b11b0ea32c62c59d61c865b7730f02cb67fc3c018594006.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/6d7d43b5e3557e212b11b0ea32c62c59d61c865b7730f02cb67fc3c018594006.jpg)

![6f60e8f22166b9d55341bf4b4519fa75f48900197dbb2c3b3877dfd3ee0e0c49.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/6f60e8f22166b9d55341bf4b4519fa75f48900197dbb2c3b3877dfd3ee0e0c49.jpg)

![742675f76339b8743e4717e5684b36b02b0bc6b6cd1322b9115a315081024119.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/742675f76339b8743e4717e5684b36b02b0bc6b6cd1322b9115a315081024119.jpg)

![749ccf98566ebfd1082f510e086a822dce3f884c11dcc1e7bf86f463d3756bdd.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/749ccf98566ebfd1082f510e086a822dce3f884c11dcc1e7bf86f463d3756bdd.jpg)

![759038a3705b29bfd79821183daabc8b910e0b0780de8c040d00963a328d6b1d.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/759038a3705b29bfd79821183daabc8b910e0b0780de8c040d00963a328d6b1d.jpg)

![7e96246efcd9db52e509938c0e0cb9f5f0490e4f341ce132695a88b26f507868.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/7e96246efcd9db52e509938c0e0cb9f5f0490e4f341ce132695a88b26f507868.jpg)

![823b47a9fdb3c19642a7f6c5b117aac937251488819f0b41716a37ce8ac6d3ba.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/823b47a9fdb3c19642a7f6c5b117aac937251488819f0b41716a37ce8ac6d3ba.jpg)

![99c7d4adb4c76bb34e28ae38af994cbe572f1a09085787e77979b2d3d604ee59.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/99c7d4adb4c76bb34e28ae38af994cbe572f1a09085787e77979b2d3d604ee59.jpg)

![99d10566aeb83d96ba645f4371afa4d5227f4e45cf93ddd6fb6e8ef6ea3e8e38.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/99d10566aeb83d96ba645f4371afa4d5227f4e45cf93ddd6fb6e8ef6ea3e8e38.jpg)

![9a6bb9a105ff465dc98835454cfe26534f7f0ff8ff94b845499c5880fa8e0340.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/9a6bb9a105ff465dc98835454cfe26534f7f0ff8ff94b845499c5880fa8e0340.jpg)

![a13ee383d076b5abefbafe142c5c8b65924bbe5af58aa0587414cb7bb2ba8788.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/a13ee383d076b5abefbafe142c5c8b65924bbe5af58aa0587414cb7bb2ba8788.jpg)

![a4086e41b1f7fdc65e91f42100ce0935f8b5544485a9d616f1135f1b855fe9da.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/a4086e41b1f7fdc65e91f42100ce0935f8b5544485a9d616f1135f1b855fe9da.jpg)

![a9a0cf22d864e7c2513c5ac4604b951f1cbc1e58139cc6fb2ab756e989aed341.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/a9a0cf22d864e7c2513c5ac4604b951f1cbc1e58139cc6fb2ab756e989aed341.jpg)

![aa8762b88698bae4a6caa52c12b5b6922dae90539044fc83359a2ab9dece25aa.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/aa8762b88698bae4a6caa52c12b5b6922dae90539044fc83359a2ab9dece25aa.jpg)

![ac0446e62003b19fab0df6d179eb4b02f764daa28a9411671868f13f21229609.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/ac0446e62003b19fab0df6d179eb4b02f764daa28a9411671868f13f21229609.jpg)

![acefd2b57756718de1ef34a5ca60c662d9283f1d5db6fa572c5222d613960a5b.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/acefd2b57756718de1ef34a5ca60c662d9283f1d5db6fa572c5222d613960a5b.jpg)

![ae719d9226c589af5cfe6085513809a00a94f00647211bad8e1ef19edc7a8b32.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/ae719d9226c589af5cfe6085513809a00a94f00647211bad8e1ef19edc7a8b32.jpg)

![af07eb8443b26a7dd6fd7c3a5d03846a26c1a87fb8313394011dc50e479e6dd8.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/af07eb8443b26a7dd6fd7c3a5d03846a26c1a87fb8313394011dc50e479e6dd8.jpg)

![b62a49e7162ca7dde29fab56d8c52e19fe6463970975b7e94d521015294f3611.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/b62a49e7162ca7dde29fab56d8c52e19fe6463970975b7e94d521015294f3611.jpg)

![b8026228e0248e05370c8699b5fa56ff1d077cf0d781fb59525f0d0fb9c9ffa5.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/b8026228e0248e05370c8699b5fa56ff1d077cf0d781fb59525f0d0fb9c9ffa5.jpg)

![b8736f2d7e26748dea4f4fa05eb4349a6e4e23a799fa974afd8ab8b4c83cbc87.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/b8736f2d7e26748dea4f4fa05eb4349a6e4e23a799fa974afd8ab8b4c83cbc87.jpg)

![bae5f4921b827e3911668e29013e5434054f5decda2d64a9af0494ff40b301f7.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/bae5f4921b827e3911668e29013e5434054f5decda2d64a9af0494ff40b301f7.jpg)

![be2845367bede5035372c7a5e6d538b2316462c413ca874b29248a6e2fd474d5.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/be2845367bede5035372c7a5e6d538b2316462c413ca874b29248a6e2fd474d5.jpg)

![c06905f8f52d600b139634bfd9ab8abc1a49b8b297ea2201ebcc8cdd5f496173.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/c06905f8f52d600b139634bfd9ab8abc1a49b8b297ea2201ebcc8cdd5f496173.jpg)

![c6b264e5c7820c8652c3a05409caad1303fa22c62d466aa790ef016ac17ea169.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/c6b264e5c7820c8652c3a05409caad1303fa22c62d466aa790ef016ac17ea169.jpg)

![d6ec0761cf3ec00b8ed90375adc03776822f24e342f1ee8c953d08eeead80f98.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/d6ec0761cf3ec00b8ed90375adc03776822f24e342f1ee8c953d08eeead80f98.jpg)

![d9c6b4a1b82324d5c42d0a7cd4457a9422e953dc2984e830ce4108ee4737d8e5.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/d9c6b4a1b82324d5c42d0a7cd4457a9422e953dc2984e830ce4108ee4737d8e5.jpg)

![ed75a3bf435703958718d46f34f333e95ae16b33cdfdc0a650e1934fb5b24798.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/ed75a3bf435703958718d46f34f333e95ae16b33cdfdc0a650e1934fb5b24798.jpg)

![ef91072250784642a56d22d41544bc8544676793537d1a73455d5051efd73d38.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/ef91072250784642a56d22d41544bc8544676793537d1a73455d5051efd73d38.jpg)

![f0130db024156f981841b1144aef169369d1ca62b347ee0face8ff001ebe0b93.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/f0130db024156f981841b1144aef169369d1ca62b347ee0face8ff001ebe0b93.jpg)

![f5f950d9da7698843b03eb6f030f35c4dd54bacac68d452b8070b635805b920e.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/f5f950d9da7698843b03eb6f030f35c4dd54bacac68d452b8070b635805b920e.jpg)

![f8cd9451b1f36eaf4e018a11761f0f32ee9e186940cf27dbda9372189fb764e3.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/images/f8cd9451b1f36eaf4e018a11761f0f32ee9e186940cf27dbda9372189fb764e3.jpg)

### Tables

![07e6ebe8b7018fa06a290e3dbb462d1550bc50e75d886e3b40362eb7b322b758.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/07e6ebe8b7018fa06a290e3dbb462d1550bc50e75d886e3b40362eb7b322b758.jpg)

![1a635e8f7841828a84726af31c747fa2e695206a4a1ce48d25a5403393e0efb3.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/1a635e8f7841828a84726af31c747fa2e695206a4a1ce48d25a5403393e0efb3.jpg)

![1bcb2099843fbff01806b79e6c24c5a1b072fbc02bc96c04295ae0c44a74d3dc.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/1bcb2099843fbff01806b79e6c24c5a1b072fbc02bc96c04295ae0c44a74d3dc.jpg)

![6de0448f4ad2b2dcb659631cca68236a0ba1212fae68f777b449e1e9b5edb64e.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/6de0448f4ad2b2dcb659631cca68236a0ba1212fae68f777b449e1e9b5edb64e.jpg)

![70dfc7ed22da3b15bea8fc04174519c01c946092d0fea171c63620befc675e19.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/70dfc7ed22da3b15bea8fc04174519c01c946092d0fea171c63620befc675e19.jpg)

![77244dbf6c27c407d99bf83b145599d904931548515f07bdedda449b6aafb6b4.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/77244dbf6c27c407d99bf83b145599d904931548515f07bdedda449b6aafb6b4.jpg)

![81692f7a206487d585e6c5d2c2675d9286852ec1ebbc0acb2ffd9e9536491267.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/81692f7a206487d585e6c5d2c2675d9286852ec1ebbc0acb2ffd9e9536491267.jpg)

![8d1ada42248dfbd937c7b20b792a5f10783286561668b0dd18bf6d1ec9731dd2.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/8d1ada42248dfbd937c7b20b792a5f10783286561668b0dd18bf6d1ec9731dd2.jpg)

![9b410d539fdd4e3e05b5bc47047598bcfd126e1a44392f98da169cb982f8dc44.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/9b410d539fdd4e3e05b5bc47047598bcfd126e1a44392f98da169cb982f8dc44.jpg)

![adc22bbd8c4492a4e39aa612b44548aa516c7bd58c5601842abee78112f721b8.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/adc22bbd8c4492a4e39aa612b44548aa516c7bd58c5601842abee78112f721b8.jpg)

![bec2773b48d31d66ce0901ae999d37ccadbfbf750457d9e93e09ffd0ac8dd2f3.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/bec2773b48d31d66ce0901ae999d37ccadbfbf750457d9e93e09ffd0ac8dd2f3.jpg)

![cbd632a1bad7d9d15fe5f167e9d2f006e6e8099e9f83b67777ab55a42bde7abf.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/cbd632a1bad7d9d15fe5f167e9d2f006e6e8099e9f83b67777ab55a42bde7abf.jpg)

![e26a52eeaa13cadfe34202cdcd3a4a7869dc3ccf4cc02dce250faa45666d02a7.jpg](acl_results/583_GODBench_ A Benchmark for Multimodal Large Language Models in Video Comment Art/tables/e26a52eeaa13cadfe34202cdcd3a4a7869dc3ccf4cc02dce250faa45666d02a7.jpg)

## UniLR: Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever

### Images

![1b236630656a50fedab8c2fb2622d2b5b88ba47b2e39f8d1b31996931e4d1a27.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/images/1b236630656a50fedab8c2fb2622d2b5b88ba47b2e39f8d1b31996931e4d1a27.jpg)

![6435455003453e570c824962038941b9fd2386a7c675315e3c0b467b7f252c69.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/images/6435455003453e570c824962038941b9fd2386a7c675315e3c0b467b7f252c69.jpg)

![861e3fc4ca8af56384de86857c1897f20d0698c0a5fe87c2307ac635d9ed703f.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/images/861e3fc4ca8af56384de86857c1897f20d0698c0a5fe87c2307ac635d9ed703f.jpg)

![bfea56e5601f67fee5d8e02e6cc55d7c77080204a1f51669b9f89de7bc1f686d.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/images/bfea56e5601f67fee5d8e02e6cc55d7c77080204a1f51669b9f89de7bc1f686d.jpg)

![caa111477ef229327429f574d8dadb0be42f369f0ce87eaada0ccf287aa75fa6.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/images/caa111477ef229327429f574d8dadb0be42f369f0ce87eaada0ccf287aa75fa6.jpg)

### Tables

![0f8332868146dd78745c61acf8de3f134d84312c95e3a0c4f11881f102cf7d79.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/tables/0f8332868146dd78745c61acf8de3f134d84312c95e3a0c4f11881f102cf7d79.jpg)

![148c03e8a6a3d1895622f427f9d384a1e1fcf9c84eece79103d1b399a3c80fe1.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/tables/148c03e8a6a3d1895622f427f9d384a1e1fcf9c84eece79103d1b399a3c80fe1.jpg)

![2649456aac439bdac2b4d3842bcdb8caa1499c1036c61d847810e4a4c138e5a9.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/tables/2649456aac439bdac2b4d3842bcdb8caa1499c1036c61d847810e4a4c138e5a9.jpg)

![3655a20183b418d916af1a1d7863c3227914b90172114ccd2eb928aa1bf2cf82.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/tables/3655a20183b418d916af1a1d7863c3227914b90172114ccd2eb928aa1bf2cf82.jpg)

![470052d977eb043d30f4d93efb72c7907de5bc26b4220c5d41367f72408d6443.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/tables/470052d977eb043d30f4d93efb72c7907de5bc26b4220c5d41367f72408d6443.jpg)

![548a965d85797a66e152eccd3b1ffb4806c2cc848db1337973b29bf3e9b002a2.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/tables/548a965d85797a66e152eccd3b1ffb4806c2cc848db1337973b29bf3e9b002a2.jpg)

![6e46bd809131b800682fb65f7d0aefcdcd1b4feed5616bf14613e434ee94cc75.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/tables/6e46bd809131b800682fb65f7d0aefcdcd1b4feed5616bf14613e434ee94cc75.jpg)

![728ba7e0f3023d58a36a2e7afdf466a357efbd140355bd1c84d7641ef43afb23.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/tables/728ba7e0f3023d58a36a2e7afdf466a357efbd140355bd1c84d7641ef43afb23.jpg)

![7d75320962331f8a90f51bf1c14d998b27fd40ed0baccefe23fdcec8b8c2b874.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/tables/7d75320962331f8a90f51bf1c14d998b27fd40ed0baccefe23fdcec8b8c2b874.jpg)

![cef72ecbd5d6480f08e5d55eef3517545674fcc04cb361054d13a7ad8ae5be79.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/tables/cef72ecbd5d6480f08e5d55eef3517545674fcc04cb361054d13a7ad8ae5be79.jpg)

![d08dc5dbddb798ca39e9f9ccb8a3077047ccf74c0a32cade42f670292f0cd649.jpg](acl_results/584_UniLR_ Unleashing the Power ofLLMs on Multiple Legal Tasks with a Unified Legal Retriever/tables/d08dc5dbddb798ca39e9f9ccb8a3077047ccf74c0a32cade42f670292f0cd649.jpg)

## Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models

### Images

![04459fee21234c6625069a24f06e12a6bb9f6f917d51dd9229ad149baf32c445.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/images/04459fee21234c6625069a24f06e12a6bb9f6f917d51dd9229ad149baf32c445.jpg)

![5876318e12bea99e863a2a63e3455c71b27c4ae10be5ef8edcbd85df72774c47.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/images/5876318e12bea99e863a2a63e3455c71b27c4ae10be5ef8edcbd85df72774c47.jpg)

![5ca37487f3a67c2431ae320776b06649d6791593cb197125966c29cd1781307c.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/images/5ca37487f3a67c2431ae320776b06649d6791593cb197125966c29cd1781307c.jpg)

![b6b5152719e5d0ef1974431f2e639b114efdbc85131e5fa0dbc7df7bcf5569da.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/images/b6b5152719e5d0ef1974431f2e639b114efdbc85131e5fa0dbc7df7bcf5569da.jpg)

![bc715e76a61bcfa2a721ae94c7534ed9f33ceb329a767ea7d080cb315459bdc0.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/images/bc715e76a61bcfa2a721ae94c7534ed9f33ceb329a767ea7d080cb315459bdc0.jpg)

![cad1d69956662f0b92131e1f43232c3659f7d84cdb0dba69df4b0b269c1a8d1d.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/images/cad1d69956662f0b92131e1f43232c3659f7d84cdb0dba69df4b0b269c1a8d1d.jpg)

### Tables

![031f4e0d3a2fe025895e0836384b8c252d6d23e85294c64a736679385f5bc2c6.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/031f4e0d3a2fe025895e0836384b8c252d6d23e85294c64a736679385f5bc2c6.jpg)

![085b9d4df7f0f79ab7db7751a27986a083b1fc3930134730db188dae0fb70105.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/085b9d4df7f0f79ab7db7751a27986a083b1fc3930134730db188dae0fb70105.jpg)

![08e48865c9c87ef953bb7cd5fd0ef76cd5469718954bea15d92d64f0e984adaa.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/08e48865c9c87ef953bb7cd5fd0ef76cd5469718954bea15d92d64f0e984adaa.jpg)

![11a3e6aa11d1789a0c369c0a6efbae724584919c41df2b4897043409135328ef.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/11a3e6aa11d1789a0c369c0a6efbae724584919c41df2b4897043409135328ef.jpg)

![1ccf61896523e4291f8684f81898f42d3e2856dd3980b8316612021fd73f4301.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/1ccf61896523e4291f8684f81898f42d3e2856dd3980b8316612021fd73f4301.jpg)

![6339f614e80157cc9071bd3b84b3258c1f8df889cef9fcd6e58b4b0888c223dc.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/6339f614e80157cc9071bd3b84b3258c1f8df889cef9fcd6e58b4b0888c223dc.jpg)

![6bc6134158eea86a3327181c0b25fab647d36adddbb1cea655fcf8fc5182bbd0.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/6bc6134158eea86a3327181c0b25fab647d36adddbb1cea655fcf8fc5182bbd0.jpg)

![768cb803adef40b2063b3f7b15ea8e68030271755a29f6d352e5fa91979c6b4d.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/768cb803adef40b2063b3f7b15ea8e68030271755a29f6d352e5fa91979c6b4d.jpg)

![79b3c17859224df882d765c659aaa6421d1a65ce3c85cd7749fe2200b39904ea.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/79b3c17859224df882d765c659aaa6421d1a65ce3c85cd7749fe2200b39904ea.jpg)

![7bf9daa952501267d1f832a8098916cf15ef538e667d303fd9b051ba58557e2a.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/7bf9daa952501267d1f832a8098916cf15ef538e667d303fd9b051ba58557e2a.jpg)

![87e19229886c48eb5fa492504766155156c619b629d37a7d76b3103e3cde9c7e.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/87e19229886c48eb5fa492504766155156c619b629d37a7d76b3103e3cde9c7e.jpg)

![de9bd1d9673255c57691530acecf203592ef66cefb2a6afc3bbe950e5954cd98.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/de9bd1d9673255c57691530acecf203592ef66cefb2a6afc3bbe950e5954cd98.jpg)

![f4c5f48e5a6f0827cfc9f2177f9c20166eb2cdc4abac5c1af791ddc03fc0206c.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/f4c5f48e5a6f0827cfc9f2177f9c20166eb2cdc4abac5c1af791ddc03fc0206c.jpg)

![ff9a0d3895c7af6ffbe9613e0c55b85e4209bece1bba53a6de3f72d331ec7ed4.jpg](acl_results/585_Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models/tables/ff9a0d3895c7af6ffbe9613e0c55b85e4209bece1bba53a6de3f72d331ec7ed4.jpg)

## Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model

### Images

![03ba6926c8b4ccc86e64cbc087ffea2b4ab58a759fb41b897df4d84ae83e2d1e.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/03ba6926c8b4ccc86e64cbc087ffea2b4ab58a759fb41b897df4d84ae83e2d1e.jpg)

![07f6e8d269d42eb22a59d5de6bd19bca1bcf66befee653dd6090d4bbee1119b9.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/07f6e8d269d42eb22a59d5de6bd19bca1bcf66befee653dd6090d4bbee1119b9.jpg)

![1fc0244df5c38353100a0529790becfe8c14ecdd17bcc09cdcaba6348fbf9917.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/1fc0244df5c38353100a0529790becfe8c14ecdd17bcc09cdcaba6348fbf9917.jpg)

![5702033be94f82ad2a3b71ea4160f05e40e118da55f145641c31e347baa8fbe5.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/5702033be94f82ad2a3b71ea4160f05e40e118da55f145641c31e347baa8fbe5.jpg)

![5b4dd2cb6dcacba63df8862057b870224838e38fec788f92c326996b3ab5c633.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/5b4dd2cb6dcacba63df8862057b870224838e38fec788f92c326996b3ab5c633.jpg)

![5f3214d5fb8a8afa1a2ce5327b6637ea49ca9f9db2d8a8646c985e589c262a78.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/5f3214d5fb8a8afa1a2ce5327b6637ea49ca9f9db2d8a8646c985e589c262a78.jpg)

![6334fae90ff9affcada3a6ec953e963c05f3c4506e085e1a7836a2769de412f2.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/6334fae90ff9affcada3a6ec953e963c05f3c4506e085e1a7836a2769de412f2.jpg)

![6824be638353d2da5fde27b35886415c3749c990f9a49bf739e94f36c881a44b.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/6824be638353d2da5fde27b35886415c3749c990f9a49bf739e94f36c881a44b.jpg)

![7b7e2ab2aa733e28e84c6dd9da1b2bdd974dbb9d0ec0815e5063d32ab334cd2e.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/7b7e2ab2aa733e28e84c6dd9da1b2bdd974dbb9d0ec0815e5063d32ab334cd2e.jpg)

![823687c2d6a0bdffe8328efd71df8366cc2d0aa454d3d942012b0d4d6ded9e35.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/823687c2d6a0bdffe8328efd71df8366cc2d0aa454d3d942012b0d4d6ded9e35.jpg)

![82d69756e6c64fed31950451c2e5b36ea5f7eb1879def8fd732cf090273d09d3.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/82d69756e6c64fed31950451c2e5b36ea5f7eb1879def8fd732cf090273d09d3.jpg)

![9fa27c3dec35f506209864a21ef0fd4f8fe8db27831bfdf4f74cb631720ec709.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/9fa27c3dec35f506209864a21ef0fd4f8fe8db27831bfdf4f74cb631720ec709.jpg)

![a9a2c73ddb8c29898b73ab81f6ce99528618c27d055a9c72f664be7a618e987e.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/a9a2c73ddb8c29898b73ab81f6ce99528618c27d055a9c72f664be7a618e987e.jpg)

![b1b4f7ac4eda8a011b7524e8c1d20697382fccdf6e6f863bf150c2b99fd17231.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/b1b4f7ac4eda8a011b7524e8c1d20697382fccdf6e6f863bf150c2b99fd17231.jpg)

![c0ad7aa9c1adb6579f1ef49f066034e6528c4e6b17199c6e2c5998ffd065e775.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/c0ad7aa9c1adb6579f1ef49f066034e6528c4e6b17199c6e2c5998ffd065e775.jpg)

![e1aac5aff5a4e7eeda7da9c61ec650d1218e27d2affdae91a5f73576f1796034.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/e1aac5aff5a4e7eeda7da9c61ec650d1218e27d2affdae91a5f73576f1796034.jpg)

![e96cac215ba8d970b56113570b78ced946d163c8971315a246efa5f0b278d9af.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/images/e96cac215ba8d970b56113570b78ced946d163c8971315a246efa5f0b278d9af.jpg)

### Tables

![02ecde2ca2f9ef02853573d5554d214e43eaced681a17cad49a68b9b0ebf2012.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/02ecde2ca2f9ef02853573d5554d214e43eaced681a17cad49a68b9b0ebf2012.jpg)

![06d890ec63a19405c8dc8d24136ccaeb8c6234b7378a558b36e8fffd5bca1b40.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/06d890ec63a19405c8dc8d24136ccaeb8c6234b7378a558b36e8fffd5bca1b40.jpg)

![0a9c45990b7eb9c2b2296394a2408a442ef46688169360ab45c19945080f838c.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/0a9c45990b7eb9c2b2296394a2408a442ef46688169360ab45c19945080f838c.jpg)

![0e4fe99e5a7688fc928d988926c77e24b18f7b54e398c4af1ce521935d3fc313.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/0e4fe99e5a7688fc928d988926c77e24b18f7b54e398c4af1ce521935d3fc313.jpg)

![128b550c4a4972fa3a7ba5a3a65e415be27daef0690328440c715d9bd63194a5.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/128b550c4a4972fa3a7ba5a3a65e415be27daef0690328440c715d9bd63194a5.jpg)

![29215cc5063e97f751bccaa621a38c68ee2ee4369ef42a04003ccbd5c935cda3.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/29215cc5063e97f751bccaa621a38c68ee2ee4369ef42a04003ccbd5c935cda3.jpg)

![3486a5f11485409ba0356ef706f52dfb13e688b46473cc9c3c120c0a5a627182.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/3486a5f11485409ba0356ef706f52dfb13e688b46473cc9c3c120c0a5a627182.jpg)

![49fb3030aa47aad9dfca2e82e386c3ce823ecfdd147330f252dfb3a5a4d7a54d.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/49fb3030aa47aad9dfca2e82e386c3ce823ecfdd147330f252dfb3a5a4d7a54d.jpg)

![57289b83101626339851a3bfe33759c24a1df31ba56480abc22f25b81fc35b6f.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/57289b83101626339851a3bfe33759c24a1df31ba56480abc22f25b81fc35b6f.jpg)

![6013acb03c073e4acf735118aba853cc4e1b13c667afe6e23f17cb91ed377824.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/6013acb03c073e4acf735118aba853cc4e1b13c667afe6e23f17cb91ed377824.jpg)

![6da8fc430151e1b0ac762338e1ca6cddde2f8f3848c2dcb10e94bc2f75d07992.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/6da8fc430151e1b0ac762338e1ca6cddde2f8f3848c2dcb10e94bc2f75d07992.jpg)

![84e8551c4cfe755a3c705e0ee1ef6e8bd4cd72b548f0d18db0dfa7dc0d6738d7.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/84e8551c4cfe755a3c705e0ee1ef6e8bd4cd72b548f0d18db0dfa7dc0d6738d7.jpg)

![9fde76aa1e3a381264ce46dfdfc09f8218ab82d5ba113fc10880fa38a2a9a617.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/9fde76aa1e3a381264ce46dfdfc09f8218ab82d5ba113fc10880fa38a2a9a617.jpg)

![a11e5f31d9ac6cecafa32fca51f7e51d7f749d5ad10935c9d7930d748b1ec1a8.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/a11e5f31d9ac6cecafa32fca51f7e51d7f749d5ad10935c9d7930d748b1ec1a8.jpg)

![ae27c8e957765dee56890d31db63ec10b9c581fe8e06d4e64c72cc2fd42a0040.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/ae27c8e957765dee56890d31db63ec10b9c581fe8e06d4e64c72cc2fd42a0040.jpg)

![b01651bbb36fa109fbd116f7dd3ce1fa6b7cb61767feb58eb42e80d1eb98c10e.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/b01651bbb36fa109fbd116f7dd3ce1fa6b7cb61767feb58eb42e80d1eb98c10e.jpg)

![b06f94678e2221385725e6d60fee9de38a3201aa06dbb24918ad3767565ff8d5.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/b06f94678e2221385725e6d60fee9de38a3201aa06dbb24918ad3767565ff8d5.jpg)

![c6e26def7fc3154527467f47c9a4d9ca0fb19b5f2236b06457a5392dd76c00f1.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/c6e26def7fc3154527467f47c9a4d9ca0fb19b5f2236b06457a5392dd76c00f1.jpg)

![cb6fcbb1ddf7a8bb2535e8077eaae7f1219bd8c3656b9223805f917231d11bd1.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/cb6fcbb1ddf7a8bb2535e8077eaae7f1219bd8c3656b9223805f917231d11bd1.jpg)

![df31bac8305da8ccd330d880067618ccd3d25e4028b14c4e62a621e25bd1ea1e.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/df31bac8305da8ccd330d880067618ccd3d25e4028b14c4e62a621e25bd1ea1e.jpg)

![e35c24a0252922c1594bde588a2e42744bd729c3212affb4fe53456a3c99709a.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/e35c24a0252922c1594bde588a2e42744bd729c3212affb4fe53456a3c99709a.jpg)

![e9c440d9b7c941aef6db7c449de45e7ecf8e932a25595cd4049d24b6b21c51c4.jpg](acl_results/586_Beyond Dialogue_ A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model/tables/e9c440d9b7c941aef6db7c449de45e7ecf8e932a25595cd4049d24b6b21c51c4.jpg)

## ACECODER: Acing CoderRLvia Automated Test-Case Synthesis

### Images

![2e8a01c3d30312e6f55faa097acc38b9a6a8c6aaf25e8d031221663dc90afe05.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/images/2e8a01c3d30312e6f55faa097acc38b9a6a8c6aaf25e8d031221663dc90afe05.jpg)

![55980a9aa9735f45fa04e9783119f6cf3947afc02c3b7b407d0d50f1f0306dbf.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/images/55980a9aa9735f45fa04e9783119f6cf3947afc02c3b7b407d0d50f1f0306dbf.jpg)

![ffa295f5ce1dba0465b9109742c52826e0057f515873be989c445060f2dee302.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/images/ffa295f5ce1dba0465b9109742c52826e0057f515873be989c445060f2dee302.jpg)

### Tables

![014c086375ae273f890539185e2d8b06cc44cd2c095cc769220ed2189eec9b14.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/tables/014c086375ae273f890539185e2d8b06cc44cd2c095cc769220ed2189eec9b14.jpg)

![01b177056c13bbc07436fc9d12f2892ef6149b40c1f7c414448d4689eb52b50e.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/tables/01b177056c13bbc07436fc9d12f2892ef6149b40c1f7c414448d4689eb52b50e.jpg)

![06b382153a7f518a2dd8664a3dddd902c279ebe3fe05358b1580da4a5d63dbb6.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/tables/06b382153a7f518a2dd8664a3dddd902c279ebe3fe05358b1580da4a5d63dbb6.jpg)

![0fbe7320a51a72348b609185aaeedbb648a758cf96f748bf1f8ceaf670b89800.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/tables/0fbe7320a51a72348b609185aaeedbb648a758cf96f748bf1f8ceaf670b89800.jpg)

![3f0cb28c35556c634c3eae2996af83aa0885288251d9b3adcbaf04222ed23e60.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/tables/3f0cb28c35556c634c3eae2996af83aa0885288251d9b3adcbaf04222ed23e60.jpg)

![48a4d1a754d52accff7d410e95f9449c81e4474e0dec7f75eb269509ef5f5528.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/tables/48a4d1a754d52accff7d410e95f9449c81e4474e0dec7f75eb269509ef5f5528.jpg)

![5b781c7e92353855a50207e5a19563bc7bd9d98b9d8b1a307ae28fde21ffd45c.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/tables/5b781c7e92353855a50207e5a19563bc7bd9d98b9d8b1a307ae28fde21ffd45c.jpg)

![73f358f0ecc9b7c8fb51e0484f0ece660a4f8ff10212e52f28e98acfea06daa1.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/tables/73f358f0ecc9b7c8fb51e0484f0ece660a4f8ff10212e52f28e98acfea06daa1.jpg)

![754059f43a138912fc43cbcfb26a7234508c4041ab2bad767e2a1c04e004e9b2.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/tables/754059f43a138912fc43cbcfb26a7234508c4041ab2bad767e2a1c04e004e9b2.jpg)

![dafb69a6afb1508b41d046c2e25e491bd63fa15a66c2dcad9c541f65e1038356.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/tables/dafb69a6afb1508b41d046c2e25e491bd63fa15a66c2dcad9c541f65e1038356.jpg)

![e95cc270e5d72a374883f65ba7743a0cdff314ed651371e6dc325568a10dc0a9.jpg](acl_results/587_ACECODER_ Acing CoderRLvia Automated Test-Case Synthesis/tables/e95cc270e5d72a374883f65ba7743a0cdff314ed651371e6dc325568a10dc0a9.jpg)

## Quantifying Semantic Emergence in Language Models

### Images

![021f05c497d6d142f3f321ed7cf87a26fc18cd93d2fe5ee4e5addee0fbf42a15.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/images/021f05c497d6d142f3f321ed7cf87a26fc18cd93d2fe5ee4e5addee0fbf42a15.jpg)

![041720d11aec9d2616069484b62757acd1a8906efe89b0a271f80a088244803d.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/images/041720d11aec9d2616069484b62757acd1a8906efe89b0a271f80a088244803d.jpg)

![4a5491e291c8f774c7bf1c1be36f21c748dfde921dfbd7b6c78fcb12090cac74.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/images/4a5491e291c8f774c7bf1c1be36f21c748dfde921dfbd7b6c78fcb12090cac74.jpg)

![7dd5cff2a4937975ac89197509f7487d13188f1cfd1facc9b2fb11b9fded78ca.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/images/7dd5cff2a4937975ac89197509f7487d13188f1cfd1facc9b2fb11b9fded78ca.jpg)

![a096dfe2484e6a114f2d4dfb40b6ab918c031f7db0505711c784991ecd477994.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/images/a096dfe2484e6a114f2d4dfb40b6ab918c031f7db0505711c784991ecd477994.jpg)

![d760ff86de8e04a9cf668dbc09c5761dde0c2df9ee15ddae069cd885d93c8d67.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/images/d760ff86de8e04a9cf668dbc09c5761dde0c2df9ee15ddae069cd885d93c8d67.jpg)

### Tables

![1f12be39801ee449e9df404be2bf2d16b01bd63e9b0ea3d722e9c107a8fa7a53.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/tables/1f12be39801ee449e9df404be2bf2d16b01bd63e9b0ea3d722e9c107a8fa7a53.jpg)

![2db05400012f423536095eefe74bfbd43d636d4627da8c8b356ece4e7251860d.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/tables/2db05400012f423536095eefe74bfbd43d636d4627da8c8b356ece4e7251860d.jpg)

![46750ce4f24ca3a1c5df735059e384ac730585a5afc0cd0b934d07b04470e9a0.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/tables/46750ce4f24ca3a1c5df735059e384ac730585a5afc0cd0b934d07b04470e9a0.jpg)

![4ed973c8a5068fedd7a155d43d4b3fd47c95f3658b30f95e5796582661d61565.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/tables/4ed973c8a5068fedd7a155d43d4b3fd47c95f3658b30f95e5796582661d61565.jpg)

![646ef6ba9259ab7aab510584eb357275d041d81dac81fce0d7e3b2d3aec3f8f7.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/tables/646ef6ba9259ab7aab510584eb357275d041d81dac81fce0d7e3b2d3aec3f8f7.jpg)

![7b4b801132938211b4455d0487a5ea93112a490883c472e0233b021ae1933c5d.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/tables/7b4b801132938211b4455d0487a5ea93112a490883c472e0233b021ae1933c5d.jpg)

![95c1ed388ea3ab1c2ee8023fa9f95360cc6f5e76780b77f156ed26edfd26a434.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/tables/95c1ed388ea3ab1c2ee8023fa9f95360cc6f5e76780b77f156ed26edfd26a434.jpg)

![a99696f45b70431e903310ddc0345302137b110d445fb613d8786c093adc06d2.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/tables/a99696f45b70431e903310ddc0345302137b110d445fb613d8786c093adc06d2.jpg)

![b277d2fcb96f973d37ee7072887b755bc024ad6fccdc52d8b91c0200b53b1bb1.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/tables/b277d2fcb96f973d37ee7072887b755bc024ad6fccdc52d8b91c0200b53b1bb1.jpg)

![bf911cc086c1b49ca8adc7db282a8094e30db7179d9bb9dbafe22c3de4a95ffa.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/tables/bf911cc086c1b49ca8adc7db282a8094e30db7179d9bb9dbafe22c3de4a95ffa.jpg)

![ca6435c9a9b7770e61b52d85e457e156f61109ab515b80d8a112f5019adca189.jpg](acl_results/588_Quantifying Semantic Emergence in Language Models/tables/ca6435c9a9b7770e61b52d85e457e156f61109ab515b80d8a112f5019adca189.jpg)

## DebateCoder: Towards Collective Intelligence ofLLMs via Test Case DrivenLLMDebate for Code Generation

### Images

![1d83e637edf5613e788a03a99bd2c10dbff8ad846a42d4a07183d3190a80ddf4.jpg](acl_results/589_DebateCoder_ Towards Collective Intelligence ofLLMs via Test Case DrivenLLMDebate for Code Generatio/images/1d83e637edf5613e788a03a99bd2c10dbff8ad846a42d4a07183d3190a80ddf4.jpg)

![54aaf7300ccf88c7e6076d021379b579283b893f7e0909fce7691d7a616252d0.jpg](acl_results/589_DebateCoder_ Towards Collective Intelligence ofLLMs via Test Case DrivenLLMDebate for Code Generatio/images/54aaf7300ccf88c7e6076d021379b579283b893f7e0909fce7691d7a616252d0.jpg)

![a4541420668f7e01e19dedf7a307fb81ac5fa9fb5a66922cf04e34eb6d5a74f9.jpg](acl_results/589_DebateCoder_ Towards Collective Intelligence ofLLMs via Test Case DrivenLLMDebate for Code Generatio/images/a4541420668f7e01e19dedf7a307fb81ac5fa9fb5a66922cf04e34eb6d5a74f9.jpg)

![a76085ae5b919b21c254d94fc7448c27aa2595c038d89884871c79eff2760ebc.jpg](acl_results/589_DebateCoder_ Towards Collective Intelligence ofLLMs via Test Case DrivenLLMDebate for Code Generatio/images/a76085ae5b919b21c254d94fc7448c27aa2595c038d89884871c79eff2760ebc.jpg)

![e8a1348e13984646ce38d623c55b5b5305e9f4eb88a8b68fd0ba591c17b8ee72.jpg](acl_results/589_DebateCoder_ Towards Collective Intelligence ofLLMs via Test Case DrivenLLMDebate for Code Generatio/images/e8a1348e13984646ce38d623c55b5b5305e9f4eb88a8b68fd0ba591c17b8ee72.jpg)

### Tables

![6b13abf8521f7f43f898eaca9e8dc3729dfe7a34d05eadb286802adff82d47a7.jpg](acl_results/589_DebateCoder_ Towards Collective Intelligence ofLLMs via Test Case DrivenLLMDebate for Code Generatio/tables/6b13abf8521f7f43f898eaca9e8dc3729dfe7a34d05eadb286802adff82d47a7.jpg)

![89669e217a55fad0692f928e8a507ccd776366996962e15efecb0a859ea2f39b.jpg](acl_results/589_DebateCoder_ Towards Collective Intelligence ofLLMs via Test Case DrivenLLMDebate for Code Generatio/tables/89669e217a55fad0692f928e8a507ccd776366996962e15efecb0a859ea2f39b.jpg)

![be219b5a7fd4c3936504757e513455fc85fa807a09d4b557f201e5c01bf69003.jpg](acl_results/589_DebateCoder_ Towards Collective Intelligence ofLLMs via Test Case DrivenLLMDebate for Code Generatio/tables/be219b5a7fd4c3936504757e513455fc85fa807a09d4b557f201e5c01bf69003.jpg)

![ca2b969a570b2b4d87368415cd67496dd5f430e645719dc70e06111a3890a1f8.jpg](acl_results/589_DebateCoder_ Towards Collective Intelligence ofLLMs via Test Case DrivenLLMDebate for Code Generatio/tables/ca2b969a570b2b4d87368415cd67496dd5f430e645719dc70e06111a3890a1f8.jpg)

## The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models

### Images

![0422e37f963facdf9662ad26fc30e3314d3ba399e2e267f866f89d89a8523c2b.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/images/0422e37f963facdf9662ad26fc30e3314d3ba399e2e267f866f89d89a8523c2b.jpg)

![0ccb3c7c31111d9c7bfcf01e8ba4f0346c87160c5e67a6bbb36a517fdc037c66.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/images/0ccb3c7c31111d9c7bfcf01e8ba4f0346c87160c5e67a6bbb36a517fdc037c66.jpg)

![678b3852bac56568f2599de4b25add391b58ea8ce0f75163af2fc81a7ddc6a95.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/images/678b3852bac56568f2599de4b25add391b58ea8ce0f75163af2fc81a7ddc6a95.jpg)

![682ad310650ea1670d02c05eb8dc1d08913ee3aac4f0c61a87cdeb517b6ac7ec.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/images/682ad310650ea1670d02c05eb8dc1d08913ee3aac4f0c61a87cdeb517b6ac7ec.jpg)

![68c57805077936fc066c6597ed6016b90e2b21e87e75006a1a767cc84ae74437.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/images/68c57805077936fc066c6597ed6016b90e2b21e87e75006a1a767cc84ae74437.jpg)

![6b39981960dffef4d97f749fd2f8e8b93bb5cc90608812ef464228e9138f27db.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/images/6b39981960dffef4d97f749fd2f8e8b93bb5cc90608812ef464228e9138f27db.jpg)

![89094b4d58cd56df3df56444b1dfd81221db7ce6d8e0f0832ed75c5345f54c5b.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/images/89094b4d58cd56df3df56444b1dfd81221db7ce6d8e0f0832ed75c5345f54c5b.jpg)

![92b778381acf97ad65125048359b23f21d21b6b577b596c9fd4408a3b0ef3140.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/images/92b778381acf97ad65125048359b23f21d21b6b577b596c9fd4408a3b0ef3140.jpg)

![f3f03b65d21f7e17e78e2fba361e132b682774e1b0eb9d451e7c1d08a5b96ea4.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/images/f3f03b65d21f7e17e78e2fba361e132b682774e1b0eb9d451e7c1d08a5b96ea4.jpg)

### Tables

![062850536897be2404efa65ac6d71ed68640031ac8bdc2f284ad9abd3130a2af.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/tables/062850536897be2404efa65ac6d71ed68640031ac8bdc2f284ad9abd3130a2af.jpg)

![089910cba3e6b30bc22728aeec2c2c2c93c57eddc40d70afb817777e25002cc9.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/tables/089910cba3e6b30bc22728aeec2c2c2c93c57eddc40d70afb817777e25002cc9.jpg)

![0da36e3775f8041814cf3cc2f7beb0e6c104f514ff6851605cb4c88ef2ff6bcf.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/tables/0da36e3775f8041814cf3cc2f7beb0e6c104f514ff6851605cb4c88ef2ff6bcf.jpg)

![21a01bbd0a4401c44ad6a6d69ee4ecd7d3616ac9c1994d2a2f7919221bac00c8.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/tables/21a01bbd0a4401c44ad6a6d69ee4ecd7d3616ac9c1994d2a2f7919221bac00c8.jpg)

![5cccbd6f6f95322d7d8700cf77de8b661df5e682787d7fc5044a7ad3f14c2fae.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/tables/5cccbd6f6f95322d7d8700cf77de8b661df5e682787d7fc5044a7ad3f14c2fae.jpg)

![72110ef1060c1ed35785f214fb4210a976fd19d613d28bf086bb68a8693e805c.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/tables/72110ef1060c1ed35785f214fb4210a976fd19d613d28bf086bb68a8693e805c.jpg)

![cab40b7756393722ed1ebf79d4ea7b4c853c56ddaeedb828b53f76d4a83caf09.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/tables/cab40b7756393722ed1ebf79d4ea7b4c853c56ddaeedb828b53f76d4a83caf09.jpg)

![e1f4fb1d073aeeb485937708a77111e4491dc84bb3f84c55446b151aaed18432.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/tables/e1f4fb1d073aeeb485937708a77111e4491dc84bb3f84c55446b151aaed18432.jpg)

![fc8d6cf8cc2780abaf7bace735ca27c371694714aaaffc62d3850da73cec0ed3.jpg](acl_results/590_The Tug of War Within_ Mitigating the Fairness-Privacy Conflicts in Large Language Models/tables/fc8d6cf8cc2780abaf7bace735ca27c371694714aaaffc62d3850da73cec0ed3.jpg)

## GraphInsight: Unlocking Insights in Large Language Models for Graph Structure Understanding

### Images

![1cb533fe46037175ffa8ede2f58426e7ddb237553acaea42aafb2d2ca4b8ad88.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/images/1cb533fe46037175ffa8ede2f58426e7ddb237553acaea42aafb2d2ca4b8ad88.jpg)

![2e1de16f4a973d033a8e0798677ae35b1d24019d07628b7847ce67da4fd8d697.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/images/2e1de16f4a973d033a8e0798677ae35b1d24019d07628b7847ce67da4fd8d697.jpg)

![2efaadbadb840553f38a60a7c09cbb31c25137e182a4c53220ad5ff63fb0d4d6.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/images/2efaadbadb840553f38a60a7c09cbb31c25137e182a4c53220ad5ff63fb0d4d6.jpg)

![48a13826d95f088913589519ee0cbe34e43c1098077b1ea5dc18c5fa0e4db3dd.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/images/48a13826d95f088913589519ee0cbe34e43c1098077b1ea5dc18c5fa0e4db3dd.jpg)

![4c817e281bb52d7296456aa1fa1a6e537d51215474012fb167fa6047e5b97249.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/images/4c817e281bb52d7296456aa1fa1a6e537d51215474012fb167fa6047e5b97249.jpg)

![969f28e04fcd60f56d9f92308de844a4ae3e13686d7f5ebe337e99f54dec49e2.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/images/969f28e04fcd60f56d9f92308de844a4ae3e13686d7f5ebe337e99f54dec49e2.jpg)

![9c5f01e7fdd6dc937741cb27ab84d95f4213df2183ac01feea1467e8cbfc5295.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/images/9c5f01e7fdd6dc937741cb27ab84d95f4213df2183ac01feea1467e8cbfc5295.jpg)

![a821875501ffbca02cd0da9cc8aae6ae4c46a5a2531d38e123fce9ccc0dd5f41.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/images/a821875501ffbca02cd0da9cc8aae6ae4c46a5a2531d38e123fce9ccc0dd5f41.jpg)

![d95a1ea19c31832237e792d6c5ac9d53221425726d297a9819ac80862a734d8f.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/images/d95a1ea19c31832237e792d6c5ac9d53221425726d297a9819ac80862a734d8f.jpg)

![eb7e642357be1e581656acb40e24e9d4cf55b3ec815d5444ce3f77474053cfe6.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/images/eb7e642357be1e581656acb40e24e9d4cf55b3ec815d5444ce3f77474053cfe6.jpg)

![ec16ac5eeabc9198c6cbfec541e548b860ed23b5d6240bce59af9172452371b1.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/images/ec16ac5eeabc9198c6cbfec541e548b860ed23b5d6240bce59af9172452371b1.jpg)

![f136f9bad2fbc26c91084cbbf6743fb9976166e207f40e508ac617bb686b44d8.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/images/f136f9bad2fbc26c91084cbbf6743fb9976166e207f40e508ac617bb686b44d8.jpg)

### Tables

![03f8e4428cc7b7842aaf5d97787b268b1007f0d56162b387e1e69cad599f98dd.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/03f8e4428cc7b7842aaf5d97787b268b1007f0d56162b387e1e69cad599f98dd.jpg)

![0f0a43cdbd842fa85c0f58f5b78ef9daa2d12fb41d2d3145b2b517e2804b0d99.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/0f0a43cdbd842fa85c0f58f5b78ef9daa2d12fb41d2d3145b2b517e2804b0d99.jpg)

![1705449a8a85cecf0bc62c1b61d1545bc3f8d27b62b960e1c02d4e96bf95759f.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/1705449a8a85cecf0bc62c1b61d1545bc3f8d27b62b960e1c02d4e96bf95759f.jpg)

![45ba93e71caadca7ebf294d75ba3f60a4f579c597bb1df152e4c9047fea4d4d3.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/45ba93e71caadca7ebf294d75ba3f60a4f579c597bb1df152e4c9047fea4d4d3.jpg)

![5c8fee18fa5b9faf8abb77f661a45cfca0891f71e3e4f82cd44907d187a773bf.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/5c8fee18fa5b9faf8abb77f661a45cfca0891f71e3e4f82cd44907d187a773bf.jpg)

![65229a43ed2c3e382b0230425a05ffce5627d80cd37ea801ddf98318ff7255a7.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/65229a43ed2c3e382b0230425a05ffce5627d80cd37ea801ddf98318ff7255a7.jpg)

![6c3cbdba9465168ceb156ed11b00097bf09d0eb59c8961615e76ef070a3e9a66.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/6c3cbdba9465168ceb156ed11b00097bf09d0eb59c8961615e76ef070a3e9a66.jpg)

![6efefd45c0fe1ef4ca0f17aa4c3fd5de45bb4746b6b7d70be91ca879346af1a1.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/6efefd45c0fe1ef4ca0f17aa4c3fd5de45bb4746b6b7d70be91ca879346af1a1.jpg)

![7ff0b98c2e1af9f047dadcb9a43e9e1f9ef209117b83f3cf809bc957984012b1.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/7ff0b98c2e1af9f047dadcb9a43e9e1f9ef209117b83f3cf809bc957984012b1.jpg)

![8a2a8a608c2418943dfbeb64c0ad25c631498e8c10cf68f4e51d36b369bbaf52.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/8a2a8a608c2418943dfbeb64c0ad25c631498e8c10cf68f4e51d36b369bbaf52.jpg)

![8e0ac4fab5a1ca1314e3e1a6c274fb21b77138306c6f84d40936edac23349a6b.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/8e0ac4fab5a1ca1314e3e1a6c274fb21b77138306c6f84d40936edac23349a6b.jpg)

![939802730b1f247cbb7d8de251773e03b51dac7024b351397a749ed4d514e1a0.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/939802730b1f247cbb7d8de251773e03b51dac7024b351397a749ed4d514e1a0.jpg)

![96d5b45a2c51f9e6a22de8412dec800db808cc508f6cce3ef906b26313ac5b33.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/96d5b45a2c51f9e6a22de8412dec800db808cc508f6cce3ef906b26313ac5b33.jpg)

![99207c576b5312561cba517de95a3891a2539e1f178c102a7708049cff04d70b.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/99207c576b5312561cba517de95a3891a2539e1f178c102a7708049cff04d70b.jpg)

![cc706ff242623f48d9b485ebebb842d660d4bb40b5777642ff394bd92516f981.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/cc706ff242623f48d9b485ebebb842d660d4bb40b5777642ff394bd92516f981.jpg)

![cdf920d6937625debab5c7abb490e6081b341e01e1289d4e4f0e7296c05c6bbe.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/cdf920d6937625debab5c7abb490e6081b341e01e1289d4e4f0e7296c05c6bbe.jpg)

![e32dcd931cedb7e8a2388bf8b493a2c5c15c3351027945adc4cb15f1f5b66b8d.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/e32dcd931cedb7e8a2388bf8b493a2c5c15c3351027945adc4cb15f1f5b66b8d.jpg)

![f614e6c6ae36bca2a1e1e4ce71015cf3474c19d6e368dbd52aa0cdb2f4ed04c8.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/f614e6c6ae36bca2a1e1e4ce71015cf3474c19d6e368dbd52aa0cdb2f4ed04c8.jpg)

![fd213164982e60e245cd7fffa50bb63e9578a37138baed6d930a6d33ba84df93.jpg](acl_results/591_GraphInsight_ Unlocking Insights in Large Language Models for Graph Structure Understanding/tables/fd213164982e60e245cd7fffa50bb63e9578a37138baed6d930a6d33ba84df93.jpg)

## Phonotomizer: A Compact, Unsupervised, Online Training Approach to Real-Time, Multilingual Phonetic Segmentation

### Images

![2cf8c5d5ece3909fd7064e371570675e313a0f3cee93c0b6c943ceb656992f18.jpg](acl_results/592_Phonotomizer_ A Compact, Unsupervised, Online Training Approach to Real-Time, Multilingual Phonetic /images/2cf8c5d5ece3909fd7064e371570675e313a0f3cee93c0b6c943ceb656992f18.jpg)

![323cefcf87a22b04aafcede2fb6d98dde8b04a039b8730d81ff2b0f68c58d5ed.jpg](acl_results/592_Phonotomizer_ A Compact, Unsupervised, Online Training Approach to Real-Time, Multilingual Phonetic /images/323cefcf87a22b04aafcede2fb6d98dde8b04a039b8730d81ff2b0f68c58d5ed.jpg)

![36379bafa907dedfcddb3c5eb4343229a88540cb0563f5c929610009cbcac88d.jpg](acl_results/592_Phonotomizer_ A Compact, Unsupervised, Online Training Approach to Real-Time, Multilingual Phonetic /images/36379bafa907dedfcddb3c5eb4343229a88540cb0563f5c929610009cbcac88d.jpg)

![8f6dbfc3aa46ee241dc37185ef0d0146dcf9cb132bbf581d0c23ff0a337e58f3.jpg](acl_results/592_Phonotomizer_ A Compact, Unsupervised, Online Training Approach to Real-Time, Multilingual Phonetic /images/8f6dbfc3aa46ee241dc37185ef0d0146dcf9cb132bbf581d0c23ff0a337e58f3.jpg)

![d7e03a335278cd57bb8fdcdc8c68af4b12c0284b8f5d0e8db534326ed45a3625.jpg](acl_results/592_Phonotomizer_ A Compact, Unsupervised, Online Training Approach to Real-Time, Multilingual Phonetic /images/d7e03a335278cd57bb8fdcdc8c68af4b12c0284b8f5d0e8db534326ed45a3625.jpg)

### Tables

![34bbd39f04edaeb5e4760da7d19a6c9adbf486df42dde37ce3db6166dd274fd4.jpg](acl_results/592_Phonotomizer_ A Compact, Unsupervised, Online Training Approach to Real-Time, Multilingual Phonetic /tables/34bbd39f04edaeb5e4760da7d19a6c9adbf486df42dde37ce3db6166dd274fd4.jpg)

![748738a44e048136406383c1e098f31f062eaca6049da2e3e790df60f92ad86b.jpg](acl_results/592_Phonotomizer_ A Compact, Unsupervised, Online Training Approach to Real-Time, Multilingual Phonetic /tables/748738a44e048136406383c1e098f31f062eaca6049da2e3e790df60f92ad86b.jpg)

![a3698175f44a8ff9ceafabd5f96f45242612c4c3d3e33df018f88e0997fe1dda.jpg](acl_results/592_Phonotomizer_ A Compact, Unsupervised, Online Training Approach to Real-Time, Multilingual Phonetic /tables/a3698175f44a8ff9ceafabd5f96f45242612c4c3d3e33df018f88e0997fe1dda.jpg)

![a48d53400f912257ed142410bfa4b2c9f9b5262f3a59ed9d02ec3f046f1190a5.jpg](acl_results/592_Phonotomizer_ A Compact, Unsupervised, Online Training Approach to Real-Time, Multilingual Phonetic /tables/a48d53400f912257ed142410bfa4b2c9f9b5262f3a59ed9d02ec3f046f1190a5.jpg)

![ff5060cb70321f90e3a43ec087fac97d1973d598b3682b277c05c6d4a14a0418.jpg](acl_results/592_Phonotomizer_ A Compact, Unsupervised, Online Training Approach to Real-Time, Multilingual Phonetic /tables/ff5060cb70321f90e3a43ec087fac97d1973d598b3682b277c05c6d4a14a0418.jpg)

## A Multi-persona Framework for Argument Quality Assessment

### Images

![4b598a66b52d6f6b798bc8dd478ce684f480a7fb6aae5e433bdaf23569ad4b8d.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/images/4b598a66b52d6f6b798bc8dd478ce684f480a7fb6aae5e433bdaf23569ad4b8d.jpg)

![64cad8581b15c0c9471f16b01f9b0735efd58ada661cdf369329d2dc0dd3f9fb.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/images/64cad8581b15c0c9471f16b01f9b0735efd58ada661cdf369329d2dc0dd3f9fb.jpg)

![773ee23b292b50ac903621c81e1599a3ccbebfaad6e022159b586b23658786ab.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/images/773ee23b292b50ac903621c81e1599a3ccbebfaad6e022159b586b23658786ab.jpg)

![820c1a96947fdee21741d639c1b760e15188533ab3cc48f755fa57bda7705e96.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/images/820c1a96947fdee21741d639c1b760e15188533ab3cc48f755fa57bda7705e96.jpg)

![a35f43c1c47cb19114f150b2209a904803c3b9195cc309c20f1c3f64a2f82ee1.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/images/a35f43c1c47cb19114f150b2209a904803c3b9195cc309c20f1c3f64a2f82ee1.jpg)

![be4855a9e9298b07261a49bd749e5015933c9efcecc8a43b5aaf98070186fa04.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/images/be4855a9e9298b07261a49bd749e5015933c9efcecc8a43b5aaf98070186fa04.jpg)

![d58693e64760bb59a04e1ef81c84d1272c035e0e377e6958cab83195e51b861c.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/images/d58693e64760bb59a04e1ef81c84d1272c035e0e377e6958cab83195e51b861c.jpg)

![f462b42d55c7eb93254f3419c37c5876cc4355ccd8cd707b34d0371c94961ae2.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/images/f462b42d55c7eb93254f3419c37c5876cc4355ccd8cd707b34d0371c94961ae2.jpg)

### Tables

![07c16b7edc7e0100c2374d0d9aad4c6e6833ebed0112dde48e6fe6aaf7a4c317.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/07c16b7edc7e0100c2374d0d9aad4c6e6833ebed0112dde48e6fe6aaf7a4c317.jpg)

![08f2ec1bcd236138572831a70a7569aa7fb1beacd7911650632a5020a7ec77f8.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/08f2ec1bcd236138572831a70a7569aa7fb1beacd7911650632a5020a7ec77f8.jpg)

![20aeb232c6db198e7963ae9c9b3001cb56e108085032dc2ca475a9475748f6f1.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/20aeb232c6db198e7963ae9c9b3001cb56e108085032dc2ca475a9475748f6f1.jpg)

![4c8adff3fc07cd0c120c1b1f8905eebcddf5ac9a08f71900a7af589d77e51510.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/4c8adff3fc07cd0c120c1b1f8905eebcddf5ac9a08f71900a7af589d77e51510.jpg)

![66419e7570be6d850861da60028340ca080a8faad610d29d48ef265574a905f4.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/66419e7570be6d850861da60028340ca080a8faad610d29d48ef265574a905f4.jpg)

![6fcdcf6f650f45a92b1df083ec28a5bbb070fbc3e68e7ed3b80c5504ec48dd55.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/6fcdcf6f650f45a92b1df083ec28a5bbb070fbc3e68e7ed3b80c5504ec48dd55.jpg)

![7f90a1576f6fa11cc410c4bebe01ad875bc64878de8ece369b2622f222875d30.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/7f90a1576f6fa11cc410c4bebe01ad875bc64878de8ece369b2622f222875d30.jpg)

![8e7df962c781a5bd87b0f0e288b02c53fed1e00e0cc992f5130b886b1b0d73e8.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/8e7df962c781a5bd87b0f0e288b02c53fed1e00e0cc992f5130b886b1b0d73e8.jpg)

![90a2494eb229711bc29cd3c36988ed8c2593bbce73e4575e7681d1983b3bb140.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/90a2494eb229711bc29cd3c36988ed8c2593bbce73e4575e7681d1983b3bb140.jpg)

![94c1f150520ee938b695469cc576ee5d53d5c1653ce61370f8d3850d806185a2.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/94c1f150520ee938b695469cc576ee5d53d5c1653ce61370f8d3850d806185a2.jpg)

![a71cc822564aa1f5aadb78e528fac02fa598f7e70b9219898680a2942cf2e02b.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/a71cc822564aa1f5aadb78e528fac02fa598f7e70b9219898680a2942cf2e02b.jpg)

![c75ac6b305fd0dcf467b6da8cfc70f25b8736073b0b2a28c9ef5343bc0d5d035.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/c75ac6b305fd0dcf467b6da8cfc70f25b8736073b0b2a28c9ef5343bc0d5d035.jpg)

![cf344fe8b56da3fe2754fc261a301ed572dd167ff444074867fcd5c9fdd966d4.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/cf344fe8b56da3fe2754fc261a301ed572dd167ff444074867fcd5c9fdd966d4.jpg)

![d91485eaebad0a112c8dc55ae74b97ef9fc4e97679a669d9c0d8ca95dc26d7a9.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/d91485eaebad0a112c8dc55ae74b97ef9fc4e97679a669d9c0d8ca95dc26d7a9.jpg)

![ddd6bb925aec1b96dfcc371e1b18a549782230ee083ed7dd469bcf070a0d033f.jpg](acl_results/593_A Multi-persona Framework for Argument Quality Assessment/tables/ddd6bb925aec1b96dfcc371e1b18a549782230ee083ed7dd469bcf070a0d033f.jpg)

## Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification

### Images

![0bef4703c33bffd897a4b234e9ee055b0faeae87d19c4fd8b96a33e7dc996ac1.jpg](acl_results/594_Safe_ Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal /images/0bef4703c33bffd897a4b234e9ee055b0faeae87d19c4fd8b96a33e7dc996ac1.jpg)

![0ca7dca41a42e037077d3f4374e085144d85da66914574a3bee70961867b5b20.jpg](acl_results/594_Safe_ Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal /images/0ca7dca41a42e037077d3f4374e085144d85da66914574a3bee70961867b5b20.jpg)

![319fa315d809844207fc347a4c71541d5195406b09ed6dd141ea243b29835364.jpg](acl_results/594_Safe_ Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal /images/319fa315d809844207fc347a4c71541d5195406b09ed6dd141ea243b29835364.jpg)

![80cc579225eb95bc374c3dfb11c590e7e35c3f2a940f6f8d4512072de74c32fd.jpg](acl_results/594_Safe_ Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal /images/80cc579225eb95bc374c3dfb11c590e7e35c3f2a940f6f8d4512072de74c32fd.jpg)

![927b2d1c7d98fa8e6072d588fa60117590bbaad3745917f9e502e14b5f7cbf3e.jpg](acl_results/594_Safe_ Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal /images/927b2d1c7d98fa8e6072d588fa60117590bbaad3745917f9e502e14b5f7cbf3e.jpg)

![e37b2f7bc708144146e8c5fef7d220daa84c3b11758b05ae64f5d3308ec8fdb3.jpg](acl_results/594_Safe_ Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal /images/e37b2f7bc708144146e8c5fef7d220daa84c3b11758b05ae64f5d3308ec8fdb3.jpg)

### Tables

![6d9b89d8476d2cd7d33821ed09530a3fbc052ea3cd492f0f63f7e9217feea40a.jpg](acl_results/594_Safe_ Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal /tables/6d9b89d8476d2cd7d33821ed09530a3fbc052ea3cd492f0f63f7e9217feea40a.jpg)

![6e31dd4f8cf69a138ee1d529a86be0ead140bc0b2b9912e4f00131892a1cafa3.jpg](acl_results/594_Safe_ Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal /tables/6e31dd4f8cf69a138ee1d529a86be0ead140bc0b2b9912e4f00131892a1cafa3.jpg)

![88cf4db8a4eeba7bfc54d6fc2202358e517bd339d302d88da97f4c0e8ced1e56.jpg](acl_results/594_Safe_ Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal /tables/88cf4db8a4eeba7bfc54d6fc2202358e517bd339d302d88da97f4c0e8ced1e56.jpg)

![b73c011a417d395630455b7cd787fa483833903a5030f22afa42d5573a2688a5.jpg](acl_results/594_Safe_ Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal /tables/b73c011a417d395630455b7cd787fa483833903a5030f22afa42d5573a2688a5.jpg)

![f8ba073c548789a79640b0bdd5b5213ed4984d55ed7201ae4240cdb06c904cd6.jpg](acl_results/594_Safe_ Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal /tables/f8ba073c548789a79640b0bdd5b5213ed4984d55ed7201ae4240cdb06c904cd6.jpg)

## SAMDecoding: Speculative Decoding via Suffix Automaton

### Images

![00c9f60f9b009ac42b5fd43723d3b9190ce09567113d5165928d068a967d9f32.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/images/00c9f60f9b009ac42b5fd43723d3b9190ce09567113d5165928d068a967d9f32.jpg)

![074409e3620cbeab1393c1e767742291ff7f445073ec13491e15d69b90523475.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/images/074409e3620cbeab1393c1e767742291ff7f445073ec13491e15d69b90523475.jpg)

![0fec469b77aecb2e42d10d2a06af1b88454765d0cd7936d6a30754d9276c39ff.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/images/0fec469b77aecb2e42d10d2a06af1b88454765d0cd7936d6a30754d9276c39ff.jpg)

![12a829ad67d2a49e1c464fb94f8a3fa815d0786386eef4570a4c489c43869f16.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/images/12a829ad67d2a49e1c464fb94f8a3fa815d0786386eef4570a4c489c43869f16.jpg)

![2547aa44b613d4f3b0451cb09174956db4c0d66613d1cc670b37cfd74fccd7bf.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/images/2547aa44b613d4f3b0451cb09174956db4c0d66613d1cc670b37cfd74fccd7bf.jpg)

![2ffec63d0b70beceb0098fcb8582b858fda1d299595a2bed5e3403235ca8291a.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/images/2ffec63d0b70beceb0098fcb8582b858fda1d299595a2bed5e3403235ca8291a.jpg)

![4cc6c86f9d0b473ddbbd3a5a8393480471727e3bfa54d64101d4415c552b8f87.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/images/4cc6c86f9d0b473ddbbd3a5a8393480471727e3bfa54d64101d4415c552b8f87.jpg)

![5b398c39c292e2be0e5c5a1fc82cc3de83b3bb22d804595034f0a2795250602e.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/images/5b398c39c292e2be0e5c5a1fc82cc3de83b3bb22d804595034f0a2795250602e.jpg)

![831ddf936d9f1959ee779ba4ca257f2c08877fbda0e4bd47aac8b5caa2f5aac8.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/images/831ddf936d9f1959ee779ba4ca257f2c08877fbda0e4bd47aac8b5caa2f5aac8.jpg)

### Tables

![1ee4946b62c7b16111d26c32ef0fff6875ee6c3df0b897a1c2e1184f27a19c50.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/1ee4946b62c7b16111d26c32ef0fff6875ee6c3df0b897a1c2e1184f27a19c50.jpg)

![2a301e6ed8b8df48cd215de6087a62e56e3ab023bd6378bcafd6fe743c6fcbe1.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/2a301e6ed8b8df48cd215de6087a62e56e3ab023bd6378bcafd6fe743c6fcbe1.jpg)

![39ec3d18055b9f5d2cfbb548507a7ed04a61894451a5039661e164884fa8fde2.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/39ec3d18055b9f5d2cfbb548507a7ed04a61894451a5039661e164884fa8fde2.jpg)

![611f54521b9edd8bc62fcb0b8b7f27d9b6cbb3f06fc51e27bcf31da33dcd16e1.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/611f54521b9edd8bc62fcb0b8b7f27d9b6cbb3f06fc51e27bcf31da33dcd16e1.jpg)

![6d4f9b8b67d16ca8a214409ca7e0f26d3bb88d11ce126ba9f42098d1e0b3fd54.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/6d4f9b8b67d16ca8a214409ca7e0f26d3bb88d11ce126ba9f42098d1e0b3fd54.jpg)

![6f54ee7f109afcf9d14ed63d26431c3981da35322c48792d3cef74fd5d732466.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/6f54ee7f109afcf9d14ed63d26431c3981da35322c48792d3cef74fd5d732466.jpg)

![8ab3fa38f70906d25dbd6ce8e1b3564fd3b4a619c246a4cd6ddaa239192aa8d7.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/8ab3fa38f70906d25dbd6ce8e1b3564fd3b4a619c246a4cd6ddaa239192aa8d7.jpg)

![a224e1d52112e3fef9db29aad84cc9265d2d7ac8e84b5c3c887ca5b533af76f6.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/a224e1d52112e3fef9db29aad84cc9265d2d7ac8e84b5c3c887ca5b533af76f6.jpg)

![a8ec4a95b225a6970546153f607aae54781c84a1df3924a69713f5cabf528a6b.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/a8ec4a95b225a6970546153f607aae54781c84a1df3924a69713f5cabf528a6b.jpg)

![b9026b52726047746f19539573a08b3609f3b3f40faac0b04be3e42eb9dd5e36.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/b9026b52726047746f19539573a08b3609f3b3f40faac0b04be3e42eb9dd5e36.jpg)

![bd32428da5ed091da0af699314fb1681ed29a894e82e0bfc10d23115ffd610b5.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/bd32428da5ed091da0af699314fb1681ed29a894e82e0bfc10d23115ffd610b5.jpg)

![c27ee4dbbe45b79dd02eddbd44128c793c20a4968eda30a74e03526f10bf0010.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/c27ee4dbbe45b79dd02eddbd44128c793c20a4968eda30a74e03526f10bf0010.jpg)

![c2b9d165330c1f600fa6b2d533aa1e6c73b0cbbe0a6e1b02fc38828ed737b7b5.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/c2b9d165330c1f600fa6b2d533aa1e6c73b0cbbe0a6e1b02fc38828ed737b7b5.jpg)

![f4406664a17f24e75a0ddcde835cc4191e4d5bfd682b8d52e0c3b1b04aab762b.jpg](acl_results/595_SAMDecoding_ Speculative Decoding via Suffix Automaton/tables/f4406664a17f24e75a0ddcde835cc4191e4d5bfd682b8d52e0c3b1b04aab762b.jpg)

## PsyAdvisor: A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conversations

### Images

![3a210e58ed4d5fd8f50f073164f972e91544e4604b118f7bf70e6fdb636c9dd4.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/images/3a210e58ed4d5fd8f50f073164f972e91544e4604b118f7bf70e6fdb636c9dd4.jpg)

![a16c087acbcbd1326f30622296bab1f04d122d2c6567c21b70b071074f1f6886.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/images/a16c087acbcbd1326f30622296bab1f04d122d2c6567c21b70b071074f1f6886.jpg)

![a3dbcb059fe080a98315969a08e77a982ca7625da7550f8ac4832f6ede6b5941.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/images/a3dbcb059fe080a98315969a08e77a982ca7625da7550f8ac4832f6ede6b5941.jpg)

![a7131fadc6e85b5b7c05e6dd0358f0550013d6ae9412a25f295e89c4590a2a3c.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/images/a7131fadc6e85b5b7c05e6dd0358f0550013d6ae9412a25f295e89c4590a2a3c.jpg)

![c7f0cde483feebc4ace5170f5d07269bec3c82e4be57a150bc57d7c4902db938.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/images/c7f0cde483feebc4ace5170f5d07269bec3c82e4be57a150bc57d7c4902db938.jpg)

![d9903c7170396d92674334b5e2dbb3eb47d7f4d276b1237899ba1f3998ec485b.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/images/d9903c7170396d92674334b5e2dbb3eb47d7f4d276b1237899ba1f3998ec485b.jpg)

### Tables

![0675033487d9b2f2fb11f67dd45c6c27631f481d639b9526731c40ad33534883.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/tables/0675033487d9b2f2fb11f67dd45c6c27631f481d639b9526731c40ad33534883.jpg)

![200b9c73a4edaf051566ba41bae7962474507c798f5945b5a2c44473ce324931.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/tables/200b9c73a4edaf051566ba41bae7962474507c798f5945b5a2c44473ce324931.jpg)

![5d8952b9a49f86286f2b6bbfd7d31363d106676c66afe4b5a178a2a523f4f1c9.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/tables/5d8952b9a49f86286f2b6bbfd7d31363d106676c66afe4b5a178a2a523f4f1c9.jpg)

![6d3c1e56411e66176d43496ee2744b3559b83c5c75bc8f2bb356968f8b72ac63.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/tables/6d3c1e56411e66176d43496ee2744b3559b83c5c75bc8f2bb356968f8b72ac63.jpg)

![a33caf9eeead3971ed5d519516e53ce995d017354486a441f8f742abbea51871.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/tables/a33caf9eeead3971ed5d519516e53ce995d017354486a441f8f742abbea51871.jpg)

![a5b5dc3a470e165079633deeabcc904bcab6664d657d864be0cc326102371c29.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/tables/a5b5dc3a470e165079633deeabcc904bcab6664d657d864be0cc326102371c29.jpg)

![f52bc19f09a8d1ccf9aedeb052d86cf626dee6918c3d4be91c3529882a4c88d8.jpg](acl_results/596_PsyAdvisor_ A Plug-and-Play Strategy Advice Planner with Proactive Questioning in Psychological Conv/tables/f52bc19f09a8d1ccf9aedeb052d86cf626dee6918c3d4be91c3529882a4c88d8.jpg)

## HomeBench: EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multiple Devices

### Images

![1090ac33c24b52869fafdaca94d4f0681f17679cd10a64c77dde98b1e9745209.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/1090ac33c24b52869fafdaca94d4f0681f17679cd10a64c77dde98b1e9745209.jpg)

![1e9ee35a980791db85034f5583633f1c84a6dec4341340d52cd65e89a49df19a.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/1e9ee35a980791db85034f5583633f1c84a6dec4341340d52cd65e89a49df19a.jpg)

![20d36326579efc142be89db843ef4fb82ea1f3f43e0fc0bdae86b9c2b46ce2c9.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/20d36326579efc142be89db843ef4fb82ea1f3f43e0fc0bdae86b9c2b46ce2c9.jpg)

![291b9fde2e76f4374eed5c2947750d661e29e4fb0657d6b38a6c0d399448d18b.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/291b9fde2e76f4374eed5c2947750d661e29e4fb0657d6b38a6c0d399448d18b.jpg)

![46a1aaf6159f4f07984c8ec98a4321c720d25039be13ed9f3950e8b643e9fce1.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/46a1aaf6159f4f07984c8ec98a4321c720d25039be13ed9f3950e8b643e9fce1.jpg)

![79dd2bdf663e066b5a188d92da0ac6ba63d85d13e7f0fa9ad10c04438becffb4.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/79dd2bdf663e066b5a188d92da0ac6ba63d85d13e7f0fa9ad10c04438becffb4.jpg)

![7d129ed02d2947b459dbd1d164edff93e192b26478a2f884e4ba232b7e1ff2a1.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/7d129ed02d2947b459dbd1d164edff93e192b26478a2f884e4ba232b7e1ff2a1.jpg)

![b5975733afaca289f94ddb7bcabf15994adf6d7bf5c0faa7bdfdc4d14b2f70bc.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/b5975733afaca289f94ddb7bcabf15994adf6d7bf5c0faa7bdfdc4d14b2f70bc.jpg)

![c4ed421f944fe98e440f232cc13977287357109fe4a7ee25180132accb9ed7be.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/c4ed421f944fe98e440f232cc13977287357109fe4a7ee25180132accb9ed7be.jpg)

![c7257b75c659668af011d5d976a08104745cdb03c284b4c16470461a3daf8712.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/c7257b75c659668af011d5d976a08104745cdb03c284b4c16470461a3daf8712.jpg)

![c98fbeb35dbbab2e5f863559ad0928b425c4ea1c306ed2a2a2a0b1245dac03bd.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/c98fbeb35dbbab2e5f863559ad0928b425c4ea1c306ed2a2a2a0b1245dac03bd.jpg)

![d78d48594c0ee5e8ad13ec079bc883e357373ee8995047ba4dc7a36e49628666.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/d78d48594c0ee5e8ad13ec079bc883e357373ee8995047ba4dc7a36e49628666.jpg)

![dfda3de2bf2372b698069bf53ed52ec9a785d62c7bc0036beb866b7832a94fb9.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/dfda3de2bf2372b698069bf53ed52ec9a785d62c7bc0036beb866b7832a94fb9.jpg)

![ecf22f32effa0e899ded5057d48bd77c05c63abae211df33769529ee50ee2d5c.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/images/ecf22f32effa0e899ded5057d48bd77c05c63abae211df33769529ee50ee2d5c.jpg)

### Tables

![0185a9d97eea3192618057ece181872d6a84bac0405c43ea196a9c1e8a17765a.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/tables/0185a9d97eea3192618057ece181872d6a84bac0405c43ea196a9c1e8a17765a.jpg)

![01b5acfc52604b4dc9e6bd280aa5a485b68e2685d53efdabc7214dae5f394246.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/tables/01b5acfc52604b4dc9e6bd280aa5a485b68e2685d53efdabc7214dae5f394246.jpg)

![76e65a79054de7c4d492c1ff36bd9c244d6724c400ce1caa215bb9e904ba2b8b.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/tables/76e65a79054de7c4d492c1ff36bd9c244d6724c400ce1caa215bb9e904ba2b8b.jpg)

![960d15ea240f663efcfdfade338ba67bab561455bb2f82243a4cc7ccc6fea6d4.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/tables/960d15ea240f663efcfdfade338ba67bab561455bb2f82243a4cc7ccc6fea6d4.jpg)

![9d732a1f3e2808792c569c537e48d152cf47dffd2c973652b16cc615913c6acf.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/tables/9d732a1f3e2808792c569c537e48d152cf47dffd2c973652b16cc615913c6acf.jpg)

![a40eebe8ade229d405b2be3a2eed7478153640c0e15e3818709e556244fe454d.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/tables/a40eebe8ade229d405b2be3a2eed7478153640c0e15e3818709e556244fe454d.jpg)

![a4df46a536d6f8c3f14c93f452c08013b8402ab7508199382d3d035e8d51d600.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/tables/a4df46a536d6f8c3f14c93f452c08013b8402ab7508199382d3d035e8d51d600.jpg)

![c01190a264d7b5f5fe46510b9a1d2a86c821159bba902a0c5c67fe25bcde666b.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/tables/c01190a264d7b5f5fe46510b9a1d2a86c821159bba902a0c5c67fe25bcde666b.jpg)

![d253b3f10c444776ee0c0292ea124d72886b3843f5164d0574bdb62172776e8c.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/tables/d253b3f10c444776ee0c0292ea124d72886b3843f5164d0574bdb62172776e8c.jpg)

![dac907fe3797a5be0f2cb28e87701c7e7dece7c01f822878d9ba30b12779d78c.jpg](acl_results/597_HomeBench_ EvaluatingLLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multi/tables/dac907fe3797a5be0f2cb28e87701c7e7dece7c01f822878d9ba30b12779d78c.jpg)

## Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment

### Images

![53ba17c348ec6939d4935b506c29e311988749cbb55e221eb036d3844b79c561.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/images/53ba17c348ec6939d4935b506c29e311988749cbb55e221eb036d3844b79c561.jpg)

![5e2c7e557d72a4f139fde7bcf9c374390cce29c41020db92c088c7e1d842c6ac.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/images/5e2c7e557d72a4f139fde7bcf9c374390cce29c41020db92c088c7e1d842c6ac.jpg)

![9b6f8fecdc77530f678798363c2608e95f196a177b3c877ff90f67a2b961dd61.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/images/9b6f8fecdc77530f678798363c2608e95f196a177b3c877ff90f67a2b961dd61.jpg)

![f612572380048e928922d8316dcea5ba211fda3268773d1fcb38c69224e1c26c.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/images/f612572380048e928922d8316dcea5ba211fda3268773d1fcb38c69224e1c26c.jpg)

### Tables

![036241a43aa556df8dcfd3a101b874e399fbbfe2dd27d6b855057af030c8d04e.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/tables/036241a43aa556df8dcfd3a101b874e399fbbfe2dd27d6b855057af030c8d04e.jpg)

![476f4dfae164559d25e2ac3d5cb6602d83fdb31910d23f15b3d47439622fb9b1.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/tables/476f4dfae164559d25e2ac3d5cb6602d83fdb31910d23f15b3d47439622fb9b1.jpg)

![4f7febfb7eb50e4c9b9010d0a7dbba7b381dda1e72453592b04c096f0e7a7983.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/tables/4f7febfb7eb50e4c9b9010d0a7dbba7b381dda1e72453592b04c096f0e7a7983.jpg)

![69eb1064ee09cf961eaf5e22d7146190fe0453780d287b29afb3343d1f0bc9d4.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/tables/69eb1064ee09cf961eaf5e22d7146190fe0453780d287b29afb3343d1f0bc9d4.jpg)

![7f55a012a402f8757b0be23338a4435492fea3a4de453535539db686eca13357.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/tables/7f55a012a402f8757b0be23338a4435492fea3a4de453535539db686eca13357.jpg)

![8e01bb184a591844a0d746ae6e87b8dc0d90c204a47428f898573ee5eb2700be.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/tables/8e01bb184a591844a0d746ae6e87b8dc0d90c204a47428f898573ee5eb2700be.jpg)

![aa46de39606f58f302955db82b9d424c7e948f1cec332534e548d3ceaa9f6de8.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/tables/aa46de39606f58f302955db82b9d424c7e948f1cec332534e548d3ceaa9f6de8.jpg)

![c13e747a3c27860e939c44d4dc0aaa8e356979246a2b4fbcaafed3dbd2be992a.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/tables/c13e747a3c27860e939c44d4dc0aaa8e356979246a2b4fbcaafed3dbd2be992a.jpg)

![d80c49d9379dfd7c5b522c10055630df40fadf06257aaba5c76eb50df023af1c.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/tables/d80c49d9379dfd7c5b522c10055630df40fadf06257aaba5c76eb50df023af1c.jpg)

![d957866ca5ec466d0f5854fcb1c8ac44a8b9406d78b08eefc60abb7a78a6abcf.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/tables/d957866ca5ec466d0f5854fcb1c8ac44a8b9406d78b08eefc60abb7a78a6abcf.jpg)

![f010c2b8f7fed3fd095ada6f92de33069ab290f9c52234b3262b059dd2afef0c.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/tables/f010c2b8f7fed3fd095ada6f92de33069ab290f9c52234b3262b059dd2afef0c.jpg)

![fe709971c3799944d1ccfb1d7ffb18bbc4b62297d93f3fead3fc56ef25dbeedc.jpg](acl_results/598_Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment/tables/fe709971c3799944d1ccfb1d7ffb18bbc4b62297d93f3fead3fc56ef25dbeedc.jpg)

## GiFT:Gibbs Fine-Tuning for Code Generation

### Images

![09c1b84bf6b21a7ff5a2651378654963ac59ca7c3d0a7be884b8544153a4e00b.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/images/09c1b84bf6b21a7ff5a2651378654963ac59ca7c3d0a7be884b8544153a4e00b.jpg)

![0c2ff94b2c8999756d2764cae6cc52ab37e18e6ed8613d98b3600c0961c56942.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/images/0c2ff94b2c8999756d2764cae6cc52ab37e18e6ed8613d98b3600c0961c56942.jpg)

![1aceadb63610e75f6ee84a0f89db29fa2e2f3ab91a457e115928fc31c1c56041.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/images/1aceadb63610e75f6ee84a0f89db29fa2e2f3ab91a457e115928fc31c1c56041.jpg)

![1cd3f63cf3323adce404fe150c489a9f12180d7b0e8a079dc61b092209c17c0e.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/images/1cd3f63cf3323adce404fe150c489a9f12180d7b0e8a079dc61b092209c17c0e.jpg)

![20b03a340e9acd6f083aad94c2013c9fb9fb1e230c822935bd53085113cc1ac3.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/images/20b03a340e9acd6f083aad94c2013c9fb9fb1e230c822935bd53085113cc1ac3.jpg)

![26bc162b97b5a5afe91f1ffca9d8f53b80243c3529a34746ef6fae19f34ac640.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/images/26bc162b97b5a5afe91f1ffca9d8f53b80243c3529a34746ef6fae19f34ac640.jpg)

![2f94f2bbb69463efc9b9ea36a41f2f91f1c9fd00c169db36e4d043f3fe5619e5.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/images/2f94f2bbb69463efc9b9ea36a41f2f91f1c9fd00c169db36e4d043f3fe5619e5.jpg)

![5e4360a9eb5e536b47f018f6d231f364fe075ebdd7cc2bb33f9c9184dade111c.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/images/5e4360a9eb5e536b47f018f6d231f364fe075ebdd7cc2bb33f9c9184dade111c.jpg)

![828862140699f12d4517327240961015b4ee3763def00241aa5094db909a802c.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/images/828862140699f12d4517327240961015b4ee3763def00241aa5094db909a802c.jpg)

![8b8168d0b1bec468d777ba9684483e9b4af2c2942a980afc3cf3b67e38e3ee04.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/images/8b8168d0b1bec468d777ba9684483e9b4af2c2942a980afc3cf3b67e38e3ee04.jpg)

![a1a8fba3a0d18d5323ef070e0fa3c4e495a79f17170946197e11a37352c86fb9.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/images/a1a8fba3a0d18d5323ef070e0fa3c4e495a79f17170946197e11a37352c86fb9.jpg)

![ec73823c76c31eb02b39f01433e5d469b8124b7ed4d34aabc7543c935b1b36d0.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/images/ec73823c76c31eb02b39f01433e5d469b8124b7ed4d34aabc7543c935b1b36d0.jpg)

### Tables

![7578ac0eed3b2f064489707f1cd25582ecc1ddef1f2c92780dd1fbc5aaa89fd7.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/tables/7578ac0eed3b2f064489707f1cd25582ecc1ddef1f2c92780dd1fbc5aaa89fd7.jpg)

![80502759b9116c2064df8db963d3ffce049011403acf795e6fa472d31df2ea0a.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/tables/80502759b9116c2064df8db963d3ffce049011403acf795e6fa472d31df2ea0a.jpg)

![b704dd88bee7bccfbb290a95951fb786de38b4f1c2fa83bc80610008fe935f0f.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/tables/b704dd88bee7bccfbb290a95951fb786de38b4f1c2fa83bc80610008fe935f0f.jpg)

![de1301c6770deaae0f835ce53596a95f0a6579b05833a01b71e25bf894d70b4f.jpg](acl_results/599_GiFT_Gibbs Fine-Tuning for Code Generation/tables/de1301c6770deaae0f835ce53596a95f0a6579b05833a01b71e25bf894d70b4f.jpg)

## Enhancing Interpretable Image Classification ThroughLLMAgents and Conditional Concept Bottleneck Models

### Images

![425de8464749d28c33e55386327a10f4bbb53eda32537825a4e74df5ac48306e.jpg](acl_results/600_Enhancing Interpretable Image Classification ThroughLLMAgents and Conditional Concept Bottleneck Mod/images/425de8464749d28c33e55386327a10f4bbb53eda32537825a4e74df5ac48306e.jpg)

![939618288a360259a91377566c1daeb666c803610a6412c408525d8fd4674801.jpg](acl_results/600_Enhancing Interpretable Image Classification ThroughLLMAgents and Conditional Concept Bottleneck Mod/images/939618288a360259a91377566c1daeb666c803610a6412c408525d8fd4674801.jpg)

![a0f6dc8dd6f251a7685d525d535bdf0899e0b7a3845daa27b54e1618ec84f14c.jpg](acl_results/600_Enhancing Interpretable Image Classification ThroughLLMAgents and Conditional Concept Bottleneck Mod/images/a0f6dc8dd6f251a7685d525d535bdf0899e0b7a3845daa27b54e1618ec84f14c.jpg)

![b1c045b17382a2c321ebd7246ace6b78789922cc496a281572ea3a0dbeb67fcf.jpg](acl_results/600_Enhancing Interpretable Image Classification ThroughLLMAgents and Conditional Concept Bottleneck Mod/images/b1c045b17382a2c321ebd7246ace6b78789922cc496a281572ea3a0dbeb67fcf.jpg)

![f221ad31c6f518df20007ff3b49576d290820288bda95d890c16a123c5b9988c.jpg](acl_results/600_Enhancing Interpretable Image Classification ThroughLLMAgents and Conditional Concept Bottleneck Mod/images/f221ad31c6f518df20007ff3b49576d290820288bda95d890c16a123c5b9988c.jpg)

### Tables

![40615ac63821835b7c60f21be30c10fc944504c0e209947f6966f833efc570f8.jpg](acl_results/600_Enhancing Interpretable Image Classification ThroughLLMAgents and Conditional Concept Bottleneck Mod/tables/40615ac63821835b7c60f21be30c10fc944504c0e209947f6966f833efc570f8.jpg)

![77664f78688fe7db7c4bc77df719bdb864dc45ca149517b09246c0d8d2e131f2.jpg](acl_results/600_Enhancing Interpretable Image Classification ThroughLLMAgents and Conditional Concept Bottleneck Mod/tables/77664f78688fe7db7c4bc77df719bdb864dc45ca149517b09246c0d8d2e131f2.jpg)

## Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction

### Images

![4bfb9c7b2e3876eed22b0a7e0cef83fda74d88e0ae37272c519b728552a0a385.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/images/4bfb9c7b2e3876eed22b0a7e0cef83fda74d88e0ae37272c519b728552a0a385.jpg)

![4c5ca12dc119acbac6bc33f07f947006ecdcc744aa8b22331bb199cdd7e16ff9.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/images/4c5ca12dc119acbac6bc33f07f947006ecdcc744aa8b22331bb199cdd7e16ff9.jpg)

![6cbd1e12f61f537dc75acce537edba3eda8bc643e6c147813c0cab0ecace146f.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/images/6cbd1e12f61f537dc75acce537edba3eda8bc643e6c147813c0cab0ecace146f.jpg)

![9d056d7ea7a44fb3c988c0d46f05ef5d1d1c15004b3249dbd9271ae17f07ee6c.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/images/9d056d7ea7a44fb3c988c0d46f05ef5d1d1c15004b3249dbd9271ae17f07ee6c.jpg)

![a00db9f331f6a1e21390a523ecffb79e47310e225cc376d98d1d89eb99c76211.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/images/a00db9f331f6a1e21390a523ecffb79e47310e225cc376d98d1d89eb99c76211.jpg)

![a3132b4e7be3b72e2d1cb209641e8ee82bfe41d37d198df1669cd6edf4593246.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/images/a3132b4e7be3b72e2d1cb209641e8ee82bfe41d37d198df1669cd6edf4593246.jpg)

![bd272861c10b54751a1af5628577bcf003325806392690350dc76ca5c345a639.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/images/bd272861c10b54751a1af5628577bcf003325806392690350dc76ca5c345a639.jpg)

![ce08fed7522067057dd58b45c15ce533e3ec95ca9d456f21fb4c95dfd1aef0c8.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/images/ce08fed7522067057dd58b45c15ce533e3ec95ca9d456f21fb4c95dfd1aef0c8.jpg)

![d829c55d58d2bc13a316ab52296cd36271dee717b39e898f443653907050ea00.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/images/d829c55d58d2bc13a316ab52296cd36271dee717b39e898f443653907050ea00.jpg)

![dd53830f71178e951ed7f32736ebd8591b1950f977937d9fa9200fd69dddbdb7.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/images/dd53830f71178e951ed7f32736ebd8591b1950f977937d9fa9200fd69dddbdb7.jpg)

### Tables

![1b13bce4b45e6b87bab2a822f8c8f0a0af8807da3d8f0d953970b0ac2e4723d2.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/1b13bce4b45e6b87bab2a822f8c8f0a0af8807da3d8f0d953970b0ac2e4723d2.jpg)

![369cdb948fe114485d787e6efb96989420f6c77e5007d3c68184c9952951d2ae.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/369cdb948fe114485d787e6efb96989420f6c77e5007d3c68184c9952951d2ae.jpg)

![3fcb798ee5df0bde46f09e0ff6af4183c713c475e848dfeb71963682eb9ccb83.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/3fcb798ee5df0bde46f09e0ff6af4183c713c475e848dfeb71963682eb9ccb83.jpg)

![48699360f0319b1c868337113df0c915f6de8f88d01dfe2685b578f2a3c2bccd.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/48699360f0319b1c868337113df0c915f6de8f88d01dfe2685b578f2a3c2bccd.jpg)

![4e6713f761e4f3c9ffb8c7880dcaefebdc33392064470c57d1907c47cd624c75.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/4e6713f761e4f3c9ffb8c7880dcaefebdc33392064470c57d1907c47cd624c75.jpg)

![516b659158725ce636f5dddb6600390274eb1799c79020d2ff5d2a07088c969f.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/516b659158725ce636f5dddb6600390274eb1799c79020d2ff5d2a07088c969f.jpg)

![6356bbb40d9b4295bcd730e2c750c28071f1f4b00bfef6634b69551be395af58.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/6356bbb40d9b4295bcd730e2c750c28071f1f4b00bfef6634b69551be395af58.jpg)

![6c650318c3eb86a7f25c6f93d52f6efec06112f51f3de3f5ee2c1549cd053823.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/6c650318c3eb86a7f25c6f93d52f6efec06112f51f3de3f5ee2c1549cd053823.jpg)

![7bffde5cb8c368078dd5b8ece3d3f53c795e216eb7670e05797116da65c9e87d.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/7bffde5cb8c368078dd5b8ece3d3f53c795e216eb7670e05797116da65c9e87d.jpg)

![91a2c439eadffb64f9acccd7064e507fb76f7f22cdfa9241fef443252d424d10.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/91a2c439eadffb64f9acccd7064e507fb76f7f22cdfa9241fef443252d424d10.jpg)

![99b17756d0c6660a171b283d410226ad3d73b98ac2793a9fec93a7eeb0ab50e5.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/99b17756d0c6660a171b283d410226ad3d73b98ac2793a9fec93a7eeb0ab50e5.jpg)

![b08606a10d494da5f6f471a00117267351e2d1daa61ab07c1c3602573d83ffa0.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/b08606a10d494da5f6f471a00117267351e2d1daa61ab07c1c3602573d83ffa0.jpg)

![b9cf03a58556d8ee90ef815724d11989d54063bb064b5f0ebba0c1cb3bd031c2.jpg](acl_results/601_Reliably Bounding False Positives_ A Zero-Shot Machine-Generated Text Detection Framework via Multis/tables/b9cf03a58556d8ee90ef815724d11989d54063bb064b5f0ebba0c1cb3bd031c2.jpg)

## RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph

### Images

![048953fb388ded8a3e3ffe467fa73112c39212714df682f6a0c61a56023f35b5.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/images/048953fb388ded8a3e3ffe467fa73112c39212714df682f6a0c61a56023f35b5.jpg)

![13945803e656b5df71a015b3991ec4da3ae33204dc9580f8ac2aaa13a7c6ad12.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/images/13945803e656b5df71a015b3991ec4da3ae33204dc9580f8ac2aaa13a7c6ad12.jpg)

![14b52c8f98c9da0f01e68ab576f0267299279f3e9ef5a4acc55fb74299bf9dfe.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/images/14b52c8f98c9da0f01e68ab576f0267299279f3e9ef5a4acc55fb74299bf9dfe.jpg)

![54bcdd278e9ce64c8b5028143fb15e63fe8f620aadc13acef532a0d709ff5e62.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/images/54bcdd278e9ce64c8b5028143fb15e63fe8f620aadc13acef532a0d709ff5e62.jpg)

![7160f6ba76aa73fa084201d395d329c12ed377d8435516b54fa8e6f6cc731d48.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/images/7160f6ba76aa73fa084201d395d329c12ed377d8435516b54fa8e6f6cc731d48.jpg)

![757e0e023d81bbb236eb7c7f0454c06d0e03d4ed32daf07f6cf6be878145db9a.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/images/757e0e023d81bbb236eb7c7f0454c06d0e03d4ed32daf07f6cf6be878145db9a.jpg)

![898f2906c3a90f679301c051e234351dd1d9bb66e8a7dd36f8300da68667a313.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/images/898f2906c3a90f679301c051e234351dd1d9bb66e8a7dd36f8300da68667a313.jpg)

![a8f050338d17bb61613cd99b8fa24a1d1e626e3ea18bca5d6f4b873d51561c51.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/images/a8f050338d17bb61613cd99b8fa24a1d1e626e3ea18bca5d6f4b873d51561c51.jpg)

![aa76d7787837b2c8d039d5178fc6e09b76f48e32db6ede8822fee09d444b69c5.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/images/aa76d7787837b2c8d039d5178fc6e09b76f48e32db6ede8822fee09d444b69c5.jpg)

![d076e14868c90189e2c1faf1e3754e2eaba669bcf5bc843e76820b3c281cbf60.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/images/d076e14868c90189e2c1faf1e3754e2eaba669bcf5bc843e76820b3c281cbf60.jpg)

### Tables

![0618da0fe375f6b1bd6d3081d01b2a3f8e2314719835e547dd431ede8a4ade0e.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/0618da0fe375f6b1bd6d3081d01b2a3f8e2314719835e547dd431ede8a4ade0e.jpg)

![06a42ba928ef9f410113c252fe370155b48a516d4e0aebca1ce5cf2b3de32b3f.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/06a42ba928ef9f410113c252fe370155b48a516d4e0aebca1ce5cf2b3de32b3f.jpg)

![09c61969740988aad07c3215c31861a6a50665be1105c5517bbfa8cb402d0f83.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/09c61969740988aad07c3215c31861a6a50665be1105c5517bbfa8cb402d0f83.jpg)

![0d0764927777f3c4ed8e390af6ed881f3fc3e3a8df4fb27c7f5cfe697a3e9695.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/0d0764927777f3c4ed8e390af6ed881f3fc3e3a8df4fb27c7f5cfe697a3e9695.jpg)

![246f9ad149a0fc38da1f380770ba8d14ca8d180ad5004afb8ef27a483eb372e0.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/246f9ad149a0fc38da1f380770ba8d14ca8d180ad5004afb8ef27a483eb372e0.jpg)

![48da00b52c1c269c88b6169f004ffa5ae2cede96f20a193970d4bacf6e3dd727.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/48da00b52c1c269c88b6169f004ffa5ae2cede96f20a193970d4bacf6e3dd727.jpg)

![4c412d61d3714f7e3bb6cea76a28bf6945f77068e275e610a8fba3a084cb87a9.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/4c412d61d3714f7e3bb6cea76a28bf6945f77068e275e610a8fba3a084cb87a9.jpg)

![4d38ffe02228447d6c9a0c76b7861dc10112c990b272f9423288fcfab25a1834.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/4d38ffe02228447d6c9a0c76b7861dc10112c990b272f9423288fcfab25a1834.jpg)

![7b38744233c9bfc5ea8c216b63d65878f3597bffb3a3697e593293a5cf096f06.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/7b38744233c9bfc5ea8c216b63d65878f3597bffb3a3697e593293a5cf096f06.jpg)

![7d271a9890591abf7604edc3d6277812bdc49e6fb6530011cbdcc0b09dd35380.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/7d271a9890591abf7604edc3d6277812bdc49e6fb6530011cbdcc0b09dd35380.jpg)

![8c2b88e9097fd563e10713962e2c2a4bba956c43d7d612d5e5dadb7feb983cd6.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/8c2b88e9097fd563e10713962e2c2a4bba956c43d7d612d5e5dadb7feb983cd6.jpg)

![9daf44c3c712f9b06fabc77d400e1a0800ee712f34f0ab8188a93309315875e3.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/9daf44c3c712f9b06fabc77d400e1a0800ee712f34f0ab8188a93309315875e3.jpg)

![c4c985c725ffc6667dd7156284b02ed81d7d4cc878fb37379e740efc089b21d2.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/c4c985c725ffc6667dd7156284b02ed81d7d4cc878fb37379e740efc089b21d2.jpg)

![d751f76572bc079c550cace3c0bebd71a91c518bcbbfd307560b1209ade86578.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/d751f76572bc079c550cace3c0bebd71a91c518bcbbfd307560b1209ade86578.jpg)

![dffabd37a6b8f8d18465e9c962c217e8eec39fefe4d09794cb9fa5828a1a9b64.jpg](acl_results/602_RSCF_ Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph/tables/dffabd37a6b8f8d18465e9c962c217e8eec39fefe4d09794cb9fa5828a1a9b64.jpg)

## RolePlot: A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of Role-Playing Agents

### Images

![17a46e218b846cd37cf249e9ecad8c8f1bc13e9141bafd07db5e3d0f594fe19e.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/images/17a46e218b846cd37cf249e9ecad8c8f1bc13e9141bafd07db5e3d0f594fe19e.jpg)

![2672e1b04e13e7d60dd82a03a59334d9da8d8129b6a7f30bd276d852e6819158.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/images/2672e1b04e13e7d60dd82a03a59334d9da8d8129b6a7f30bd276d852e6819158.jpg)

![4271fa10c363da9c31deba5d7c46196d694c82fec55ec1a7ab8375f6b52bb547.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/images/4271fa10c363da9c31deba5d7c46196d694c82fec55ec1a7ab8375f6b52bb547.jpg)

![7b451f959d1921668b547d77dac3a19e262b28136f6c77747d185fdeaeef780e.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/images/7b451f959d1921668b547d77dac3a19e262b28136f6c77747d185fdeaeef780e.jpg)

![7e9bc699c97c6926a8efb6b345372ab13be76a1dbb48515b3dd28ed868ea9ab9.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/images/7e9bc699c97c6926a8efb6b345372ab13be76a1dbb48515b3dd28ed868ea9ab9.jpg)

![a275940b67c95d412fbc068eaa714c469257d017106e98230760e808592287d3.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/images/a275940b67c95d412fbc068eaa714c469257d017106e98230760e808592287d3.jpg)

![b057ee3a8903a9104a703f0850db2e40b1d0181fd6561c8db554181bf6b77626.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/images/b057ee3a8903a9104a703f0850db2e40b1d0181fd6561c8db554181bf6b77626.jpg)

![e7d36a1473bb57e3e56108da8ce4d1af3ff6e367270d4f1166395a7d3a9707e0.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/images/e7d36a1473bb57e3e56108da8ce4d1af3ff6e367270d4f1166395a7d3a9707e0.jpg)

### Tables

![5e5704fb367e00d03e46d96da3aeb66f192d6623cfead5554b595535a3bb5898.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/tables/5e5704fb367e00d03e46d96da3aeb66f192d6623cfead5554b595535a3bb5898.jpg)

![76b018d35e939eb90a88484208c3cd881d746a9c35c7c5e2917020e9e9be50a0.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/tables/76b018d35e939eb90a88484208c3cd881d746a9c35c7c5e2917020e9e9be50a0.jpg)

![7f43b4698cea294a91b5bc3180e29c1b0405920db7816285ee998941caf89488.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/tables/7f43b4698cea294a91b5bc3180e29c1b0405920db7816285ee998941caf89488.jpg)

![97aebb58d1ed3a0417e04cb39926cf0a99671b44d464068383984d12236a5e89.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/tables/97aebb58d1ed3a0417e04cb39926cf0a99671b44d464068383984d12236a5e89.jpg)

![ee6156a3b6d5d5f6768a1bab9df7b516cd7433355f6a85f844cb7acac24c712d.jpg](acl_results/603_RolePlot_ A Systematic Framework for Evaluating and Enhancing the Plot-Progression Capabilities of R/tables/ee6156a3b6d5d5f6768a1bab9df7b516cd7433355f6a85f844cb7acac24c712d.jpg)

## TreeRL:LLMReinforcement Learning with On-Policy Tree Search

### Images

![4c287f29d92e1833f4b14eeeff8d4bcfc826596aa7eff4977e4a9d6d8c21dd91.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/images/4c287f29d92e1833f4b14eeeff8d4bcfc826596aa7eff4977e4a9d6d8c21dd91.jpg)

![79367b5d577734d4c7b84c0496a0c656d79de8ac66171ee2fada02b7f41a261b.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/images/79367b5d577734d4c7b84c0496a0c656d79de8ac66171ee2fada02b7f41a261b.jpg)

![7e45690eaa76c4930c18d0105b7fd3485d88c17c6d8f0c76562716c7572edb98.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/images/7e45690eaa76c4930c18d0105b7fd3485d88c17c6d8f0c76562716c7572edb98.jpg)

![7e676d2e0e58747f11b1bda66e6a5522199a2e5e26fe363a86ee60cfb65e0b1c.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/images/7e676d2e0e58747f11b1bda66e6a5522199a2e5e26fe363a86ee60cfb65e0b1c.jpg)

![828294aa52ab78beaf130127306ca085c49f83f3e81afa8377776a5c4238754e.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/images/828294aa52ab78beaf130127306ca085c49f83f3e81afa8377776a5c4238754e.jpg)

![92b4ca0d45c86fe670b6bf70a39982ece0376d4f5eed79176585799f2e875716.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/images/92b4ca0d45c86fe670b6bf70a39982ece0376d4f5eed79176585799f2e875716.jpg)

![95549d366dca297ee4865c15fc8f2dc561cc001b65474528c2be85948ed91620.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/images/95549d366dca297ee4865c15fc8f2dc561cc001b65474528c2be85948ed91620.jpg)

![b0fb517195fa77569a57fa68ffda85bb7a677803066b9976fe459089de1bfd72.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/images/b0fb517195fa77569a57fa68ffda85bb7a677803066b9976fe459089de1bfd72.jpg)

![b878583855e6267e45b00ecd0424e6c6deae2ea25613e6e3339ff811cf61f597.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/images/b878583855e6267e45b00ecd0424e6c6deae2ea25613e6e3339ff811cf61f597.jpg)

![c70c8ac8c8bf00a8130ec1b6aa00046896eb17bee35bfe49511930afb1264c4a.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/images/c70c8ac8c8bf00a8130ec1b6aa00046896eb17bee35bfe49511930afb1264c4a.jpg)

![de2d25a19e373d48459f219767ae693a467f0f866c56fe7583ac54d65ab2cf1e.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/images/de2d25a19e373d48459f219767ae693a467f0f866c56fe7583ac54d65ab2cf1e.jpg)

### Tables

![55e3918afc01dabfdd14c0bf06943274e5a5a92fcf7efeeb1f8dd752dd8d81f9.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/tables/55e3918afc01dabfdd14c0bf06943274e5a5a92fcf7efeeb1f8dd752dd8d81f9.jpg)

![7f4bd8900af94518bac2e385d8dedf3a08f35d40a4d97e08cfc8cd01ddb5227c.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/tables/7f4bd8900af94518bac2e385d8dedf3a08f35d40a4d97e08cfc8cd01ddb5227c.jpg)

![97759b44b7c7f36859eaf281119121eaa6879a818f93685abc31171aafc6c4ad.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/tables/97759b44b7c7f36859eaf281119121eaa6879a818f93685abc31171aafc6c4ad.jpg)

![99572393fc842b1b564c8b7b4fd95e9e4eab0a29cb791901e857788ed8433a5e.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/tables/99572393fc842b1b564c8b7b4fd95e9e4eab0a29cb791901e857788ed8433a5e.jpg)

![c54b2b793841abdf892fd4132b0aadb80d51191d5d0b1558f98fcc57a4318d53.jpg](acl_results/604_TreeRL_LLMReinforcement Learning with On-Policy Tree Search/tables/c54b2b793841abdf892fd4132b0aadb80d51191d5d0b1558f98fcc57a4318d53.jpg)

## Can a Single Model Master Both Multi-turn Conversations and Tool Use?CoALM: A Unified Conversational Agentic Language Model

### Images

![5725db264adff701b8fd90919f61247a3e308be3f4a1955705f2c52bb1c662b2.jpg](acl_results/605_Can a Single Model Master Both Multi-turn Conversations and Tool Use_CoALM_ A Unified Conversational/images/5725db264adff701b8fd90919f61247a3e308be3f4a1955705f2c52bb1c662b2.jpg)

![69f3c7144687b4db5ef67285d27ffe04e870b708cc4568a6dfa1563d3531e438.jpg](acl_results/605_Can a Single Model Master Both Multi-turn Conversations and Tool Use_CoALM_ A Unified Conversational/images/69f3c7144687b4db5ef67285d27ffe04e870b708cc4568a6dfa1563d3531e438.jpg)

![7117ef73c5382ca646c3fa59778a674e568c94c904d62848e4ca468998c1ba6e.jpg](acl_results/605_Can a Single Model Master Both Multi-turn Conversations and Tool Use_CoALM_ A Unified Conversational/images/7117ef73c5382ca646c3fa59778a674e568c94c904d62848e4ca468998c1ba6e.jpg)

![c7134096627bdce6d9aa44aabd41a5f0601bd479e176ba8729a0652de4ba79ef.jpg](acl_results/605_Can a Single Model Master Both Multi-turn Conversations and Tool Use_CoALM_ A Unified Conversational/images/c7134096627bdce6d9aa44aabd41a5f0601bd479e176ba8729a0652de4ba79ef.jpg)

### Tables

![19c116089fcc233d8af68c12700662cdfdad98e0722eedbae523a01db80ae98a.jpg](acl_results/605_Can a Single Model Master Both Multi-turn Conversations and Tool Use_CoALM_ A Unified Conversational/tables/19c116089fcc233d8af68c12700662cdfdad98e0722eedbae523a01db80ae98a.jpg)

![9c3d9672a28dc95bceef5ce8846deae9bdd2d18e922b2c095e3961d5c9c56d0f.jpg](acl_results/605_Can a Single Model Master Both Multi-turn Conversations and Tool Use_CoALM_ A Unified Conversational/tables/9c3d9672a28dc95bceef5ce8846deae9bdd2d18e922b2c095e3961d5c9c56d0f.jpg)

![b97f795c62c93eea2b71a617f55280c3413f9e669d60c5cb310da51a40bed960.jpg](acl_results/605_Can a Single Model Master Both Multi-turn Conversations and Tool Use_CoALM_ A Unified Conversational/tables/b97f795c62c93eea2b71a617f55280c3413f9e669d60c5cb310da51a40bed960.jpg)

![c9c3554776be95d7ff89cd13d47e509971b5e54ff5636fcc5b7a1195880613ca.jpg](acl_results/605_Can a Single Model Master Both Multi-turn Conversations and Tool Use_CoALM_ A Unified Conversational/tables/c9c3554776be95d7ff89cd13d47e509971b5e54ff5636fcc5b7a1195880613ca.jpg)

![de6b7cdaa562bd2c06fba6f05b8de39b45d987aca0e6c76c8f11ad69a186817f.jpg](acl_results/605_Can a Single Model Master Both Multi-turn Conversations and Tool Use_CoALM_ A Unified Conversational/tables/de6b7cdaa562bd2c06fba6f05b8de39b45d987aca0e6c76c8f11ad69a186817f.jpg)

## Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation

### Images

![084d70a71a475a95ec1bfed0289e02bf856549d81767173b1a04fb63ef8b2df2.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/images/084d70a71a475a95ec1bfed0289e02bf856549d81767173b1a04fb63ef8b2df2.jpg)

![0eaf44b74c22c881b2ca7e51a98d79bad7b37aa09e44fe14042b37c5bdcf65e8.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/images/0eaf44b74c22c881b2ca7e51a98d79bad7b37aa09e44fe14042b37c5bdcf65e8.jpg)

![1260eaa3c0274294aacb633c02034e2e23ba887e2b34254760c85e77e278a5dd.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/images/1260eaa3c0274294aacb633c02034e2e23ba887e2b34254760c85e77e278a5dd.jpg)

![546c07213a1e4b4cea76b7d9c6750a7b7797a62f7c7e8228b0e53e479c49f9c7.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/images/546c07213a1e4b4cea76b7d9c6750a7b7797a62f7c7e8228b0e53e479c49f9c7.jpg)

![5ab2c42254f1fdf87c03919ef92ae6fe304c05eba3c950794639fe7e95c261ee.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/images/5ab2c42254f1fdf87c03919ef92ae6fe304c05eba3c950794639fe7e95c261ee.jpg)

![5f3a143ec9ef9ca2a288810bb2d194f3b5ffbd1f51c053ef962afadfb2cfa0a2.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/images/5f3a143ec9ef9ca2a288810bb2d194f3b5ffbd1f51c053ef962afadfb2cfa0a2.jpg)

![952d9e8921e873a734dfe19cc77cacff9ed7b3f3fab936bd9a837dd2901a3531.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/images/952d9e8921e873a734dfe19cc77cacff9ed7b3f3fab936bd9a837dd2901a3531.jpg)

![c376eebf4d2d7579f2a93c6b95944b9c3436bd7f30b60692c7b88d25eac706f4.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/images/c376eebf4d2d7579f2a93c6b95944b9c3436bd7f30b60692c7b88d25eac706f4.jpg)

![cef06c9399d3e9f16c96e7e63aa0eafd1f7af01a6b6b9cba90e57a1057835f7c.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/images/cef06c9399d3e9f16c96e7e63aa0eafd1f7af01a6b6b9cba90e57a1057835f7c.jpg)

![dd0b0d5042b8c30250ad86f5f31d9b69c4b2c26658430fdbcaa227aa35ebc552.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/images/dd0b0d5042b8c30250ad86f5f31d9b69c4b2c26658430fdbcaa227aa35ebc552.jpg)

![f7e0828a12bf2fe451020f17772ba4a4eb42ee298de94e83046ae98f7618f362.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/images/f7e0828a12bf2fe451020f17772ba4a4eb42ee298de94e83046ae98f7618f362.jpg)

### Tables

![1da79367d76be1111b2c785ab6a045c3508c6ea88ac2acf1f99333f4dfd0d5dc.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/1da79367d76be1111b2c785ab6a045c3508c6ea88ac2acf1f99333f4dfd0d5dc.jpg)

![27eb9de830c3e070672ba3eba37afa515bfcd522df2d9223a42d3e523c150b57.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/27eb9de830c3e070672ba3eba37afa515bfcd522df2d9223a42d3e523c150b57.jpg)

![2a9b6cb20f2fea4d0983cd233c1eaa57ed685137dff7605a17cfb1fa2c67a3d1.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/2a9b6cb20f2fea4d0983cd233c1eaa57ed685137dff7605a17cfb1fa2c67a3d1.jpg)

![4b1d6eb406625383361b1db4b14aef35078b7e434d5f8aafb9c7c2b6c655eccd.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/4b1d6eb406625383361b1db4b14aef35078b7e434d5f8aafb9c7c2b6c655eccd.jpg)

![4d65cab00ccc8c7de44fc9b0e290be8c412a7cd253d074d64f335b57f1539dc9.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/4d65cab00ccc8c7de44fc9b0e290be8c412a7cd253d074d64f335b57f1539dc9.jpg)

![4f2c31a19e46f8d95ff07000ab975f88c75d67dd0582b980920a8b0ac2c25b85.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/4f2c31a19e46f8d95ff07000ab975f88c75d67dd0582b980920a8b0ac2c25b85.jpg)

![5e3e76a9315c5c59c4f1844a6ec8f492dab1dd102c66a73aef8972da47da72a1.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/5e3e76a9315c5c59c4f1844a6ec8f492dab1dd102c66a73aef8972da47da72a1.jpg)

![734212d715337acaf8efd953fbaf6c39720765b04568026500e844c8da21ba12.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/734212d715337acaf8efd953fbaf6c39720765b04568026500e844c8da21ba12.jpg)

![792cb392c71b9ae8aaaae949f6f057e05cc6f12f42eb65333f367be8badc1d0f.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/792cb392c71b9ae8aaaae949f6f057e05cc6f12f42eb65333f367be8badc1d0f.jpg)

![86ad490d9404d3e333f1bdf8601b900766f0490ebe3ded9dfcf0880a6a35fd80.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/86ad490d9404d3e333f1bdf8601b900766f0490ebe3ded9dfcf0880a6a35fd80.jpg)

![a01f18b68c15d9e53d5fc53c4ef5a967ca74552576c82f457a53c564a00bd765.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/a01f18b68c15d9e53d5fc53c4ef5a967ca74552576c82f457a53c564a00bd765.jpg)

![c30b27a6612bcd4538d0e7f60301ebc1fa687dcde3ab94e86233e0e8959984d5.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/c30b27a6612bcd4538d0e7f60301ebc1fa687dcde3ab94e86233e0e8959984d5.jpg)

![e6ea40d61e00a7c2160039bf47c03ece641f0d7c598dd9a44ee0869bcb86ab5d.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/e6ea40d61e00a7c2160039bf47c03ece641f0d7c598dd9a44ee0869bcb86ab5d.jpg)

![ffdc9a078f17047d9806f5dbffab6d84e8b261335c9d9246a16b7a1cdad88389.jpg](acl_results/606_Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Tra/tables/ffdc9a078f17047d9806f5dbffab6d84e8b261335c9d9246a16b7a1cdad88389.jpg)

## SDPO: Segment-Level Direct Preference Optimization for Social Agents

### Images

![121962015b79467f98fb8fb0a2c54441a023cfeea4fbce5c6799ea3afa67de8f.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/images/121962015b79467f98fb8fb0a2c54441a023cfeea4fbce5c6799ea3afa67de8f.jpg)

![32fd9ed7c5c334559ac38c174d9304bf8c74c6579f31e245a37b8f38324f1f90.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/images/32fd9ed7c5c334559ac38c174d9304bf8c74c6579f31e245a37b8f38324f1f90.jpg)

![5a1dddd004f7bcb7afb67f1f67bbc77deeea23705a4c882d062d530364da88a4.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/images/5a1dddd004f7bcb7afb67f1f67bbc77deeea23705a4c882d062d530364da88a4.jpg)

![ee1d0f5ddd2dbea67c3969628e6f741ca1069e2d81f09f7b7ad05c66ff48161d.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/images/ee1d0f5ddd2dbea67c3969628e6f741ca1069e2d81f09f7b7ad05c66ff48161d.jpg)

![fefad7b6a0ed3d67a66eb0824d625edbbbc69c497fb1e3c59512179ee80c1cbe.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/images/fefad7b6a0ed3d67a66eb0824d625edbbbc69c497fb1e3c59512179ee80c1cbe.jpg)

### Tables

![065f8e1280b4980726e5be89d65864ec926522892553b0e93dccc34f4a16bf52.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/tables/065f8e1280b4980726e5be89d65864ec926522892553b0e93dccc34f4a16bf52.jpg)

![10f88e158585176085a2bb67ffc6f376e3d3bfd2c69a405b58e52f104dfc7d5b.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/tables/10f88e158585176085a2bb67ffc6f376e3d3bfd2c69a405b58e52f104dfc7d5b.jpg)

![11feba78b18b5b0c4b6c539326585f47e6ef09e45b8a274526573942cdb7e933.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/tables/11feba78b18b5b0c4b6c539326585f47e6ef09e45b8a274526573942cdb7e933.jpg)

![6e907810f5c7a87873e81ae2b684c4462071abc856c515882e179325e2899b2d.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/tables/6e907810f5c7a87873e81ae2b684c4462071abc856c515882e179325e2899b2d.jpg)

![78bb6b3eead5ec4c92e8f469bfbe14f355ca59e9f5f2867de2a927656d32585f.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/tables/78bb6b3eead5ec4c92e8f469bfbe14f355ca59e9f5f2867de2a927656d32585f.jpg)

![96dafc9274164eb1a7600ecf4c2ec84be466aa6bc1d04fb8a5d7fd7c6b6b20f4.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/tables/96dafc9274164eb1a7600ecf4c2ec84be466aa6bc1d04fb8a5d7fd7c6b6b20f4.jpg)

![9d36602340e9fef24d044f84927cdbf595bf38f5929371729e74b0a53f012b1e.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/tables/9d36602340e9fef24d044f84927cdbf595bf38f5929371729e74b0a53f012b1e.jpg)

![c6681aafabc2d6b45851931eee5d51a26c9c0093397c6823376d76f1b9fea59f.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/tables/c6681aafabc2d6b45851931eee5d51a26c9c0093397c6823376d76f1b9fea59f.jpg)

![d0d74a71508d9dc1ef8e5d0331b9c227a272f7bc0fff3b8aa49737ac7137f07a.jpg](acl_results/607_SDPO_ Segment-Level Direct Preference Optimization for Social Agents/tables/d0d74a71508d9dc1ef8e5d0331b9c227a272f7bc0fff3b8aa49737ac7137f07a.jpg)

## KokoroChat: AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors

### Images

![137bc1c1ee767f4ce6c4617342574964728d5dbc7ff36f10360acbd1b1318699.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/137bc1c1ee767f4ce6c4617342574964728d5dbc7ff36f10360acbd1b1318699.jpg)

![1d78de0d359b579363b3b0958e7d4b17ca397f2a13a68740be89b88325f9209f.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/1d78de0d359b579363b3b0958e7d4b17ca397f2a13a68740be89b88325f9209f.jpg)

![21d57a068e11c518952b6cd3b723caacb3dac61ae0bfacff55644f27fcdfb455.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/21d57a068e11c518952b6cd3b723caacb3dac61ae0bfacff55644f27fcdfb455.jpg)

![37332372e8b0059153f28efdc24bd7ddb7bcbbb35cfddce39366558fec0fe318.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/37332372e8b0059153f28efdc24bd7ddb7bcbbb35cfddce39366558fec0fe318.jpg)

![6d241c5fc663aa9d2ccebe192eea9747f4dca5b065cc7f624eef8f421b087a7a.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/6d241c5fc663aa9d2ccebe192eea9747f4dca5b065cc7f624eef8f421b087a7a.jpg)

![7b020c41b766ca251ed8752cfad2e5afbd35e43eadab71de1f0e96b0d55a0665.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/7b020c41b766ca251ed8752cfad2e5afbd35e43eadab71de1f0e96b0d55a0665.jpg)

![8dfecd71e1c145a16bc9d68350be1075c7f3f0e45911ed16a4331193269cd8e2.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/8dfecd71e1c145a16bc9d68350be1075c7f3f0e45911ed16a4331193269cd8e2.jpg)

![b36d0f9231407011feff9c157e13b1ad025c6b356f9e9e43a8ce74914f34d2ae.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/b36d0f9231407011feff9c157e13b1ad025c6b356f9e9e43a8ce74914f34d2ae.jpg)

![bbe0e69006b033fc748e56a5aa795891e1bcc39aeb7b27f9a11f26dad657fca5.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/bbe0e69006b033fc748e56a5aa795891e1bcc39aeb7b27f9a11f26dad657fca5.jpg)

![df3be1f3ef598dba43524b4dbd66db04e81049f2f66c87b95124b1dd0566dd22.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/df3be1f3ef598dba43524b4dbd66db04e81049f2f66c87b95124b1dd0566dd22.jpg)

![e2dd7d9df0f5d3991669e362c2731482ad817d5ea246d56910b5dc73d32fe4f3.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/e2dd7d9df0f5d3991669e362c2731482ad817d5ea246d56910b5dc73d32fe4f3.jpg)

![e6904c05583d90d498195da7f9d4e02c405e9654d3b186fc7d8899b008a74cd0.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/e6904c05583d90d498195da7f9d4e02c405e9654d3b186fc7d8899b008a74cd0.jpg)

![ea5c49b6a5afbbe8feb91cca2c73582ed026b84c182890ece8b8b69b4812b1a2.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/ea5c49b6a5afbbe8feb91cca2c73582ed026b84c182890ece8b8b69b4812b1a2.jpg)

![eb264ff222e51d0a2fcb9a8b9c6009493ae0ffe8d93252f7f19283ae9cba8ed5.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/eb264ff222e51d0a2fcb9a8b9c6009493ae0ffe8d93252f7f19283ae9cba8ed5.jpg)

![f10468d9099b2703b8d63939a572b8222978ab4d439605f693dfdd15b12764be.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/f10468d9099b2703b8d63939a572b8222978ab4d439605f693dfdd15b12764be.jpg)

![fa7d5fe6b421ced4a787ee526f8f8695992ff43f5e6d88334dd8eee0b3890dd0.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/fa7d5fe6b421ced4a787ee526f8f8695992ff43f5e6d88334dd8eee0b3890dd0.jpg)

![fad4830d159abc43c9ba25906faeb7f305b3704cf7ac2560de0afe1b37294a7e.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/fad4830d159abc43c9ba25906faeb7f305b3704cf7ac2560de0afe1b37294a7e.jpg)

![fffa8a98bb8e83288e5232534c96fb60d2ed76217384ce6276e8c80557a88013.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/images/fffa8a98bb8e83288e5232534c96fb60d2ed76217384ce6276e8c80557a88013.jpg)

### Tables

![05f6e274bd5925cf2525d9885cd51cc4c88b0fad32c4b397bc549f525af6ca8f.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/tables/05f6e274bd5925cf2525d9885cd51cc4c88b0fad32c4b397bc549f525af6ca8f.jpg)

![15fc947c25d3555cc262f117ebb60e1c6397dfdcd66201df19816599646bc0b8.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/tables/15fc947c25d3555cc262f117ebb60e1c6397dfdcd66201df19816599646bc0b8.jpg)

![5a6a9ddae8de7c34d4ccae266febfb06d92b41057c40ec17a4d004f46956a554.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/tables/5a6a9ddae8de7c34d4ccae266febfb06d92b41057c40ec17a4d004f46956a554.jpg)

![983e4b580e31e961911b32459cc915fd03a0fda4a606129e98f9b2fa344850b9.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/tables/983e4b580e31e961911b32459cc915fd03a0fda4a606129e98f9b2fa344850b9.jpg)

![a27f894d4a5b219a938be6ea017301df7c8a7a187a49f34a96c1172bb69685dd.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/tables/a27f894d4a5b219a938be6ea017301df7c8a7a187a49f34a96c1172bb69685dd.jpg)

![a77f7a1e297e3ed508dfaf24b5578a978aafb191fbf735f6dbdc4188cfcce00c.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/tables/a77f7a1e297e3ed508dfaf24b5578a978aafb191fbf735f6dbdc4188cfcce00c.jpg)

![d29617ca0c5717b40711edd29140baf5fbf342cb0497a802216f6350c9175b11.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/tables/d29617ca0c5717b40711edd29140baf5fbf342cb0497a802216f6350c9175b11.jpg)

![d9d085193eb5eb4889e96bac82602346488f6a18e062fd38923d09754d128bb0.jpg](acl_results/608_KokoroChat_ AJapanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Traine/tables/d9d085193eb5eb4889e96bac82602346488f6a18e062fd38923d09754d128bb0.jpg)