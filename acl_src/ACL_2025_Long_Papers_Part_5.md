# ACL 2025 Long Papers


**Summary:** 1599 papers with extracted content:
- 📊 Total images: 13877
- 📋 Total tables: 16805
- 📄 Total files: 3068
---

# ACL 2025 Long Papers - Part 5 of 50

## 目录 (Table of Contents)

1. [D.Va: Validate Your Demonstration First Before You Use It](#DVa-Validate-Your-Demonstration-First-Before-You-Use-It)
2. [Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?](#Are-Any-to-Any-Models-More-Consistent-Across-Modality-Transfers-Than-Specialists)
3. [MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation](#MAIN-RAG-Multi-Agent-Filtering-Retrieval-Augmented-Generation)
4. [Unraveling the Mechanics of Learning-Based Demonstration Selection for In-Context Learning](#Unraveling-the-Mechanics-of-Learning-Based-Demonstration-Selection-for-In-Context-Learning)
5. [Direct Prompt Optimization with Continuous Representations](#Direct-Prompt-Optimization-with-Continuous-Representations)
6. [uMedSum: A Unified Framework for Clinical Abstractive Summarization](#uMedSum-A-Unified-Framework-for-Clinical-Abstractive-Summarization)
7. [GigaSpeech 2: An Evolving, Large-Scale and Multi-domainASRCorpus for Low-Resource Languages with Automated Crawling, Transcription and Refinement](#GigaSpeech-2-An-Evolving-Large-Scale-and-Multi-domainASRCorpus-for-Low-Resource-Languages-with-Automated-Crawling-Transcription-and-Refinement)
8. [Context-Aware Sentiment Forecasting viaLLM-based Multi-Perspective Role-Playing Agents](#Context-Aware-Sentiment-Forecasting-viaLLM-based-Multi-Perspective-Role-Playing-Agents)
9. [TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](#TARGA-Targeted-Synthetic-Data-Generation-for-Practical-Reasoning-over-Structured-Data)
10. [AndroidGen: Building an Android Language Agent under Data Scarcity](#AndroidGen-Building-an-Android-Language-Agent-under-Data-Scarcity)
11. [Prompt Candidates, then Distill: A Teacher-Student Framework forLLM-driven Data Annotation](#Prompt-Candidates-then-Distill-A-Teacher-Student-Framework-forLLM-driven-Data-Annotation)
12. [A Survey of Post-Training Scaling in Large Language Models](#A-Survey-of-Post-Training-Scaling-in-Large-Language-Models)
13. [Position-aware Automatic Circuit Discovery](#Position-aware-Automatic-Circuit-Discovery)
14. [HyperFM: Fact-Centric Multimodal Fusion for Link Prediction over Hyper-Relational Knowledge Graphs](#HyperFM-Fact-Centric-Multimodal-Fusion-for-Link-Prediction-over-Hyper-Relational-Knowledge-Graphs)
15. [Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model](#Centurio-On-Drivers-of-Multilingual-Ability-of-Large-Vision-Language-Model)
16. [Less for More: Enhanced Feedback-aligned MixedLLMs for Molecule Caption Generation and Fine-GrainedNLIEvaluation](#Less-for-More-Enhanced-Feedback-aligned-MixedLLMs-for-Molecule-Caption-Generation-and-Fine-GrainedNLIEvaluation)
17. [Ensemble Watermarks for Large Language Models](#Ensemble-Watermarks-for-Large-Language-Models)
18. [ConInstruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities](#ConInstruction-Universal-Jailbreaking-of-Multimodal-Large-Language-Models-via-Non-Textual-Modalities)
19. [TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning forLLM-as-a-Judge](#TRACT-Regression-Aware-Fine-tuning-Meets-Chain-of-Thought-Reasoning-forLLM-as-a-Judge)
20. [DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization for Dynamic Retrieval-Augmented Generation](#DioR-Adaptive-Cognitive-Detection-and-Contextual-Retrieval-Optimization-for-Dynamic-Retrieval-Augmented-Generation)
21. [Unveiling the Power of Source: Source-based MinimumBayes Risk Decoding for Neural Machine Translation](#Unveiling-the-Power-of-Source-Source-based-MinimumBayes-Risk-Decoding-for-Neural-Machine-Translation)
22. [ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](#ToolHop-A-Query-Driven-Benchmark-for-Evaluating-Large-Language-Models-in-Multi-Hop-Tool-Use)
23. [Mixture of insighTful Experts (MoTE): The Synergy of Reasoning Chains and Expert Mixtures in Self-Alignment](#Mixture-of-insighTful-Experts-MoTE-The-Synergy-of-Reasoning-Chains-and-Expert-Mixtures-in-Self-Alignment)
24. [MAPS: Motivation-Aware Personalized Search viaLLM-Driven Consultation Alignment](#MAPS-Motivation-Aware-Personalized-Search-viaLLM-Driven-Consultation-Alignment)
25. [Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework](#Aristotle-Mastering-Logical-Reasoning-with-A-Logic-Complete-Decompose-Search-Resolve-Framework)
26. [LADM: Long-context Training Data Selection with Attention-based Dependency Measurement forLLMs](#LADM-Long-context-Training-Data-Selection-with-Attention-based-Dependency-Measurement-forLLMs)
27. [Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training](#Iron-Sharpens-Iron-Defending-Against-Attacks-in-Machine-Generated-Text-Detection-with-Adversarial-Training)
28. [Cultural Learning-Based Culture Adaptation of Language Models](#Cultural-Learning-Based-Culture-Adaptation-of-Language-Models)
29. [A-TASC:AsianTED-Based Automatic Subtitling Corpus](#A-TASCAsianTED-Based-Automatic-Subtitling-Corpus)
30. [Refuse Whenever You Feel Unsafe: Improving Safety inLLMs via Decoupled Refusal Training](#Refuse-Whenever-You-Feel-Unsafe-Improving-Safety-inLLMs-via-Decoupled-Refusal-Training)
31. [Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings fromLLMs](#Token-Prepending-A-Training-Free-Approach-for-Eliciting-Better-Sentence-Embeddings-fromLLMs)
32. [No Questions are Stupid, but some are Poorly Posed: Understanding Poorly-Posed Information-Seeking Questions](#No-Questions-are-Stupid-but-some-are-Poorly-Posed-Understanding-Poorly-Posed-Information-Seeking-Questions)

---


## D.Va: Validate Your Demonstration First Before You Use It

### Images

![340698c31bcf5d7ea7501300a6b2447bfcecd0b6484a2dd0dc70de4d4c67a0ce.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/images/340698c31bcf5d7ea7501300a6b2447bfcecd0b6484a2dd0dc70de4d4c67a0ce.jpg)

![40f4eff108d981d5f67163805599b8cdc91630e7153024f967f449276db7415b.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/images/40f4eff108d981d5f67163805599b8cdc91630e7153024f967f449276db7415b.jpg)

![590d1cbcaad53d3b3b3e06316d5680fa3caffe752eb846e1ece51339cde0464e.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/images/590d1cbcaad53d3b3b3e06316d5680fa3caffe752eb846e1ece51339cde0464e.jpg)

![9a87e243c61ad15ba7f84a56a3acfb54f1fdda81a135ea8ff0e0e9f1e6734164.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/images/9a87e243c61ad15ba7f84a56a3acfb54f1fdda81a135ea8ff0e0e9f1e6734164.jpg)

![c78c9993a50bfdd1e39c43cb76acc8e4384324369ff5ffe309cf590e733c5740.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/images/c78c9993a50bfdd1e39c43cb76acc8e4384324369ff5ffe309cf590e733c5740.jpg)

![d351029b4a14f4ebf6cd27e1f65b84b3aafd002c6d48732f45856554abc610c0.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/images/d351029b4a14f4ebf6cd27e1f65b84b3aafd002c6d48732f45856554abc610c0.jpg)

### Tables

![32e18979c8000624159c90219409f2b71f4dd1c6fb8ee05d70d253dc19436621.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/32e18979c8000624159c90219409f2b71f4dd1c6fb8ee05d70d253dc19436621.jpg)

![56e346b0d92d8fbf0e3f7268ccc55eae4aef1f51ab5b3ffdc732cbb2133a2204.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/56e346b0d92d8fbf0e3f7268ccc55eae4aef1f51ab5b3ffdc732cbb2133a2204.jpg)

![5dfe856c87a72ffdda7b9325ac286ccfbeea53931f9d77d577652a9eac7fcea0.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/5dfe856c87a72ffdda7b9325ac286ccfbeea53931f9d77d577652a9eac7fcea0.jpg)

![7d38fe10f1ffefa728f84c4adc1199fc35b5d8285dce7557901f3d9c3c39e860.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/7d38fe10f1ffefa728f84c4adc1199fc35b5d8285dce7557901f3d9c3c39e860.jpg)

![88d2a5fdc90a69c05a66082bc70be7f65d0a5b55db8d27225c380921ea21f734.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/88d2a5fdc90a69c05a66082bc70be7f65d0a5b55db8d27225c380921ea21f734.jpg)

![9c8cf981f73975cd0272b2b51603c0451aff10905a899ddac2dac4c5eb04018d.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/9c8cf981f73975cd0272b2b51603c0451aff10905a899ddac2dac4c5eb04018d.jpg)

![a71b7b23bd12cb0f3ad2e02573a033a591f8bdb1b8e88b4fa27c2c5770d4d1ec.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/a71b7b23bd12cb0f3ad2e02573a033a591f8bdb1b8e88b4fa27c2c5770d4d1ec.jpg)

![ab94ac5e1173e78229c47fbf21ff7c9644f0f5d7cc019030d1bd4626cae6bf0b.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/ab94ac5e1173e78229c47fbf21ff7c9644f0f5d7cc019030d1bd4626cae6bf0b.jpg)

![abd4dbbb9d3e00ea0f9f686d80da34d40b07fa8d94f7427f8b1ab4d56bdde0e5.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/abd4dbbb9d3e00ea0f9f686d80da34d40b07fa8d94f7427f8b1ab4d56bdde0e5.jpg)

![af2a73e51c26291675ec04681119e0dd513cd3fb7b57984f3306189261910017.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/af2a73e51c26291675ec04681119e0dd513cd3fb7b57984f3306189261910017.jpg)

![ba6d53519c6c3720147b1c5c371fb7a9d54810675b5fc8817db1c298ee38d4cb.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/ba6d53519c6c3720147b1c5c371fb7a9d54810675b5fc8817db1c298ee38d4cb.jpg)

![c3fc798592c372ba0192f15b8b9d07aee0bb5cd120f37f556d2f84b87c1c65ca.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/c3fc798592c372ba0192f15b8b9d07aee0bb5cd120f37f556d2f84b87c1c65ca.jpg)

![fd4a634be10a6cc74c45f8768367188a9cec7d46efeae9d8e06e9dc578f422e4.jpg](../acl_results/129_D.Va_%20Validate%20Your%20Demonstration%20First%20Before%20You%20Use%20It/tables/fd4a634be10a6cc74c45f8768367188a9cec7d46efeae9d8e06e9dc578f422e4.jpg)

## Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?

### Images

![20f810605c7b36226ffc89e1c62c638114e3fe34573c699278a771cf8d6bbf16.jpg](../acl_results/130_Are%20Any-to-Any%20Models%20More%20Consistent%20Across%20Modality%20Transfers%20Than%20Specialists_/images/20f810605c7b36226ffc89e1c62c638114e3fe34573c699278a771cf8d6bbf16.jpg)

![4eb7d4024356c632692cf5ef149f505acb59294ab66402112c7f4e76187fc751.jpg](../acl_results/130_Are%20Any-to-Any%20Models%20More%20Consistent%20Across%20Modality%20Transfers%20Than%20Specialists_/images/4eb7d4024356c632692cf5ef149f505acb59294ab66402112c7f4e76187fc751.jpg)

![beb1d8187646a64da4b6d38b8559fd05c194754501a2e1e49d7634f6ba172203.jpg](../acl_results/130_Are%20Any-to-Any%20Models%20More%20Consistent%20Across%20Modality%20Transfers%20Than%20Specialists_/images/beb1d8187646a64da4b6d38b8559fd05c194754501a2e1e49d7634f6ba172203.jpg)

![dee3fc6650d7e31cde13e130947f0f1cf5e5c577df5b40684596e456a259baa9.jpg](../acl_results/130_Are%20Any-to-Any%20Models%20More%20Consistent%20Across%20Modality%20Transfers%20Than%20Specialists_/images/dee3fc6650d7e31cde13e130947f0f1cf5e5c577df5b40684596e456a259baa9.jpg)

![e3916cbe38158e55b72675f0902c482d776cdc1a18c374f2cd10f23345b7ed3f.jpg](../acl_results/130_Are%20Any-to-Any%20Models%20More%20Consistent%20Across%20Modality%20Transfers%20Than%20Specialists_/images/e3916cbe38158e55b72675f0902c482d776cdc1a18c374f2cd10f23345b7ed3f.jpg)

![f31ddd57807a7191745ac6426be31e9f39cace25473b6290b354c8bf78ab1821.jpg](../acl_results/130_Are%20Any-to-Any%20Models%20More%20Consistent%20Across%20Modality%20Transfers%20Than%20Specialists_/images/f31ddd57807a7191745ac6426be31e9f39cace25473b6290b354c8bf78ab1821.jpg)

### Tables

![0ceecf77b595b7bf0d81d4379997b96c65b17816a9fc80fb71c322c50931b386.jpg](../acl_results/130_Are%20Any-to-Any%20Models%20More%20Consistent%20Across%20Modality%20Transfers%20Than%20Specialists_/tables/0ceecf77b595b7bf0d81d4379997b96c65b17816a9fc80fb71c322c50931b386.jpg)

## MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation

### Images

![0e24a9bc18446747120bde407e5bbb936121c14bdb7c0a38b14e59140d80b087.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/images/0e24a9bc18446747120bde407e5bbb936121c14bdb7c0a38b14e59140d80b087.jpg)

![1cdd9875d6b08af457dd08a57a7c457d6c3bd033a598fc31109d4248767ac706.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/images/1cdd9875d6b08af457dd08a57a7c457d6c3bd033a598fc31109d4248767ac706.jpg)

![40b34ee38f8f4fcbec3dbc488ee71836a214607d02aa12cc0fc63a8f919c3b48.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/images/40b34ee38f8f4fcbec3dbc488ee71836a214607d02aa12cc0fc63a8f919c3b48.jpg)

![7250fc25e182f79aef4ae88da09ca9625972e1fcae470ab6293cac227de2609c.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/images/7250fc25e182f79aef4ae88da09ca9625972e1fcae470ab6293cac227de2609c.jpg)

![98ae6c1e25cbf7b5c615b68bf6246b89d731e8d93e29d47d60a326b0397dab91.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/images/98ae6c1e25cbf7b5c615b68bf6246b89d731e8d93e29d47d60a326b0397dab91.jpg)

![9db8b2783b5d0a67aa6d08ae3d6a6647be86252faf38c587a376b72bee611525.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/images/9db8b2783b5d0a67aa6d08ae3d6a6647be86252faf38c587a376b72bee611525.jpg)

![c37968db31ba2c5d2bc55b7d4e8c7310f2a8ad8603cf0b00951ae2458ca44c80.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/images/c37968db31ba2c5d2bc55b7d4e8c7310f2a8ad8603cf0b00951ae2458ca44c80.jpg)

![c55f0556127be03ce84fd666aefd19ce4239ef5d58051f69958e67210f45b427.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/images/c55f0556127be03ce84fd666aefd19ce4239ef5d58051f69958e67210f45b427.jpg)

![db2139b02d028c8d2b04aa21c17f4382ca589f1464327a6fb980db3822328193.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/images/db2139b02d028c8d2b04aa21c17f4382ca589f1464327a6fb980db3822328193.jpg)

![ddfcaec01e5d669518c4e80f35e0348582bd32ccaee9980d4374f66abaa0a15e.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/images/ddfcaec01e5d669518c4e80f35e0348582bd32ccaee9980d4374f66abaa0a15e.jpg)

![f8d19823a9cda0dcc023f201cc27d045fb5a631e3bfe8e4da252c5b2587c4e8f.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/images/f8d19823a9cda0dcc023f201cc27d045fb5a631e3bfe8e4da252c5b2587c4e8f.jpg)

### Tables

![0ddb47994e49ba3f7a758b53b8bc91c79172a0534bb719fcd2471a1a9e4171b9.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/tables/0ddb47994e49ba3f7a758b53b8bc91c79172a0534bb719fcd2471a1a9e4171b9.jpg)

![80ee41a94978bb6ef4bd0d49ce87b4b85f2870756a2b2f1742a150a44e1295c8.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/tables/80ee41a94978bb6ef4bd0d49ce87b4b85f2870756a2b2f1742a150a44e1295c8.jpg)

![877997e83ec11c26fead139eaff1d381be3417507ca54720b14dbe3ef76af850.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/tables/877997e83ec11c26fead139eaff1d381be3417507ca54720b14dbe3ef76af850.jpg)

![9b4d82ecfc9dfbcaf3e5883fd18dfa77d8851ad045f5386573a27a387b395948.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/tables/9b4d82ecfc9dfbcaf3e5883fd18dfa77d8851ad045f5386573a27a387b395948.jpg)

![fa874964f287583e47554ba86d32bd70ca47b3c03634e1ba7ebda8ac964eeb19.jpg](../acl_results/131_MAIN-RAG_%20Multi-Agent%20Filtering%20Retrieval-Augmented%20Generation/tables/fa874964f287583e47554ba86d32bd70ca47b3c03634e1ba7ebda8ac964eeb19.jpg)

## Unraveling the Mechanics of Learning-Based Demonstration Selection for In-Context Learning

### Images

![18e0418c951363911f7277bd282d4d0b2e1f011d1cafec742213906217a2c58b.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/images/18e0418c951363911f7277bd282d4d0b2e1f011d1cafec742213906217a2c58b.jpg)

![25368113e0cba4e2025121f98f9ecb0260ea992254eaf5156f82cd68ddd1aafc.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/images/25368113e0cba4e2025121f98f9ecb0260ea992254eaf5156f82cd68ddd1aafc.jpg)

![3c50abef80be284df00d82b0c2560e04d059833dcca86830df34ccce5d837ffb.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/images/3c50abef80be284df00d82b0c2560e04d059833dcca86830df34ccce5d837ffb.jpg)

![547b6d47f2fec13dadcfa01d94b17cd394dd535c70aac817813f04dd7215a21f.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/images/547b6d47f2fec13dadcfa01d94b17cd394dd535c70aac817813f04dd7215a21f.jpg)

![b10aa113f908e5f6c8733b9ee729ffb1243a16fa446373439ab902cc6a94ccae.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/images/b10aa113f908e5f6c8733b9ee729ffb1243a16fa446373439ab902cc6a94ccae.jpg)

![d9436d6e530ee0677307ba2be0a58ba003df40c8eb0bc4085473330af5ce732a.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/images/d9436d6e530ee0677307ba2be0a58ba003df40c8eb0bc4085473330af5ce732a.jpg)

![ee7e13afe8b64fbe326162cb8448234d0bbda0280ceed2b61657a528d541125c.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/images/ee7e13afe8b64fbe326162cb8448234d0bbda0280ceed2b61657a528d541125c.jpg)

### Tables

![05d3d28645e62d05c02804608137eb958d93996271155a9c302a02ca8d0a4786.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/05d3d28645e62d05c02804608137eb958d93996271155a9c302a02ca8d0a4786.jpg)

![0cfd2915e7f26469ec9e22b0a8fff038c274b76d80542355bd261fc284ec9613.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/0cfd2915e7f26469ec9e22b0a8fff038c274b76d80542355bd261fc284ec9613.jpg)

![280b29ecbd228513f885397297ee3d7b80e0262368cefa8c7d925e0b6f4f7191.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/280b29ecbd228513f885397297ee3d7b80e0262368cefa8c7d925e0b6f4f7191.jpg)

![2ac4d9224176c4c500b828d3edc5a2eceeeeb86d4f51f2b94f6560a5b6290f94.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/2ac4d9224176c4c500b828d3edc5a2eceeeeb86d4f51f2b94f6560a5b6290f94.jpg)

![47e6020e21b817fba1ca84304425a2e943157d0d236eadb8889678c986a959d0.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/47e6020e21b817fba1ca84304425a2e943157d0d236eadb8889678c986a959d0.jpg)

![65d145fc265f9df86b777bc93435abe60f12eaa208977232c316515ea010680f.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/65d145fc265f9df86b777bc93435abe60f12eaa208977232c316515ea010680f.jpg)

![8772a7ebb4d3914a960eb7c3143f8c116ca7f70e844b5cf41d93576ab0486cfc.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/8772a7ebb4d3914a960eb7c3143f8c116ca7f70e844b5cf41d93576ab0486cfc.jpg)

![a3632a2ff452f761ed95ea7c75d49679dd521e247edb9bfbdca1da9d02698995.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/a3632a2ff452f761ed95ea7c75d49679dd521e247edb9bfbdca1da9d02698995.jpg)

![b7e1938b5764edd513316d3ad20cf5ca73a07d18022087322d7b80dbc10a09e0.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/b7e1938b5764edd513316d3ad20cf5ca73a07d18022087322d7b80dbc10a09e0.jpg)

![c511e9f2448723a0d9af379db3d2c40b01be9cd597221d9499e1929c9607ec21.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/c511e9f2448723a0d9af379db3d2c40b01be9cd597221d9499e1929c9607ec21.jpg)

![e2a523686db27096e90d9a5d4a28ad1a05f4c576ae571f9731386ea204edef2e.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/e2a523686db27096e90d9a5d4a28ad1a05f4c576ae571f9731386ea204edef2e.jpg)

![e6c23daaa4b0d0457d18183fcaa3f69d97f5b8bf8d6df6dbaa9c9c53d076a9d4.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/e6c23daaa4b0d0457d18183fcaa3f69d97f5b8bf8d6df6dbaa9c9c53d076a9d4.jpg)

![f3bcbfbb38c226d1f161d3e9b6f2f8eff1b22ac8c34b92d730518f6fee879360.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/f3bcbfbb38c226d1f161d3e9b6f2f8eff1b22ac8c34b92d730518f6fee879360.jpg)

![fe57f437dd5f376c84e0a0fb0e2ca962b52bac48fa860e8b5d77f6070b952e52.jpg](../acl_results/132_Unraveling%20the%20Mechanics%20of%20Learning-Based%20Demonstration%20Selection%20for%20In-Context%20Learning/tables/fe57f437dd5f376c84e0a0fb0e2ca962b52bac48fa860e8b5d77f6070b952e52.jpg)

## Direct Prompt Optimization with Continuous Representations

### Images

![2602f47b237b7db55aee90241e83c11d0c115dd5fa454b684a009f7417e6d1bb.jpg](../acl_results/133_Direct%20Prompt%20Optimization%20with%20Continuous%20Representations/images/2602f47b237b7db55aee90241e83c11d0c115dd5fa454b684a009f7417e6d1bb.jpg)

### Tables

![034a8144010a03723cc101de6fbe3baa470dfeb91e9d1e3634d32323959cf4d4.jpg](../acl_results/133_Direct%20Prompt%20Optimization%20with%20Continuous%20Representations/tables/034a8144010a03723cc101de6fbe3baa470dfeb91e9d1e3634d32323959cf4d4.jpg)

![64167bd67ebd434b12434169ec6c0b9191a6fe0baaa9e6e1dacded789cf07c33.jpg](../acl_results/133_Direct%20Prompt%20Optimization%20with%20Continuous%20Representations/tables/64167bd67ebd434b12434169ec6c0b9191a6fe0baaa9e6e1dacded789cf07c33.jpg)

![6ed94ca45ff11ffaed7461033afd98231e52e5e3f06f7435fc87ff9d91e8fbe2.jpg](../acl_results/133_Direct%20Prompt%20Optimization%20with%20Continuous%20Representations/tables/6ed94ca45ff11ffaed7461033afd98231e52e5e3f06f7435fc87ff9d91e8fbe2.jpg)

![b2513ca02230050a66fa6c01ed55728e97dfc0a0bf740ff403009049be2417c2.jpg](../acl_results/133_Direct%20Prompt%20Optimization%20with%20Continuous%20Representations/tables/b2513ca02230050a66fa6c01ed55728e97dfc0a0bf740ff403009049be2417c2.jpg)

![ba2e92c8257562349f832176cbb25ba14de4537a2e657aae4ff16ec82c4fad07.jpg](../acl_results/133_Direct%20Prompt%20Optimization%20with%20Continuous%20Representations/tables/ba2e92c8257562349f832176cbb25ba14de4537a2e657aae4ff16ec82c4fad07.jpg)

## uMedSum: A Unified Framework for Clinical Abstractive Summarization

### Images

![50512cbec1e429ad7deb83258135358c92ba67972e099d04806ca17a33607aa7.jpg](../acl_results/134_uMedSum_%20A%20Unified%20Framework%20for%20Clinical%20Abstractive%20Summarization/images/50512cbec1e429ad7deb83258135358c92ba67972e099d04806ca17a33607aa7.jpg)

![50b80044469c0e2a5e540e2b8df251d3a63c3be19b85aa2e05cf87130ca1d66c.jpg](../acl_results/134_uMedSum_%20A%20Unified%20Framework%20for%20Clinical%20Abstractive%20Summarization/images/50b80044469c0e2a5e540e2b8df251d3a63c3be19b85aa2e05cf87130ca1d66c.jpg)

![5ee70b2fdbf133737e4498563450db8379596ea67e9e080d998ced8145682c63.jpg](../acl_results/134_uMedSum_%20A%20Unified%20Framework%20for%20Clinical%20Abstractive%20Summarization/images/5ee70b2fdbf133737e4498563450db8379596ea67e9e080d998ced8145682c63.jpg)

### Tables

![31cd69b1f7249923bc6dd88b0ed7df120a582066c49389e1b8164f2825940819.jpg](../acl_results/134_uMedSum_%20A%20Unified%20Framework%20for%20Clinical%20Abstractive%20Summarization/tables/31cd69b1f7249923bc6dd88b0ed7df120a582066c49389e1b8164f2825940819.jpg)

![3bb61c1476fa78fd4e7387fcfa806dcff3f24f600d134bc429a6807054dab042.jpg](../acl_results/134_uMedSum_%20A%20Unified%20Framework%20for%20Clinical%20Abstractive%20Summarization/tables/3bb61c1476fa78fd4e7387fcfa806dcff3f24f600d134bc429a6807054dab042.jpg)

![4bc42cfbd80f7f7c2f07198fbcbab48be3df52e0e2fa911e1d1c697743b51ba1.jpg](../acl_results/134_uMedSum_%20A%20Unified%20Framework%20for%20Clinical%20Abstractive%20Summarization/tables/4bc42cfbd80f7f7c2f07198fbcbab48be3df52e0e2fa911e1d1c697743b51ba1.jpg)

![6714ca35b7e18d118f607b43135638086d7cc0d6d3ee5a2853633478095add43.jpg](../acl_results/134_uMedSum_%20A%20Unified%20Framework%20for%20Clinical%20Abstractive%20Summarization/tables/6714ca35b7e18d118f607b43135638086d7cc0d6d3ee5a2853633478095add43.jpg)

![7fe99a73359ecf793b5542dbb486197d334cb7cdd131accd56d3ebe8c17ff438.jpg](../acl_results/134_uMedSum_%20A%20Unified%20Framework%20for%20Clinical%20Abstractive%20Summarization/tables/7fe99a73359ecf793b5542dbb486197d334cb7cdd131accd56d3ebe8c17ff438.jpg)

![83fdc061e624ac8681beb44957bb9c60e0e5672661763278f71a4efff01e0a9b.jpg](../acl_results/134_uMedSum_%20A%20Unified%20Framework%20for%20Clinical%20Abstractive%20Summarization/tables/83fdc061e624ac8681beb44957bb9c60e0e5672661763278f71a4efff01e0a9b.jpg)

![9f68c1941758651fa18e07aaa4943d492780b147e49c3e90f6e75150555b3dd2.jpg](../acl_results/134_uMedSum_%20A%20Unified%20Framework%20for%20Clinical%20Abstractive%20Summarization/tables/9f68c1941758651fa18e07aaa4943d492780b147e49c3e90f6e75150555b3dd2.jpg)

![d651b59f216fe77d02d9ff16bd10ca09fa09141427ef99b914148182a97c63cf.jpg](../acl_results/134_uMedSum_%20A%20Unified%20Framework%20for%20Clinical%20Abstractive%20Summarization/tables/d651b59f216fe77d02d9ff16bd10ca09fa09141427ef99b914148182a97c63cf.jpg)

## GigaSpeech 2: An Evolving, Large-Scale and Multi-domainASRCorpus for Low-Resource Languages with Automated Crawling, Transcription and Refinement

### Images

![7802aac07eb0ce2da9c51d65401700b080adca24b77e2b36c26e436bb2191308.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/images/7802aac07eb0ce2da9c51d65401700b080adca24b77e2b36c26e436bb2191308.jpg)

![d83ce406f8a63c272d5ec7b98d29e846b570660eb91c1e85c3febd59511254a6.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/images/d83ce406f8a63c272d5ec7b98d29e846b570660eb91c1e85c3febd59511254a6.jpg)

![fcb5a0168e396579d0098dff764877530e58e4c56de8d5d5af2440dc11aff9e5.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/images/fcb5a0168e396579d0098dff764877530e58e4c56de8d5d5af2440dc11aff9e5.jpg)

### Tables

![0b3d41e92ef56bed53e47068016d4a81b029db9a8085066c45708f5553c2c8c3.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/tables/0b3d41e92ef56bed53e47068016d4a81b029db9a8085066c45708f5553c2c8c3.jpg)

![648b1b05e93e4ae334ae39255f931d8d8a39feb2833551d1f0a9d76ab6a31d2f.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/tables/648b1b05e93e4ae334ae39255f931d8d8a39feb2833551d1f0a9d76ab6a31d2f.jpg)

![7645514e5558c636f9ee4267ba5e126653c5ca72632d784c7835b07e1d16b179.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/tables/7645514e5558c636f9ee4267ba5e126653c5ca72632d784c7835b07e1d16b179.jpg)

![7f7d43c0f5cf8cf3f35b369bab137f86950703e5f6d5e2287627ef5e416a1b5a.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/tables/7f7d43c0f5cf8cf3f35b369bab137f86950703e5f6d5e2287627ef5e416a1b5a.jpg)

![847592e7d1fb38498498905a7a9b230114bffd55f7e4580d2a9fb654b6b094cb.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/tables/847592e7d1fb38498498905a7a9b230114bffd55f7e4580d2a9fb654b6b094cb.jpg)

![9249fe19845a5cb34f280892778ee2fd7fb8247d5ed3b71581f482b2a945c6e1.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/tables/9249fe19845a5cb34f280892778ee2fd7fb8247d5ed3b71581f482b2a945c6e1.jpg)

![b8ccc719e12d4553920474227f0f4d0b7bddbc13e3383fbde0cc1942ccef23a5.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/tables/b8ccc719e12d4553920474227f0f4d0b7bddbc13e3383fbde0cc1942ccef23a5.jpg)

![de5313bcdbde5801a1773b4ff7ee0fbf423f04bbf8ff375edee2fddd6bc22777.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/tables/de5313bcdbde5801a1773b4ff7ee0fbf423f04bbf8ff375edee2fddd6bc22777.jpg)

![edd89e3f3e6a70f7ff2ac9b68c7b43a6730f524a93039cb60a79b7c651e9b471.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/tables/edd89e3f3e6a70f7ff2ac9b68c7b43a6730f524a93039cb60a79b7c651e9b471.jpg)

![f6218d6f8e858144210071cee51fceac849b1d12049d6b6110767ca9ec8093c8.jpg](../acl_results/135_GigaSpeech%202_%20An%20Evolving%2C%20Large-Scale%20and%20Multi-domainASRCorpus%20for%20Low-Resource%20Languages%20with%20Aut/tables/f6218d6f8e858144210071cee51fceac849b1d12049d6b6110767ca9ec8093c8.jpg)

## Context-Aware Sentiment Forecasting viaLLM-based Multi-Perspective Role-Playing Agents

### Images

![07e290439f5965232852569babdb2e4105afcf42f2b6e5f1bafdc433c665270b.jpg](../acl_results/136_Context-Aware%20Sentiment%20Forecasting%20viaLLM-based%20Multi-Perspective%20Role-Playing%20Agents/images/07e290439f5965232852569babdb2e4105afcf42f2b6e5f1bafdc433c665270b.jpg)

![5736a1664ff234921c0840e8e1025037ae8d49c255bb6842e8bcb7b937887622.jpg](../acl_results/136_Context-Aware%20Sentiment%20Forecasting%20viaLLM-based%20Multi-Perspective%20Role-Playing%20Agents/images/5736a1664ff234921c0840e8e1025037ae8d49c255bb6842e8bcb7b937887622.jpg)

![6082db13a382af3a4fe0776891e87e6bc054f3072f3aaf5b5afd9385dd6d5a29.jpg](../acl_results/136_Context-Aware%20Sentiment%20Forecasting%20viaLLM-based%20Multi-Perspective%20Role-Playing%20Agents/images/6082db13a382af3a4fe0776891e87e6bc054f3072f3aaf5b5afd9385dd6d5a29.jpg)

![c99291263e9b838eaa8736375ac09e50b32a9f1597a0244e61452e56c2b775ef.jpg](../acl_results/136_Context-Aware%20Sentiment%20Forecasting%20viaLLM-based%20Multi-Perspective%20Role-Playing%20Agents/images/c99291263e9b838eaa8736375ac09e50b32a9f1597a0244e61452e56c2b775ef.jpg)

### Tables

![46650032fb3a6f1865b19fa27d2d90f4e716e2cc2824ae557a4828f416657f04.jpg](../acl_results/136_Context-Aware%20Sentiment%20Forecasting%20viaLLM-based%20Multi-Perspective%20Role-Playing%20Agents/tables/46650032fb3a6f1865b19fa27d2d90f4e716e2cc2824ae557a4828f416657f04.jpg)

![5c0a3d7fd0c5c627bdfe2198b9236c14572ae4acc1e5a74acd521fdf6d760a14.jpg](../acl_results/136_Context-Aware%20Sentiment%20Forecasting%20viaLLM-based%20Multi-Perspective%20Role-Playing%20Agents/tables/5c0a3d7fd0c5c627bdfe2198b9236c14572ae4acc1e5a74acd521fdf6d760a14.jpg)

![86eb083a65d2fc74dbfa7af1724f6839901d6bc56461eaaa0c1c2b5ade009acd.jpg](../acl_results/136_Context-Aware%20Sentiment%20Forecasting%20viaLLM-based%20Multi-Perspective%20Role-Playing%20Agents/tables/86eb083a65d2fc74dbfa7af1724f6839901d6bc56461eaaa0c1c2b5ade009acd.jpg)

![bc032b1ce728f2440ba6af9fcc05821f73efb2213de16ac68a7b50a41901fa55.jpg](../acl_results/136_Context-Aware%20Sentiment%20Forecasting%20viaLLM-based%20Multi-Perspective%20Role-Playing%20Agents/tables/bc032b1ce728f2440ba6af9fcc05821f73efb2213de16ac68a7b50a41901fa55.jpg)

![d5b810a609209e7a229745bf8179a911faaa54f1e6495806918552f1e2b47844.jpg](../acl_results/136_Context-Aware%20Sentiment%20Forecasting%20viaLLM-based%20Multi-Perspective%20Role-Playing%20Agents/tables/d5b810a609209e7a229745bf8179a911faaa54f1e6495806918552f1e2b47844.jpg)

## TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data

### Images

![27083398b9caff7e7da68fcedc2e37a3f8a18e24332fbe41b148a4e7f594a7e9.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/images/27083398b9caff7e7da68fcedc2e37a3f8a18e24332fbe41b148a4e7f594a7e9.jpg)

![5c8a3d4de429cbaaff18be752ae9353d6f6734da33c7e02df731fc57ea47a528.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/images/5c8a3d4de429cbaaff18be752ae9353d6f6734da33c7e02df731fc57ea47a528.jpg)

![7fd55a03c960a1fda44980121ef740fb33bae8ad40d7ab82326d2f23a73d564c.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/images/7fd55a03c960a1fda44980121ef740fb33bae8ad40d7ab82326d2f23a73d564c.jpg)

![cf11916d96339956fd13222291f41087fd02e2a99238f88b719b9f66a4bbe0da.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/images/cf11916d96339956fd13222291f41087fd02e2a99238f88b719b9f66a4bbe0da.jpg)

![e193d644a431a0b5ac141955db42475818239b798e2a8dc15fd0a8fbe0fb3aad.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/images/e193d644a431a0b5ac141955db42475818239b798e2a8dc15fd0a8fbe0fb3aad.jpg)

### Tables

![0ef52bc5ba712f801086767fc64d47012d6843c9e27bbd9b722180e51c2153af.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/0ef52bc5ba712f801086767fc64d47012d6843c9e27bbd9b722180e51c2153af.jpg)

![20314b53efcf540b372bb8e1e45c9a1c1f8a1ceba3a8471819bc1ded6bdce750.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/20314b53efcf540b372bb8e1e45c9a1c1f8a1ceba3a8471819bc1ded6bdce750.jpg)

![21740a8bde56b662a5ae906490e111aedc71bc171b6769e7520af3c2b83358f6.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/21740a8bde56b662a5ae906490e111aedc71bc171b6769e7520af3c2b83358f6.jpg)

![41064df4b5ba8ca0cc4c2c3e436009dd091443a301fe0e2ded3596cadb52df53.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/41064df4b5ba8ca0cc4c2c3e436009dd091443a301fe0e2ded3596cadb52df53.jpg)

![54d1d97d3ddeaa1da60e031df3c35bd29a8d5ceafbb57db2a2387ed9abc50783.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/54d1d97d3ddeaa1da60e031df3c35bd29a8d5ceafbb57db2a2387ed9abc50783.jpg)

![5b7e6f7ed712df6e3b3abd91664a3de2ea588cb88228bd92bab585bb6ddf292e.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/5b7e6f7ed712df6e3b3abd91664a3de2ea588cb88228bd92bab585bb6ddf292e.jpg)

![5ef95f6bd45b5512ff8bd90e7a20aff78fca60488ac88ee03cbf49e8e2e9af31.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/5ef95f6bd45b5512ff8bd90e7a20aff78fca60488ac88ee03cbf49e8e2e9af31.jpg)

![676c213a7e56daa882955ac58347181ccbf784f72a39798cf3957fcacad33f5a.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/676c213a7e56daa882955ac58347181ccbf784f72a39798cf3957fcacad33f5a.jpg)

![b5a5cd4d3f52ac469bbe17697fe787b3149e21b7b590412c0e61f8af4a9553fa.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/b5a5cd4d3f52ac469bbe17697fe787b3149e21b7b590412c0e61f8af4a9553fa.jpg)

![b65fda58c6e00580d6c5ff2c4f79758ca7cc8d431ac9232f76ed9ea870d8f2d3.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/b65fda58c6e00580d6c5ff2c4f79758ca7cc8d431ac9232f76ed9ea870d8f2d3.jpg)

![c6b398fc286dc26ebd70ad8e3cb7ef95e0c624b8affb083c638c1ee0eb284b21.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/c6b398fc286dc26ebd70ad8e3cb7ef95e0c624b8affb083c638c1ee0eb284b21.jpg)

![d715c1a163d1070fb0a4c9eb58a0b57f1411500fe6e31e5584036b8d7a779d0c.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/d715c1a163d1070fb0a4c9eb58a0b57f1411500fe6e31e5584036b8d7a779d0c.jpg)

![e0e881446d9a0077c7bc70b3ff41e2ed0a7a4e218fc5f9abe6c02c1f37a910a2.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/e0e881446d9a0077c7bc70b3ff41e2ed0a7a4e218fc5f9abe6c02c1f37a910a2.jpg)

![f68eb3e09f5a12b571c569fadeb0e5394d45dfaaffc50234dfb21441c50415aa.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/f68eb3e09f5a12b571c569fadeb0e5394d45dfaaffc50234dfb21441c50415aa.jpg)

![fb98d2628a02668163699b937a0c2d7ae138228135eac5d17cfccf5cfaffd28f.jpg](../acl_results/137_TARGA_%20Targeted%20Synthetic%20Data%20Generation%20for%20Practical%20Reasoning%20over%20Structured%20Data/tables/fb98d2628a02668163699b937a0c2d7ae138228135eac5d17cfccf5cfaffd28f.jpg)

## AndroidGen: Building an Android Language Agent under Data Scarcity

### Images

![04dd92069c9189facff8265948eb09081f0a78099cf9984c41589ae43d140d42.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/04dd92069c9189facff8265948eb09081f0a78099cf9984c41589ae43d140d42.jpg)

![0bdd65f498144335a5f95556caf56765b4434cbe98b6c63182b6543efabd63c0.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/0bdd65f498144335a5f95556caf56765b4434cbe98b6c63182b6543efabd63c0.jpg)

![129fc24fd4840bcf03ec1e8131bc7813c60ef103512eea44a5cf686a0d6d3287.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/129fc24fd4840bcf03ec1e8131bc7813c60ef103512eea44a5cf686a0d6d3287.jpg)

![2347fcbfaf3169b8c82bf998b3e0093a6ebe83df8aa37e49843da1d4b3dc3f4c.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/2347fcbfaf3169b8c82bf998b3e0093a6ebe83df8aa37e49843da1d4b3dc3f4c.jpg)

![4fb89d3da6aa5079357d5990f25564b94fd7da93f5e19974ee38a0689d286f96.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/4fb89d3da6aa5079357d5990f25564b94fd7da93f5e19974ee38a0689d286f96.jpg)

![5a1a1364ce46a460f74f273a2daba475e9770b64a5f032e28eed1ee32e04f8eb.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/5a1a1364ce46a460f74f273a2daba475e9770b64a5f032e28eed1ee32e04f8eb.jpg)

![5e79a5e1bc19d8fdeae32830fba0666874cec268b965ee5c3d2a8bb93128a057.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/5e79a5e1bc19d8fdeae32830fba0666874cec268b965ee5c3d2a8bb93128a057.jpg)

![68ba2845ef7339a968656dd50521c79a55742e72addbc3d0fd81f95387c13199.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/68ba2845ef7339a968656dd50521c79a55742e72addbc3d0fd81f95387c13199.jpg)

![6f20651dc79e013c8508ee214ccf73e6b12148d26c0cc51724041c973fa99f23.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/6f20651dc79e013c8508ee214ccf73e6b12148d26c0cc51724041c973fa99f23.jpg)

![8c9afa05a2da240003e2c6412fc963949ffd55ca78b8ea4f5c89567dcdcfd524.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/8c9afa05a2da240003e2c6412fc963949ffd55ca78b8ea4f5c89567dcdcfd524.jpg)

![93a68d3a90d3ab081b25aaa835205cf9c17b285ea3281fcf9f051388371df9f8.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/93a68d3a90d3ab081b25aaa835205cf9c17b285ea3281fcf9f051388371df9f8.jpg)

![9ab7634201b21ef44c40a79e7dadd4d19131d959142a6502dc3dda3a74e387cb.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/9ab7634201b21ef44c40a79e7dadd4d19131d959142a6502dc3dda3a74e387cb.jpg)

![9b05b74e4bbb11ae41d670e875ddb87da2880f626402c6f19d9a13c6eae1f064.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/9b05b74e4bbb11ae41d670e875ddb87da2880f626402c6f19d9a13c6eae1f064.jpg)

![b1f88d2df8d8cfae75118ae1ea38d3ad963df6c5d7c9a5978030e06a0b05ed6b.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/b1f88d2df8d8cfae75118ae1ea38d3ad963df6c5d7c9a5978030e06a0b05ed6b.jpg)

![ba6282237f643c8c6d37354f76c350269dd0b9ede9273a37822c5d7949c21359.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/ba6282237f643c8c6d37354f76c350269dd0b9ede9273a37822c5d7949c21359.jpg)

![ca0912c921aa897d85145a5c1d4a4e584eefe20000a3e53bd032d1d46d0d47dd.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/ca0912c921aa897d85145a5c1d4a4e584eefe20000a3e53bd032d1d46d0d47dd.jpg)

![e4b2fd6527389520240cfef3b9111cbaf192ba7ffa9662ae8378909a4015becb.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/e4b2fd6527389520240cfef3b9111cbaf192ba7ffa9662ae8378909a4015becb.jpg)

![f485a07358df1fd105c45424badaa4caabc2a5d023cc685e80221ee2d73be308.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/images/f485a07358df1fd105c45424badaa4caabc2a5d023cc685e80221ee2d73be308.jpg)

### Tables

![0d132c11631509ad3b65b2363ce3d58564726ef7c6b72e9d4a085796b6a1e290.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/tables/0d132c11631509ad3b65b2363ce3d58564726ef7c6b72e9d4a085796b6a1e290.jpg)

![1636c716850ef75d2f73202306d5ec3d8e23884bf55e6de3a771cf069e598aa7.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/tables/1636c716850ef75d2f73202306d5ec3d8e23884bf55e6de3a771cf069e598aa7.jpg)

![4627a5d9e6ce0652a68952d2dd1d34798e060cdad4d29d05844dad87d8c49776.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/tables/4627a5d9e6ce0652a68952d2dd1d34798e060cdad4d29d05844dad87d8c49776.jpg)

![6c082cab485441e5d8f35fa435877340fbde198e7f02675fa39a41a8e47baceb.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/tables/6c082cab485441e5d8f35fa435877340fbde198e7f02675fa39a41a8e47baceb.jpg)

![705291cf156bc09e6734a3ca615033df06abcd64e17be00217abfba2d63bbb7d.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/tables/705291cf156bc09e6734a3ca615033df06abcd64e17be00217abfba2d63bbb7d.jpg)

![91ba998477e5d1b971f3b8b3eee0389082f49bce7f2896fc365dd22a5e68bbbc.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/tables/91ba998477e5d1b971f3b8b3eee0389082f49bce7f2896fc365dd22a5e68bbbc.jpg)

![ab87637561b9c28ff141635c078f13bb337b95d00e7498afe0a9a49ab35ab50d.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/tables/ab87637561b9c28ff141635c078f13bb337b95d00e7498afe0a9a49ab35ab50d.jpg)

![c392a0aabab643399eca2c09aa1e87c5b6d0c4933acaec39e78f4cbd8d6b55bf.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/tables/c392a0aabab643399eca2c09aa1e87c5b6d0c4933acaec39e78f4cbd8d6b55bf.jpg)

![d8d1d3c737841220bd0ebb79765b63219f56e8ca83407df6bbce4436043cad27.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/tables/d8d1d3c737841220bd0ebb79765b63219f56e8ca83407df6bbce4436043cad27.jpg)

![de735c92764d0753c5300665ee8c94ef3936d446da0556b06f36bd5523f4b6be.jpg](../acl_results/138_AndroidGen_%20Building%20an%20Android%20Language%20Agent%20under%20Data%20Scarcity/tables/de735c92764d0753c5300665ee8c94ef3936d446da0556b06f36bd5523f4b6be.jpg)

## Prompt Candidates, then Distill: A Teacher-Student Framework forLLM-driven Data Annotation

### Images

![8105c12e9a6a1fa6857909e638f2c96b133a9d7427e7b0786459fcead2f46ae7.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/images/8105c12e9a6a1fa6857909e638f2c96b133a9d7427e7b0786459fcead2f46ae7.jpg)

![9c78e5fc92e7a024e36f190a48d13eb6acc25da75c99f630e9cf51640cddc54d.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/images/9c78e5fc92e7a024e36f190a48d13eb6acc25da75c99f630e9cf51640cddc54d.jpg)

![a961babd68c3fbde47ceaace6320e1141e6651d9d2ea25cc44385375f4c42a8a.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/images/a961babd68c3fbde47ceaace6320e1141e6651d9d2ea25cc44385375f4c42a8a.jpg)

![b747122ff17a32eaf47f11d4e1b74f754fd7d4288279360175e453dfc30ccaa0.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/images/b747122ff17a32eaf47f11d4e1b74f754fd7d4288279360175e453dfc30ccaa0.jpg)

![d688d1e105855b5d4386b7d7abc8009772814959f348bf83644c99c36c5b9269.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/images/d688d1e105855b5d4386b7d7abc8009772814959f348bf83644c99c36c5b9269.jpg)

![edd64e4a6eaefd8ec080681615a9c839b9fac8aebdd1f05f64b50ced29c6a0e7.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/images/edd64e4a6eaefd8ec080681615a9c839b9fac8aebdd1f05f64b50ced29c6a0e7.jpg)

### Tables

![140b01968991f5ec5111621fde2bf48ca0dc30b7cefbae8ea0ad8a5c9c970a37.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/140b01968991f5ec5111621fde2bf48ca0dc30b7cefbae8ea0ad8a5c9c970a37.jpg)

![2ceea847dfa77b4e9a9ef64ab7490362255291c469e6ac0a1cd0432f462e12b2.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/2ceea847dfa77b4e9a9ef64ab7490362255291c469e6ac0a1cd0432f462e12b2.jpg)

![2d9486f3411766f59af21e99f8035e4bede78a91ac30da9ae32d5a10acb271ea.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/2d9486f3411766f59af21e99f8035e4bede78a91ac30da9ae32d5a10acb271ea.jpg)

![2ecab651acfa994bf24560d760d71f0624c409d0dfafed6c8775a598ce02c16c.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/2ecab651acfa994bf24560d760d71f0624c409d0dfafed6c8775a598ce02c16c.jpg)

![588457faa9f4e1338dab9498d09a78fab584a2395505fd6acf07410719a65e9a.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/588457faa9f4e1338dab9498d09a78fab584a2395505fd6acf07410719a65e9a.jpg)

![5aa8643b729d70d29b5d27efa5c59132461443e8a9152afe2817305d2d6106d7.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/5aa8643b729d70d29b5d27efa5c59132461443e8a9152afe2817305d2d6106d7.jpg)

![5fa3904fc24ab02a1b878a5c3109e4204d6b64569700cc77da8ecbd494e1e336.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/5fa3904fc24ab02a1b878a5c3109e4204d6b64569700cc77da8ecbd494e1e336.jpg)

![84affe63a2bc66985826235d89ae1c09d4d1fd52c417cfc765234771a97a3d73.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/84affe63a2bc66985826235d89ae1c09d4d1fd52c417cfc765234771a97a3d73.jpg)

![ae6b41303794b8450e9c7c9b6baabbe30cdb917872453ce08e24906702283b1e.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/ae6b41303794b8450e9c7c9b6baabbe30cdb917872453ce08e24906702283b1e.jpg)

![b49aedce2578e27528b81d11807f78e00b627b6fab2797e7ebc75a4b10c7c99e.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/b49aedce2578e27528b81d11807f78e00b627b6fab2797e7ebc75a4b10c7c99e.jpg)

![cfdb2e52a64bdf4d831cf433def65dfd443e896aa68f8c45ca08245d1633495a.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/cfdb2e52a64bdf4d831cf433def65dfd443e896aa68f8c45ca08245d1633495a.jpg)

![d5abbd48f7516ceb331092990f9eff7056c77203e8b6fbc98c44de1b28e11841.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/d5abbd48f7516ceb331092990f9eff7056c77203e8b6fbc98c44de1b28e11841.jpg)

![d841ac3836f7a2a8efc1789a6fb0b39c309f19db0aaa08a5e2db7147902b6c63.jpg](../acl_results/139_Prompt%20Candidates%2C%20then%20Distill_%20A%20Teacher-Student%20Framework%20forLLM-driven%20Data%20Annotation/tables/d841ac3836f7a2a8efc1789a6fb0b39c309f19db0aaa08a5e2db7147902b6c63.jpg)

## A Survey of Post-Training Scaling in Large Language Models

### Images

![4e87605c5ef4ddfbdc6617563eaf2a7233c86db666e93f8c057059aebfa819d0.jpg](../acl_results/140_A%20Survey%20of%20Post-Training%20Scaling%20in%20Large%20Language%20Models/images/4e87605c5ef4ddfbdc6617563eaf2a7233c86db666e93f8c057059aebfa819d0.jpg)

![6a7ae89e868cc95e6f375ecde1f34793782a548002fc059f109eff7cc146a2e1.jpg](../acl_results/140_A%20Survey%20of%20Post-Training%20Scaling%20in%20Large%20Language%20Models/images/6a7ae89e868cc95e6f375ecde1f34793782a548002fc059f109eff7cc146a2e1.jpg)

![8a771390d0714439cb54574a719457ded88efc7486be94c993077b3478fd5b04.jpg](../acl_results/140_A%20Survey%20of%20Post-Training%20Scaling%20in%20Large%20Language%20Models/images/8a771390d0714439cb54574a719457ded88efc7486be94c993077b3478fd5b04.jpg)

![b7c698c19ea153ecf79da7cdac8da04a747ed34d5fc5350330cdc2f83e922813.jpg](../acl_results/140_A%20Survey%20of%20Post-Training%20Scaling%20in%20Large%20Language%20Models/images/b7c698c19ea153ecf79da7cdac8da04a747ed34d5fc5350330cdc2f83e922813.jpg)

![de04db27ab20caf85795d96dafb07df62575a1d815688fac99f4c62df4d59d75.jpg](../acl_results/140_A%20Survey%20of%20Post-Training%20Scaling%20in%20Large%20Language%20Models/images/de04db27ab20caf85795d96dafb07df62575a1d815688fac99f4c62df4d59d75.jpg)

### Tables

![2ce47f95712d7b54d0d462bb7e6f845b2f0a668e297966856a3113325c74b52e.jpg](../acl_results/140_A%20Survey%20of%20Post-Training%20Scaling%20in%20Large%20Language%20Models/tables/2ce47f95712d7b54d0d462bb7e6f845b2f0a668e297966856a3113325c74b52e.jpg)

![8b0a6a955d248fa7d3a8d98ec5a5fff1b4b73655095cf431335b2a1f0e6b4b91.jpg](../acl_results/140_A%20Survey%20of%20Post-Training%20Scaling%20in%20Large%20Language%20Models/tables/8b0a6a955d248fa7d3a8d98ec5a5fff1b4b73655095cf431335b2a1f0e6b4b91.jpg)

![eaa4d9e08457511bcb8b7e936f668e0a80b62eddc0b2f8e8ec6a1684b1c1e35b.jpg](../acl_results/140_A%20Survey%20of%20Post-Training%20Scaling%20in%20Large%20Language%20Models/tables/eaa4d9e08457511bcb8b7e936f668e0a80b62eddc0b2f8e8ec6a1684b1c1e35b.jpg)

## Position-aware Automatic Circuit Discovery

### Images

![048142c8442f9a2523b19d238d92ca072105c6f5de04598f2ca71d9693357359.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/images/048142c8442f9a2523b19d238d92ca072105c6f5de04598f2ca71d9693357359.jpg)

![349892bc952a7fa826655bee9d480569f85a0218c3e455ff5d3ca8532977c162.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/images/349892bc952a7fa826655bee9d480569f85a0218c3e455ff5d3ca8532977c162.jpg)

![77a098a781300e733495dd40e570598cf3379673524dbae29c057e903df1b42d.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/images/77a098a781300e733495dd40e570598cf3379673524dbae29c057e903df1b42d.jpg)

![869df0d39bc143460beb013b1ef609194f6e34d1ff1c7c0dc6a5932c5b162586.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/images/869df0d39bc143460beb013b1ef609194f6e34d1ff1c7c0dc6a5932c5b162586.jpg)

![b4511b3d3e32bc400b4299685c9830680a7bbdaf31a1f939ba3ef4bc7b6fca04.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/images/b4511b3d3e32bc400b4299685c9830680a7bbdaf31a1f939ba3ef4bc7b6fca04.jpg)

![b6e62dd8d857c311e8c24e551c63cf69210d7882c85702db010bee1beaec4fe9.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/images/b6e62dd8d857c311e8c24e551c63cf69210d7882c85702db010bee1beaec4fe9.jpg)

![ba5c7eaa57e48cd6fe7d15d160b64960091b80594c79cfad4c976f9c713e80e9.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/images/ba5c7eaa57e48cd6fe7d15d160b64960091b80594c79cfad4c976f9c713e80e9.jpg)

![ca01c3c5977e37f0e7395f4c8a078da8f513ae78af28f6246b04679dcd82696f.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/images/ca01c3c5977e37f0e7395f4c8a078da8f513ae78af28f6246b04679dcd82696f.jpg)

![d0d3b707a309ae3bbe71d58c79dfbe567f840bed7d572307d771a4f87548d3f9.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/images/d0d3b707a309ae3bbe71d58c79dfbe567f840bed7d572307d771a4f87548d3f9.jpg)

![d7b52ba0ef1dba4a53904a7540855b43d57c97aef2e53a01c9bebb7c25ec600e.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/images/d7b52ba0ef1dba4a53904a7540855b43d57c97aef2e53a01c9bebb7c25ec600e.jpg)

![e8eff5ad740f8184a0894bbb584a8fbf14d17df561e9334dedddd679d02f08b9.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/images/e8eff5ad740f8184a0894bbb584a8fbf14d17df561e9334dedddd679d02f08b9.jpg)

### Tables

![0e54db7705f12a7676ba0a30771b9f241da1b3a6b955fb5238b82390bfd6acc7.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/tables/0e54db7705f12a7676ba0a30771b9f241da1b3a6b955fb5238b82390bfd6acc7.jpg)

![3da44798c28846678d86d2a383bcc35930ca9e326a20a100fbde8aae16700895.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/tables/3da44798c28846678d86d2a383bcc35930ca9e326a20a100fbde8aae16700895.jpg)

![3f9afb39433052f922dc4fae70a6ca08685195af183eaaaa5364fd3c559bc6d8.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/tables/3f9afb39433052f922dc4fae70a6ca08685195af183eaaaa5364fd3c559bc6d8.jpg)

![46c33fff93874ef1c1eb99b4f02d08a016c139d2d57e42342533222efafca955.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/tables/46c33fff93874ef1c1eb99b4f02d08a016c139d2d57e42342533222efafca955.jpg)

![65e7406bf946a79b21f91bf3006960464595761d3be4ec21ad5dd0aa181345b2.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/tables/65e7406bf946a79b21f91bf3006960464595761d3be4ec21ad5dd0aa181345b2.jpg)

![7836fe07d0c5f3f0e219ca43772b91c42cccc25d519eded8893ec6d6465fdf22.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/tables/7836fe07d0c5f3f0e219ca43772b91c42cccc25d519eded8893ec6d6465fdf22.jpg)

![bface8d80e572deb65d6c5bda0da18c94af1d6610999dd039fb05526fae440d8.jpg](../acl_results/141_Position-aware%20Automatic%20Circuit%20Discovery/tables/bface8d80e572deb65d6c5bda0da18c94af1d6610999dd039fb05526fae440d8.jpg)

## HyperFM: Fact-Centric Multimodal Fusion for Link Prediction over Hyper-Relational Knowledge Graphs

### Images

![3b3d465545ad69ed08adc86ad41a0e3a21eb78ca23ecd6e33d0f7666dccda028.jpg](../acl_results/142_HyperFM_%20Fact-Centric%20Multimodal%20Fusion%20for%20Link%20Prediction%20over%20Hyper-Relational%20Knowledge%20Graphs/images/3b3d465545ad69ed08adc86ad41a0e3a21eb78ca23ecd6e33d0f7666dccda028.jpg)

![487d1ad7f4c4d1c90ee011037ab791b61962fb820b7ab71ca4122cc591831cd4.jpg](../acl_results/142_HyperFM_%20Fact-Centric%20Multimodal%20Fusion%20for%20Link%20Prediction%20over%20Hyper-Relational%20Knowledge%20Graphs/images/487d1ad7f4c4d1c90ee011037ab791b61962fb820b7ab71ca4122cc591831cd4.jpg)

![94d5f7413d383f6d3260e7e201e92ddb8bcb4da11a9f5d0b70b13032d8a0545e.jpg](../acl_results/142_HyperFM_%20Fact-Centric%20Multimodal%20Fusion%20for%20Link%20Prediction%20over%20Hyper-Relational%20Knowledge%20Graphs/images/94d5f7413d383f6d3260e7e201e92ddb8bcb4da11a9f5d0b70b13032d8a0545e.jpg)

![e0888d1f0fa3058e1ce1c84b70e35001231307d5881e7e98b1afe8a83d231632.jpg](../acl_results/142_HyperFM_%20Fact-Centric%20Multimodal%20Fusion%20for%20Link%20Prediction%20over%20Hyper-Relational%20Knowledge%20Graphs/images/e0888d1f0fa3058e1ce1c84b70e35001231307d5881e7e98b1afe8a83d231632.jpg)

### Tables

![13637b2b4b9f0cb20f06a1097fd265495aa295e045788a0238ca3d2b98a6bd65.jpg](../acl_results/142_HyperFM_%20Fact-Centric%20Multimodal%20Fusion%20for%20Link%20Prediction%20over%20Hyper-Relational%20Knowledge%20Graphs/tables/13637b2b4b9f0cb20f06a1097fd265495aa295e045788a0238ca3d2b98a6bd65.jpg)

![346549395e41c73fea008aed5ac62c39c185db227a6f7557ef576fe7fff366e2.jpg](../acl_results/142_HyperFM_%20Fact-Centric%20Multimodal%20Fusion%20for%20Link%20Prediction%20over%20Hyper-Relational%20Knowledge%20Graphs/tables/346549395e41c73fea008aed5ac62c39c185db227a6f7557ef576fe7fff366e2.jpg)

![74bf1471cee1f9892918a03c65102f68958c19c190def2d84d60218220716d6c.jpg](../acl_results/142_HyperFM_%20Fact-Centric%20Multimodal%20Fusion%20for%20Link%20Prediction%20over%20Hyper-Relational%20Knowledge%20Graphs/tables/74bf1471cee1f9892918a03c65102f68958c19c190def2d84d60218220716d6c.jpg)

![af7a03cdc8db1be542c61b805a61c6ba89e99971789a0ddcf6a601345bf330b3.jpg](../acl_results/142_HyperFM_%20Fact-Centric%20Multimodal%20Fusion%20for%20Link%20Prediction%20over%20Hyper-Relational%20Knowledge%20Graphs/tables/af7a03cdc8db1be542c61b805a61c6ba89e99971789a0ddcf6a601345bf330b3.jpg)

![c436ae69eadf394e3c06e13f373ecad2ed13b806ecd9f460a494de2d5ffe9d66.jpg](../acl_results/142_HyperFM_%20Fact-Centric%20Multimodal%20Fusion%20for%20Link%20Prediction%20over%20Hyper-Relational%20Knowledge%20Graphs/tables/c436ae69eadf394e3c06e13f373ecad2ed13b806ecd9f460a494de2d5ffe9d66.jpg)

## Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model

### Images

![2c2d5ffd3f6656e1e6c7f65b7f07a3a0cb558b6e9b39677a67c7b9e94ba514a5.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/images/2c2d5ffd3f6656e1e6c7f65b7f07a3a0cb558b6e9b39677a67c7b9e94ba514a5.jpg)

![360922c1f2c9120ebc03fe810063fa74d44697174563244120f39954e2219c44.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/images/360922c1f2c9120ebc03fe810063fa74d44697174563244120f39954e2219c44.jpg)

![61bd458c229c5d4e4b8b5d9db4325feac0a465fa17286367d6bc1aa9a1dfada9.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/images/61bd458c229c5d4e4b8b5d9db4325feac0a465fa17286367d6bc1aa9a1dfada9.jpg)

![b68fec16ce6707151288645323fc6c2d3265d95a0f3a9df297f57f5ae8972643.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/images/b68fec16ce6707151288645323fc6c2d3265d95a0f3a9df297f57f5ae8972643.jpg)

![b8190a42683b914218c40a27506a2d1ed0ea36a2805a75b54892636861f8db2e.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/images/b8190a42683b914218c40a27506a2d1ed0ea36a2805a75b54892636861f8db2e.jpg)

![dfafe86b8bd29ffa545fcb83a5b32f3a05e6dbf4acdff05d9b3868bd15f10f79.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/images/dfafe86b8bd29ffa545fcb83a5b32f3a05e6dbf4acdff05d9b3868bd15f10f79.jpg)

### Tables

![000a5de3f56707a74128723363708376aa019ba21dabef4cce87ea7e6c284a05.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/000a5de3f56707a74128723363708376aa019ba21dabef4cce87ea7e6c284a05.jpg)

![028956e516f868e261945735d24b83fb18e7fde58da5a9a22871f06ffd4d6443.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/028956e516f868e261945735d24b83fb18e7fde58da5a9a22871f06ffd4d6443.jpg)

![046797915969e17d9b634eccb5a66328d10bb6d3cfc414e84e2e9bb4d96de9a1.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/046797915969e17d9b634eccb5a66328d10bb6d3cfc414e84e2e9bb4d96de9a1.jpg)

![0f6ccf04ce5b462764ce470c5c80eb7931d04801dc92ff6e96dd30571f4ab619.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/0f6ccf04ce5b462764ce470c5c80eb7931d04801dc92ff6e96dd30571f4ab619.jpg)

![158200440c277dfe35133f0d05868860373a3aeb554ec4ef5980fdb00b09c13c.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/158200440c277dfe35133f0d05868860373a3aeb554ec4ef5980fdb00b09c13c.jpg)

![1b909174f1e0c365f943e9fe8c28876e791ac67ec62bd0fafae612ed9677b54e.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/1b909174f1e0c365f943e9fe8c28876e791ac67ec62bd0fafae612ed9677b54e.jpg)

![203013775898b209d26035abf693a4ee2c5d3bf7d6d2c526ac3a579659dce75e.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/203013775898b209d26035abf693a4ee2c5d3bf7d6d2c526ac3a579659dce75e.jpg)

![20e465228adda1df4e12bfc68e610c075d26eb0b8b010eaae23c67d9f61a9da7.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/20e465228adda1df4e12bfc68e610c075d26eb0b8b010eaae23c67d9f61a9da7.jpg)

![24e6de540448ad1b207c159f84615824ee857ccc05658ad9c4daf54c04dafa50.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/24e6de540448ad1b207c159f84615824ee857ccc05658ad9c4daf54c04dafa50.jpg)

![27c9d5ace63613f9938c13a1504c66e63b40eb3054e6c20aa763ba88a4452250.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/27c9d5ace63613f9938c13a1504c66e63b40eb3054e6c20aa763ba88a4452250.jpg)

![296f21bec5b7b3ae5a5f469a85b756d921dceae78abf6034b883c5ec094edf05.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/296f21bec5b7b3ae5a5f469a85b756d921dceae78abf6034b883c5ec094edf05.jpg)

![2ad56b4c6ae12d5426e2e90f31f4d9a4efed86ff83c055d29a3bc5253dc6aa91.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/2ad56b4c6ae12d5426e2e90f31f4d9a4efed86ff83c055d29a3bc5253dc6aa91.jpg)

![3c77b220a309d5846ad737f7f77b8a74120383f21da55653aa7b29eb3347b70a.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/3c77b220a309d5846ad737f7f77b8a74120383f21da55653aa7b29eb3347b70a.jpg)

![3dd94c4df36a22c0f8a0763f0d18b5a3caa4f9fa0eb60a211dd8be6a0b585ba0.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/3dd94c4df36a22c0f8a0763f0d18b5a3caa4f9fa0eb60a211dd8be6a0b585ba0.jpg)

![40e74529a9b9b79e1b4d88bb0c5be373d705e9c62cc27679615890cf77197199.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/40e74529a9b9b79e1b4d88bb0c5be373d705e9c62cc27679615890cf77197199.jpg)

![4be44a30c8f68d6b348fde7f2a45d4c71b34447b80abe281c99258e90d4cf387.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/4be44a30c8f68d6b348fde7f2a45d4c71b34447b80abe281c99258e90d4cf387.jpg)

![4c167f85483c7fdaa950075e061040d58e3e5582b1b3902bdab41c2e80066976.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/4c167f85483c7fdaa950075e061040d58e3e5582b1b3902bdab41c2e80066976.jpg)

![5177b6d4c7b88612ed3d442c52b3bb0fb1bdfd001fc90ac6f984103a7fffe321.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/5177b6d4c7b88612ed3d442c52b3bb0fb1bdfd001fc90ac6f984103a7fffe321.jpg)

![531f44db201a8701a461fc354e76829a61ae05f9011fb17d763109d8690d23c0.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/531f44db201a8701a461fc354e76829a61ae05f9011fb17d763109d8690d23c0.jpg)

![56b7fa474f155da159f09d645d07d3f4df3745dd2c68a5bb7683d6d5a09ad3be.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/56b7fa474f155da159f09d645d07d3f4df3745dd2c68a5bb7683d6d5a09ad3be.jpg)

![57940b48fe877b832b496542979f69598846f1d9a3d7f06a7968772ea2c2c1c2.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/57940b48fe877b832b496542979f69598846f1d9a3d7f06a7968772ea2c2c1c2.jpg)

![605718e31aa309cb097f7838d27fabd597b1ed437dcd57186afc35ac37b5b248.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/605718e31aa309cb097f7838d27fabd597b1ed437dcd57186afc35ac37b5b248.jpg)

![61d0db9a74510d95f654b939bd501ba47f4c52e537cb131e457497b9690c389f.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/61d0db9a74510d95f654b939bd501ba47f4c52e537cb131e457497b9690c389f.jpg)

![62c5f89832af1e0313c27278905b655da0e8093f222943499586a150169a3f7a.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/62c5f89832af1e0313c27278905b655da0e8093f222943499586a150169a3f7a.jpg)

![7daf064f167617504727408dfcf4ce9bed9337718694c7e9a49e640b909c4cf6.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/7daf064f167617504727408dfcf4ce9bed9337718694c7e9a49e640b909c4cf6.jpg)

![7f54a1e7538cf2aaaee6626f3143d927199ad784669db3dbdfea67707a2ad69a.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/7f54a1e7538cf2aaaee6626f3143d927199ad784669db3dbdfea67707a2ad69a.jpg)

![81a00dcd5eb7d29740056d251eb94747e37890f62828d249991a066177c746c0.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/81a00dcd5eb7d29740056d251eb94747e37890f62828d249991a066177c746c0.jpg)

![8318240d3e95152080b01bb8ff82d5d975c5cd06ea0a31fdea6b6b1350ed5b3c.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/8318240d3e95152080b01bb8ff82d5d975c5cd06ea0a31fdea6b6b1350ed5b3c.jpg)

![8b04bb6a193fb49a5517fb556903438165bad5cfdb335fd4fe50dc1bdbb7cf64.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/8b04bb6a193fb49a5517fb556903438165bad5cfdb335fd4fe50dc1bdbb7cf64.jpg)

![93ccffc9b18d119950da3bfad8d7bae186089d50850528ff71aed1d47f2e686c.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/93ccffc9b18d119950da3bfad8d7bae186089d50850528ff71aed1d47f2e686c.jpg)

![95a5316ab83f25c547f0321bdc5d28309b5f2af80e77bd60ff5da0ab7d71ff6d.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/95a5316ab83f25c547f0321bdc5d28309b5f2af80e77bd60ff5da0ab7d71ff6d.jpg)

![9661a14a855b5ef0330be600e8bd5d232f4da6d91f1fda9b718ad3d00c791d16.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/9661a14a855b5ef0330be600e8bd5d232f4da6d91f1fda9b718ad3d00c791d16.jpg)

![9bfe60345946a675e1bb6867fecc5be1fad4b16848797d9e6a852cfdaaf1c0f8.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/9bfe60345946a675e1bb6867fecc5be1fad4b16848797d9e6a852cfdaaf1c0f8.jpg)

![a49c3612c029f231112bee61f1aefa2c0bc59179e0c7f75a7a0da8d953c19f4a.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/a49c3612c029f231112bee61f1aefa2c0bc59179e0c7f75a7a0da8d953c19f4a.jpg)

![a522e7c8e8695892d39a1fad228f18131069586556a1be6c7bdac8534e268648.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/a522e7c8e8695892d39a1fad228f18131069586556a1be6c7bdac8534e268648.jpg)

![bb7ea0ba2972e41cee84597b9bdb930b47b72d468009486c1ccb41581e8cbfe2.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/bb7ea0ba2972e41cee84597b9bdb930b47b72d468009486c1ccb41581e8cbfe2.jpg)

![bd2bfe1075b67be93c80e993ef8305f3dc41bade3593da428cabe423c65b7758.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/bd2bfe1075b67be93c80e993ef8305f3dc41bade3593da428cabe423c65b7758.jpg)

![c7de62acd98f09a8e4f89633311bc6d1a450dfc6ccc9f0bf1dd18ea074e4c2f0.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/c7de62acd98f09a8e4f89633311bc6d1a450dfc6ccc9f0bf1dd18ea074e4c2f0.jpg)

![d132556bbc406df50300322b647d3145b0fe86d16163d480744f8223d9f38392.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/d132556bbc406df50300322b647d3145b0fe86d16163d480744f8223d9f38392.jpg)

![d83b819a4bc823f8458a4ab5d81109529757ba28208aee728e4ff384b5fb2390.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/d83b819a4bc823f8458a4ab5d81109529757ba28208aee728e4ff384b5fb2390.jpg)

![dd0ee7398f8b448a20bd0e7d034a5e5e94bc52fda0e48a7bfee0dde98fac8714.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/dd0ee7398f8b448a20bd0e7d034a5e5e94bc52fda0e48a7bfee0dde98fac8714.jpg)

![dd1e0ef7f1ddacd4e2b45eb6b3766ef2c1d15ab65d1533f74c91b2601b90c23e.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/dd1e0ef7f1ddacd4e2b45eb6b3766ef2c1d15ab65d1533f74c91b2601b90c23e.jpg)

![df7b7edb5e78569c76388f2e98c7e2fa56e4204581d9f03d2f2ecfc71d650749.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/df7b7edb5e78569c76388f2e98c7e2fa56e4204581d9f03d2f2ecfc71d650749.jpg)

![eb733e67a9cb782c05a67037b707925d6d197de34d5512e882099685d5eb2619.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/eb733e67a9cb782c05a67037b707925d6d197de34d5512e882099685d5eb2619.jpg)

![eead16dd5befc9fdeeaeae5b2aceb9be587a95b9fbb2fb497e64d396f69e4eee.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/eead16dd5befc9fdeeaeae5b2aceb9be587a95b9fbb2fb497e64d396f69e4eee.jpg)

![f6290e1a4775246fb0dd8d9052c969bddedcce9bb699865e79423afe30696804.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/f6290e1a4775246fb0dd8d9052c969bddedcce9bb699865e79423afe30696804.jpg)

![fe6650bdd3e7fe0409c757c33903e80a533b5ff430d43170cbeafdd4e2ab4dd8.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/fe6650bdd3e7fe0409c757c33903e80a533b5ff430d43170cbeafdd4e2ab4dd8.jpg)

![fe9da9288adb6f3d4415241bc93cf76a251b6cc79eebaa10e1842ff1a2026b5e.jpg](../acl_results/143_Centurio_%20On%20Drivers%20of%20Multilingual%20Ability%20of%20Large%20Vision-Language%20Model/tables/fe9da9288adb6f3d4415241bc93cf76a251b6cc79eebaa10e1842ff1a2026b5e.jpg)

## Less for More: Enhanced Feedback-aligned MixedLLMs for Molecule Caption Generation and Fine-GrainedNLIEvaluation

### Images

![0533e2e4456621f369e411f1e1ad1c1049b4b926d580d5e08ee19db2181f129f.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/0533e2e4456621f369e411f1e1ad1c1049b4b926d580d5e08ee19db2181f129f.jpg)

![1c11e1db505fa3e66237a0faf306539004a834c0891ec0f2a5641c04069fdafe.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/1c11e1db505fa3e66237a0faf306539004a834c0891ec0f2a5641c04069fdafe.jpg)

![20980f43ddb9cc6664730797cf82430147eb49bcf21c7cd19077637ac1f342d7.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/20980f43ddb9cc6664730797cf82430147eb49bcf21c7cd19077637ac1f342d7.jpg)

![2540135e8d7d8f452377919ec14e32b395a8fc2b3c061eb3f6a43f8fb68356c4.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/2540135e8d7d8f452377919ec14e32b395a8fc2b3c061eb3f6a43f8fb68356c4.jpg)

![2cfc76ae153f170d4d9d8f4bcb287be9816aab592643805018c19140fb441722.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/2cfc76ae153f170d4d9d8f4bcb287be9816aab592643805018c19140fb441722.jpg)

![2d7184e62553a97891ecff8c0d66a16123ef4d08c9a651cde3aa40d8d18c5287.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/2d7184e62553a97891ecff8c0d66a16123ef4d08c9a651cde3aa40d8d18c5287.jpg)

![2f75f10ca66af1580b29e1aed45bcbc2145007a84bc9e2086330dffed5d57e33.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/2f75f10ca66af1580b29e1aed45bcbc2145007a84bc9e2086330dffed5d57e33.jpg)

![63522ac24487ad6ffc276d6355e6340ceb63997a1917f16a326c87e1b5910c6f.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/63522ac24487ad6ffc276d6355e6340ceb63997a1917f16a326c87e1b5910c6f.jpg)

![77475bc05abdcd4c336e951154c073eda6efdfaacdbfdf3386665fe679e5c9b6.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/77475bc05abdcd4c336e951154c073eda6efdfaacdbfdf3386665fe679e5c9b6.jpg)

![877f1583b7cdfd347a9007a8684f4d84df632a746d279c591323dac886d94a6b.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/877f1583b7cdfd347a9007a8684f4d84df632a746d279c591323dac886d94a6b.jpg)

![941e8624ca319473b750b5a09cd7ef786cf5312c455c6c89ba8cad16dd2c53fb.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/941e8624ca319473b750b5a09cd7ef786cf5312c455c6c89ba8cad16dd2c53fb.jpg)

![975dfd7bf1871bc69fa1672baa1ee1b6dc87fa283f23771abe5caf649550efbe.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/975dfd7bf1871bc69fa1672baa1ee1b6dc87fa283f23771abe5caf649550efbe.jpg)

![a7e2ff1293f695a67655ab35bf76771353de24b8fe2be543d2eededa5583da9f.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/a7e2ff1293f695a67655ab35bf76771353de24b8fe2be543d2eededa5583da9f.jpg)

![bf7aed0bf197f0458d271e466718690bd513d481491e0192ffca792417a69564.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/bf7aed0bf197f0458d271e466718690bd513d481491e0192ffca792417a69564.jpg)

![c8de2c5fb09d2a9da46150c2d7f7d0646867e8c5c0ce6fe92c39d88088e8495a.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/c8de2c5fb09d2a9da46150c2d7f7d0646867e8c5c0ce6fe92c39d88088e8495a.jpg)

![e0192847e246ce8cb7354e7c9f41743f3c197d9a0a30551d92e11efaa09fbf9d.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/e0192847e246ce8cb7354e7c9f41743f3c197d9a0a30551d92e11efaa09fbf9d.jpg)

![e5611ef5029ca543ef2c0cd4197a08d6fd86371bddd3a0376257b51fc3c2a75e.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/e5611ef5029ca543ef2c0cd4197a08d6fd86371bddd3a0376257b51fc3c2a75e.jpg)

![f0813649b39c37b6b1c13aca450388e784df47940c24e9edef55339182037197.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/f0813649b39c37b6b1c13aca450388e784df47940c24e9edef55339182037197.jpg)

![f40d8b32dca5780329847e25ea309e7db434f8f52caada4c58217de740a280ee.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/f40d8b32dca5780329847e25ea309e7db434f8f52caada4c58217de740a280ee.jpg)

![f5769923c99d7245c420fdcd506ac9c1f01f653e65701ab09a617794178165e6.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/images/f5769923c99d7245c420fdcd506ac9c1f01f653e65701ab09a617794178165e6.jpg)

### Tables

![10da6d5fe03687634fd04f6ba5c39a672598742cf7e2aa76ce6333c7e0b78124.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/tables/10da6d5fe03687634fd04f6ba5c39a672598742cf7e2aa76ce6333c7e0b78124.jpg)

![29ab7ccddcf68cba8c8285790f8d1a3247d782dfb27c2b5cb58a11ed91d840c8.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/tables/29ab7ccddcf68cba8c8285790f8d1a3247d782dfb27c2b5cb58a11ed91d840c8.jpg)

![4ca8e76a83cb8d93110362d9470073a438e3e0ab82d1b648c4463e9f35737586.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/tables/4ca8e76a83cb8d93110362d9470073a438e3e0ab82d1b648c4463e9f35737586.jpg)

![5f76886efdd5d35efc405dfc8426a1344314b05e28676f5001d6faf3324edb20.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/tables/5f76886efdd5d35efc405dfc8426a1344314b05e28676f5001d6faf3324edb20.jpg)

![b92eaca30b12cac240d704b02c740932146454191ec79ca04e942adca10b23e8.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/tables/b92eaca30b12cac240d704b02c740932146454191ec79ca04e942adca10b23e8.jpg)

![ea8646dbf1e938099af2b9401323ae2671a47a993ea27a5af2108bd5a4ee9715.jpg](../acl_results/144_Less%20for%20More_%20Enhanced%20Feedback-aligned%20MixedLLMs%20for%20Molecule%20Caption%20Generation%20and%20Fine-GrainedN/tables/ea8646dbf1e938099af2b9401323ae2671a47a993ea27a5af2108bd5a4ee9715.jpg)

## Ensemble Watermarks for Large Language Models

### Images

![2abc0f95203edfa4d09c4332a5594829a057bcf540a46e7c0d30d968cf1e53a6.jpg](../acl_results/145_Ensemble%20Watermarks%20for%20Large%20Language%20Models/images/2abc0f95203edfa4d09c4332a5594829a057bcf540a46e7c0d30d968cf1e53a6.jpg)

![3ed0f3e4e2ccabd35892f855a9dca64c67253b14f338be85cb877ae8ed2c1503.jpg](../acl_results/145_Ensemble%20Watermarks%20for%20Large%20Language%20Models/images/3ed0f3e4e2ccabd35892f855a9dca64c67253b14f338be85cb877ae8ed2c1503.jpg)

![59b9391f080a6975ad8aa97014a522dad8fecd1b5305703e785c2b122422e12f.jpg](../acl_results/145_Ensemble%20Watermarks%20for%20Large%20Language%20Models/images/59b9391f080a6975ad8aa97014a522dad8fecd1b5305703e785c2b122422e12f.jpg)

![8336edf44bca81c642496119b2fa1d6280dc50f2bfc849d6072f8aec61d97b1d.jpg](../acl_results/145_Ensemble%20Watermarks%20for%20Large%20Language%20Models/images/8336edf44bca81c642496119b2fa1d6280dc50f2bfc849d6072f8aec61d97b1d.jpg)

![9ddf1efdbec3f28ffe84cab574e4682b26ea9459c5bd1cb698a24b4f7293b823.jpg](../acl_results/145_Ensemble%20Watermarks%20for%20Large%20Language%20Models/images/9ddf1efdbec3f28ffe84cab574e4682b26ea9459c5bd1cb698a24b4f7293b823.jpg)

![d02fcf0a2b69c9176a58fe52d8da6b627ff5cfb0bf0245eea13476e582d2dcef.jpg](../acl_results/145_Ensemble%20Watermarks%20for%20Large%20Language%20Models/images/d02fcf0a2b69c9176a58fe52d8da6b627ff5cfb0bf0245eea13476e582d2dcef.jpg)

![d175e99a460e546df95fa7775f34f9605a70913fb1b4a86666087b2673b43fa1.jpg](../acl_results/145_Ensemble%20Watermarks%20for%20Large%20Language%20Models/images/d175e99a460e546df95fa7775f34f9605a70913fb1b4a86666087b2673b43fa1.jpg)

![d3333459c25ebfd55cc33d740bd3ac732360c62cbca80928a39168a85abc98de.jpg](../acl_results/145_Ensemble%20Watermarks%20for%20Large%20Language%20Models/images/d3333459c25ebfd55cc33d740bd3ac732360c62cbca80928a39168a85abc98de.jpg)

![e05a3003c3077be2907c586760ab4f6bc855b5fe1918c8aa9bae9fd88e5b346b.jpg](../acl_results/145_Ensemble%20Watermarks%20for%20Large%20Language%20Models/images/e05a3003c3077be2907c586760ab4f6bc855b5fe1918c8aa9bae9fd88e5b346b.jpg)

### Tables

![68ec111f9b9871c88f20ed3e66efddacb26ec1d5adac530a03273e9c0de52dba.jpg](../acl_results/145_Ensemble%20Watermarks%20for%20Large%20Language%20Models/tables/68ec111f9b9871c88f20ed3e66efddacb26ec1d5adac530a03273e9c0de52dba.jpg)

![923deae7a211540fd489fe883d440f3432f62c0b016d000800b4adb0bd691e0e.jpg](../acl_results/145_Ensemble%20Watermarks%20for%20Large%20Language%20Models/tables/923deae7a211540fd489fe883d440f3432f62c0b016d000800b4adb0bd691e0e.jpg)

![c82d5154b9fa961d36cfba580299de37e3696febb459c55f72e735e1f0c69b28.jpg](../acl_results/145_Ensemble%20Watermarks%20for%20Large%20Language%20Models/tables/c82d5154b9fa961d36cfba580299de37e3696febb459c55f72e735e1f0c69b28.jpg)

## ConInstruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities

### Images

![6d8996cdbd3db9d8a46a91171cab6becabff28d3fe47c0f10d5f005104bfd9b6.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/images/6d8996cdbd3db9d8a46a91171cab6becabff28d3fe47c0f10d5f005104bfd9b6.jpg)

![6e1d36ca6202b0672eb8db7068983662d93213af396a5250979229a545f0e13f.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/images/6e1d36ca6202b0672eb8db7068983662d93213af396a5250979229a545f0e13f.jpg)

![830576e0834ee2aea8b9121d57716b5a073e410e5ebf4948776b3df8e8338166.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/images/830576e0834ee2aea8b9121d57716b5a073e410e5ebf4948776b3df8e8338166.jpg)

![aed4d87cf185d0a96d2d6d39cb291d2127b76560fe0fa4ca77159acf6f2f89b6.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/images/aed4d87cf185d0a96d2d6d39cb291d2127b76560fe0fa4ca77159acf6f2f89b6.jpg)

![ba3a6a707df3fd60d0c6a47258eac47177d7f7812721e7f17be49134212b802a.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/images/ba3a6a707df3fd60d0c6a47258eac47177d7f7812721e7f17be49134212b802a.jpg)

![c427fa787a9f406528559d4238245f636179eebb03be8e3273df29267cbc965b.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/images/c427fa787a9f406528559d4238245f636179eebb03be8e3273df29267cbc965b.jpg)

![ccebb4d539c7f38c32fe1ff239ab524472bf35e271124e9fa8a4edd366806368.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/images/ccebb4d539c7f38c32fe1ff239ab524472bf35e271124e9fa8a4edd366806368.jpg)

![e98601950c0ccb64b72ffa0cc440325932c596877cf4070aebe4f465f0194516.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/images/e98601950c0ccb64b72ffa0cc440325932c596877cf4070aebe4f465f0194516.jpg)

### Tables

![31cf7f0af145b9098ead31c63a8b3ed7e843f36c4b30b5870baebf1340cffabe.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/tables/31cf7f0af145b9098ead31c63a8b3ed7e843f36c4b30b5870baebf1340cffabe.jpg)

![330cb8ee300c7918d578040efc9f2a70ddef7916cc3c90025e0a59dbaf50c188.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/tables/330cb8ee300c7918d578040efc9f2a70ddef7916cc3c90025e0a59dbaf50c188.jpg)

![5534cd8a4a62537d03259c9329771f8fa27437af50c93f96c6d0d72b518c82cf.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/tables/5534cd8a4a62537d03259c9329771f8fa27437af50c93f96c6d0d72b518c82cf.jpg)

![7d14228a037a0ce2ac0faf1f2cef1687df9939597961e5706ca0006fd1a25fa3.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/tables/7d14228a037a0ce2ac0faf1f2cef1687df9939597961e5706ca0006fd1a25fa3.jpg)

![8ee0a8eae7cb6f5f447c31dc55e8b8d2048de86c1cbc9e434defeb913d3753f0.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/tables/8ee0a8eae7cb6f5f447c31dc55e8b8d2048de86c1cbc9e434defeb913d3753f0.jpg)

![a57934db691d244f6402b2aa2fa445cc1dd7c71301a235fd1ea6c8c7e159b556.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/tables/a57934db691d244f6402b2aa2fa445cc1dd7c71301a235fd1ea6c8c7e159b556.jpg)

![ad715e0936d3a95da3ad07293635a58f42d5b1acc0fa40551d565de9a27ca45b.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/tables/ad715e0936d3a95da3ad07293635a58f42d5b1acc0fa40551d565de9a27ca45b.jpg)

![bcf927668d7c41af8983037438e0fb6db73434cea63df888f2e447577a9984cc.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/tables/bcf927668d7c41af8983037438e0fb6db73434cea63df888f2e447577a9984cc.jpg)

![f6835b0d885858b950f0ec46305d7fe3681a86b74d18861278e75810ddc9059c.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/tables/f6835b0d885858b950f0ec46305d7fe3681a86b74d18861278e75810ddc9059c.jpg)

![fc8b8b43f34eddf9c0e9e8ba1d54845234c50e67bad04e01228b4941eae4eb65.jpg](../acl_results/146_ConInstruction_%20Universal%20Jailbreaking%20of%20Multimodal%20Large%20Language%20Models%20via%20Non-Textual%20Modalitie/tables/fc8b8b43f34eddf9c0e9e8ba1d54845234c50e67bad04e01228b4941eae4eb65.jpg)

## TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning forLLM-as-a-Judge

### Images

![4eff526b94d245041a1fa4fccfed19482564d563a47f93ecb5108fc249eef49c.jpg](../acl_results/147_TRACT_%20Regression-Aware%20Fine-tuning%20Meets%20Chain-of-Thought%20Reasoning%20forLLM-as-a-Judge/images/4eff526b94d245041a1fa4fccfed19482564d563a47f93ecb5108fc249eef49c.jpg)

![e33624b1b2533345cfc138a907adda19fcd5d7965bcd81d357fd3fac85cf5ff9.jpg](../acl_results/147_TRACT_%20Regression-Aware%20Fine-tuning%20Meets%20Chain-of-Thought%20Reasoning%20forLLM-as-a-Judge/images/e33624b1b2533345cfc138a907adda19fcd5d7965bcd81d357fd3fac85cf5ff9.jpg)

![f2b040cc8d627825549848dcea9ed4f54a45b29d8cf9afd431ab8809463c1289.jpg](../acl_results/147_TRACT_%20Regression-Aware%20Fine-tuning%20Meets%20Chain-of-Thought%20Reasoning%20forLLM-as-a-Judge/images/f2b040cc8d627825549848dcea9ed4f54a45b29d8cf9afd431ab8809463c1289.jpg)

### Tables

![23cdfbdbefe486382acded9f92e0727c6f37d2e6bbeb64cd03bc15d6f3c0f640.jpg](../acl_results/147_TRACT_%20Regression-Aware%20Fine-tuning%20Meets%20Chain-of-Thought%20Reasoning%20forLLM-as-a-Judge/tables/23cdfbdbefe486382acded9f92e0727c6f37d2e6bbeb64cd03bc15d6f3c0f640.jpg)

![2fdeb3feb0d51583685511deb9fbc9723033bca9f1f883da23a61f863cf5f4bc.jpg](../acl_results/147_TRACT_%20Regression-Aware%20Fine-tuning%20Meets%20Chain-of-Thought%20Reasoning%20forLLM-as-a-Judge/tables/2fdeb3feb0d51583685511deb9fbc9723033bca9f1f883da23a61f863cf5f4bc.jpg)

![60e0a1d449b5c8a77f8b7258d5b701e0c1f868235de2b3bbafa4eb5fae5875c2.jpg](../acl_results/147_TRACT_%20Regression-Aware%20Fine-tuning%20Meets%20Chain-of-Thought%20Reasoning%20forLLM-as-a-Judge/tables/60e0a1d449b5c8a77f8b7258d5b701e0c1f868235de2b3bbafa4eb5fae5875c2.jpg)

![6f21e048211206f35bf93983900e11a74972d7ff62163de942db5e314e1c8b03.jpg](../acl_results/147_TRACT_%20Regression-Aware%20Fine-tuning%20Meets%20Chain-of-Thought%20Reasoning%20forLLM-as-a-Judge/tables/6f21e048211206f35bf93983900e11a74972d7ff62163de942db5e314e1c8b03.jpg)

![8e5feb37b1ae5ad9c96c16b41ef1272b6a49df8fd9015e1666533056161c18a7.jpg](../acl_results/147_TRACT_%20Regression-Aware%20Fine-tuning%20Meets%20Chain-of-Thought%20Reasoning%20forLLM-as-a-Judge/tables/8e5feb37b1ae5ad9c96c16b41ef1272b6a49df8fd9015e1666533056161c18a7.jpg)

![90b1fd22ce436a7170e2232f21753c226f9f8a2fce786f9acc6f27c7bba44668.jpg](../acl_results/147_TRACT_%20Regression-Aware%20Fine-tuning%20Meets%20Chain-of-Thought%20Reasoning%20forLLM-as-a-Judge/tables/90b1fd22ce436a7170e2232f21753c226f9f8a2fce786f9acc6f27c7bba44668.jpg)

![a636a53a16467913b90bca38a0e80350c8eac27296b61e6890458f307d1eb283.jpg](../acl_results/147_TRACT_%20Regression-Aware%20Fine-tuning%20Meets%20Chain-of-Thought%20Reasoning%20forLLM-as-a-Judge/tables/a636a53a16467913b90bca38a0e80350c8eac27296b61e6890458f307d1eb283.jpg)

![ae57668a3968e9867d62890083bbfd78eb35932470d8fb0282fde4227f4b3a3e.jpg](../acl_results/147_TRACT_%20Regression-Aware%20Fine-tuning%20Meets%20Chain-of-Thought%20Reasoning%20forLLM-as-a-Judge/tables/ae57668a3968e9867d62890083bbfd78eb35932470d8fb0282fde4227f4b3a3e.jpg)

![b0a48bab37ac0fe849d83a9d6272ef00240509e8711e2cf2d35a8a30f75851a0.jpg](../acl_results/147_TRACT_%20Regression-Aware%20Fine-tuning%20Meets%20Chain-of-Thought%20Reasoning%20forLLM-as-a-Judge/tables/b0a48bab37ac0fe849d83a9d6272ef00240509e8711e2cf2d35a8a30f75851a0.jpg)

## DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization for Dynamic Retrieval-Augmented Generation

### Images

![36bbd97571fdc8dd82fe6a8044d85e5e89cd50630a5eb6b740b75c90ae0ac078.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/images/36bbd97571fdc8dd82fe6a8044d85e5e89cd50630a5eb6b740b75c90ae0ac078.jpg)

![47455868027ed39851c06abfef4f9a2e9c4edfd92b9314cf0c3c42b6ad4727fa.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/images/47455868027ed39851c06abfef4f9a2e9c4edfd92b9314cf0c3c42b6ad4727fa.jpg)

![52d1c3924fc9c210dc82f132fbbae8a1746e93d042eb53f9e3fc8538da5e80b6.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/images/52d1c3924fc9c210dc82f132fbbae8a1746e93d042eb53f9e3fc8538da5e80b6.jpg)

![657c6d2e6c68ef417050ab49423d712a98513e7bc11348d4205232166a2e7f8b.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/images/657c6d2e6c68ef417050ab49423d712a98513e7bc11348d4205232166a2e7f8b.jpg)

![6859446229bb0112b758c894be2561bca1ef1ac3b7198a5307bf78e6727659f1.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/images/6859446229bb0112b758c894be2561bca1ef1ac3b7198a5307bf78e6727659f1.jpg)

![bb0d172810cc06e0271d96549e51845c82e1282da301da47489d5421fa95ba99.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/images/bb0d172810cc06e0271d96549e51845c82e1282da301da47489d5421fa95ba99.jpg)

![cfbea4450b0010b2c91134b51cb179fc4107494b6d9e9226ac2de39429727057.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/images/cfbea4450b0010b2c91134b51cb179fc4107494b6d9e9226ac2de39429727057.jpg)

![e19003f14c4b8a8e82b8735daa50f02cd9b4978767b6313343c2a1c5b91751e4.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/images/e19003f14c4b8a8e82b8735daa50f02cd9b4978767b6313343c2a1c5b91751e4.jpg)

![fcca50341f78a6d2f1af487062c55fac5cecec651219f20706e28b8ba3e4104d.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/images/fcca50341f78a6d2f1af487062c55fac5cecec651219f20706e28b8ba3e4104d.jpg)

### Tables

![0a02ade68c81e1b44dc2c2edc3d11fe09f0a1b6743d6e94a46b6015d8b8d2ebd.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/tables/0a02ade68c81e1b44dc2c2edc3d11fe09f0a1b6743d6e94a46b6015d8b8d2ebd.jpg)

![315a8b9f872de525a373f5a7f87d4e47d6537ba04ea286b009ffbcb81eae1d63.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/tables/315a8b9f872de525a373f5a7f87d4e47d6537ba04ea286b009ffbcb81eae1d63.jpg)

![3e4df2f01f87ea80da0820bd5a1c7c0cb774e5064ba26c1b9263663e0ede37b0.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/tables/3e4df2f01f87ea80da0820bd5a1c7c0cb774e5064ba26c1b9263663e0ede37b0.jpg)

![961d18fba16cfe9e9e48f52b638e065c8343bab82b6599bfa2bdcdd5c548665f.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/tables/961d18fba16cfe9e9e48f52b638e065c8343bab82b6599bfa2bdcdd5c548665f.jpg)

![b8a65bc726625d30a2b1482c8505284b9ec25a6b8e676e142ddf6d34e2185a39.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/tables/b8a65bc726625d30a2b1482c8505284b9ec25a6b8e676e142ddf6d34e2185a39.jpg)

![ca8de82408075f98e961a677b93fc3ade20cee3d8c2a8279da3b26ddc0da5e5f.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/tables/ca8de82408075f98e961a677b93fc3ade20cee3d8c2a8279da3b26ddc0da5e5f.jpg)

![cf57e50f40309ddf6da37fb4f7530eb2b8872be6989a479acacc93e2ea944ca5.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/tables/cf57e50f40309ddf6da37fb4f7530eb2b8872be6989a479acacc93e2ea944ca5.jpg)

![fa6425854b239f7cd459c7cef416e3673b986daa24cef7235931dd646492e3ef.jpg](../acl_results/148_DioR_%20Adaptive%20Cognitive%20Detection%20and%20Contextual%20Retrieval%20Optimization%20for%20Dynamic%20Retrieval-Augme/tables/fa6425854b239f7cd459c7cef416e3673b986daa24cef7235931dd646492e3ef.jpg)

## Unveiling the Power of Source: Source-based MinimumBayes Risk Decoding for Neural Machine Translation

### Images

![536f724bd1f08a4dd7cd8450e34410b57bf6aecb1bb30ed392e28d875575c18b.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/images/536f724bd1f08a4dd7cd8450e34410b57bf6aecb1bb30ed392e28d875575c18b.jpg)

![d60b9465eba1d543acc54f0820fb639e78c8845c3a88858154908a3fa4de6d09.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/images/d60b9465eba1d543acc54f0820fb639e78c8845c3a88858154908a3fa4de6d09.jpg)

![e29566b8b03ec2957be344d807d18ab6f2af26720ab935dcda8b169e78522124.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/images/e29566b8b03ec2957be344d807d18ab6f2af26720ab935dcda8b169e78522124.jpg)

### Tables

![3a99dd9966ac8d85e9a36d3ed5acb78ba4a34f9c3352aa90fcf6e02c69fd17f2.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/3a99dd9966ac8d85e9a36d3ed5acb78ba4a34f9c3352aa90fcf6e02c69fd17f2.jpg)

![3d0abbcee77882f3fdf51e34bdf3061e98b87d92840bf18481af0eaac3ac9933.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/3d0abbcee77882f3fdf51e34bdf3061e98b87d92840bf18481af0eaac3ac9933.jpg)

![3e46886891b4ed15dc8d6ef91ea9f78c636236b448c1cbe56832ea67d96aa109.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/3e46886891b4ed15dc8d6ef91ea9f78c636236b448c1cbe56832ea67d96aa109.jpg)

![50c657d29d9c7e2e5b2d66eb4305e2c9bf5bee3e22d5c836d20c5215445fec3c.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/50c657d29d9c7e2e5b2d66eb4305e2c9bf5bee3e22d5c836d20c5215445fec3c.jpg)

![88605fa0128fbc139d5c3d1d45bccc4ad334bc76737fcf67c21caab0344ea5f9.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/88605fa0128fbc139d5c3d1d45bccc4ad334bc76737fcf67c21caab0344ea5f9.jpg)

![91c783f615306e83c817e3095b631eca9faae97ca73a0610b828394bfdfa2d5a.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/91c783f615306e83c817e3095b631eca9faae97ca73a0610b828394bfdfa2d5a.jpg)

![9e06898d46a069a8b3755e28f3a242d5d19e630f8168c3a0b8e577f56181c796.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/9e06898d46a069a8b3755e28f3a242d5d19e630f8168c3a0b8e577f56181c796.jpg)

![ae9ab61ee0265addd6815504d5a9c9c0c0035d9ad52964467e5c503604b8a20b.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/ae9ab61ee0265addd6815504d5a9c9c0c0035d9ad52964467e5c503604b8a20b.jpg)

![c60f72aa5419293afb6dd672a777894408ddc85278a7cb94eb0b0fd729114d36.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/c60f72aa5419293afb6dd672a777894408ddc85278a7cb94eb0b0fd729114d36.jpg)

![d05ad4057caa31d3de04edc040a0e94f5533be0ac750544bba17fcefc7dd9ca1.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/d05ad4057caa31d3de04edc040a0e94f5533be0ac750544bba17fcefc7dd9ca1.jpg)

![db09da45bd7a8d554772c97d516132aabe5640fe78b92a154b6ae6ccf21d8e9d.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/db09da45bd7a8d554772c97d516132aabe5640fe78b92a154b6ae6ccf21d8e9d.jpg)

![e0bd3e2ad794d065ae445f346dc4331500ddde2e9d9fc7e994fcad27f7f42a29.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/e0bd3e2ad794d065ae445f346dc4331500ddde2e9d9fc7e994fcad27f7f42a29.jpg)

![e5cbabc5f41edbefdd7acc97c7302544c0f92b23c24d20583819c3061a00811b.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/e5cbabc5f41edbefdd7acc97c7302544c0f92b23c24d20583819c3061a00811b.jpg)

![f193f07be5eeb1f405187cf154aa2c25b43332d6f7962151edc23dd02ed58056.jpg](../acl_results/149_Unveiling%20the%20Power%20of%20Source_%20Source-based%20MinimumBayes%20Risk%20Decoding%20for%20Neural%20Machine%20Translatio/tables/f193f07be5eeb1f405187cf154aa2c25b43332d6f7962151edc23dd02ed58056.jpg)

## ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use

### Images

![1c8399402bfb5153eb1ef994e1ab5fa5ddad6c7712a1694ed2ad73728858defa.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/images/1c8399402bfb5153eb1ef994e1ab5fa5ddad6c7712a1694ed2ad73728858defa.jpg)

![1edce15cea8907e78b6e2b29fbf98089aa146c7f778cf45703fd972927bfd33a.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/images/1edce15cea8907e78b6e2b29fbf98089aa146c7f778cf45703fd972927bfd33a.jpg)

![66571eabcf5cef96b08f17ccc01315f6daba59c50f2236c67f6cbeee152dd16c.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/images/66571eabcf5cef96b08f17ccc01315f6daba59c50f2236c67f6cbeee152dd16c.jpg)

![6fa66c9b88a2d266e8391a54409442d4b89231536123111d0000574c607287e9.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/images/6fa66c9b88a2d266e8391a54409442d4b89231536123111d0000574c607287e9.jpg)

![937534d43a4fdedc709164aa0272d644d04a78404482670d8a94742487dcd6e5.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/images/937534d43a4fdedc709164aa0272d644d04a78404482670d8a94742487dcd6e5.jpg)

![b6407f3a194d2d6ebbb36f5b9b34c80938ad198568ff4f1ec8efda69e916acaa.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/images/b6407f3a194d2d6ebbb36f5b9b34c80938ad198568ff4f1ec8efda69e916acaa.jpg)

![eb6d8ea5bcf2759d4f423428cc40e2aa0bef5967b243f1e7c534acdd1f8be1ff.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/images/eb6d8ea5bcf2759d4f423428cc40e2aa0bef5967b243f1e7c534acdd1f8be1ff.jpg)

![f2fb10aff40779285f13ec2f8b35f4b16f7b02ee9e4c182fdc179a7419b169c3.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/images/f2fb10aff40779285f13ec2f8b35f4b16f7b02ee9e4c182fdc179a7419b169c3.jpg)

### Tables

![0111af3a635f28905d0b8147ac4c63e0311fc2802c487b41c9ddab9d229c2a51.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/tables/0111af3a635f28905d0b8147ac4c63e0311fc2802c487b41c9ddab9d229c2a51.jpg)

![0d4bcd5f105ff404b560becebb316c3c549b02ff3ca82e660708a2c3590f02d3.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/tables/0d4bcd5f105ff404b560becebb316c3c549b02ff3ca82e660708a2c3590f02d3.jpg)

![25e67e2c7b070e7818af1eaf693804e0495fee86e1ddf536bd0b366e25286aec.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/tables/25e67e2c7b070e7818af1eaf693804e0495fee86e1ddf536bd0b366e25286aec.jpg)

![b57007471107c20612268c1c8d2279a1f9032b8733264eff23e6fea2fab4528d.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/tables/b57007471107c20612268c1c8d2279a1f9032b8733264eff23e6fea2fab4528d.jpg)

![c5e8ec45e9cebb1986d0e0f38f1f1ac338d1760deff2243a1668d263a2a012fe.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/tables/c5e8ec45e9cebb1986d0e0f38f1f1ac338d1760deff2243a1668d263a2a012fe.jpg)

![c72767044916f4b4d3c7cf3b503ebf9ed4ac26abf6f6eff5531f280627db3d63.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/tables/c72767044916f4b4d3c7cf3b503ebf9ed4ac26abf6f6eff5531f280627db3d63.jpg)

![e3bd80ee6c148531b0713aedd6e61b4994d7be5490dcf51e81d6d7d333a7f872.jpg](../acl_results/150_ToolHop_%20A%20Query-Driven%20Benchmark%20for%20Evaluating%20Large%20Language%20Models%20in%20Multi-Hop%20Tool%20Use/tables/e3bd80ee6c148531b0713aedd6e61b4994d7be5490dcf51e81d6d7d333a7f872.jpg)

## Mixture of insighTful Experts (MoTE): The Synergy of Reasoning Chains and Expert Mixtures in Self-Alignment

### Images

![02c4ed90b23405bb981c0d3ebf242235d5ee83d7f211486c076a929e38743708.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/images/02c4ed90b23405bb981c0d3ebf242235d5ee83d7f211486c076a929e38743708.jpg)

![27cc8c3800a5f850731c013cbfbc49c20d79a343ead383c81031592e12e28ef2.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/images/27cc8c3800a5f850731c013cbfbc49c20d79a343ead383c81031592e12e28ef2.jpg)

![5dd39446a967ea491cdec2d8a9171a46fd2c9f2cb1fed6bc6b42c1ed46e9585e.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/images/5dd39446a967ea491cdec2d8a9171a46fd2c9f2cb1fed6bc6b42c1ed46e9585e.jpg)

![77c6e4cb26397ae2f709f2a1e8d22e4f8c7a5084e6652a6676fdd8ca52dcdfd3.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/images/77c6e4cb26397ae2f709f2a1e8d22e4f8c7a5084e6652a6676fdd8ca52dcdfd3.jpg)

![b2acd7e55a677c7c2b7088b5de231fbc39eb3d49d06b4b698e97ed54fe092d89.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/images/b2acd7e55a677c7c2b7088b5de231fbc39eb3d49d06b4b698e97ed54fe092d89.jpg)

![cca04f49c8e7cb27b9f95dcc5e85aef6831daf4c4d7926f90fb8ef1eeb66d862.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/images/cca04f49c8e7cb27b9f95dcc5e85aef6831daf4c4d7926f90fb8ef1eeb66d862.jpg)

### Tables

![2d32cce8b606db5f402883f97f1ea555f95b3fdb4734b01fb59848c4ed5b9170.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/tables/2d32cce8b606db5f402883f97f1ea555f95b3fdb4734b01fb59848c4ed5b9170.jpg)

![2fae0bf587e2adc2ceeb8cdeff4ac4c5c4224a01a3c6d39316d0f9d8af8f1e80.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/tables/2fae0bf587e2adc2ceeb8cdeff4ac4c5c4224a01a3c6d39316d0f9d8af8f1e80.jpg)

![37024376e29f8da72a3a08bc13f465151be998f8753a7fa787f4d9ab1f8b71cc.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/tables/37024376e29f8da72a3a08bc13f465151be998f8753a7fa787f4d9ab1f8b71cc.jpg)

![4df645eb99cd3c2e536cb6a3c4c78686e198f280bb28951b1a62e87837c1a695.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/tables/4df645eb99cd3c2e536cb6a3c4c78686e198f280bb28951b1a62e87837c1a695.jpg)

![5f1e6a5494a9aa4830a11babd65e58db93a73f49fc50a7bd92e6fd557d8fa82b.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/tables/5f1e6a5494a9aa4830a11babd65e58db93a73f49fc50a7bd92e6fd557d8fa82b.jpg)

![6602a7018c36830d02875c94c2cc57da7109ef6842a187aa33ba9e9857fdf77a.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/tables/6602a7018c36830d02875c94c2cc57da7109ef6842a187aa33ba9e9857fdf77a.jpg)

![8081316da19ba97bdf88382c99280f6165538726d8cd5c62e6f83d254526868c.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/tables/8081316da19ba97bdf88382c99280f6165538726d8cd5c62e6f83d254526868c.jpg)

![858e29682af2667bcdd2419af3471ff7ef051b8707e0744cfcc08f880daaa631.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/tables/858e29682af2667bcdd2419af3471ff7ef051b8707e0744cfcc08f880daaa631.jpg)

![8c6e8a69d205a4b9cbf3f7cb2fe0121517c9379d3a730b4dce746a10587301ba.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/tables/8c6e8a69d205a4b9cbf3f7cb2fe0121517c9379d3a730b4dce746a10587301ba.jpg)

![ce75c928a6724dc2dd89bd42336faeca5ddc39dd3d4f4601fba1581a023df5a5.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/tables/ce75c928a6724dc2dd89bd42336faeca5ddc39dd3d4f4601fba1581a023df5a5.jpg)

![e30291e3ab6cbda28bcb4820d8331de6debffd79da984954c2d1bbef4440e76c.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/tables/e30291e3ab6cbda28bcb4820d8331de6debffd79da984954c2d1bbef4440e76c.jpg)

![f855e4ebe653fc8203c21840518472b595d2ea60c31572f1660cc2e8da8eb8bc.jpg](../acl_results/151_Mixture%20of%20insighTful%20Experts%20%28MoTE)_ The Synergy of Reasoning Chains and Expert Mixtures in Self-Al/tables/f855e4ebe653fc8203c21840518472b595d2ea60c31572f1660cc2e8da8eb8bc.jpg)

## MAPS: Motivation-Aware Personalized Search viaLLM-Driven Consultation Alignment

### Images

![04f9fb15be093de86f469b5b72056383b3705ff221fab3305ee6bf23499c477a.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/images/04f9fb15be093de86f469b5b72056383b3705ff221fab3305ee6bf23499c477a.jpg)

![05576450885ab876cd6b8e03b4ac597130e798d28c9e5c8548d30eb8347303f4.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/images/05576450885ab876cd6b8e03b4ac597130e798d28c9e5c8548d30eb8347303f4.jpg)

![155b58ab731287c18eb28011d88378aa1d7324f63ab4f7bd1e5095c95ab24353.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/images/155b58ab731287c18eb28011d88378aa1d7324f63ab4f7bd1e5095c95ab24353.jpg)

![5967c97891185880dc7867c2e3e9f426cb32ee146ef9a94cfc956e984f429f82.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/images/5967c97891185880dc7867c2e3e9f426cb32ee146ef9a94cfc956e984f429f82.jpg)

![a22b77029c872cebf64f08e2e2ae7a08596f16c2b7ba17ae406f1abc355dca6c.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/images/a22b77029c872cebf64f08e2e2ae7a08596f16c2b7ba17ae406f1abc355dca6c.jpg)

![a48b2b440c80043e18b88a757d9bc720f589d719a25bb5796fa01f5917fd6066.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/images/a48b2b440c80043e18b88a757d9bc720f589d719a25bb5796fa01f5917fd6066.jpg)

![ea76e5d8338d2ff97bc6455a9d6a818a2f6680c87b070280e51a954f7460c529.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/images/ea76e5d8338d2ff97bc6455a9d6a818a2f6680c87b070280e51a954f7460c529.jpg)

### Tables

![010642b57d02afeb7485635a6de346d0143531462f980e8b719e0030b9efe275.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/tables/010642b57d02afeb7485635a6de346d0143531462f980e8b719e0030b9efe275.jpg)

![07037c3144aecf84df33b5034b8fbb38d2c435b50fafff5a0aaa4843a808d1f8.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/tables/07037c3144aecf84df33b5034b8fbb38d2c435b50fafff5a0aaa4843a808d1f8.jpg)

![34126879a64aa803771a8f590e11f5c135dcb15b0d7d75e48de85b4734823115.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/tables/34126879a64aa803771a8f590e11f5c135dcb15b0d7d75e48de85b4734823115.jpg)

![3bbff36cf17dde4b341712b2f05cb904dbd4df098cf037b5d60c274471a3ee52.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/tables/3bbff36cf17dde4b341712b2f05cb904dbd4df098cf037b5d60c274471a3ee52.jpg)

![50e7cb1a60091c09397d02bfd5daa6992ceee1e4fd1e283a5443b3e8db3ab46b.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/tables/50e7cb1a60091c09397d02bfd5daa6992ceee1e4fd1e283a5443b3e8db3ab46b.jpg)

![59c274bdbb0721d2d60f10f8c7fc5d1b47f6b592a92321f3d214777965e0a812.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/tables/59c274bdbb0721d2d60f10f8c7fc5d1b47f6b592a92321f3d214777965e0a812.jpg)

![c97a0ad62e1ee178a3146439e652b29fb078d387831608e4e2289e92bd220b47.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/tables/c97a0ad62e1ee178a3146439e652b29fb078d387831608e4e2289e92bd220b47.jpg)

![ee886174e23806c35efd53d67539955ea82f5ca123adade3451c533929119927.jpg](../acl_results/152_MAPS_%20Motivation-Aware%20Personalized%20Search%20viaLLM-Driven%20Consultation%20Alignment/tables/ee886174e23806c35efd53d67539955ea82f5ca123adade3451c533929119927.jpg)

## Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework

### Images

![0cf9659342c69445188628c4b5c5f90dc832fe987c3aa3c26b6b4b30d1970242.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/images/0cf9659342c69445188628c4b5c5f90dc832fe987c3aa3c26b6b4b30d1970242.jpg)

![0dc15bbad61ab1172d99ad3478a861624bfff40d3a079e9512e5953a303fa516.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/images/0dc15bbad61ab1172d99ad3478a861624bfff40d3a079e9512e5953a303fa516.jpg)

![4f7e684546c2764e0d9584e4c72d0aea913bf0c1fc9d04c3c0eb54eff7ae0af0.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/images/4f7e684546c2764e0d9584e4c72d0aea913bf0c1fc9d04c3c0eb54eff7ae0af0.jpg)

![7dcf229c9d3492b7c5ab32c6a528e65158c7bfbbfbd047a7050e9ff9ec84cd65.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/images/7dcf229c9d3492b7c5ab32c6a528e65158c7bfbbfbd047a7050e9ff9ec84cd65.jpg)

![c1f6a6208497e10123e0e14c62e3ebb7483e1f0c84e0fd555bfcf07c71c7e7e3.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/images/c1f6a6208497e10123e0e14c62e3ebb7483e1f0c84e0fd555bfcf07c71c7e7e3.jpg)

![cf83ffd1a54057271b865ea64bb5dae86f14620f87c7466fd6fbc2d3a640dce8.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/images/cf83ffd1a54057271b865ea64bb5dae86f14620f87c7466fd6fbc2d3a640dce8.jpg)

![e851d5d2f3947ca5b9231a29e33a9bbc7e0d4290024902fcb457b85e26efaa0a.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/images/e851d5d2f3947ca5b9231a29e33a9bbc7e0d4290024902fcb457b85e26efaa0a.jpg)

![fad03530725b224fa2bfb2c367a4c4a0b44e7fdacd98d0350f6c20675a4b78dc.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/images/fad03530725b224fa2bfb2c367a4c4a0b44e7fdacd98d0350f6c20675a4b78dc.jpg)

### Tables

![3af1e562f4c0be69138de38c5bfa76da93a9a34e644b5768b41e59a891a2f640.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/tables/3af1e562f4c0be69138de38c5bfa76da93a9a34e644b5768b41e59a891a2f640.jpg)

![7134f011d9789266884ea5fc45b95e061a8dfe817601785da312be821764b0a3.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/tables/7134f011d9789266884ea5fc45b95e061a8dfe817601785da312be821764b0a3.jpg)

![9308df95949626ebc1f6b8db03130e8ecfcc365fa6b92e4a936ccdad6b956775.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/tables/9308df95949626ebc1f6b8db03130e8ecfcc365fa6b92e4a936ccdad6b956775.jpg)

![a8a017f252e002c7b054458716f0f72ab574cdc25390e50ba4f0167a277d1c19.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/tables/a8a017f252e002c7b054458716f0f72ab574cdc25390e50ba4f0167a277d1c19.jpg)

![c8cbd3cdcb8c2e5b80ff581488d0592d660560e43492431fe19b11b6787c9daa.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/tables/c8cbd3cdcb8c2e5b80ff581488d0592d660560e43492431fe19b11b6787c9daa.jpg)

![de55316158a0d64d6f9f8f370c673fafe2b99f147e8d8d6d1fb2ed08b8771789.jpg](../acl_results/153_Aristotle_%20Mastering%20Logical%20Reasoning%20with%20A%20Logic-Complete%20Decompose-Search-Resolve%20Framework/tables/de55316158a0d64d6f9f8f370c673fafe2b99f147e8d8d6d1fb2ed08b8771789.jpg)

## LADM: Long-context Training Data Selection with Attention-based Dependency Measurement forLLMs

### Images

![0c7c414952c2dbcb54b89b18e51c3e9e41fc7f715346aa9eed75a93ee912757d.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/images/0c7c414952c2dbcb54b89b18e51c3e9e41fc7f715346aa9eed75a93ee912757d.jpg)

![34227a14418ca56ae6125d537bddb999b5bec1e3b42f0e0dd77c1bb6767efe4a.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/images/34227a14418ca56ae6125d537bddb999b5bec1e3b42f0e0dd77c1bb6767efe4a.jpg)

![36bd0dff5d81d97c4c483484f32b632d8551cde7411f3f38b6209836343d28b9.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/images/36bd0dff5d81d97c4c483484f32b632d8551cde7411f3f38b6209836343d28b9.jpg)

![7420bfceaa3eb70e672e6bbc6e5574793fc7f40043f68878e2a257cfe206c7b5.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/images/7420bfceaa3eb70e672e6bbc6e5574793fc7f40043f68878e2a257cfe206c7b5.jpg)

![75e9c6e92f28211453af5b101473ef68cc790b5e71bc33d61967e831b097a08f.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/images/75e9c6e92f28211453af5b101473ef68cc790b5e71bc33d61967e831b097a08f.jpg)

![9fe131506c9bbfa4e8f552ef00a7acff9e1928a64f7cbe9f6d0235e2763c18b2.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/images/9fe131506c9bbfa4e8f552ef00a7acff9e1928a64f7cbe9f6d0235e2763c18b2.jpg)

![b6615b60c10bca9a6d060699667d1ec1311f2df15e6fa56dd62aa0cf028c848c.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/images/b6615b60c10bca9a6d060699667d1ec1311f2df15e6fa56dd62aa0cf028c848c.jpg)

![dd3049e77aa672b27b09f85fedee5225f7cf337f38efacc675617eac00809647.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/images/dd3049e77aa672b27b09f85fedee5225f7cf337f38efacc675617eac00809647.jpg)

![ee5c1254ed59bdd89defcd1bd0d6d5f2bd3e54398cd8cbaa2547a0f3c610df4a.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/images/ee5c1254ed59bdd89defcd1bd0d6d5f2bd3e54398cd8cbaa2547a0f3c610df4a.jpg)

### Tables

![073c4fd5523993af0aafb76392a5c987576f18d2d338e370e8a256e3e0023213.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/tables/073c4fd5523993af0aafb76392a5c987576f18d2d338e370e8a256e3e0023213.jpg)

![09fa8d12427bbcb8527e14faefb6f2f5868aa80c909871dcbb99d0c17b536e8a.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/tables/09fa8d12427bbcb8527e14faefb6f2f5868aa80c909871dcbb99d0c17b536e8a.jpg)

![168c400b8ea67b0a601432de21e67d6cddaec240c7867adf188bd753aeea0934.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/tables/168c400b8ea67b0a601432de21e67d6cddaec240c7867adf188bd753aeea0934.jpg)

![20285e42ff792ee8e71420f46533cf46c09486cb693dec824fcb2821035f59a8.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/tables/20285e42ff792ee8e71420f46533cf46c09486cb693dec824fcb2821035f59a8.jpg)

![27ce872d7fb911abf66acbd455c8e00c9f38433ce2d771bc4232d115fc11acd1.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/tables/27ce872d7fb911abf66acbd455c8e00c9f38433ce2d771bc4232d115fc11acd1.jpg)

![8003d188fff469e2b26aedad2aa8c8beace8ce6bc231916305ffac4b6d73e4ed.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/tables/8003d188fff469e2b26aedad2aa8c8beace8ce6bc231916305ffac4b6d73e4ed.jpg)

![8c1760806b2998a3f5cb4cce3023bcd0908a6b0f9baa7bd5e9bd21596d1e7ac2.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/tables/8c1760806b2998a3f5cb4cce3023bcd0908a6b0f9baa7bd5e9bd21596d1e7ac2.jpg)

![90c9e267fc7a76eeef7792dc38d0a91d7cff5a26aec34a1e5dc6d4d70fff5e50.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/tables/90c9e267fc7a76eeef7792dc38d0a91d7cff5a26aec34a1e5dc6d4d70fff5e50.jpg)

![9fe2f5b1527f5fb1bf230cf6d3fed977f0143e068bce77eb5ca884e84b170d0a.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/tables/9fe2f5b1527f5fb1bf230cf6d3fed977f0143e068bce77eb5ca884e84b170d0a.jpg)

![ce4b671e7124310642dd9a985cb05d1b078d4a4f810f4369247a426e58ac9a23.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/tables/ce4b671e7124310642dd9a985cb05d1b078d4a4f810f4369247a426e58ac9a23.jpg)

![d4137d8c8f6a8b1663aeffe3cdda61b7cd284c9ec635444fc278d453c453642c.jpg](../acl_results/154_LADM_%20Long-context%20Training%20Data%20Selection%20with%20Attention-based%20Dependency%20Measurement%20forLLMs/tables/d4137d8c8f6a8b1663aeffe3cdda61b7cd284c9ec635444fc278d453c453642c.jpg)

## Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training

### Images

![1c99bd24efcec78dcbfe2bf5e1851af715caeaf37c7a85ae3365575b1a61df85.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/images/1c99bd24efcec78dcbfe2bf5e1851af715caeaf37c7a85ae3365575b1a61df85.jpg)

![727bb75dca5c34ae2f2d2aa7f17dbc5dca161ed3cbf1fd08d6e211044fc63af8.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/images/727bb75dca5c34ae2f2d2aa7f17dbc5dca161ed3cbf1fd08d6e211044fc63af8.jpg)

![827d38a8d30a7d4dee748be47411092d00fc19f7f00a5fc47224d258022aab54.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/images/827d38a8d30a7d4dee748be47411092d00fc19f7f00a5fc47224d258022aab54.jpg)

![b7e22e674dc57fc4a8e024de65f6c89ad29efda21bd087b4f85e91542f3c2b38.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/images/b7e22e674dc57fc4a8e024de65f6c89ad29efda21bd087b4f85e91542f3c2b38.jpg)

### Tables

![2cd97db990db3f57e889a798cb31f9247caf72752ee90567c0fe61cb0d384be2.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/2cd97db990db3f57e889a798cb31f9247caf72752ee90567c0fe61cb0d384be2.jpg)

![450f7f62d6067421aa9ea758e53a8c9298169c8ff60f10f9309cc1a82fd2a859.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/450f7f62d6067421aa9ea758e53a8c9298169c8ff60f10f9309cc1a82fd2a859.jpg)

![61f5be14677bdd28755c29f7b4a7b32856520ddeeac7345dfdfd0d672436ea29.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/61f5be14677bdd28755c29f7b4a7b32856520ddeeac7345dfdfd0d672436ea29.jpg)

![65d307d82c4d37e99afb26175726404a8c625c788fa6cdc1694fa49317611b0c.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/65d307d82c4d37e99afb26175726404a8c625c788fa6cdc1694fa49317611b0c.jpg)

![68913ffc44449601583c825ac8adcc7bd4168c412ec2f4530ea6dac466c59085.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/68913ffc44449601583c825ac8adcc7bd4168c412ec2f4530ea6dac466c59085.jpg)

![6a2df87159be594591251aef7dab205caacbe9999158294f21e61897d551cc3e.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/6a2df87159be594591251aef7dab205caacbe9999158294f21e61897d551cc3e.jpg)

![78e89e1b919e2e0a8a176433036c5bc96574a91fcd9c38d2a52a8f5afb3459c1.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/78e89e1b919e2e0a8a176433036c5bc96574a91fcd9c38d2a52a8f5afb3459c1.jpg)

![84ff254452ab8687589228ba60569b357b5a11b42b94ee830fb74399d7659022.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/84ff254452ab8687589228ba60569b357b5a11b42b94ee830fb74399d7659022.jpg)

![aa7bc93abfb5d0e46c10ab2a0e0169de7219a45338fbbdf0df8d03424f803930.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/aa7bc93abfb5d0e46c10ab2a0e0169de7219a45338fbbdf0df8d03424f803930.jpg)

![bfccf9c1e4f7482b2e378863aa832daa054ec088b47d1ae2613e64edb8d3a100.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/bfccf9c1e4f7482b2e378863aa832daa054ec088b47d1ae2613e64edb8d3a100.jpg)

![cb8eb65a71a43922df2092f7b57ad4e6482cf517650e46dbdfe2b6f5612d4786.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/cb8eb65a71a43922df2092f7b57ad4e6482cf517650e46dbdfe2b6f5612d4786.jpg)

![cf21b397b1d1a526a2ab7e479d447d9c96623cab9a4a988de06050c9ba89af28.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/cf21b397b1d1a526a2ab7e479d447d9c96623cab9a4a988de06050c9ba89af28.jpg)

![f7c838fdd40d3332925338abe067cfc940b69090b8b23b48b568bb237604111a.jpg](../acl_results/155_Iron%20Sharpens%20Iron_%20Defending%20Against%20Attacks%20in%20Machine-Generated%20Text%20Detection%20with%20Adversarial%20T/tables/f7c838fdd40d3332925338abe067cfc940b69090b8b23b48b568bb237604111a.jpg)

## Cultural Learning-Based Culture Adaptation of Language Models

### Images

![5c251c7f41e1016a0c4f5e37fff6616cf6183f02cda1ced90081629f7ee1350c.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/images/5c251c7f41e1016a0c4f5e37fff6616cf6183f02cda1ced90081629f7ee1350c.jpg)

![6254cd649be25b3699440e5ec86bfbcf1ae53995174da9fc68823b68b97f1925.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/images/6254cd649be25b3699440e5ec86bfbcf1ae53995174da9fc68823b68b97f1925.jpg)

![8070e326721fc5935a656ad3a620e2bb0153b89866684a235935114aeb9678ac.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/images/8070e326721fc5935a656ad3a620e2bb0153b89866684a235935114aeb9678ac.jpg)

![b517c70ad123b9caadc567efbbd32c027a2e0e9f25bc644f4b51c209a4ef9417.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/images/b517c70ad123b9caadc567efbbd32c027a2e0e9f25bc644f4b51c209a4ef9417.jpg)

![c04b1c53b20bada575590b0305cedc00b38589bc5501c45401d1f46ae1883f78.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/images/c04b1c53b20bada575590b0305cedc00b38589bc5501c45401d1f46ae1883f78.jpg)

![c75c74d6bc2bf60e4e8be214de5a2ee6feec309e5feaef364030fa7ccbe9b97e.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/images/c75c74d6bc2bf60e4e8be214de5a2ee6feec309e5feaef364030fa7ccbe9b97e.jpg)

![f17ba35be2f22739601eb9c9b7b88e73aad0a37fc70b3f21fc297e4f681d95df.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/images/f17ba35be2f22739601eb9c9b7b88e73aad0a37fc70b3f21fc297e4f681d95df.jpg)

### Tables

![1e5b2d6bce07df65f640b964c1d214b503913ff6041fcb35556af25d52d04738.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/1e5b2d6bce07df65f640b964c1d214b503913ff6041fcb35556af25d52d04738.jpg)

![20bc239075ec47d6a04610a20cc059df1359498da001beb5e63b7bd931b6c3ec.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/20bc239075ec47d6a04610a20cc059df1359498da001beb5e63b7bd931b6c3ec.jpg)

![25eccdd8f3bd6ec6e3dbf7b5b279e626704ea9d2ef5208f3a75c5b6ca41d0dec.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/25eccdd8f3bd6ec6e3dbf7b5b279e626704ea9d2ef5208f3a75c5b6ca41d0dec.jpg)

![2ca6d9c239b351fa3364319f4d250b5a01f07214a8a1a2d7ff37d8ca1207ac33.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/2ca6d9c239b351fa3364319f4d250b5a01f07214a8a1a2d7ff37d8ca1207ac33.jpg)

![5a50299b45755e0f078219275a01da2b0ad6ea77e5a9055e4fe86cf04a7fc43f.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/5a50299b45755e0f078219275a01da2b0ad6ea77e5a9055e4fe86cf04a7fc43f.jpg)

![63edb29db3fe0111b7d1cf89082eb641461f4793ff8daa5249738a079f791992.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/63edb29db3fe0111b7d1cf89082eb641461f4793ff8daa5249738a079f791992.jpg)

![68736810d88a30ef15348bb1a6b4df8a4bc5ffd39492d7efde5d6742bc76d63f.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/68736810d88a30ef15348bb1a6b4df8a4bc5ffd39492d7efde5d6742bc76d63f.jpg)

![692bc4ddbd747e1140f2aeef917b5dcac1d8f4433b6b026bcb63538311c5df18.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/692bc4ddbd747e1140f2aeef917b5dcac1d8f4433b6b026bcb63538311c5df18.jpg)

![6c7b575c3dc83bdd3ddaff5b0e5dd9899c4dcd6c17b0bc7eb11a800c1fce0e91.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/6c7b575c3dc83bdd3ddaff5b0e5dd9899c4dcd6c17b0bc7eb11a800c1fce0e91.jpg)

![6e01aa1aefc88978e6fb12719f8a442146372c63d71aa44499ae7b075160b2fa.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/6e01aa1aefc88978e6fb12719f8a442146372c63d71aa44499ae7b075160b2fa.jpg)

![7449fe32597444cc20ddbb8093aefc6209e73862df69593770c522cf45bcffee.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/7449fe32597444cc20ddbb8093aefc6209e73862df69593770c522cf45bcffee.jpg)

![7d0f932ff2a4fd2e2af98502c9c6225000d8899be763ab4f11634d9b2d2e9f4e.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/7d0f932ff2a4fd2e2af98502c9c6225000d8899be763ab4f11634d9b2d2e9f4e.jpg)

![8018ac9823c54398fca6548ad0e90d26821be6e2e76d0d2c7b3aeb683dfc2fb6.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/8018ac9823c54398fca6548ad0e90d26821be6e2e76d0d2c7b3aeb683dfc2fb6.jpg)

![82c415fe059d7bb38568df3e04b675b29508f1c0e7a990e89c95e47a2297978c.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/82c415fe059d7bb38568df3e04b675b29508f1c0e7a990e89c95e47a2297978c.jpg)

![83bbb92c723cbac4728cb0202f27e323acc2edcd51dc16ad9bf9522d4437fcd2.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/83bbb92c723cbac4728cb0202f27e323acc2edcd51dc16ad9bf9522d4437fcd2.jpg)

![b68932b3f12646d21f64c42458eae4a177747d20b35e04d8dbbf8292a1d6a58c.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/b68932b3f12646d21f64c42458eae4a177747d20b35e04d8dbbf8292a1d6a58c.jpg)

![b7f91b9222c15e2bba9dbde2d3c699d8da17e33cc77152f7ccf0355832ce28bd.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/b7f91b9222c15e2bba9dbde2d3c699d8da17e33cc77152f7ccf0355832ce28bd.jpg)

![bf383116b6b04f98d0686a9622462944fc3aa9976f65ec68a202709d844c7d27.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/bf383116b6b04f98d0686a9622462944fc3aa9976f65ec68a202709d844c7d27.jpg)

![c573109e42502539c6e0f7572d7cdb1fc368f075d1c7b4833c10f2de817ba07f.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/c573109e42502539c6e0f7572d7cdb1fc368f075d1c7b4833c10f2de817ba07f.jpg)

![cb33b70f5832255da9719283fc809cc6850cc86768bed73f46b29717a32e0924.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/cb33b70f5832255da9719283fc809cc6850cc86768bed73f46b29717a32e0924.jpg)

![cfdefbad1304469330739254a546fb491db67fd118ef3025d7d810caf2b26b92.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/cfdefbad1304469330739254a546fb491db67fd118ef3025d7d810caf2b26b92.jpg)

![d6bef13ea698c75fe19f2658d27c37c73bd73c0963e3e58d78425af64904c643.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/d6bef13ea698c75fe19f2658d27c37c73bd73c0963e3e58d78425af64904c643.jpg)

![e25d1ce92a98f06a2cfe3ce6e0e726ffdf565a8f93fe01866ea742dde4aad08f.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/e25d1ce92a98f06a2cfe3ce6e0e726ffdf565a8f93fe01866ea742dde4aad08f.jpg)

![ef280b98c318def481ddb4481a1a789bf8833eaecd4829c089e4996a24a8e45f.jpg](../acl_results/156_Cultural%20Learning-Based%20Culture%20Adaptation%20of%20Language%20Models/tables/ef280b98c318def481ddb4481a1a789bf8833eaecd4829c089e4996a24a8e45f.jpg)

## A-TASC:AsianTED-Based Automatic Subtitling Corpus

### Images

![0aeadcc2e83d875d64b43be9010e5cb7ef33d3cd689e399bb301ef094c74c058.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/images/0aeadcc2e83d875d64b43be9010e5cb7ef33d3cd689e399bb301ef094c74c058.jpg)

![0fee75fefb3698be8580d63934a62dd9ec2f0293132ccb35d628f1998ec69405.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/images/0fee75fefb3698be8580d63934a62dd9ec2f0293132ccb35d628f1998ec69405.jpg)

![37e117a23a84d5dbbcee61347cf006a1473479e3fee9742ebe30368c7dd5e987.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/images/37e117a23a84d5dbbcee61347cf006a1473479e3fee9742ebe30368c7dd5e987.jpg)

![ac41541a7433f5c647296d57d71b4f16e7b2cfaf9022dfeec8f846a49c36c923.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/images/ac41541a7433f5c647296d57d71b4f16e7b2cfaf9022dfeec8f846a49c36c923.jpg)

### Tables

![01c32c3666267ece3cd6836f7431ca5ac4a3673d201768b5171e0b5010cc7dc0.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/01c32c3666267ece3cd6836f7431ca5ac4a3673d201768b5171e0b5010cc7dc0.jpg)

![20923e9d0ce8dd8146428d490d749114ed6f8b9679bda26438cbd41be8ac74e6.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/20923e9d0ce8dd8146428d490d749114ed6f8b9679bda26438cbd41be8ac74e6.jpg)

![3c13a6f7cf4318fbf7dfda83ef6576c5c1ab7befbd6466a2bb4a048f3f2f1cc1.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/3c13a6f7cf4318fbf7dfda83ef6576c5c1ab7befbd6466a2bb4a048f3f2f1cc1.jpg)

![4669ae8792290e1bc0086762bdc081c687f4b0abce498653c5e747ea7b24f7a4.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/4669ae8792290e1bc0086762bdc081c687f4b0abce498653c5e747ea7b24f7a4.jpg)

![4715ed63da5a56df95b457818015c26282669fadd9c3272ab60248ddf19ebbcc.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/4715ed63da5a56df95b457818015c26282669fadd9c3272ab60248ddf19ebbcc.jpg)

![5d2655c3b212f9df10d7edcb4a81dd119d0bf12de085d19bfaea7484431180f0.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/5d2655c3b212f9df10d7edcb4a81dd119d0bf12de085d19bfaea7484431180f0.jpg)

![64c3de3304bf026058b1dd9643d1a5f56db75d88324125c9c330b4d072c5800c.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/64c3de3304bf026058b1dd9643d1a5f56db75d88324125c9c330b4d072c5800c.jpg)

![7b1c10fadae877fc00f184e09e6ad575f817a36b6dd4ea555a11d58db3d574c7.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/7b1c10fadae877fc00f184e09e6ad575f817a36b6dd4ea555a11d58db3d574c7.jpg)

![7cf4e958432685106f42179f68ebed7b90d0197cfff81404bca65f52ce93d0fd.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/7cf4e958432685106f42179f68ebed7b90d0197cfff81404bca65f52ce93d0fd.jpg)

![84e1d398d882d860c8525265377c32a31ab3ace648193ac19ecf79a672d7452e.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/84e1d398d882d860c8525265377c32a31ab3ace648193ac19ecf79a672d7452e.jpg)

![f38df44c652232782fdb69b109fa4881f5cc2c575bd7e0baccdf2f149acc9094.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/f38df44c652232782fdb69b109fa4881f5cc2c575bd7e0baccdf2f149acc9094.jpg)

![f5bfe2535bcf72ef9db39ffa40a4806fd3705b3b16ca8d9aa46dbfc32e77f159.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/f5bfe2535bcf72ef9db39ffa40a4806fd3705b3b16ca8d9aa46dbfc32e77f159.jpg)

![fdd0e6b3f9ad0ac445c69ed00cf266d25cd60eb1670fbf7e42271635661f64f7.jpg](../acl_results/157_A-TASC_AsianTED-Based%20Automatic%20Subtitling%20Corpus/tables/fdd0e6b3f9ad0ac445c69ed00cf266d25cd60eb1670fbf7e42271635661f64f7.jpg)

## Refuse Whenever You Feel Unsafe: Improving Safety inLLMs via Decoupled Refusal Training

### Images

![08a41b173649ea8d8d2ff14cc778ad72fb1bfe9fa9bc56f6deb44238f2ecccfb.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/images/08a41b173649ea8d8d2ff14cc778ad72fb1bfe9fa9bc56f6deb44238f2ecccfb.jpg)

![3302e5305f24874be54af9c798302d6f449e8e026c9cc09cd14f293e6e5180da.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/images/3302e5305f24874be54af9c798302d6f449e8e026c9cc09cd14f293e6e5180da.jpg)

![35b1d960d956d392db2406920711775430d78261c22958ca8624a7aa1739fc6e.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/images/35b1d960d956d392db2406920711775430d78261c22958ca8624a7aa1739fc6e.jpg)

![5d8aa230441d45432a17a08d25e24a9b0ec49928248d3d6d8c67a4fc2cd04f5c.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/images/5d8aa230441d45432a17a08d25e24a9b0ec49928248d3d6d8c67a4fc2cd04f5c.jpg)

![6c1fc6b64e9dac8d430e66735565417797f7d0f781950fe044ce5e5c7629f049.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/images/6c1fc6b64e9dac8d430e66735565417797f7d0f781950fe044ce5e5c7629f049.jpg)

![86d2b5166ddda8387d782c62a15dde995ef73bb18b5b6d15427249b5aeb941f1.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/images/86d2b5166ddda8387d782c62a15dde995ef73bb18b5b6d15427249b5aeb941f1.jpg)

![90784ecd0c302d7cf575f5077ef77c191294fdc29e4e791e9af77cfa04aac16b.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/images/90784ecd0c302d7cf575f5077ef77c191294fdc29e4e791e9af77cfa04aac16b.jpg)

![981e04dcefccc0bb86b316b98eeff3126aacd9a34fc557bd301996e73eec0c43.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/images/981e04dcefccc0bb86b316b98eeff3126aacd9a34fc557bd301996e73eec0c43.jpg)

![ad412edae098854e2d28a13f946ef4931183871af0095e871b15181da080ee6b.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/images/ad412edae098854e2d28a13f946ef4931183871af0095e871b15181da080ee6b.jpg)

![dc5b55bf1b0410b36b214269fe16d6e0cdb9d72dadcab27251f9c77e1c0c5a53.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/images/dc5b55bf1b0410b36b214269fe16d6e0cdb9d72dadcab27251f9c77e1c0c5a53.jpg)

![face4b4f856d852aecfb2d6d2dc623418defecd40155b31f04b1c8b6aba3ee2b.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/images/face4b4f856d852aecfb2d6d2dc623418defecd40155b31f04b1c8b6aba3ee2b.jpg)

### Tables

![2f3089925357be140c1eba8bf709428d6ec406b8070a8428e491df41a657e684.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/tables/2f3089925357be140c1eba8bf709428d6ec406b8070a8428e491df41a657e684.jpg)

![35c4d8c7008cbf61a084a0d965ca0889c654228111149ca408995389feb4a2a6.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/tables/35c4d8c7008cbf61a084a0d965ca0889c654228111149ca408995389feb4a2a6.jpg)

![35dc6d4dcbe121bc65ecf6b30d4ba1ebacfed60be3bad33d8975748d31bdc31f.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/tables/35dc6d4dcbe121bc65ecf6b30d4ba1ebacfed60be3bad33d8975748d31bdc31f.jpg)

![44ba2776900d6b9c04003b586dbdd1c59ef614275d8f51a38ec811b8644dca9a.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/tables/44ba2776900d6b9c04003b586dbdd1c59ef614275d8f51a38ec811b8644dca9a.jpg)

![6bb70748e00892808b7f371b6ec0d726008701a257c589b05685482733a5970f.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/tables/6bb70748e00892808b7f371b6ec0d726008701a257c589b05685482733a5970f.jpg)

![87a7421d6abc92d5a511162e9bfd5efd04ddcfe3131d43d74605148de0d9a597.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/tables/87a7421d6abc92d5a511162e9bfd5efd04ddcfe3131d43d74605148de0d9a597.jpg)

![ac71bb375ba17137dd662fa6e48e6796f57b549f1aacdea13e6c86407bd965eb.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/tables/ac71bb375ba17137dd662fa6e48e6796f57b549f1aacdea13e6c86407bd965eb.jpg)

![c0714bbf4be5caee8280e8e8eafe344190d6e042cf820bf0d8d5a1b31adbd89d.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/tables/c0714bbf4be5caee8280e8e8eafe344190d6e042cf820bf0d8d5a1b31adbd89d.jpg)

![e8cc3e824d7f0411a7009a6b59f0d797338b4864d3d809a180678895bedc2db6.jpg](../acl_results/158_Refuse%20Whenever%20You%20Feel%20Unsafe_%20Improving%20Safety%20inLLMs%20via%20Decoupled%20Refusal%20Training/tables/e8cc3e824d7f0411a7009a6b59f0d797338b4864d3d809a180678895bedc2db6.jpg)

## Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings fromLLMs

### Images

![0ed5a014e08d301f26d19ae823cf7d82db7c3e8e273ab2df68de83f20060b096.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/images/0ed5a014e08d301f26d19ae823cf7d82db7c3e8e273ab2df68de83f20060b096.jpg)

![181baf7b276ce86502e081348fcbc6f8179b9037e6ab99750655f53dc13f6840.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/images/181baf7b276ce86502e081348fcbc6f8179b9037e6ab99750655f53dc13f6840.jpg)

![3620f9fb67c2dfda76eb5f5578a3c128d416f1685526ac97342a4b94823416c0.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/images/3620f9fb67c2dfda76eb5f5578a3c128d416f1685526ac97342a4b94823416c0.jpg)

![3d4c5467ec110f66380047737b7816bd1d9d4303f4a15ab3dc9afc94ba14a500.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/images/3d4c5467ec110f66380047737b7816bd1d9d4303f4a15ab3dc9afc94ba14a500.jpg)

![7403e1b511c9360801f2c95058680e6ae1795510464887fe4796b7006c863663.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/images/7403e1b511c9360801f2c95058680e6ae1795510464887fe4796b7006c863663.jpg)

### Tables

![1f4e47051d5d26aa9fcddcf151acfd6d0738ed7bb91e63cea616cdf5225b034a.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/1f4e47051d5d26aa9fcddcf151acfd6d0738ed7bb91e63cea616cdf5225b034a.jpg)

![28339e21e0f12b79485af95fa27e614b38cbd37efba770e4dbb4af8768eefbf0.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/28339e21e0f12b79485af95fa27e614b38cbd37efba770e4dbb4af8768eefbf0.jpg)

![287fb1ce1554026c9b6d0a9565c38749994a98603daf9afb39f1c413dbe732c9.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/287fb1ce1554026c9b6d0a9565c38749994a98603daf9afb39f1c413dbe732c9.jpg)

![385dbf8a2685ee01481bc326c9392b4d19221686a46aa6253546241fb2588ddd.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/385dbf8a2685ee01481bc326c9392b4d19221686a46aa6253546241fb2588ddd.jpg)

![59d21aa5dbc72e19f98de42650072ed2c6ce79cb8fd9be913fa58c8d21107e28.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/59d21aa5dbc72e19f98de42650072ed2c6ce79cb8fd9be913fa58c8d21107e28.jpg)

![65599077777247d0a86092652f9542c79f016fb85ce3046b74206b1cc806220b.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/65599077777247d0a86092652f9542c79f016fb85ce3046b74206b1cc806220b.jpg)

![6712f8651ec8d424d3adc404c5765ccefc6a685fd962905f0f237af6c51625ad.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/6712f8651ec8d424d3adc404c5765ccefc6a685fd962905f0f237af6c51625ad.jpg)

![6a04210d53a2c3fb4366bd725656c765fe929f85a17ed2da18d68190efc92989.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/6a04210d53a2c3fb4366bd725656c765fe929f85a17ed2da18d68190efc92989.jpg)

![76d4bf25bc04bb9fe51e092a3c6ec2a9c040a6dc9575570bca27115bc1019ac8.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/76d4bf25bc04bb9fe51e092a3c6ec2a9c040a6dc9575570bca27115bc1019ac8.jpg)

![8a5d1223da72b27d7fc5f1f05b050c3c1eae8a1de2da032922dbd838982896b6.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/8a5d1223da72b27d7fc5f1f05b050c3c1eae8a1de2da032922dbd838982896b6.jpg)

![8cb953f2a85c77dc9443bf5aa6e6dd8f6b3aff8d7cd5368db7834c7322d6cfba.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/8cb953f2a85c77dc9443bf5aa6e6dd8f6b3aff8d7cd5368db7834c7322d6cfba.jpg)

![b5dee9ca60a18dd13382ec707d55372cbb5745c718013af92c98704854302a9c.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/b5dee9ca60a18dd13382ec707d55372cbb5745c718013af92c98704854302a9c.jpg)

![b91e2302198df16e6df3c46095d72b7d4dbe84df12be776686f58d13ed5359ce.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/b91e2302198df16e6df3c46095d72b7d4dbe84df12be776686f58d13ed5359ce.jpg)

![c1930f54d2dd6336bfc97172f41986fe5b8d89d2904f065216b6a17709264d5c.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/c1930f54d2dd6336bfc97172f41986fe5b8d89d2904f065216b6a17709264d5c.jpg)

![c77838aa765febcf3f30880b8aceab64ff9315e4e015a5db285cbcbc586f8124.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/c77838aa765febcf3f30880b8aceab64ff9315e4e015a5db285cbcbc586f8124.jpg)

![d07aec700d2ba55e83c6fd1429558baf8d9d918ca4ffccbe1cbc06150dcd6e6b.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/d07aec700d2ba55e83c6fd1429558baf8d9d918ca4ffccbe1cbc06150dcd6e6b.jpg)

![e50a96ae4e82c0ae4224e1acfcaba0105204bf1e39fb0ab01700516939f0e134.jpg](../acl_results/159_Token%20Prepending_%20A%20Training-Free%20Approach%20for%20Eliciting%20Better%20Sentence%20Embeddings%20fromLLMs/tables/e50a96ae4e82c0ae4224e1acfcaba0105204bf1e39fb0ab01700516939f0e134.jpg)

## No Questions are Stupid, but some are Poorly Posed: Understanding Poorly-Posed Information-Seeking Questions

### Images

![126c913c95a1ba510a3cd8f1d9518a83ed5dc8166b9e1b005ba84e576309507a.jpg](../acl_results/160_No%20Questions%20are%20Stupid%2C%20but%20some%20are%20Poorly%20Posed_%20Understanding%20Poorly-Posed%20Information-Seeking%20Q/images/126c913c95a1ba510a3cd8f1d9518a83ed5dc8166b9e1b005ba84e576309507a.jpg)

![25bac3c52e6f6e257aed4768a562dd9bab0633dbf1131ffc3553dd713dd6b3a8.jpg](../acl_results/160_No%20Questions%20are%20Stupid%2C%20but%20some%20are%20Poorly%20Posed_%20Understanding%20Poorly-Posed%20Information-Seeking%20Q/images/25bac3c52e6f6e257aed4768a562dd9bab0633dbf1131ffc3553dd713dd6b3a8.jpg)

![4d11e0a14398b229b8e806c193827985b11250d2c16ba4465cf6628415f97872.jpg](../acl_results/160_No%20Questions%20are%20Stupid%2C%20but%20some%20are%20Poorly%20Posed_%20Understanding%20Poorly-Posed%20Information-Seeking%20Q/images/4d11e0a14398b229b8e806c193827985b11250d2c16ba4465cf6628415f97872.jpg)

![5747c070e47aba2bcabea6c5dee974c6a3b94d84c6bc485c869bf643ef9cb38d.jpg](../acl_results/160_No%20Questions%20are%20Stupid%2C%20but%20some%20are%20Poorly%20Posed_%20Understanding%20Poorly-Posed%20Information-Seeking%20Q/images/5747c070e47aba2bcabea6c5dee974c6a3b94d84c6bc485c869bf643ef9cb38d.jpg)

![5d55fc5c5c6dc9a737df16e4787588ebcb81c1af31b4abd18793278ca71e44d5.jpg](../acl_results/160_No%20Questions%20are%20Stupid%2C%20but%20some%20are%20Poorly%20Posed_%20Understanding%20Poorly-Posed%20Information-Seeking%20Q/images/5d55fc5c5c6dc9a737df16e4787588ebcb81c1af31b4abd18793278ca71e44d5.jpg)

![bbb2491671e50f73bbed2adc1a4e6dfa35f1f8718465aed2fced44ce4c901744.jpg](../acl_results/160_No%20Questions%20are%20Stupid%2C%20but%20some%20are%20Poorly%20Posed_%20Understanding%20Poorly-Posed%20Information-Seeking%20Q/images/bbb2491671e50f73bbed2adc1a4e6dfa35f1f8718465aed2fced44ce4c901744.jpg)

![e53e72b48397fd57c4edfd10f35228f78a845b1b035bdc420b37a80c6e654a9b.jpg](../acl_results/160_No%20Questions%20are%20Stupid%2C%20but%20some%20are%20Poorly%20Posed_%20Understanding%20Poorly-Posed%20Information-Seeking%20Q/images/e53e72b48397fd57c4edfd10f35228f78a845b1b035bdc420b37a80c6e654a9b.jpg)

![ee0e193cb5004ee8ad2bcb02769d535adf30325e07273620ff3b7c8cd36c2047.jpg](../acl_results/160_No%20Questions%20are%20Stupid%2C%20but%20some%20are%20Poorly%20Posed_%20Understanding%20Poorly-Posed%20Information-Seeking%20Q/images/ee0e193cb5004ee8ad2bcb02769d535adf30325e07273620ff3b7c8cd36c2047.jpg)

### Tables

![bb3949dc1016c000e00f3126dfd3610e3b5f7845ecfb87d96530c8c5a81951e8.jpg](../acl_results/160_No%20Questions%20are%20Stupid%2C%20but%20some%20are%20Poorly%20Posed_%20Understanding%20Poorly-Posed%20Information-Seeking%20Q/tables/bb3949dc1016c000e00f3126dfd3610e3b5f7845ecfb87d96530c8c5a81951e8.jpg)

![c84ac0237086c343297dc35e9ce02f133e5cc4444ffb1e26a9a86dacab7b775a.jpg](../acl_results/160_No%20Questions%20are%20Stupid%2C%20but%20some%20are%20Poorly%20Posed_%20Understanding%20Poorly-Posed%20Information-Seeking%20Q/tables/c84ac0237086c343297dc35e9ce02f133e5cc4444ffb1e26a9a86dacab7b775a.jpg)

![d698ed17586cf9548819a75449a9fa6faf73c247fe87174368d23d100f984d0f.jpg](../acl_results/160_No%20Questions%20are%20Stupid%2C%20but%20some%20are%20Poorly%20Posed_%20Understanding%20Poorly-Posed%20Information-Seeking%20Q/tables/d698ed17586cf9548819a75449a9fa6faf73c247fe87174368d23d100f984d0f.jpg)