# ACL 2025 Long Papers


**Summary:** 1599 papers with extracted content:
- 📊 Total images: 13877
- 📋 Total tables: 16805
- 📄 Total files: 3068
---

# ACL 2025 Long Papers - Part 26 of 50

## 目录 (Table of Contents)

1. [MMBoundary: AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration](#MMBoundary-AdvancingMLLMKnowledge-Boundary-Awareness-through-Reasoning-Step-Confidence-Calibration)
2. [LIFBench: Evaluating the Instruction Following Performance and Stability of Large Language Models in Long-Context Scenarios](#LIFBench-Evaluating-the-Instruction-Following-Performance-and-Stability-of-Large-Language-Models-in-Long-Context-Scenarios)
3. [Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering](#Aligning-Large-Language-Models-to-Follow-Instructions-and-Hallucinate-Less-via-Effective-Data-Filtering)
4. [M2S: Multi-turn to Single-turn jailbreak in Red Teaming forLLMs](#M2S-Multi-turn-to-Single-turn-jailbreak-in-Red-Teaming-forLLMs)
5. [RAEmoLLM: Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information](#RAEmoLLM-Retrieval-AugmentedLLMs-for-Cross-Domain-Misinformation-Detection-Using-In-Context-Learning-Based-on-Emotional-Information)
6. [Task-Specific Information Decomposition for End-to-End Dense Video Captioning](#Task-Specific-Information-Decomposition-for-End-to-End-Dense-Video-Captioning)
7. [CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges](#CalibraEval-Calibrating-Prediction-Distribution-to-Mitigate-Selection-Bias-inLLMs-as-Judges)
8. [Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection](#Explaining-Matters-Leveraging-Definitions-and-Semantic-Expansion-for-Sexism-Detection)
9. [Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models](#Private-Memorization-Editing-Turning-Memorization-into-a-Defense-to-Strengthen-Data-Privacy-in-Large-Language-Models)
10. [PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](#PhysReason-A-Comprehensive-Benchmark-towards-Physics-Based-Reasoning)
11. [Does Time Have Its Place? Temporal Heads: Where Language Models Recall Time-specific Information](#Does-Time-Have-Its-Place-Temporal-Heads-Where-Language-Models-Recall-Time-specific-Information)
12. [Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training](#Velocitune-A-Velocity-based-Dynamic-Domain-Reweighting-Method-for-Continual-Pre-training)
13. [Sheep’s Skin, Wolf’s Deeds: AreLLMs Ready for Metaphorical Implicit Hate Speech?](#Sheeps-Skin-Wolfs-Deeds-AreLLMs-Ready-for-Metaphorical-Implicit-Hate-Speech)
14. [Neuron-Level Sequential Editing for Large Language Models](#Neuron-Level-Sequential-Editing-for-Large-Language-Models)
15. [Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts](#Automatic-Expert-Discovery-inLLMUpcycling-via-Sparse-Interpolated-Mixture-of-Experts)
16. [SimulS2S-LLM: Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation](#SimulS2S-LLM-Unlocking-Simultaneous-Inference-of-SpeechLLMs-for-Speech-to-Speech-Translation)
17. [VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models](#VoxEval-Benchmarking-the-Knowledge-Understanding-Capabilities-of-End-to-End-Spoken-Language-Models)
18. [RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation](#RetroLLM-Empowering-Large-Language-Models-to-Retrieve-Fine-grained-Evidence-within-Generation)
19. [The Role of Deductive and Inductive Reasoning in Large Language Models](#The-Role-of-Deductive-and-Inductive-Reasoning-in-Large-Language-Models)
20. [Disentangling the Roles of Representation and Selection in Data Pruning](#Disentangling-the-Roles-of-Representation-and-Selection-in-Data-Pruning)
21. [FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](#FRACTAL-Fine-Grained-Scoring-from-Aggregate-Text-Labels)
22. [ACT: Knowledgeable Agents to Design and Perform Complex Tasks](#ACT-Knowledgeable-Agents-to-Design-and-Perform-Complex-Tasks)
23. [Logical forms complement probability in understanding language model (and human) performance](#Logical-forms-complement-probability-in-understanding-language-model-and-human-performance)
24. [Length Controlled Generation for Black-boxLLMs](#Length-Controlled-Generation-for-Black-boxLLMs)
25. [Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization](#Improving-Contextual-Faithfulness-of-Large-Language-Models-via-Retrieval-Heads-Induced-Optimization)
26. [Global Eye: Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process](#Global-Eye-Breaking-the-Fixed-Thinking-Pattern-during-the-Instruction-Expansion-Process)
27. [On Synthesizing Data for Context Attribution in Question Answering](#On-Synthesizing-Data-for-Context-Attribution-in-Question-Answering)
28. [TST: A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks](#TST-A-Schema-Based-Top-Down-and-Dynamic-Aware-Agent-of-Text-to-Table-Tasks)
29. [EventRAG: EnhancingLLMGeneration with Event Knowledge Graphs](#EventRAG-EnhancingLLMGeneration-with-Event-Knowledge-Graphs)
30. [Analyzing the Rapid Generalization ofSFTvia the Perspective of Attention Head Activation Patterns](#Analyzing-the-Rapid-Generalization-ofSFTvia-the-Perspective-of-Attention-Head-Activation-Patterns)
31. [Can’t See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for MultimodalLLMs](#Cant-See-the-Forest-for-the-Trees-Benchmarking-Multimodal-Safety-Awareness-for-MultimodalLLMs)
32. [Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling](#Mis-prompt-Benchmarking-Large-Language-Models-for-Proactive-Error-Handling)

---


## MMBoundary: AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration

### Images

![10821a6ff88216c42ddff1856038b8276bffd591c38004604adf9c3987ec8797.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/images/10821a6ff88216c42ddff1856038b8276bffd591c38004604adf9c3987ec8797.jpg)

![1620df1455e3d6d28c214fa7e9b21307efabd354d2d3d17b0f460d0d7999a2c0.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/images/1620df1455e3d6d28c214fa7e9b21307efabd354d2d3d17b0f460d0d7999a2c0.jpg)

![1f50d5ff5de7b9a4aea48117651d1b353f9ff55fc166de01c7f5c8a346fb555e.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/images/1f50d5ff5de7b9a4aea48117651d1b353f9ff55fc166de01c7f5c8a346fb555e.jpg)

![315b29c15672f9efa6ecdcec1e1e133fd7da4fc29e0ba27713fefb96aa6c782a.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/images/315b29c15672f9efa6ecdcec1e1e133fd7da4fc29e0ba27713fefb96aa6c782a.jpg)

![c964f100c7763778db035825947ba36519245ab9dfe35ce763a0d9daf2a2b9e8.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/images/c964f100c7763778db035825947ba36519245ab9dfe35ce763a0d9daf2a2b9e8.jpg)

![f83d61e7fca1a68b8ef414785a70b7e9eee0f64a6b920e733d0713304052d9b6.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/images/f83d61e7fca1a68b8ef414785a70b7e9eee0f64a6b920e733d0713304052d9b6.jpg)

### Tables

![2317314b1367fddd77b0cb2963d81689f92075dcf240590dbe5fb50e47206fd7.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/tables/2317314b1367fddd77b0cb2963d81689f92075dcf240590dbe5fb50e47206fd7.jpg)

![2c90235ae05e1610f705d242c03bba5a34d81f3a261b59f32752b9e990e9d70c.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/tables/2c90235ae05e1610f705d242c03bba5a34d81f3a261b59f32752b9e990e9d70c.jpg)

![4d19e7dd00dfe1540f996b049b92b168de217100c697e1f3292ae16d4670b9ff.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/tables/4d19e7dd00dfe1540f996b049b92b168de217100c697e1f3292ae16d4670b9ff.jpg)

![5a66dba4a8eb7820a3a01857ea51ddb8053616583d1fd8da6cd09194406fcf24.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/tables/5a66dba4a8eb7820a3a01857ea51ddb8053616583d1fd8da6cd09194406fcf24.jpg)

![7b9678cc164f07fd2d02679421f7aeb1c54aab6147ac736842a6f190120faa94.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/tables/7b9678cc164f07fd2d02679421f7aeb1c54aab6147ac736842a6f190120faa94.jpg)

![7dc5c17b6daf87f298c186acd8974cae70a1fd786942cde54657d0d17339a139.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/tables/7dc5c17b6daf87f298c186acd8974cae70a1fd786942cde54657d0d17339a139.jpg)

![910955d082c8c924527c618a05f5bcf890f7e87ac65e996321a468516fa1225d.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/tables/910955d082c8c924527c618a05f5bcf890f7e87ac65e996321a468516fa1225d.jpg)

![b9ca4af1625430fc9a452120262f8c04cff019d87f0b9909360d8290dae39d3e.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/tables/b9ca4af1625430fc9a452120262f8c04cff019d87f0b9909360d8290dae39d3e.jpg)

![d6003c292c4c415bd4552f941cbabd49f84447d6f817af606c34ce6a5e95cace.jpg](acl_results/802_MMBoundary_ AdvancingMLLMKnowledge Boundary Awareness through Reasoning Step Confidence Calibration/tables/d6003c292c4c415bd4552f941cbabd49f84447d6f817af606c34ce6a5e95cace.jpg)

## LIFBench: Evaluating the Instruction Following Performance and Stability of Large Language Models in Long-Context Scenarios

### Images

![1ab5733c810aa39a08703fd99b2a031c6e796fc440ff75e72b30ea96ef93c8a4.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/images/1ab5733c810aa39a08703fd99b2a031c6e796fc440ff75e72b30ea96ef93c8a4.jpg)

![27cd6eaaf2ff5674783248bd946bc7e1489df2253ba2e56dcec3081b1257ed82.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/images/27cd6eaaf2ff5674783248bd946bc7e1489df2253ba2e56dcec3081b1257ed82.jpg)

![520001c094d37659b31543482fbb0f73e38d399806e63e49cfa12a21e28030d1.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/images/520001c094d37659b31543482fbb0f73e38d399806e63e49cfa12a21e28030d1.jpg)

![62b51380bd4cc3ae03a8941a39066fa60a31274f94d4e690784dcd20dacf11e0.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/images/62b51380bd4cc3ae03a8941a39066fa60a31274f94d4e690784dcd20dacf11e0.jpg)

![8327c7d4ac91adfc7db1768ea099c49a26bca7c83a410bdba6adee3ad1a330e4.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/images/8327c7d4ac91adfc7db1768ea099c49a26bca7c83a410bdba6adee3ad1a330e4.jpg)

![8a7a7fd4c9de1df73e1a96df1df659b6752bfdd865a7eb6755e53a8ded89c835.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/images/8a7a7fd4c9de1df73e1a96df1df659b6752bfdd865a7eb6755e53a8ded89c835.jpg)

![a2a1bfa7345b5983d951d586c49d98c2916e00497ae5fb1bb5ad93e51bcafa21.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/images/a2a1bfa7345b5983d951d586c49d98c2916e00497ae5fb1bb5ad93e51bcafa21.jpg)

### Tables

![18f029e75f758e02a7296df8aa1aa5d539d783c1f56680c071379ec77525bb25.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/18f029e75f758e02a7296df8aa1aa5d539d783c1f56680c071379ec77525bb25.jpg)

![3eee23ab6cca62e251e687f7f8850efc9e64061ff836cf12222158671703eb4f.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/3eee23ab6cca62e251e687f7f8850efc9e64061ff836cf12222158671703eb4f.jpg)

![44b15c1a6e5bfc6b14a6401ce5d29150c7f3a630bf1e1daec60339ed19270717.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/44b15c1a6e5bfc6b14a6401ce5d29150c7f3a630bf1e1daec60339ed19270717.jpg)

![8d7b38f9a3d8dc5238f223eb731ff202bd12922dedb6cd53484fd97dfe32f3ff.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/8d7b38f9a3d8dc5238f223eb731ff202bd12922dedb6cd53484fd97dfe32f3ff.jpg)

![92b0481e84a8a85ffa331e86080c0ebfc28aaad191b4f91f19a0bb56fc15aec6.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/92b0481e84a8a85ffa331e86080c0ebfc28aaad191b4f91f19a0bb56fc15aec6.jpg)

![9893c0f3fcb8b0334a471332f6f0d3ad2cff0c12c190148d8c8a579aa4b9663d.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/9893c0f3fcb8b0334a471332f6f0d3ad2cff0c12c190148d8c8a579aa4b9663d.jpg)

![9aa37f1e1df3b1a91d0e70ac2188d5fd032c75a69cb002d7246c7fac184bc881.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/9aa37f1e1df3b1a91d0e70ac2188d5fd032c75a69cb002d7246c7fac184bc881.jpg)

![9ae22fb9f33fe09b3796f44846f98325329356cdb7d798c294ec25d2aea3f6d1.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/9ae22fb9f33fe09b3796f44846f98325329356cdb7d798c294ec25d2aea3f6d1.jpg)

![a8d986e5791f4f36d8c663b7ddab37f3edc203d40426f449475eaa31f0c440bf.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/a8d986e5791f4f36d8c663b7ddab37f3edc203d40426f449475eaa31f0c440bf.jpg)

![c026fa5f4cf0693ff7866c139ac5550c48e618831db30c51ce401ffc0a61b687.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/c026fa5f4cf0693ff7866c139ac5550c48e618831db30c51ce401ffc0a61b687.jpg)

![c68342e4dc6ad7fdc208bb5cf7cb3fe3a770388b9b5610d24fa1bab84851f69e.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/c68342e4dc6ad7fdc208bb5cf7cb3fe3a770388b9b5610d24fa1bab84851f69e.jpg)

![db8c1302599b9f1d89b2e59b99c1b3106ae2475cb724fdc2023133dd06228473.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/db8c1302599b9f1d89b2e59b99c1b3106ae2475cb724fdc2023133dd06228473.jpg)

![df69d83c663e00048c9bdc987e8ef2dc167eda7e016df2306d498e82b78491fb.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/df69d83c663e00048c9bdc987e8ef2dc167eda7e016df2306d498e82b78491fb.jpg)

![f0aee7baaa883dcd374a107901b1c7ee37739a7ccfbff2d73b9b66e1653b4a4b.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/f0aee7baaa883dcd374a107901b1c7ee37739a7ccfbff2d73b9b66e1653b4a4b.jpg)

![f7505eb351bb7f42002939cd8f3b455e0b365a9c30c6063ccf493da806c6d5d9.jpg](acl_results/803_LIFBench_ Evaluating the Instruction Following Performance and Stability of Large Language Models in/tables/f7505eb351bb7f42002939cd8f3b455e0b365a9c30c6063ccf493da806c6d5d9.jpg)

## Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering

### Images

![508d3f2124cf1fa1e3fa9dac4faee362748a158214c248e47de36fef694b8593.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/images/508d3f2124cf1fa1e3fa9dac4faee362748a158214c248e47de36fef694b8593.jpg)

![b73d5b3ddf5b9e4593c881f7c8118da36b4669ac4d34288e2ef1dfa53431e3ae.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/images/b73d5b3ddf5b9e4593c881f7c8118da36b4669ac4d34288e2ef1dfa53431e3ae.jpg)

![ce48e31e0fb94bca80a2937a376f57c986e4758a80beff34493b0226987c434a.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/images/ce48e31e0fb94bca80a2937a376f57c986e4758a80beff34493b0226987c434a.jpg)

![d753509145c0afc83d60194e96efd159a19567591d62dd5fc56d6d5e8abae39f.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/images/d753509145c0afc83d60194e96efd159a19567591d62dd5fc56d6d5e8abae39f.jpg)

![f96b740e01f4af2b401837c51ba7c900567994e07c7166d66c32eab2e95ea642.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/images/f96b740e01f4af2b401837c51ba7c900567994e07c7166d66c32eab2e95ea642.jpg)

### Tables

![06d36f614325f182cc9fe13619266ae334f869024242a1412288e44fd73508e2.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/tables/06d36f614325f182cc9fe13619266ae334f869024242a1412288e44fd73508e2.jpg)

![0ea1f719b06f91ee1e6f813793da208db257b7d1e7a770e6998fd8b12ddb44ff.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/tables/0ea1f719b06f91ee1e6f813793da208db257b7d1e7a770e6998fd8b12ddb44ff.jpg)

![12629568e82633be2e09155ecd8589e261e405664b1f9e2c808e61cf4ce4a67d.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/tables/12629568e82633be2e09155ecd8589e261e405664b1f9e2c808e61cf4ce4a67d.jpg)

![28472c53febdcb24ee64010679d1ae07a8d488f48deab952662fd9a007859e37.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/tables/28472c53febdcb24ee64010679d1ae07a8d488f48deab952662fd9a007859e37.jpg)

![2c104728fbe370686ba9d52ed09146e1fc770ac863a403d271f005a48ccb33e1.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/tables/2c104728fbe370686ba9d52ed09146e1fc770ac863a403d271f005a48ccb33e1.jpg)

![3c9e100435764678bdf90df78dda1d05a85259005183cd5bdee555c1c05b6ceb.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/tables/3c9e100435764678bdf90df78dda1d05a85259005183cd5bdee555c1c05b6ceb.jpg)

![59024d5a2098bc8f4517e5cc69da9b80e91425d2310a1165c00d30fcae779177.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/tables/59024d5a2098bc8f4517e5cc69da9b80e91425d2310a1165c00d30fcae779177.jpg)

![82ea35d9a8960fc7096c81bafbe4efc1bc87f53eee6ef3a8a788c5eda676cbd8.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/tables/82ea35d9a8960fc7096c81bafbe4efc1bc87f53eee6ef3a8a788c5eda676cbd8.jpg)

![846e55fc69967e83a18aea2438dfe8bd371dae012d2d5babac744bbe68e98b88.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/tables/846e55fc69967e83a18aea2438dfe8bd371dae012d2d5babac744bbe68e98b88.jpg)

![ab456d439d8c3eddf2ce0331a4156970a47ebee0a932cffd71d92fb4c8e5c19d.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/tables/ab456d439d8c3eddf2ce0331a4156970a47ebee0a932cffd71d92fb4c8e5c19d.jpg)

![c47a79ffe5f6dcc20a7bf99adb57634695b368f3252951ca7107c1ab942afbfc.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/tables/c47a79ffe5f6dcc20a7bf99adb57634695b368f3252951ca7107c1ab942afbfc.jpg)

![f09133a329b609f7be2f78b37afa82f9423a902aa79a12c05e541951d582f2e6.jpg](acl_results/804_Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filter/tables/f09133a329b609f7be2f78b37afa82f9423a902aa79a12c05e541951d582f2e6.jpg)

## M2S: Multi-turn to Single-turn jailbreak in Red Teaming forLLMs

### Images

![03fe4663f7a816f2e6371d856a2a97b769e6c48ebd8835370206eb118e5d7452.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/images/03fe4663f7a816f2e6371d856a2a97b769e6c48ebd8835370206eb118e5d7452.jpg)

![0c1ea2529c11e32b6fcd23dea8e93e99235149d2dcf3eca4df61e7a5bda8a9cb.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/images/0c1ea2529c11e32b6fcd23dea8e93e99235149d2dcf3eca4df61e7a5bda8a9cb.jpg)

![11cbae4f950e18f69b9c8acc6b57509a773fe0b909abb49e68a02ef396f4c40f.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/images/11cbae4f950e18f69b9c8acc6b57509a773fe0b909abb49e68a02ef396f4c40f.jpg)

![2c81f6ede62ffaff69b56014be27a201602bdcd640d451b4bd62fb1ffc9dabdb.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/images/2c81f6ede62ffaff69b56014be27a201602bdcd640d451b4bd62fb1ffc9dabdb.jpg)

![5c47eb1f2bfaf4643a73a0073b403c2d23b52f50a17807f3f5d7c7be2a48e433.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/images/5c47eb1f2bfaf4643a73a0073b403c2d23b52f50a17807f3f5d7c7be2a48e433.jpg)

![6c943fad9d0108e401fd9aedfca8e0b2f54b349343a4a74194ca5dc9de531efd.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/images/6c943fad9d0108e401fd9aedfca8e0b2f54b349343a4a74194ca5dc9de531efd.jpg)

![75f889bc345d1d1c664df5dcae7fa8030ce40a05b47dbe39d3a4946944266c35.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/images/75f889bc345d1d1c664df5dcae7fa8030ce40a05b47dbe39d3a4946944266c35.jpg)

![792296db8e97efc1d1e20e21fdf0e3ae8eab7aafcf2945e7285c547428742120.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/images/792296db8e97efc1d1e20e21fdf0e3ae8eab7aafcf2945e7285c547428742120.jpg)

![8b31f33a07a6ef66d1b6bf9b34d76f304a4965413c9c2ec81a987d4ce2d6e5b5.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/images/8b31f33a07a6ef66d1b6bf9b34d76f304a4965413c9c2ec81a987d4ce2d6e5b5.jpg)

![b3309083291f2759aa69696d93cc034444404adbb81a63f2c930c3afb64f962d.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/images/b3309083291f2759aa69696d93cc034444404adbb81a63f2c930c3afb64f962d.jpg)

![bdd0a376dc062a189a8eec42e898dc2bb91d203cbc06ea3cd5d02171f33287fe.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/images/bdd0a376dc062a189a8eec42e898dc2bb91d203cbc06ea3cd5d02171f33287fe.jpg)

![f1ce9c4ef9d9ce32c38d8ea1106530c806b5a3e597826e068c4fac59b00eb8bf.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/images/f1ce9c4ef9d9ce32c38d8ea1106530c806b5a3e597826e068c4fac59b00eb8bf.jpg)

### Tables

![1834b840ee37326e56227582c2765148474523eed6e807b9504fc4b3a6465b27.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/tables/1834b840ee37326e56227582c2765148474523eed6e807b9504fc4b3a6465b27.jpg)

![30bc7c2b3284063f1a126b3d3f0933a72a7ac383a8499f06b60c8e6aecfc2d4c.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/tables/30bc7c2b3284063f1a126b3d3f0933a72a7ac383a8499f06b60c8e6aecfc2d4c.jpg)

![385c2f7458b832ae29a05e35d00d5b4b4acfe01d8c9fc8647b37d1eb74e9e081.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/tables/385c2f7458b832ae29a05e35d00d5b4b4acfe01d8c9fc8647b37d1eb74e9e081.jpg)

![43aea7fd73f885d470e1325d3e3c341fba3f56685779334a838794f82ec0c444.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/tables/43aea7fd73f885d470e1325d3e3c341fba3f56685779334a838794f82ec0c444.jpg)

![552dfb129562565cf4e2d5b83360aabda9e2c28c34d5faeebfb5d49e6f4a5227.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/tables/552dfb129562565cf4e2d5b83360aabda9e2c28c34d5faeebfb5d49e6f4a5227.jpg)

![5bb7599d1c09389fbe25fe053186b9500bccb628ebf42421e4b79542ded397e9.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/tables/5bb7599d1c09389fbe25fe053186b9500bccb628ebf42421e4b79542ded397e9.jpg)

![68af95bd5d047f0fe51dc5aebd6de56c09704f11bbb7a51fe224f3186e23bab0.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/tables/68af95bd5d047f0fe51dc5aebd6de56c09704f11bbb7a51fe224f3186e23bab0.jpg)

![810342857b5a38bf3625db2a056604a7553aea4bfa5c26088d8af61fa209ad8e.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/tables/810342857b5a38bf3625db2a056604a7553aea4bfa5c26088d8af61fa209ad8e.jpg)

![851ab6d78c4b5ce7e7fe55624074ace6ea00244eb97081fbf480b6ce984c5007.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/tables/851ab6d78c4b5ce7e7fe55624074ace6ea00244eb97081fbf480b6ce984c5007.jpg)

![a8468b0f1d0f205086853dfb3c84ee6f72d703e2eccea681020d83cb452d6a6d.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/tables/a8468b0f1d0f205086853dfb3c84ee6f72d703e2eccea681020d83cb452d6a6d.jpg)

![d1aee0aae966b71b6fc694717ca0243f639deb0ed906c02fcad45effb1b62735.jpg](acl_results/805_M2S_ Multi-turn to Single-turn jailbreak in Red Teaming forLLMs/tables/d1aee0aae966b71b6fc694717ca0243f639deb0ed906c02fcad45effb1b62735.jpg)

## RAEmoLLM: Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information

### Images

![0287393ebcbe84fcaba6431b8320629b44f0e127321baf301323d44ee09dcf83.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/images/0287393ebcbe84fcaba6431b8320629b44f0e127321baf301323d44ee09dcf83.jpg)

![103a4495d5c91cb6260fa0617af5e18674a484cadede9d74b689789fd7ee21b2.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/images/103a4495d5c91cb6260fa0617af5e18674a484cadede9d74b689789fd7ee21b2.jpg)

![3cab2b7e29211bc1fe4a107b46108f12be925031aa3068b1235c95daed573d67.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/images/3cab2b7e29211bc1fe4a107b46108f12be925031aa3068b1235c95daed573d67.jpg)

![5db9dad68a9f6e8c600e13b4d25b245bf81828539af2db6463afa5e55400a2df.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/images/5db9dad68a9f6e8c600e13b4d25b245bf81828539af2db6463afa5e55400a2df.jpg)

![67e0d5b8be6837301c77a84692966cf614fddb08375d044dc1bfb73ca506067c.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/images/67e0d5b8be6837301c77a84692966cf614fddb08375d044dc1bfb73ca506067c.jpg)

![86863362c8ba739ae575a0dda0ef0360d9f56b6dfd01c7cc4cd14c30ca5f943d.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/images/86863362c8ba739ae575a0dda0ef0360d9f56b6dfd01c7cc4cd14c30ca5f943d.jpg)

![9dc703ed1f5907f0af2f69f8ced444e2c319d26e2048dd8aa62095469c3da35c.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/images/9dc703ed1f5907f0af2f69f8ced444e2c319d26e2048dd8aa62095469c3da35c.jpg)

![a07d8fe44981e0de2006ca39074f12bd91e7472c636ffb63b233f00c0b9a0374.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/images/a07d8fe44981e0de2006ca39074f12bd91e7472c636ffb63b233f00c0b9a0374.jpg)

![eaa04e9ad88296346d0b79edb0a6fb14579241770a8cf459dc40925346bd6b78.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/images/eaa04e9ad88296346d0b79edb0a6fb14579241770a8cf459dc40925346bd6b78.jpg)

![f5c7a7fd226379a85a75e7dddec3b70f44813e92b01f69fd3258f7a3685813f3.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/images/f5c7a7fd226379a85a75e7dddec3b70f44813e92b01f69fd3258f7a3685813f3.jpg)

### Tables

![0b200eefb229437a37201333bc61fa85d76af8871caf8861241490ffce35efd0.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/0b200eefb229437a37201333bc61fa85d76af8871caf8861241490ffce35efd0.jpg)

![12e237191654775e3dfccff066c6f630b15aa3db6d426d85f439489be04f8f33.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/12e237191654775e3dfccff066c6f630b15aa3db6d426d85f439489be04f8f33.jpg)

![143552f3621c4cf87e0924fbc64fcf23578f7346e04590974b59806f35222b84.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/143552f3621c4cf87e0924fbc64fcf23578f7346e04590974b59806f35222b84.jpg)

![77f7a30895acde87886623e40932bbbd5795ff81cbe4a925ce76bac2326e5ff3.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/77f7a30895acde87886623e40932bbbd5795ff81cbe4a925ce76bac2326e5ff3.jpg)

![93d2df5dd525836d2d6cfc72863d8ae4d47380d01a8ce94e9b140dd61a480b40.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/93d2df5dd525836d2d6cfc72863d8ae4d47380d01a8ce94e9b140dd61a480b40.jpg)

![a27bd1749714238b48deb0864afda9d34feef29b82ae8f416604c31a68c63788.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/a27bd1749714238b48deb0864afda9d34feef29b82ae8f416604c31a68c63788.jpg)

![a32cb247ff41bee9353ba22b45b0acf96cbf31d905d6bafdf27cca840a162e62.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/a32cb247ff41bee9353ba22b45b0acf96cbf31d905d6bafdf27cca840a162e62.jpg)

![b3cfc67a57c07709a8fd6d9844bdfed45e75ad3efcc865723fd8f19820817985.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/b3cfc67a57c07709a8fd6d9844bdfed45e75ad3efcc865723fd8f19820817985.jpg)

![b99698e60c23c6cf050db63503a917def1bd92e8b629cff2f6018b7f79d31c57.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/b99698e60c23c6cf050db63503a917def1bd92e8b629cff2f6018b7f79d31c57.jpg)

![bd7e1905a0e0c1362b0f9ef0a857a834e78498db4ec93de3390e431e3cdda546.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/bd7e1905a0e0c1362b0f9ef0a857a834e78498db4ec93de3390e431e3cdda546.jpg)

![db3068e63b2d788eea57896c2769dcf9a98604423512e8ff8302e54bdc0db20f.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/db3068e63b2d788eea57896c2769dcf9a98604423512e8ff8302e54bdc0db20f.jpg)

![e28aa13e0370df3a23c91607269c59e350b204f6575b99c7ef77998a26af6615.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/e28aa13e0370df3a23c91607269c59e350b204f6575b99c7ef77998a26af6615.jpg)

![f95c5224bec8ff0ce500fb04291edd502bdd74329e87b267eb06bd3a4dbf9846.jpg](acl_results/806_RAEmoLLM_ Retrieval AugmentedLLMs for Cross-Domain Misinformation Detection Using In-Context Learnin/tables/f95c5224bec8ff0ce500fb04291edd502bdd74329e87b267eb06bd3a4dbf9846.jpg)

## Task-Specific Information Decomposition for End-to-End Dense Video Captioning

### Images

![32f317e704fd674522fa177202e2797ada53705f72a3426f03540b5240f4357e.jpg](acl_results/807_Task-Specific Information Decomposition for End-to-End Dense Video Captioning/images/32f317e704fd674522fa177202e2797ada53705f72a3426f03540b5240f4357e.jpg)

![6b92577776ceaf2eab44c74eedf90e44260625986520eceb0e2d87e81f83942f.jpg](acl_results/807_Task-Specific Information Decomposition for End-to-End Dense Video Captioning/images/6b92577776ceaf2eab44c74eedf90e44260625986520eceb0e2d87e81f83942f.jpg)

![7b438279ddaae24cbb966654b65adec133bda66392b69fd0127975c5f4dca64b.jpg](acl_results/807_Task-Specific Information Decomposition for End-to-End Dense Video Captioning/images/7b438279ddaae24cbb966654b65adec133bda66392b69fd0127975c5f4dca64b.jpg)

![ddf8f5f48df0f515acf2f916832d3933934cbdc36836d1f101839b1cb2fa580b.jpg](acl_results/807_Task-Specific Information Decomposition for End-to-End Dense Video Captioning/images/ddf8f5f48df0f515acf2f916832d3933934cbdc36836d1f101839b1cb2fa580b.jpg)

### Tables

![38c72d45653b34b963e076105272f245772a63a3eb193f82d092482f5554bee7.jpg](acl_results/807_Task-Specific Information Decomposition for End-to-End Dense Video Captioning/tables/38c72d45653b34b963e076105272f245772a63a3eb193f82d092482f5554bee7.jpg)

![4d64df5124a1cf95ceb6cd627edee7252620179b584d00580ba0ee2fd91e751a.jpg](acl_results/807_Task-Specific Information Decomposition for End-to-End Dense Video Captioning/tables/4d64df5124a1cf95ceb6cd627edee7252620179b584d00580ba0ee2fd91e751a.jpg)

![7807f6c357360a4eb4f17e556b9337892d982b03c52b8f3aab78ed2b1960cab8.jpg](acl_results/807_Task-Specific Information Decomposition for End-to-End Dense Video Captioning/tables/7807f6c357360a4eb4f17e556b9337892d982b03c52b8f3aab78ed2b1960cab8.jpg)

![f629500ef7bcf7b058d9929d9a22e293121c8b619eca7001c157283b10cac9ad.jpg](acl_results/807_Task-Specific Information Decomposition for End-to-End Dense Video Captioning/tables/f629500ef7bcf7b058d9929d9a22e293121c8b619eca7001c157283b10cac9ad.jpg)

![fb8aa5580b993bc8d3f2fea0d0814ca45fd5189461d1e171f35700bc6d624a74.jpg](acl_results/807_Task-Specific Information Decomposition for End-to-End Dense Video Captioning/tables/fb8aa5580b993bc8d3f2fea0d0814ca45fd5189461d1e171f35700bc6d624a74.jpg)

## CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges

### Images

![0960bdec20641e68893c3a370e6422cead1c59110fe54725bda171d926a82e4f.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/images/0960bdec20641e68893c3a370e6422cead1c59110fe54725bda171d926a82e4f.jpg)

![49d5bce05f07c25544de82a48f2f0c69baf78572de20fa001417454f2d194412.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/images/49d5bce05f07c25544de82a48f2f0c69baf78572de20fa001417454f2d194412.jpg)

![53c5d6bca2f30f1d3b333df6c1a66fe64f55ce9cb262da217dc5c0344622c0e7.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/images/53c5d6bca2f30f1d3b333df6c1a66fe64f55ce9cb262da217dc5c0344622c0e7.jpg)

![5fcd9544cdef9d695e113e9b59dcd0a4309b6372e73c3e830a8b8a449716d8e6.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/images/5fcd9544cdef9d695e113e9b59dcd0a4309b6372e73c3e830a8b8a449716d8e6.jpg)

![acfe5dd4d4b77882509aaf1c8e84ad8499a1a25442f4011a50a8a642d80934fd.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/images/acfe5dd4d4b77882509aaf1c8e84ad8499a1a25442f4011a50a8a642d80934fd.jpg)

![ad73829611f8e41dc6645c9c5d44d61df95accabaa3a6d0aeba4d6c2bde4e020.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/images/ad73829611f8e41dc6645c9c5d44d61df95accabaa3a6d0aeba4d6c2bde4e020.jpg)

### Tables

![0d7bd2309f8bd7a4ec58e2e2bceabcc6b04710bfbd7e296cacbfc7b6fbb821af.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/tables/0d7bd2309f8bd7a4ec58e2e2bceabcc6b04710bfbd7e296cacbfc7b6fbb821af.jpg)

![6e3c18bcd43588b6b31c116c02c52720fb016e9f6def89fdc7d3e820b649de18.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/tables/6e3c18bcd43588b6b31c116c02c52720fb016e9f6def89fdc7d3e820b649de18.jpg)

![6f433021f4f8445962fdc9655f2db7cb012e458df6826427fdec104deb94b1b1.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/tables/6f433021f4f8445962fdc9655f2db7cb012e458df6826427fdec104deb94b1b1.jpg)

![a167f64fb1b2daaf3526804a669ae52b777b444c707df7a01f0a10b9458b2a9b.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/tables/a167f64fb1b2daaf3526804a669ae52b777b444c707df7a01f0a10b9458b2a9b.jpg)

![b823ff972025e73a9263c05b424c7b04b39a3b908c4d5ebb973e29563ab24c1e.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/tables/b823ff972025e73a9263c05b424c7b04b39a3b908c4d5ebb973e29563ab24c1e.jpg)

![db9b11a4aeeb21b65a6b196a337752a6241b1a1bbcd5718214733e15d660c33e.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/tables/db9b11a4aeeb21b65a6b196a337752a6241b1a1bbcd5718214733e15d660c33e.jpg)

![e1f98ab490892ae6f84f62b94e4244eea00be47eed7e69181fe58508497f2766.jpg](acl_results/808_CalibraEval_ Calibrating Prediction Distribution to Mitigate Selection Bias inLLMs-as-Judges/tables/e1f98ab490892ae6f84f62b94e4244eea00be47eed7e69181fe58508497f2766.jpg)

## Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection

### Images

![4df4ba410ea4573138b8543f70c02842576fe6e4ccabe980d4291d72ea9ed784.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/images/4df4ba410ea4573138b8543f70c02842576fe6e4ccabe980d4291d72ea9ed784.jpg)

![6a64253bc083fa6dea6d48e042f9411ca86c794f79a832098d5b3a66dd221a2b.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/images/6a64253bc083fa6dea6d48e042f9411ca86c794f79a832098d5b3a66dd221a2b.jpg)

![9526e62b44e2d547a3873ee400e7fe3f4d69776423fa4281d2998d67eb913ee2.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/images/9526e62b44e2d547a3873ee400e7fe3f4d69776423fa4281d2998d67eb913ee2.jpg)

![97f9ba6a38341d5b6a7609099a4e54d31505279e89e16460ade1850051ec3630.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/images/97f9ba6a38341d5b6a7609099a4e54d31505279e89e16460ade1850051ec3630.jpg)

![9cee6027d5cca03201d1bd5c6ab58380905e857a4fb9037be2663588ea3c38c0.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/images/9cee6027d5cca03201d1bd5c6ab58380905e857a4fb9037be2663588ea3c38c0.jpg)

![a955c8ad5a2493a406f587e08773b6c714d993058e1c985b720ccd320b11e02f.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/images/a955c8ad5a2493a406f587e08773b6c714d993058e1c985b720ccd320b11e02f.jpg)

![b07cf29a2f73026c340c5c652285c5dae947cb044a4dc25c4f1b0d930293b578.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/images/b07cf29a2f73026c340c5c652285c5dae947cb044a4dc25c4f1b0d930293b578.jpg)

### Tables

![166186069993f4c94da9a43594cfd58b2728ac34414f2d66724cfa6af67352b8.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/tables/166186069993f4c94da9a43594cfd58b2728ac34414f2d66724cfa6af67352b8.jpg)

![33588206d234a50f295058873c199dabfaed165666f4caeea8e6156b3235f08d.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/tables/33588206d234a50f295058873c199dabfaed165666f4caeea8e6156b3235f08d.jpg)

![6371dda446e450603753f89538d5f8814a9e7b469381bd5d4a2ae82aeedd534d.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/tables/6371dda446e450603753f89538d5f8814a9e7b469381bd5d4a2ae82aeedd534d.jpg)

![77c183a9bca25e19be28b9d9bb1dc5b4c51d65ead63a1963c8af57958a9d5fd2.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/tables/77c183a9bca25e19be28b9d9bb1dc5b4c51d65ead63a1963c8af57958a9d5fd2.jpg)

![899a771e3c0cb2de5557e43e3b10861c3bd98937143a33111a3293f2ea00c0e2.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/tables/899a771e3c0cb2de5557e43e3b10861c3bd98937143a33111a3293f2ea00c0e2.jpg)

![b32acf39da2e46ddd460dc23a36988b424d73a098c5199cc455444e6ac157b41.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/tables/b32acf39da2e46ddd460dc23a36988b424d73a098c5199cc455444e6ac157b41.jpg)

![d214cd647a386df06cb2ba07e0020686bfbd2a64aea76df68cd5554ba2d3be8a.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/tables/d214cd647a386df06cb2ba07e0020686bfbd2a64aea76df68cd5554ba2d3be8a.jpg)

![d8996ddfb9b7cff6f023cce466675c87fbcdd28cb90c3fa0f2982459942b3a7f.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/tables/d8996ddfb9b7cff6f023cce466675c87fbcdd28cb90c3fa0f2982459942b3a7f.jpg)

![e0463bb7d9c4d3111ecc3c3afa6c4a10a5f5c6a41e14d262b022b5f7b3991381.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/tables/e0463bb7d9c4d3111ecc3c3afa6c4a10a5f5c6a41e14d262b022b5f7b3991381.jpg)

![ea4a3719cd416e49fa1ff82b8fe8f820f3f777845bc4ba6d3d1e11b9f6eb8e1f.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/tables/ea4a3719cd416e49fa1ff82b8fe8f820f3f777845bc4ba6d3d1e11b9f6eb8e1f.jpg)

![ed40fdbca7fa4fc10cb70dff1c026272fcb8e2ad2c56b886857a4f29b4ef1e56.jpg](acl_results/809_Explaining Matters_ Leveraging Definitions and Semantic Expansion for Sexism Detection/tables/ed40fdbca7fa4fc10cb70dff1c026272fcb8e2ad2c56b886857a4f29b4ef1e56.jpg)

## Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models

### Images

![2d078d6a0ef1cdbfaf2e79c444ae6ee331ed5c11768cbf70fa3d846ac7b61e9e.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/images/2d078d6a0ef1cdbfaf2e79c444ae6ee331ed5c11768cbf70fa3d846ac7b61e9e.jpg)

![355ba80e11a02d0656abc602e3fa3972728bcbec10fd9e13e7a922448a827f91.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/images/355ba80e11a02d0656abc602e3fa3972728bcbec10fd9e13e7a922448a827f91.jpg)

![71fe160b131102dcd24c528620fffaff4d59ab07e0183967c79be9ec78c338ee.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/images/71fe160b131102dcd24c528620fffaff4d59ab07e0183967c79be9ec78c338ee.jpg)

![88db44a4c822f4e58b504c8bb4d407a7bc42ee598d01632b520b2e50e83a0f0d.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/images/88db44a4c822f4e58b504c8bb4d407a7bc42ee598d01632b520b2e50e83a0f0d.jpg)

![a3d54a6177c11324a279f26d83b05dfc34b76f456475cfd41c8712ce04132ea0.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/images/a3d54a6177c11324a279f26d83b05dfc34b76f456475cfd41c8712ce04132ea0.jpg)

### Tables

![64dc3f864dbba8d17584553a1abf8ef0af042213847529f0c85e6cf2ba8b693b.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/tables/64dc3f864dbba8d17584553a1abf8ef0af042213847529f0c85e6cf2ba8b693b.jpg)

![7068e561f00f70ccf23e5fa8d1be25a95666a7c6313e674107975e6aa11a78e6.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/tables/7068e561f00f70ccf23e5fa8d1be25a95666a7c6313e674107975e6aa11a78e6.jpg)

![77598a14a14941e040c7df4c599a747aa8cbd9ec05e6abae3e7e4a8d10fbe7f3.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/tables/77598a14a14941e040c7df4c599a747aa8cbd9ec05e6abae3e7e4a8d10fbe7f3.jpg)

![848734e7eb41ad2ee3eb83a71b4b8d2e4ae19ca8e4cc00d69b54ec97680c63e8.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/tables/848734e7eb41ad2ee3eb83a71b4b8d2e4ae19ca8e4cc00d69b54ec97680c63e8.jpg)

![87e1c34c08a45ece285a6b933b8d940b184f99c872abdfda2989070517c24004.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/tables/87e1c34c08a45ece285a6b933b8d940b184f99c872abdfda2989070517c24004.jpg)

![a8bd3978b32aa9c7c8f5df825f73a08c65754791b916e4f8501f574b7415d70a.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/tables/a8bd3978b32aa9c7c8f5df825f73a08c65754791b916e4f8501f574b7415d70a.jpg)

![b5e65645561ad91a2f4f699ff409546b3b4cdb85d81e5518097728b06bd57723.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/tables/b5e65645561ad91a2f4f699ff409546b3b4cdb85d81e5518097728b06bd57723.jpg)

![b7db37338941cbb93f117283ec428f29d68c4c76c41b7ea7a91268ba0e02c3d8.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/tables/b7db37338941cbb93f117283ec428f29d68c4c76c41b7ea7a91268ba0e02c3d8.jpg)

![d5c1efdad8ceb3e7dfc4b7daf488f36132b3b72701abdc0404ddda51a5f8c256.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/tables/d5c1efdad8ceb3e7dfc4b7daf488f36132b3b72701abdc0404ddda51a5f8c256.jpg)

![f4b37e9d02478abe9adaf6e89cc875c6ac5f5a35194e5cf8c389d67d932e4528.jpg](acl_results/810_Private Memorization Editing_ Turning Memorization into a Defense to Strengthen Data Privacy in Larg/tables/f4b37e9d02478abe9adaf6e89cc875c6ac5f5a35194e5cf8c389d67d932e4528.jpg)

## PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning

### Images

![39e6c45df794485cfdc7587f8dce2d932e392d2a60e025a92a3d87b401848aca.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/images/39e6c45df794485cfdc7587f8dce2d932e392d2a60e025a92a3d87b401848aca.jpg)

![3f880cb55f564414e792c2ce910a15396717e02a342417730fea52076c9f8124.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/images/3f880cb55f564414e792c2ce910a15396717e02a342417730fea52076c9f8124.jpg)

![4d460e5185fd50d03b1b3cd429f4737cc23800979ae5796179783c20e5e2a0bb.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/images/4d460e5185fd50d03b1b3cd429f4737cc23800979ae5796179783c20e5e2a0bb.jpg)

![7d7716243278b9b180e3227c04e3efae74fe6359b1cd1bf34074c6efe63ded70.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/images/7d7716243278b9b180e3227c04e3efae74fe6359b1cd1bf34074c6efe63ded70.jpg)

![a5329a83f8fc6a8bf2bbfcef9ac08020923cb41cc04e2bf525634b99667a7382.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/images/a5329a83f8fc6a8bf2bbfcef9ac08020923cb41cc04e2bf525634b99667a7382.jpg)

![b640e726bc6542331c0e29b308ef2ecabff553f537f1c9eb0c71edd511d16811.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/images/b640e726bc6542331c0e29b308ef2ecabff553f537f1c9eb0c71edd511d16811.jpg)

![c51babaded85f45fe57ac385435bf5c31401b891121bf38f7acc90751cf1f59d.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/images/c51babaded85f45fe57ac385435bf5c31401b891121bf38f7acc90751cf1f59d.jpg)

![cfc3925df9c356b614f06552c2ce04426c20a1126631d3949914f35d190376b0.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/images/cfc3925df9c356b614f06552c2ce04426c20a1126631d3949914f35d190376b0.jpg)

![d4391218d2d10d7cf1f216b16d6246e749c55aa5bb85e82ec26dfb44bed90712.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/images/d4391218d2d10d7cf1f216b16d6246e749c55aa5bb85e82ec26dfb44bed90712.jpg)

![ece1d9a19d422fb6228276cf0a57c8ac46a804db56d6dfc11375652c81e132d4.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/images/ece1d9a19d422fb6228276cf0a57c8ac46a804db56d6dfc11375652c81e132d4.jpg)

### Tables

![06ce6eedd9f759183440e3b610bc14a275aab7e29b5da6a288982f51d0012b1e.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/tables/06ce6eedd9f759183440e3b610bc14a275aab7e29b5da6a288982f51d0012b1e.jpg)

![28eedcbfeda364f5b7634e49a5a145d220c68ed58852225945537764f52a77b2.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/tables/28eedcbfeda364f5b7634e49a5a145d220c68ed58852225945537764f52a77b2.jpg)

![4e1581a39e5db7a1c2285cd8af673f5e701ae98bfca47c649fdf2cab40e9c856.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/tables/4e1581a39e5db7a1c2285cd8af673f5e701ae98bfca47c649fdf2cab40e9c856.jpg)

![5a8c98c5ab209732bf98acd5c77caeef5e7a034c12710d2f4688603d9b0e7ea6.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/tables/5a8c98c5ab209732bf98acd5c77caeef5e7a034c12710d2f4688603d9b0e7ea6.jpg)

![65409ff7ef40adc3569d999d6bd44178f3510977a239939b26c7143708074709.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/tables/65409ff7ef40adc3569d999d6bd44178f3510977a239939b26c7143708074709.jpg)

![6bb9cd252dcdbaa6a81ee31db17bc1605d3da6c642dfcd800c259f9808579b42.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/tables/6bb9cd252dcdbaa6a81ee31db17bc1605d3da6c642dfcd800c259f9808579b42.jpg)

![864c093081194f584b04b6355f441afeee10cada3ad26593e055a6b7340e7059.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/tables/864c093081194f584b04b6355f441afeee10cada3ad26593e055a6b7340e7059.jpg)

![cb25a2766c315efca3ad55cd5d4096f811ba0943a896cdaac0cd780f3b2438e1.jpg](acl_results/811_PhysReason_ A Comprehensive Benchmark towards Physics-Based Reasoning/tables/cb25a2766c315efca3ad55cd5d4096f811ba0943a896cdaac0cd780f3b2438e1.jpg)

## Does Time Have Its Place? Temporal Heads: Where Language Models Recall Time-specific Information

### Images

![0720d1ea3ac3b5391e6409e03f45f2a5696fd50a3ede8c299d51bc30364226f1.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/0720d1ea3ac3b5391e6409e03f45f2a5696fd50a3ede8c299d51bc30364226f1.jpg)

![093e0b26d73822c7d3b5b33a847f1f2068b82d72c6f0515265e11f23efd99f4d.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/093e0b26d73822c7d3b5b33a847f1f2068b82d72c6f0515265e11f23efd99f4d.jpg)

![0cacab3c0655f3d9a0a59b7f229e7df0804dfa09f87397dfd777e1a93685b7be.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/0cacab3c0655f3d9a0a59b7f229e7df0804dfa09f87397dfd777e1a93685b7be.jpg)

![1d4e379a400b93efff5b6e0217b93929c747c782d3c95b1b9594ce5aa0f8a2db.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/1d4e379a400b93efff5b6e0217b93929c747c782d3c95b1b9594ce5aa0f8a2db.jpg)

![24a70ac8e50e29dd8e4fdee5b58222eb7d89f86993aacc48ade92c1a89a81cf3.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/24a70ac8e50e29dd8e4fdee5b58222eb7d89f86993aacc48ade92c1a89a81cf3.jpg)

![2aa5abd9b9617d1eb84794a90a4ae00828696ce4130fa27692c93273e3a9e1fe.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/2aa5abd9b9617d1eb84794a90a4ae00828696ce4130fa27692c93273e3a9e1fe.jpg)

![3759c8f5334d19eaad74ecbf5a4728983ee308096e34788c2852bf8cc9769707.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/3759c8f5334d19eaad74ecbf5a4728983ee308096e34788c2852bf8cc9769707.jpg)

![47ae0ef8d19b729fbac724f3d27fab35a43e74f2eadcacd3609240512367845b.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/47ae0ef8d19b729fbac724f3d27fab35a43e74f2eadcacd3609240512367845b.jpg)

![55593d8f57595d68e32dbda1e6ec273c43b5540f53f2476d1ebbaf5494fbcdbf.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/55593d8f57595d68e32dbda1e6ec273c43b5540f53f2476d1ebbaf5494fbcdbf.jpg)

![67dcf7d74f843eff96e9b1c415a63d8f3cd07e3f75ec056854fb1e94dfd6359d.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/67dcf7d74f843eff96e9b1c415a63d8f3cd07e3f75ec056854fb1e94dfd6359d.jpg)

![738b9500c82a6a73ab7280a4327b8c0b2ac0da358739351480685bb292c2e03e.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/738b9500c82a6a73ab7280a4327b8c0b2ac0da358739351480685bb292c2e03e.jpg)

![7ed339f5301d916c957fbd233163f985d57025ff9830e8051b319a22a804e949.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/7ed339f5301d916c957fbd233163f985d57025ff9830e8051b319a22a804e949.jpg)

![9376a1a2472b059b914c05adb79ec4d2f879d908a424ba7261c738f258130869.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/9376a1a2472b059b914c05adb79ec4d2f879d908a424ba7261c738f258130869.jpg)

![c9c0c54e722b8bad97a7097f3c0f3d607752d061dab023f51c768e0c8955ad6a.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/c9c0c54e722b8bad97a7097f3c0f3d607752d061dab023f51c768e0c8955ad6a.jpg)

![c9f1a5e1a6ef2cc03bc37ee72e623558ce783ae9013cef611ab2de7c0b9e50a5.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/images/c9f1a5e1a6ef2cc03bc37ee72e623558ce783ae9013cef611ab2de7c0b9e50a5.jpg)

### Tables

![348e9afe722b4fd71cb90947f9eb41b3963be31d44797bc6a68c35c9819b3005.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/tables/348e9afe722b4fd71cb90947f9eb41b3963be31d44797bc6a68c35c9819b3005.jpg)

![4d7c20f39237523374c7e4a3e1e5c1a7138abbad554f6a0565d07e3f9367f8e6.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/tables/4d7c20f39237523374c7e4a3e1e5c1a7138abbad554f6a0565d07e3f9367f8e6.jpg)

![89c021df7d6c14306ba422b27d0dc0e80f4272c901da68864412973409058421.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/tables/89c021df7d6c14306ba422b27d0dc0e80f4272c901da68864412973409058421.jpg)

![9722c640053d6eb19010c0a244dd16d48a0ad482602f421607de4ceacd06da28.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/tables/9722c640053d6eb19010c0a244dd16d48a0ad482602f421607de4ceacd06da28.jpg)

![aa46c72f27ad84fbeb07af94b82c12f69433984745e83f625e3c056054a62150.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/tables/aa46c72f27ad84fbeb07af94b82c12f69433984745e83f625e3c056054a62150.jpg)

![aefa544d6c2639ddafa55a22a7da217f288396a42b6bfb4c80a415dc761e460b.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/tables/aefa544d6c2639ddafa55a22a7da217f288396a42b6bfb4c80a415dc761e460b.jpg)

![d6ed21da1d61b3ebf21dd3e63ebb9c73e940aebd49fbd448d98e38e763fdcc66.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/tables/d6ed21da1d61b3ebf21dd3e63ebb9c73e940aebd49fbd448d98e38e763fdcc66.jpg)

![e2d34a778ded6baf0f9ecd3e23b19f0fd70ca5010097c79ec09d5266bffd5f18.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/tables/e2d34a778ded6baf0f9ecd3e23b19f0fd70ca5010097c79ec09d5266bffd5f18.jpg)

![ff144aa5ae991789d4acec72e9b77b1916c6713b96c5e855b5c7cfe3d02efe00.jpg](acl_results/812_Does Time Have Its Place_ Temporal Heads_ Where Language Models Recall Time-specific Information/tables/ff144aa5ae991789d4acec72e9b77b1916c6713b96c5e855b5c7cfe3d02efe00.jpg)

## Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training

### Images

![24703de205f60b5d5429da98c7c4be92d77b42f8c4769cd593ec3acaf4a1f256.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/images/24703de205f60b5d5429da98c7c4be92d77b42f8c4769cd593ec3acaf4a1f256.jpg)

![4546a601d59c83e57700125716d05dc5b763048f97fcb375cc5b5a1bbd9a1ba9.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/images/4546a601d59c83e57700125716d05dc5b763048f97fcb375cc5b5a1bbd9a1ba9.jpg)

![b03d8569c27dab9ae705b36dc52aaf69f18ab587e3dad042362c4afc6f3d434d.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/images/b03d8569c27dab9ae705b36dc52aaf69f18ab587e3dad042362c4afc6f3d434d.jpg)

![cd0bf05debbe447afd0b63631b4a1d9666b0db9e74bf88fc13d719085078aacf.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/images/cd0bf05debbe447afd0b63631b4a1d9666b0db9e74bf88fc13d719085078aacf.jpg)

### Tables

![063c1dcbcddd923c6f273bcca31a4f9551dd5fc32b280a5ca82a82f305abf841.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/tables/063c1dcbcddd923c6f273bcca31a4f9551dd5fc32b280a5ca82a82f305abf841.jpg)

![0b036f6947bd4d2f155a1926701fd39d676538c55dc3f6b949c4294f96b5737f.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/tables/0b036f6947bd4d2f155a1926701fd39d676538c55dc3f6b949c4294f96b5737f.jpg)

![1d44b995988b09a830305f4313d4ab05ad23d4cbe25f4d404d0ec22f82c73334.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/tables/1d44b995988b09a830305f4313d4ab05ad23d4cbe25f4d404d0ec22f82c73334.jpg)

![33e68cd27bd655d5ab9c0c949ecd32f55e00861eb5a973099099c1760ad0b53c.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/tables/33e68cd27bd655d5ab9c0c949ecd32f55e00861eb5a973099099c1760ad0b53c.jpg)

![4172ffc08df5b5b5a6b042a74dd180c9e710a6b483d59950ba8e40e161a141bd.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/tables/4172ffc08df5b5b5a6b042a74dd180c9e710a6b483d59950ba8e40e161a141bd.jpg)

![59c8bcf21ec645c677585a11f17506a97a73e93a9c82b8d9e48a773faa8269a2.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/tables/59c8bcf21ec645c677585a11f17506a97a73e93a9c82b8d9e48a773faa8269a2.jpg)

![647f7f1f14c3a5b30a75c3ca86a6dfb8e4b0e68dae969e282d86ceaf9304e478.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/tables/647f7f1f14c3a5b30a75c3ca86a6dfb8e4b0e68dae969e282d86ceaf9304e478.jpg)

![681a75a7102819ff347561b8e7bc3465516b90f5f276f483ccc49e47e8dd1d9a.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/tables/681a75a7102819ff347561b8e7bc3465516b90f5f276f483ccc49e47e8dd1d9a.jpg)

![7b0543b0b03fa023c997b8b16cf4158d1092adaddd2f18bfabbd34c708fb94da.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/tables/7b0543b0b03fa023c997b8b16cf4158d1092adaddd2f18bfabbd34c708fb94da.jpg)

![cf5a6b7f766bb818b6597d2220201da144390b07ec61586807cf24ab584c87b7.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/tables/cf5a6b7f766bb818b6597d2220201da144390b07ec61586807cf24ab584c87b7.jpg)

![f049f1e1b860dc3c31d194199dfd72a67e53231c6f025c34e44df2fc11650748.jpg](acl_results/813_Velocitune_ A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training/tables/f049f1e1b860dc3c31d194199dfd72a67e53231c6f025c34e44df2fc11650748.jpg)

## Sheep’s Skin, Wolf’s Deeds: AreLLMs Ready for Metaphorical Implicit Hate Speech?

### Images

![498e8c775c0a1411dc0c71c12211ec68822a8b15a4d1a128bb440ccdfbab6bcf.jpg](acl_results/814_Sheep’s Skin, Wolf’s Deeds_ AreLLMs Ready for Metaphorical Implicit Hate Speech_/images/498e8c775c0a1411dc0c71c12211ec68822a8b15a4d1a128bb440ccdfbab6bcf.jpg)

![586c62747429a4a203c3d56a52937b327d5e296200b0bf0489e549551ea25b4b.jpg](acl_results/814_Sheep’s Skin, Wolf’s Deeds_ AreLLMs Ready for Metaphorical Implicit Hate Speech_/images/586c62747429a4a203c3d56a52937b327d5e296200b0bf0489e549551ea25b4b.jpg)

![db83c214104b650f7189c40cb43ad9069f22a1cb89d06306235f29bdb3c15f16.jpg](acl_results/814_Sheep’s Skin, Wolf’s Deeds_ AreLLMs Ready for Metaphorical Implicit Hate Speech_/images/db83c214104b650f7189c40cb43ad9069f22a1cb89d06306235f29bdb3c15f16.jpg)

![de8b8900a673b315c072c1004c24a5d2316127af87c16a724c9b21556e9ad7dd.jpg](acl_results/814_Sheep’s Skin, Wolf’s Deeds_ AreLLMs Ready for Metaphorical Implicit Hate Speech_/images/de8b8900a673b315c072c1004c24a5d2316127af87c16a724c9b21556e9ad7dd.jpg)

### Tables

![cda4f10077010bcb4c096a05b13846e39944437e415b4637c952bc77294f8e80.jpg](acl_results/814_Sheep’s Skin, Wolf’s Deeds_ AreLLMs Ready for Metaphorical Implicit Hate Speech_/tables/cda4f10077010bcb4c096a05b13846e39944437e415b4637c952bc77294f8e80.jpg)

![d0cd996a781b2d54e0673638d44b937ea0d03a19c7547c8e74dbb8f614e1930d.jpg](acl_results/814_Sheep’s Skin, Wolf’s Deeds_ AreLLMs Ready for Metaphorical Implicit Hate Speech_/tables/d0cd996a781b2d54e0673638d44b937ea0d03a19c7547c8e74dbb8f614e1930d.jpg)

## Neuron-Level Sequential Editing for Large Language Models

### Images

![3491fa94abd77dbb8820e08cbd96305b9b96d90d1a2a6c950e9521183f873517.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/images/3491fa94abd77dbb8820e08cbd96305b9b96d90d1a2a6c950e9521183f873517.jpg)

![4f775ab1149a150270868a551c61fb1d8801811cec2449833d74e4421dd40ddf.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/images/4f775ab1149a150270868a551c61fb1d8801811cec2449833d74e4421dd40ddf.jpg)

![510ef85f30030a86f7bc35928d41b462286a4f137d134241121c1982f62c8b95.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/images/510ef85f30030a86f7bc35928d41b462286a4f137d134241121c1982f62c8b95.jpg)

![57d603015943b91c292fab75a981ef90988aa86b0f32a0c16ce085a4b43db78d.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/images/57d603015943b91c292fab75a981ef90988aa86b0f32a0c16ce085a4b43db78d.jpg)

![da06218f5285430b242178c69a75149ccae972ffb13a4927babc41b8268f3ea8.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/images/da06218f5285430b242178c69a75149ccae972ffb13a4927babc41b8268f3ea8.jpg)

![fb73497301407c5aa65a72f1acce532e7c2b5f6f1b48269b83e62a973ddfe4ba.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/images/fb73497301407c5aa65a72f1acce532e7c2b5f6f1b48269b83e62a973ddfe4ba.jpg)

### Tables

![276c3765801cfdbad43d7a4650f722633d83aa408f3670adc9d4b5fe7ffdfb4c.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/tables/276c3765801cfdbad43d7a4650f722633d83aa408f3670adc9d4b5fe7ffdfb4c.jpg)

![61e857e70ecd641d6295ccf311e02a7b77ad78722303d280e5f19dfafc02a21f.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/tables/61e857e70ecd641d6295ccf311e02a7b77ad78722303d280e5f19dfafc02a21f.jpg)

![68335734add19fb9580a6d446556aff729d5702ef724b701310e21bb3ac76126.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/tables/68335734add19fb9580a6d446556aff729d5702ef724b701310e21bb3ac76126.jpg)

![73ce389e5ba24c1e7a7ddda8fddf6b2ad37b869eece37992579cfdf40126ca25.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/tables/73ce389e5ba24c1e7a7ddda8fddf6b2ad37b869eece37992579cfdf40126ca25.jpg)

![79c92e2442ff36718c3a95d2d98ef37120fc523e6a57022c2879fc7c218c7131.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/tables/79c92e2442ff36718c3a95d2d98ef37120fc523e6a57022c2879fc7c218c7131.jpg)

![7bdf2b86e7b2ddd60a5b7827b6e8a45a45b94c3f8442922c0f3b0684c3f369d4.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/tables/7bdf2b86e7b2ddd60a5b7827b6e8a45a45b94c3f8442922c0f3b0684c3f369d4.jpg)

![b4b0497e0d9928e53e9e85187026c7da49e04809925547ed006e4205d87cd098.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/tables/b4b0497e0d9928e53e9e85187026c7da49e04809925547ed006e4205d87cd098.jpg)

![e4a7410c34a8f9f1506bb50c80e4e066bf78dde26cd5156a305f0190e008bd9d.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/tables/e4a7410c34a8f9f1506bb50c80e4e066bf78dde26cd5156a305f0190e008bd9d.jpg)

![e8a54158e240924e29df9ad4ffb3de829640f01de2c5c971b0c7889904ec4b72.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/tables/e8a54158e240924e29df9ad4ffb3de829640f01de2c5c971b0c7889904ec4b72.jpg)

![eb05ed58a138e56003b1090ecca3511322d0c194ed64a1032606ad72ad44616e.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/tables/eb05ed58a138e56003b1090ecca3511322d0c194ed64a1032606ad72ad44616e.jpg)

![f32dd156f7ad928a35fc8177754d9c46ced2f9e1ab10e45a0f01fc5acbbbff6f.jpg](acl_results/815_Neuron-Level Sequential Editing for Large Language Models/tables/f32dd156f7ad928a35fc8177754d9c46ced2f9e1ab10e45a0f01fc5acbbbff6f.jpg)

## Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts

### Images

![1e1a0125d072825cf008181a3cfd3b96fad1115145683e027193e1f3c3922908.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/images/1e1a0125d072825cf008181a3cfd3b96fad1115145683e027193e1f3c3922908.jpg)

![2667bf9fe5a0f30762464466174ca48f0b5e0b1830f97582bcdaadbc176bb8cb.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/images/2667bf9fe5a0f30762464466174ca48f0b5e0b1830f97582bcdaadbc176bb8cb.jpg)

![54745735a29d50e316919150e3d1c1126882dcfa4c28b59133014f77f2c51e21.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/images/54745735a29d50e316919150e3d1c1126882dcfa4c28b59133014f77f2c51e21.jpg)

![6dc19f839bee7afd7484a13d57c34fafba3dd901acb6d1d1cf83c0ce9060e3ae.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/images/6dc19f839bee7afd7484a13d57c34fafba3dd901acb6d1d1cf83c0ce9060e3ae.jpg)

![d86d043ab3633cbffb94ece9513299ce0bfc5e7f771d3760e892a577fbd44aa0.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/images/d86d043ab3633cbffb94ece9513299ce0bfc5e7f771d3760e892a577fbd44aa0.jpg)

### Tables

![13f021a873f26e9b1f969df4c31e8f330f30a083f931c3c2eed9bcf717ee98e7.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/13f021a873f26e9b1f969df4c31e8f330f30a083f931c3c2eed9bcf717ee98e7.jpg)

![497316078af22212e632acdea3b5c0d049b7739f33a44e2e39de8078f99a6976.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/497316078af22212e632acdea3b5c0d049b7739f33a44e2e39de8078f99a6976.jpg)

![570c5e42073df99ff95259be2e47a9b4b784706d2daf94b48a050b10f760d238.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/570c5e42073df99ff95259be2e47a9b4b784706d2daf94b48a050b10f760d238.jpg)

![7d847de1c0168cb22bc6eedb2c796227249335212a3723d913e629ad9d8a56d3.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/7d847de1c0168cb22bc6eedb2c796227249335212a3723d913e629ad9d8a56d3.jpg)

![8ad5c20495be791c7ea67c77e729c24e85f2b54b40c8df94783e3832a36cd99d.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/8ad5c20495be791c7ea67c77e729c24e85f2b54b40c8df94783e3832a36cd99d.jpg)

![9479bfec3982894016c4d193ad997a3cc4329e9ecd326082efd3a31693557fd3.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/9479bfec3982894016c4d193ad997a3cc4329e9ecd326082efd3a31693557fd3.jpg)

![d811293ee9b44338549f1114011b001af0b64d3edc3774637bb03eac4486e633.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/d811293ee9b44338549f1114011b001af0b64d3edc3774637bb03eac4486e633.jpg)

![d87d8fb9a247896a46c0fb977ba217a17a131ff6016dd3103a6ebecc8ace2a94.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/d87d8fb9a247896a46c0fb977ba217a17a131ff6016dd3103a6ebecc8ace2a94.jpg)

![dd458612515a4e9ffadc67a90b13613433d1641fea9edd0bc732406c6085be17.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/dd458612515a4e9ffadc67a90b13613433d1641fea9edd0bc732406c6085be17.jpg)

![df5534f3256c6ed51163e3a4cfc6e5379a0d8299eb563d10d7bf21129480f44a.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/df5534f3256c6ed51163e3a4cfc6e5379a0d8299eb563d10d7bf21129480f44a.jpg)

![e0fa7783f08adc477079ebc003e7eda8dc001d04bb39a2ffa06675dfe9e7ea11.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/e0fa7783f08adc477079ebc003e7eda8dc001d04bb39a2ffa06675dfe9e7ea11.jpg)

![e9782be9e1f4114a90691e2ef7d0557b95c41da7a1ace7c42d643d20ff1e9e8e.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/e9782be9e1f4114a90691e2ef7d0557b95c41da7a1ace7c42d643d20ff1e9e8e.jpg)

![ea47853a7fd48c7fce2f0426b6bccf1af3a1e8e8eee94abe01d9cc7c614344e9.jpg](acl_results/816_Automatic Expert Discovery inLLMUpcycling via Sparse Interpolated Mixture-of-Experts/tables/ea47853a7fd48c7fce2f0426b6bccf1af3a1e8e8eee94abe01d9cc7c614344e9.jpg)

## SimulS2S-LLM: Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation

### Images

![18ccef3c30f493d3b50af25b1a7bf67a54993b4ef0b6eea47c89df865dd6aef2.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/images/18ccef3c30f493d3b50af25b1a7bf67a54993b4ef0b6eea47c89df865dd6aef2.jpg)

![1b27b6d3438553be13003e77d10b02d9e414ca2d2e4810d842ff24289e0bb918.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/images/1b27b6d3438553be13003e77d10b02d9e414ca2d2e4810d842ff24289e0bb918.jpg)

![28b22e08ced19778e19fc890a57add916cba9bd50380dd6fa4d9ee51e1775922.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/images/28b22e08ced19778e19fc890a57add916cba9bd50380dd6fa4d9ee51e1775922.jpg)

![52536f8fed151860e1512d7a118287eff3719fd75363e988f4259db0e3c708c0.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/images/52536f8fed151860e1512d7a118287eff3719fd75363e988f4259db0e3c708c0.jpg)

![58eeb854e8182bfc9fb5ca978eadb094a4a477a5240b242e91d66b5fbafc97c1.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/images/58eeb854e8182bfc9fb5ca978eadb094a4a477a5240b242e91d66b5fbafc97c1.jpg)

![8509a3c4b33c59058f7dcd5da8f4acb5c15e230a7f32058aab8ff8a8b86b03e1.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/images/8509a3c4b33c59058f7dcd5da8f4acb5c15e230a7f32058aab8ff8a8b86b03e1.jpg)

![a02ede6b444d4badf94f0fc1c128b6a52b1e07f69288d55d88717056f9a4c177.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/images/a02ede6b444d4badf94f0fc1c128b6a52b1e07f69288d55d88717056f9a4c177.jpg)

![a59ec5008e9529baee63afadaf4eafe0c47cca56864037a632ad8ef4aaf8b53f.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/images/a59ec5008e9529baee63afadaf4eafe0c47cca56864037a632ad8ef4aaf8b53f.jpg)

### Tables

![07d3d60e8cbe9d598fe5b1af867fd804456b2c60621b7c5a6b4b8c404a2a4040.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/tables/07d3d60e8cbe9d598fe5b1af867fd804456b2c60621b7c5a6b4b8c404a2a4040.jpg)

![0fe5faba582045d0a01ee645bd650958202300fd33c96ce7ab57e823307c1d06.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/tables/0fe5faba582045d0a01ee645bd650958202300fd33c96ce7ab57e823307c1d06.jpg)

![14802e46c0ed297e8070970baa14142210f669d0cca9273015b670f485725853.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/tables/14802e46c0ed297e8070970baa14142210f669d0cca9273015b670f485725853.jpg)

![1699f6909df27006c1b12330ef034f460f1212c029f1366a053b2c00fafff47f.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/tables/1699f6909df27006c1b12330ef034f460f1212c029f1366a053b2c00fafff47f.jpg)

![24b8283422a0e6a8d27ecf1fab5d41b5ac6d5b61336646cf6b36262d7c502676.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/tables/24b8283422a0e6a8d27ecf1fab5d41b5ac6d5b61336646cf6b36262d7c502676.jpg)

![3eee196397e481916f2c522ac3b84b58ebd06402d1e736c39b819c8da11f9a7c.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/tables/3eee196397e481916f2c522ac3b84b58ebd06402d1e736c39b819c8da11f9a7c.jpg)

![5181941b8b15cc853568ada7f5974d273395319acd93debbaa019306d2d4ea9e.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/tables/5181941b8b15cc853568ada7f5974d273395319acd93debbaa019306d2d4ea9e.jpg)

![9a3985d4f1f5054ffc1e2ed01e0a610b6945b851d76f7a5d65d324e75572a072.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/tables/9a3985d4f1f5054ffc1e2ed01e0a610b6945b851d76f7a5d65d324e75572a072.jpg)

![a0418aa8a3d8e9a8b242dc9589b19518571d6f893fbb302bca326fea7b777758.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/tables/a0418aa8a3d8e9a8b242dc9589b19518571d6f893fbb302bca326fea7b777758.jpg)

![e32542e99b5f74c1009f7195fafce64ca640926ff944701e43d68b6cf067dc4a.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/tables/e32542e99b5f74c1009f7195fafce64ca640926ff944701e43d68b6cf067dc4a.jpg)

![e4ebc7283e6a5108187908076af678b9f5fa0187ee0e5cf1ce54dacaadd96dce.jpg](acl_results/817_SimulS2S-LLM_ Unlocking Simultaneous Inference of SpeechLLMs for Speech-to-Speech Translation/tables/e4ebc7283e6a5108187908076af678b9f5fa0187ee0e5cf1ce54dacaadd96dce.jpg)

## VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models

### Images

![ce42e4245146b513322b60a63ff698d20ed6017984f7add49f203472b6b3746e.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/images/ce42e4245146b513322b60a63ff698d20ed6017984f7add49f203472b6b3746e.jpg)

![d2948d0a7198aab56c18a8806ee829f71f0c76412b0074e865b1affc37b6d5b8.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/images/d2948d0a7198aab56c18a8806ee829f71f0c76412b0074e865b1affc37b6d5b8.jpg)

![dc5c5296da4a677092bc246d4734bccb131267512c183c55e30400f68beed6ef.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/images/dc5c5296da4a677092bc246d4734bccb131267512c183c55e30400f68beed6ef.jpg)

![f0e77823b8ba0449a3d67fea30d6ff65b27fe6c2d3263755caf2122a598c6a04.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/images/f0e77823b8ba0449a3d67fea30d6ff65b27fe6c2d3263755caf2122a598c6a04.jpg)

### Tables

![14a417a988cfd7317d1c283713ab193e81af36a3ed60c8600d369388dc72be57.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/14a417a988cfd7317d1c283713ab193e81af36a3ed60c8600d369388dc72be57.jpg)

![26c80cc93b5863e2321cf65fc2f21c868282fab94065aab0a162d858802772a8.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/26c80cc93b5863e2321cf65fc2f21c868282fab94065aab0a162d858802772a8.jpg)

![2abae9d6429043b6683501bee01d2a6e7771a67dcc0cff753e6e5c2a921e5ca8.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/2abae9d6429043b6683501bee01d2a6e7771a67dcc0cff753e6e5c2a921e5ca8.jpg)

![40f433f83c6f043d433b68d33f4ccfbe0fc4ff9d03a79a4ce71324ae2b3c3e6d.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/40f433f83c6f043d433b68d33f4ccfbe0fc4ff9d03a79a4ce71324ae2b3c3e6d.jpg)

![58d52121d3a7271e812842e8b2c7b0fc213f048364949919d05c28c1e8f88300.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/58d52121d3a7271e812842e8b2c7b0fc213f048364949919d05c28c1e8f88300.jpg)

![5925cec8f5773e17e760cf7e44effbd4a5ed2262b56ca59204c3f88ea464e5e8.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/5925cec8f5773e17e760cf7e44effbd4a5ed2262b56ca59204c3f88ea464e5e8.jpg)

![5b19f025d876b2ab2a82f9073fb83bcfed78fd6c11f36c1cddb991ed49a4a0b0.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/5b19f025d876b2ab2a82f9073fb83bcfed78fd6c11f36c1cddb991ed49a4a0b0.jpg)

![625d49c685687166fabe4b8939144ebdb4761abe7b958a2411a98e08b545d5c0.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/625d49c685687166fabe4b8939144ebdb4761abe7b958a2411a98e08b545d5c0.jpg)

![6662e0bcdea0421f985e7aa567ebf68494d7ab8b2c3f94c2445398cc1687d028.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/6662e0bcdea0421f985e7aa567ebf68494d7ab8b2c3f94c2445398cc1687d028.jpg)

![70772d85e511ca1d25a004b1d03de0a395ac059c2d6008471d5fcf5b893c7f1c.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/70772d85e511ca1d25a004b1d03de0a395ac059c2d6008471d5fcf5b893c7f1c.jpg)

![73ccb11eb71047e9371687caa07f31849837e04c74eb9cfbfef943c4fedee0ad.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/73ccb11eb71047e9371687caa07f31849837e04c74eb9cfbfef943c4fedee0ad.jpg)

![78dd7d1e323269f656928f7c26d10e680428cdf0148a1fa2ae00576db388cd48.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/78dd7d1e323269f656928f7c26d10e680428cdf0148a1fa2ae00576db388cd48.jpg)

![850b676b5082cb4594d8ace72034c8abd274fded88e503d6d8d76d4d5da37bcf.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/850b676b5082cb4594d8ace72034c8abd274fded88e503d6d8d76d4d5da37bcf.jpg)

![ab04478556357819c43bd4692c5dccb385f0a3d508ddca1b47152a008cdeef4d.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/ab04478556357819c43bd4692c5dccb385f0a3d508ddca1b47152a008cdeef4d.jpg)

![bfd3a72747d6908d3d70cba6b57ac09e73f3c73506f5576d23dd1e9e21ff2fc5.jpg](acl_results/818_VoxEval_ Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models/tables/bfd3a72747d6908d3d70cba6b57ac09e73f3c73506f5576d23dd1e9e21ff2fc5.jpg)

## RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation

### Images

![1b1619b1a6e1fb0ad4c869fb92c4608c7cfdbef16ae6bf96790a643df82afbfd.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/images/1b1619b1a6e1fb0ad4c869fb92c4608c7cfdbef16ae6bf96790a643df82afbfd.jpg)

![5ffe25922ae2611108771bcf5a081341b2c8a1d5aa32c3ee75478d95b26cf552.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/images/5ffe25922ae2611108771bcf5a081341b2c8a1d5aa32c3ee75478d95b26cf552.jpg)

![683f8dcf11bde916e7c096da06a3a9d3954f74fd0164f1c21ad00fd8327dc6f4.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/images/683f8dcf11bde916e7c096da06a3a9d3954f74fd0164f1c21ad00fd8327dc6f4.jpg)

![d877395bddb407691fd6c887ed81592e0a1aae5c40720977e30b5415b917b645.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/images/d877395bddb407691fd6c887ed81592e0a1aae5c40720977e30b5415b917b645.jpg)

![f6d99cc927c13fd7dc9babf30d861f726c94a11369b01ea6dbf35001d326dcd5.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/images/f6d99cc927c13fd7dc9babf30d861f726c94a11369b01ea6dbf35001d326dcd5.jpg)

### Tables

![28747e4ceb5248036a596ceb47fe3addb54a755126fb313569c4d76a34b94106.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/tables/28747e4ceb5248036a596ceb47fe3addb54a755126fb313569c4d76a34b94106.jpg)

![4a95d754ae0720b71a36452eae38defeb1f5a7380e2be2e8b9685d7e97961aba.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/tables/4a95d754ae0720b71a36452eae38defeb1f5a7380e2be2e8b9685d7e97961aba.jpg)

![63f38075d88c31dfc9bc2281e38ea69808dcdd704184ba849843d899c26e855e.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/tables/63f38075d88c31dfc9bc2281e38ea69808dcdd704184ba849843d899c26e855e.jpg)

![6f569214b56fafb8f3a5ec2d2d522d7812c45f431343be5a469e8c834843d295.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/tables/6f569214b56fafb8f3a5ec2d2d522d7812c45f431343be5a469e8c834843d295.jpg)

![7fe9e3e7498c8709afcbdfba942a7d0f0aa0e5298fc8f70b6c66a57cd69daaaf.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/tables/7fe9e3e7498c8709afcbdfba942a7d0f0aa0e5298fc8f70b6c66a57cd69daaaf.jpg)

![94e9ce2728bf3b9fb5caf9984e5eb81ce7fca2baee123fb93368f362068a6bc0.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/tables/94e9ce2728bf3b9fb5caf9984e5eb81ce7fca2baee123fb93368f362068a6bc0.jpg)

![d458e9e481d43c36e6799eb1178b4bcd9b9bdc19569da22f3ede3b33f28da872.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/tables/d458e9e481d43c36e6799eb1178b4bcd9b9bdc19569da22f3ede3b33f28da872.jpg)

![e01c5b647a231bb652d7c3cdfe4b724e8f0c3e7810c1041978402a91c8520510.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/tables/e01c5b647a231bb652d7c3cdfe4b724e8f0c3e7810c1041978402a91c8520510.jpg)

![e110c8dde98a31d9a1d75cfa5511f8037bd0b8e5de57113b1c589c1f16c2a5b9.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/tables/e110c8dde98a31d9a1d75cfa5511f8037bd0b8e5de57113b1c589c1f16c2a5b9.jpg)

![f2388d955d5b65138dfe3049dee6de82767c8c88b0af09f7e54c614b119c5082.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/tables/f2388d955d5b65138dfe3049dee6de82767c8c88b0af09f7e54c614b119c5082.jpg)

![fb20e85cfd187e26c7569c0386189f9ba31d404de7d178aef8464ad48784d1bd.jpg](acl_results/819_RetroLLM_ Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation/tables/fb20e85cfd187e26c7569c0386189f9ba31d404de7d178aef8464ad48784d1bd.jpg)

## The Role of Deductive and Inductive Reasoning in Large Language Models

### Images

![1b77d689caef0f4ef9937e50887743a1252facca3c188628b249a620a129c0b0.jpg](acl_results/820_The Role of Deductive and Inductive Reasoning in Large Language Models/images/1b77d689caef0f4ef9937e50887743a1252facca3c188628b249a620a129c0b0.jpg)

![7c43d5b356ad07a40d2bf78224076c615ae61f226b264fc1684c5ecfbc717dbb.jpg](acl_results/820_The Role of Deductive and Inductive Reasoning in Large Language Models/images/7c43d5b356ad07a40d2bf78224076c615ae61f226b264fc1684c5ecfbc717dbb.jpg)

![83236ec928b41d08b85feaf0cee72f89be47ac1bb866f7dcc8163174ad7a1963.jpg](acl_results/820_The Role of Deductive and Inductive Reasoning in Large Language Models/images/83236ec928b41d08b85feaf0cee72f89be47ac1bb866f7dcc8163174ad7a1963.jpg)

![d55ffe5d4d62a833748c254d96fb3751c659a7b430988a93a527470b75708346.jpg](acl_results/820_The Role of Deductive and Inductive Reasoning in Large Language Models/images/d55ffe5d4d62a833748c254d96fb3751c659a7b430988a93a527470b75708346.jpg)

### Tables

![1ef1486ca838f0e2deab6354b79fd9589e5f6dc5c9444fa35399a46c564f5425.jpg](acl_results/820_The Role of Deductive and Inductive Reasoning in Large Language Models/tables/1ef1486ca838f0e2deab6354b79fd9589e5f6dc5c9444fa35399a46c564f5425.jpg)

![e4a8bb32a1736e54b9bbea5da934bdf7720a751bbbba440ba4fc453e5bd7d24b.jpg](acl_results/820_The Role of Deductive and Inductive Reasoning in Large Language Models/tables/e4a8bb32a1736e54b9bbea5da934bdf7720a751bbbba440ba4fc453e5bd7d24b.jpg)

## Disentangling the Roles of Representation and Selection in Data Pruning

### Images

![074c52268287e54165e915db36462509c3d187d58c9e6e0aebda01b046b66755.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/074c52268287e54165e915db36462509c3d187d58c9e6e0aebda01b046b66755.jpg)

![0ac8ec42917201364363ec194aae74558146edd3bfc293934b104828dffa5e00.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/0ac8ec42917201364363ec194aae74558146edd3bfc293934b104828dffa5e00.jpg)

![103b92f0a3982443797db1dcccf2a2fa124bb0eb31bd6a405d428314015c5bce.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/103b92f0a3982443797db1dcccf2a2fa124bb0eb31bd6a405d428314015c5bce.jpg)

![25b6b2ebbe3ccd214e8827bddea084d3314c1f225c7c920adfba2591e4443221.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/25b6b2ebbe3ccd214e8827bddea084d3314c1f225c7c920adfba2591e4443221.jpg)

![28e2269017c673bd269a70823c0ee8a50be6e4f70e8ed264aeec0524a334e44f.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/28e2269017c673bd269a70823c0ee8a50be6e4f70e8ed264aeec0524a334e44f.jpg)

![2a274e19fc603f2d1b1678a68c6c1ea76841938d4e7e739fc32593cc5f3ccdcc.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/2a274e19fc603f2d1b1678a68c6c1ea76841938d4e7e739fc32593cc5f3ccdcc.jpg)

![2a920ab0832f58b0809c39d27a30b6ae7a922cb7f48537d8732a14151312b22e.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/2a920ab0832f58b0809c39d27a30b6ae7a922cb7f48537d8732a14151312b22e.jpg)

![2f680f176ed134f8e561a737aac7fee4f763b45640b7eb0abdfdd26be53c290b.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/2f680f176ed134f8e561a737aac7fee4f763b45640b7eb0abdfdd26be53c290b.jpg)

![3cab5d75f3a9a1d2c0ff26e4e8987db6564f1a2097e421304ca049ad82e83022.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/3cab5d75f3a9a1d2c0ff26e4e8987db6564f1a2097e421304ca049ad82e83022.jpg)

![76b517db6213ed6215eaeb28eca4eb2aa35399f512e609d5468202f567ee4a23.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/76b517db6213ed6215eaeb28eca4eb2aa35399f512e609d5468202f567ee4a23.jpg)

![7a04aae8aa64e9b1e649337c0b53b859e62cb4148f9619527244c6cae00f2a3a.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/7a04aae8aa64e9b1e649337c0b53b859e62cb4148f9619527244c6cae00f2a3a.jpg)

![7b1c812b3d2805c6532cd3b8841d7e12f989ef9ad25b0a9e976a456335f73f18.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/7b1c812b3d2805c6532cd3b8841d7e12f989ef9ad25b0a9e976a456335f73f18.jpg)

![88c7c49cf23b868b30737e9b1cc1e60ef3d510039bc60ab154af195cd9c01bfa.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/88c7c49cf23b868b30737e9b1cc1e60ef3d510039bc60ab154af195cd9c01bfa.jpg)

![a2722e01be87402720222dbc30b6011fff29d451dccea5323932126df59f5b8a.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/a2722e01be87402720222dbc30b6011fff29d451dccea5323932126df59f5b8a.jpg)

![a32ecdc462f9de728f6c3fe4958550d92e89f8980330f7fbe1203f8d51313b37.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/a32ecdc462f9de728f6c3fe4958550d92e89f8980330f7fbe1203f8d51313b37.jpg)

![a646a27deb7c9e845733be250714a55a5156533bafcef80a2124c206b8c05d6c.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/a646a27deb7c9e845733be250714a55a5156533bafcef80a2124c206b8c05d6c.jpg)

![b0cbb01beaa286f5ed5c4c17327419bcfc91c93ad9b33dd3050691f7e7046a3f.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/b0cbb01beaa286f5ed5c4c17327419bcfc91c93ad9b33dd3050691f7e7046a3f.jpg)

![bd807c6b999d5a9055f4b82ff28459c51ddba861631714b68dde0e4e5e7ed110.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/bd807c6b999d5a9055f4b82ff28459c51ddba861631714b68dde0e4e5e7ed110.jpg)

![ca3cc8adc3a60370c7b23536c9f69e884feabdcdf5dcb26588e76fd0967e4ad0.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/ca3cc8adc3a60370c7b23536c9f69e884feabdcdf5dcb26588e76fd0967e4ad0.jpg)

![cefc44c136e0bfa2290abac078e2f4d9f3642ddf03956e1806e87de474236e6d.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/cefc44c136e0bfa2290abac078e2f4d9f3642ddf03956e1806e87de474236e6d.jpg)

![e3b378fd108eeba0ecbf928bc2cbea13b54b4e6c092da5a26ba1f9de4ca9e0f1.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/e3b378fd108eeba0ecbf928bc2cbea13b54b4e6c092da5a26ba1f9de4ca9e0f1.jpg)

![e7a80b0483f4b4d3d34188497f9328bcbce90aa8d2d17b527eac9bcbfb75bff7.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/e7a80b0483f4b4d3d34188497f9328bcbce90aa8d2d17b527eac9bcbfb75bff7.jpg)

![e827d6b7d8b853bbc64e15be2771b28d8234cb682308680f10c7f96046994056.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/e827d6b7d8b853bbc64e15be2771b28d8234cb682308680f10c7f96046994056.jpg)

![ec136019f19aebbe2644a8b02ded97e45dd9a23f14a38fe3190716237f646d30.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/images/ec136019f19aebbe2644a8b02ded97e45dd9a23f14a38fe3190716237f646d30.jpg)

### Tables

![55ca7d0ede9adaed3ac788f1ecd287f3c915c43b7dea77c1204aa7e82cbb51c9.jpg](acl_results/821_Disentangling the Roles of Representation and Selection in Data Pruning/tables/55ca7d0ede9adaed3ac788f1ecd287f3c915c43b7dea77c1204aa7e82cbb51c9.jpg)

## FRACTAL: Fine-Grained Scoring from Aggregate Text Labels

### Images

![08da1083d14ba7e9cb68530462f68074f70e80b563e1be071b4a2973bb00a5a2.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/images/08da1083d14ba7e9cb68530462f68074f70e80b563e1be071b4a2973bb00a5a2.jpg)

![e74b1612cfce7d64d9964a62fdcf28bc7e800cea186e73e4fb195f5857720f28.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/images/e74b1612cfce7d64d9964a62fdcf28bc7e800cea186e73e4fb195f5857720f28.jpg)

### Tables

![358c4b9db41466122fb9d5fdfb776f5f47959ff1bdd340d71bba933148d3ed4a.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/358c4b9db41466122fb9d5fdfb776f5f47959ff1bdd340d71bba933148d3ed4a.jpg)

![3a792273a912ba9a0785eb4942d0f2e2ea6703a031c68514d8dc1ba786705463.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/3a792273a912ba9a0785eb4942d0f2e2ea6703a031c68514d8dc1ba786705463.jpg)

![48378e270af1ff3bd98458d3edf7878458fd8d65bf1b1dfcdd9035ddd1d7b009.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/48378e270af1ff3bd98458d3edf7878458fd8d65bf1b1dfcdd9035ddd1d7b009.jpg)

![55d7aa878ec4dbb946edddc5251dbf5d2a045faec6665bb2eacce1dc875e2d0d.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/55d7aa878ec4dbb946edddc5251dbf5d2a045faec6665bb2eacce1dc875e2d0d.jpg)

![5846a5e0643c63d17e7fb1ad969ebe03c4cb9cfcb64d33bc6ce3d4044ebbd811.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/5846a5e0643c63d17e7fb1ad969ebe03c4cb9cfcb64d33bc6ce3d4044ebbd811.jpg)

![5e768ed5ad14e5b2c6474dbe203dbbebcbcb9f282fb4902cdb8c5b7d89298bcf.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/5e768ed5ad14e5b2c6474dbe203dbbebcbcb9f282fb4902cdb8c5b7d89298bcf.jpg)

![6e6ac03082f61f7b66ad6c842c299328ea1dc5559de72851e4d416af305d9573.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/6e6ac03082f61f7b66ad6c842c299328ea1dc5559de72851e4d416af305d9573.jpg)

![75bc11542fa5840add92dfd5380fff7426d71e73d7b2ab915d37f8acfb36e3c5.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/75bc11542fa5840add92dfd5380fff7426d71e73d7b2ab915d37f8acfb36e3c5.jpg)

![87c281838d1ca3276102bfb3bca3506cda8b56769e31db2ca511ecd7c69f0aab.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/87c281838d1ca3276102bfb3bca3506cda8b56769e31db2ca511ecd7c69f0aab.jpg)

![9e2ad6fcc900b8f8f579e6108208c227395efe1a2e16951ea3f965f75fa40660.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/9e2ad6fcc900b8f8f579e6108208c227395efe1a2e16951ea3f965f75fa40660.jpg)

![a2dcad663ff2c1852393034e4c3ff1955a3f38488bb80b6edfea401cd7b34ba1.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/a2dcad663ff2c1852393034e4c3ff1955a3f38488bb80b6edfea401cd7b34ba1.jpg)

![b257b111f273ca139823ffe9c639818129af5927c994babef9f4afabd72dcf20.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/b257b111f273ca139823ffe9c639818129af5927c994babef9f4afabd72dcf20.jpg)

![b856415e09513f83a09f312fb535ffe7d4a8c36d1b628452b6fb03e9c6af4b03.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/b856415e09513f83a09f312fb535ffe7d4a8c36d1b628452b6fb03e9c6af4b03.jpg)

![bb5da64d686889bef2fc19ccad7ec2722c70776c8d9c9ff4c6e48e7f591d4586.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/bb5da64d686889bef2fc19ccad7ec2722c70776c8d9c9ff4c6e48e7f591d4586.jpg)

![c389b37834b7cea810785ca238a8dd0ae3d513670e521d5c64950fd404c4fb1b.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/c389b37834b7cea810785ca238a8dd0ae3d513670e521d5c64950fd404c4fb1b.jpg)

![ebf2bb065c3ab61f7803fa00becac9f4af5d314090cac4706949f9077972ea45.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/ebf2bb065c3ab61f7803fa00becac9f4af5d314090cac4706949f9077972ea45.jpg)

![f429f428560a2248644a653987bf1b19ccf358e87a1d6896dd383e86030ac3f2.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/f429f428560a2248644a653987bf1b19ccf358e87a1d6896dd383e86030ac3f2.jpg)

![f8e6a4f34e7640a5fcd8a6352edc065a7e4335654491fe85edb739e76632be61.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/f8e6a4f34e7640a5fcd8a6352edc065a7e4335654491fe85edb739e76632be61.jpg)

![fc3b945201e3f34f6133eff8a6acd859a328663b3e60f7372ff7b4f22dab2ac9.jpg](acl_results/822_FRACTAL_ Fine-Grained Scoring from Aggregate Text Labels/tables/fc3b945201e3f34f6133eff8a6acd859a328663b3e60f7372ff7b4f22dab2ac9.jpg)

## ACT: Knowledgeable Agents to Design and Perform Complex Tasks

### Images

![11e14483afa93c1a7b1c724c609bd46d4b46a0ac7a25bbfb4e0df41ffc047ce7.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/images/11e14483afa93c1a7b1c724c609bd46d4b46a0ac7a25bbfb4e0df41ffc047ce7.jpg)

![1631dae6185333052abf4b81f030f201f2730d150f10a05a4c6677a89a6c4cf0.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/images/1631dae6185333052abf4b81f030f201f2730d150f10a05a4c6677a89a6c4cf0.jpg)

### Tables

![0efde2d6783badd7ee938795a08ab39e2a1a2e20871f0122931f34a90e0150f3.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/0efde2d6783badd7ee938795a08ab39e2a1a2e20871f0122931f34a90e0150f3.jpg)

![127117dd2b02364c821a6e2953e2c4f3ea989eac797ee334dd6cd39f867e75e7.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/127117dd2b02364c821a6e2953e2c4f3ea989eac797ee334dd6cd39f867e75e7.jpg)

![14a5c5399de81e2cc9315a443b240291fedff56008d6a87b45ed9c9faea7896a.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/14a5c5399de81e2cc9315a443b240291fedff56008d6a87b45ed9c9faea7896a.jpg)

![217465f4ff8c458b63476bd62fb589ef134eabf87882271da3cd87e43bd13912.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/217465f4ff8c458b63476bd62fb589ef134eabf87882271da3cd87e43bd13912.jpg)

![288632133e61aeb084b92e1cebbd38ecaa3bb36272572747fef12962647e6a9a.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/288632133e61aeb084b92e1cebbd38ecaa3bb36272572747fef12962647e6a9a.jpg)

![3112b60f0b371654487bb3a40cbbf975b42d31607206d15a574c997ce05dd988.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/3112b60f0b371654487bb3a40cbbf975b42d31607206d15a574c997ce05dd988.jpg)

![34cba875596d0b6cd5c0b00c434fc6757e91b94e755d0c9a9d7fe7100ee8087d.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/34cba875596d0b6cd5c0b00c434fc6757e91b94e755d0c9a9d7fe7100ee8087d.jpg)

![3e7d598e7b31ff734dd08e19fca987113088985ae11599642b6a70b75b118649.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/3e7d598e7b31ff734dd08e19fca987113088985ae11599642b6a70b75b118649.jpg)

![3fc714e8ce0aa5780ccd31f28152ce269ceee40ea596475883019d53d254a5cf.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/3fc714e8ce0aa5780ccd31f28152ce269ceee40ea596475883019d53d254a5cf.jpg)

![47a91b96f24c328b7bbd4168a1098d33d6918cbefce140238e74b3d06f2b4b60.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/47a91b96f24c328b7bbd4168a1098d33d6918cbefce140238e74b3d06f2b4b60.jpg)

![4b8b145155071d851fbcc0c98d4169774f8ef64a24f2a0148559235b8104e304.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/4b8b145155071d851fbcc0c98d4169774f8ef64a24f2a0148559235b8104e304.jpg)

![4c20623905006b9c2a9886f5864437fc1d87eee94753ad390af27371df7705b2.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/4c20623905006b9c2a9886f5864437fc1d87eee94753ad390af27371df7705b2.jpg)

![513156fd1d1bf4ec750cc5631e29e42f9119275679086086e21df13e654995bf.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/513156fd1d1bf4ec750cc5631e29e42f9119275679086086e21df13e654995bf.jpg)

![6003819b50080859eba3f9180c13d65c297e1d9970a46e64db70eced37d0543e.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/6003819b50080859eba3f9180c13d65c297e1d9970a46e64db70eced37d0543e.jpg)

![6a6416aeab08b2a6262c5964cc28abe5da0c36849cdf6f5bd681679332952635.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/6a6416aeab08b2a6262c5964cc28abe5da0c36849cdf6f5bd681679332952635.jpg)

![7bea99c15c6725539c75506f5ede6052b5d4283d22935edca9d947b6790dfb92.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/7bea99c15c6725539c75506f5ede6052b5d4283d22935edca9d947b6790dfb92.jpg)

![7cad23db080150b9368f1b105c829550e04118a8a0f054b1e99f57c278424fdc.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/7cad23db080150b9368f1b105c829550e04118a8a0f054b1e99f57c278424fdc.jpg)

![88ae137fbc2c24576bacb61b32bcde05081964c231a2108097fc9d21c4b02867.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/88ae137fbc2c24576bacb61b32bcde05081964c231a2108097fc9d21c4b02867.jpg)

![9c282a5fd05c8c3305489e4552d674705b2a7cf1216383636d7bcadd5944c79d.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/9c282a5fd05c8c3305489e4552d674705b2a7cf1216383636d7bcadd5944c79d.jpg)

![b024b14d2f2997656cf1e210805a025720ba37d6ae7495a89708d251156592e2.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/b024b14d2f2997656cf1e210805a025720ba37d6ae7495a89708d251156592e2.jpg)

![b2a7320dfb25399515443ea222145cbc185b7469982d38d4cb78c0cbcbf4f6d9.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/b2a7320dfb25399515443ea222145cbc185b7469982d38d4cb78c0cbcbf4f6d9.jpg)

![b638bbf0a24af75b2c801963ae75d6efb7ac4e62909f77076c0ff0f2ba25e724.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/b638bbf0a24af75b2c801963ae75d6efb7ac4e62909f77076c0ff0f2ba25e724.jpg)

![d6286f74257ee287de88f7e200084c18f424d361730510400c3e1a5c7d1db3f8.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/d6286f74257ee287de88f7e200084c18f424d361730510400c3e1a5c7d1db3f8.jpg)

![e69a7e9c9702cd83c4367092e0a89f2045163800066a3a45a4368f8abf07b182.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/e69a7e9c9702cd83c4367092e0a89f2045163800066a3a45a4368f8abf07b182.jpg)

![eaf8d33a3aba3c41bf739dba8bd5e4c7c15ae0588aca8595df42132ab4814fef.jpg](acl_results/823_ACT_ Knowledgeable Agents to Design and Perform Complex Tasks/tables/eaf8d33a3aba3c41bf739dba8bd5e4c7c15ae0588aca8595df42132ab4814fef.jpg)

## Logical forms complement probability in understanding language model (and human) performance

### Images

![10474c5ea887ee235874a55801ea6291de492c044443d55bf023d6b87a16243f.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/images/10474c5ea887ee235874a55801ea6291de492c044443d55bf023d6b87a16243f.jpg)

![110d73460a3e00ee87554fbb766197684fd5907d4ed0f599949149124c6012e1.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/images/110d73460a3e00ee87554fbb766197684fd5907d4ed0f599949149124c6012e1.jpg)

![187b5e4634dfe9d7a9842abd4c13afffd272d682555b1bf05378bee16be84768.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/images/187b5e4634dfe9d7a9842abd4c13afffd272d682555b1bf05378bee16be84768.jpg)

![244fd6fcb7bc50c81fd9589608d790d1852d2e010dbfbe0e4c6a4f78cd1dbd40.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/images/244fd6fcb7bc50c81fd9589608d790d1852d2e010dbfbe0e4c6a4f78cd1dbd40.jpg)

![902c9fd8710092eacb865d6412d7392a660464ec1df9d51e8db070a1a0bdfccb.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/images/902c9fd8710092eacb865d6412d7392a660464ec1df9d51e8db070a1a0bdfccb.jpg)

![ee20bfdc963b33a672c0a09a205f339062928081b171515d9e360822e6afbfcb.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/images/ee20bfdc963b33a672c0a09a205f339062928081b171515d9e360822e6afbfcb.jpg)

![fc3341776b2c98bc351a4675b8d0c9f09da2cb7eed196b672c107a5add4f6ab8.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/images/fc3341776b2c98bc351a4675b8d0c9f09da2cb7eed196b672c107a5add4f6ab8.jpg)

### Tables

![03b83eb8a21939f4ac364f250a8216deb520e105ffc55b2fcbf78485413324e0.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/tables/03b83eb8a21939f4ac364f250a8216deb520e105ffc55b2fcbf78485413324e0.jpg)

![6687ede5515b7b455d931354123cb2697ddb7db4c765343b765c024d5e4f5023.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/tables/6687ede5515b7b455d931354123cb2697ddb7db4c765343b765c024d5e4f5023.jpg)

![68dfd4916c429db77eb822869fb202696f74b293c0de368560527a25a638ef23.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/tables/68dfd4916c429db77eb822869fb202696f74b293c0de368560527a25a638ef23.jpg)

![93d8519a05a94716d1b971f68fb8b5d1da2c900074bd4dc9c8f8713af3d78591.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/tables/93d8519a05a94716d1b971f68fb8b5d1da2c900074bd4dc9c8f8713af3d78591.jpg)

![b44c265b53909864945f7fc18a8c990ccb8cb7f50ffd64c1a3eb353d5aca6d81.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/tables/b44c265b53909864945f7fc18a8c990ccb8cb7f50ffd64c1a3eb353d5aca6d81.jpg)

![e8ad53100113d18816f4a500bfe9587c93919b7597815ddd08536551d352d055.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/tables/e8ad53100113d18816f4a500bfe9587c93919b7597815ddd08536551d352d055.jpg)

![ff013c8888a3af4d395324f8f6d2a87898fc2b04e9131238d9a03c2ef7376bbc.jpg](acl_results/824_Logical forms complement probability in understanding language model (and human) performance/tables/ff013c8888a3af4d395324f8f6d2a87898fc2b04e9131238d9a03c2ef7376bbc.jpg)

## Length Controlled Generation for Black-boxLLMs

### Images

![9e0a3bc41a1a52b775d40eaa4a91a8c63daad5acec592e005651c9cadf44aefb.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/images/9e0a3bc41a1a52b775d40eaa4a91a8c63daad5acec592e005651c9cadf44aefb.jpg)

### Tables

![1b178cf11d5460c919057734ba5bfefa8644f48a048a712db3342a9a3fcfe135.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/1b178cf11d5460c919057734ba5bfefa8644f48a048a712db3342a9a3fcfe135.jpg)

![23544d64ece78f93c7179a62bd5c40dbec7278e021ff6737b2b22357e5246be1.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/23544d64ece78f93c7179a62bd5c40dbec7278e021ff6737b2b22357e5246be1.jpg)

![288f3d2a8f5c008c5e97ca17b76668da809813d9560ec1ca173a9871dc99e139.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/288f3d2a8f5c008c5e97ca17b76668da809813d9560ec1ca173a9871dc99e139.jpg)

![308527612cfca479b03a0a59bed2bb3008a9396fc67ff812cfcf08a49d726498.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/308527612cfca479b03a0a59bed2bb3008a9396fc67ff812cfcf08a49d726498.jpg)

![317c10e350a3bf6b28a50e1adebb1a71c1c8f40b5a37658af5d70cdd0695c0f2.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/317c10e350a3bf6b28a50e1adebb1a71c1c8f40b5a37658af5d70cdd0695c0f2.jpg)

![33f18941302f4c0aafbab5e0ef949313ed1e747e1fdfd0db2b740be0ec16fe53.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/33f18941302f4c0aafbab5e0ef949313ed1e747e1fdfd0db2b740be0ec16fe53.jpg)

![35afa72f61104934e7d9b6aaabced9a22d372f118fa3215bfd8ef2ec690c2f77.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/35afa72f61104934e7d9b6aaabced9a22d372f118fa3215bfd8ef2ec690c2f77.jpg)

![47682b5af832b433de9d3345c0ba45117a7c16fd726b59e5e3dc959b7ca4b372.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/47682b5af832b433de9d3345c0ba45117a7c16fd726b59e5e3dc959b7ca4b372.jpg)

![4b3b225424fde64198b0daef4330f83f8bad362021e9104b063a8eed6ca7c43e.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/4b3b225424fde64198b0daef4330f83f8bad362021e9104b063a8eed6ca7c43e.jpg)

![4dc1f3a26bd3f08657c2e8744dee1479c8a5f633541574390f5347c63a697b0b.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/4dc1f3a26bd3f08657c2e8744dee1479c8a5f633541574390f5347c63a697b0b.jpg)

![54e02741a8175757a0e3e430ea3fd60d96e8a5e903007b628a7948678bb9c9f8.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/54e02741a8175757a0e3e430ea3fd60d96e8a5e903007b628a7948678bb9c9f8.jpg)

![649b74059044ce73d86c534534b80bd4276b54e2d98a7d7e13cbd748ba884b50.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/649b74059044ce73d86c534534b80bd4276b54e2d98a7d7e13cbd748ba884b50.jpg)

![713a238f7004f68e229b614e834b4384dc6483dc2150547b229ca4c57c0deeac.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/713a238f7004f68e229b614e834b4384dc6483dc2150547b229ca4c57c0deeac.jpg)

![750d01744ccc497558de3e4f2e9a26f7caa510ec8f8016bee23b5627b2fbcfff.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/750d01744ccc497558de3e4f2e9a26f7caa510ec8f8016bee23b5627b2fbcfff.jpg)

![77599b18483fa0d8adf95a484668217a49eeb829a3d12f4360909d0f952962e6.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/77599b18483fa0d8adf95a484668217a49eeb829a3d12f4360909d0f952962e6.jpg)

![81b8915659f98863c4af6afb083cc45f6b2ca70571b61ab7dc8245f58cd44eec.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/81b8915659f98863c4af6afb083cc45f6b2ca70571b61ab7dc8245f58cd44eec.jpg)

![82d49d649c0eb92c547eaea76d82703f35cdd067ac90b3efe1deb96d96fd916a.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/82d49d649c0eb92c547eaea76d82703f35cdd067ac90b3efe1deb96d96fd916a.jpg)

![840a4ed1d728fcc2354786444dd9e7e426b3c554a246107b9f8484d8b0f06e18.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/840a4ed1d728fcc2354786444dd9e7e426b3c554a246107b9f8484d8b0f06e18.jpg)

![931e1980fc3cd39c698995642dd1569a09449d4365e8e7cfa752aa779543275d.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/931e1980fc3cd39c698995642dd1569a09449d4365e8e7cfa752aa779543275d.jpg)

![95ae16adc1ef23369577c4be8089a349d31b0732be455b2b7b2becd3e11ff3e9.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/95ae16adc1ef23369577c4be8089a349d31b0732be455b2b7b2becd3e11ff3e9.jpg)

![a1c4ba74b67dc43eff28df1fd5f6af080bb78ebc8e8263b79decbe479dd97dcc.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/a1c4ba74b67dc43eff28df1fd5f6af080bb78ebc8e8263b79decbe479dd97dcc.jpg)

![a502ef96e2b7c54086cc4fbd78b8062310c56432c88e92b54a5eb59d92157070.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/a502ef96e2b7c54086cc4fbd78b8062310c56432c88e92b54a5eb59d92157070.jpg)

![acb0f8e13ded83bb8ffaecce49b1fdc4188da7a40679e792178b5779ab975187.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/acb0f8e13ded83bb8ffaecce49b1fdc4188da7a40679e792178b5779ab975187.jpg)

![b0af975524c720c206f836b26bebb2746a2696ea6c7abacbb7740fc5a6466636.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/b0af975524c720c206f836b26bebb2746a2696ea6c7abacbb7740fc5a6466636.jpg)

![bc6e21ff378d5e07e24ad1131c7208938eae93152ae1016d6cbb103de9ca1575.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/bc6e21ff378d5e07e24ad1131c7208938eae93152ae1016d6cbb103de9ca1575.jpg)

![bda99f3cacdd92203134468a4ae24a4ebd014eb14a4ce3b7bb75d5efbabfd2a6.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/bda99f3cacdd92203134468a4ae24a4ebd014eb14a4ce3b7bb75d5efbabfd2a6.jpg)

![c92be511257157e1402f42dead1ea5be2ee6d1d4ed417b9410e724b03a70b01f.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/c92be511257157e1402f42dead1ea5be2ee6d1d4ed417b9410e724b03a70b01f.jpg)

![cb824abe409aad69c4c584c5feb9beb9d984bea9b99ff5dc308ad0ba46f5b9b5.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/cb824abe409aad69c4c584c5feb9beb9d984bea9b99ff5dc308ad0ba46f5b9b5.jpg)

![e773d40de05e0dac3362e1e5fb7d039b0bb68f28d5fa785fecfccd212ca484dc.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/e773d40de05e0dac3362e1e5fb7d039b0bb68f28d5fa785fecfccd212ca484dc.jpg)

![faac069b20bb7c9f61e7100e8caf32c59973ecce2728d9a6fffa1cc6ba35a8af.jpg](acl_results/825_Length Controlled Generation for Black-boxLLMs/tables/faac069b20bb7c9f61e7100e8caf32c59973ecce2728d9a6fffa1cc6ba35a8af.jpg)

## Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization

### Images

![06f29ca348187bcdfda30b53dfc1d96100311e2bbee281f872f0e82dcb39c948.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/images/06f29ca348187bcdfda30b53dfc1d96100311e2bbee281f872f0e82dcb39c948.jpg)

![237088a8ff4706a7b6f503f5cf6aa3aff335cbb02848902393abb524bd19d1db.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/images/237088a8ff4706a7b6f503f5cf6aa3aff335cbb02848902393abb524bd19d1db.jpg)

![2c8063bda14813d3ae9eb7f3971da2c83adda1c733f8beeb904e8782759f8199.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/images/2c8063bda14813d3ae9eb7f3971da2c83adda1c733f8beeb904e8782759f8199.jpg)

![738e9329185357ab1d6c69991a870487d668be7081a0cec66428d157ae40bacd.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/images/738e9329185357ab1d6c69991a870487d668be7081a0cec66428d157ae40bacd.jpg)

![836961427b05581cf59b0153f375ba89439faa64a2890cc7c7442711596773a4.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/images/836961427b05581cf59b0153f375ba89439faa64a2890cc7c7442711596773a4.jpg)

![961874832446b8123fc1119e2f917acf3e8a005c649f48abaddb1d40a6326830.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/images/961874832446b8123fc1119e2f917acf3e8a005c649f48abaddb1d40a6326830.jpg)

![c239a0ad5f49904df526b0893890d6967a4405612b33eaaa86c13f5a24cbc535.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/images/c239a0ad5f49904df526b0893890d6967a4405612b33eaaa86c13f5a24cbc535.jpg)

![dda33e4094c5e97644646bc4ed6754c3d2e1397af44da63c50a810a64980a578.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/images/dda33e4094c5e97644646bc4ed6754c3d2e1397af44da63c50a810a64980a578.jpg)

![f7e491942ab68f82d801206603c177e04d34f4dfdea3240f6f1e29efff6548c0.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/images/f7e491942ab68f82d801206603c177e04d34f4dfdea3240f6f1e29efff6548c0.jpg)

### Tables

![2bf1f92ecca540291f80e3017d1faacb8b3e26e4158f5958fb2271c6e9df9519.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/tables/2bf1f92ecca540291f80e3017d1faacb8b3e26e4158f5958fb2271c6e9df9519.jpg)

![2ca22f773dd7f4a45b44294213079d39de5a12a7ba468afe2c0f9f2c3faa95b4.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/tables/2ca22f773dd7f4a45b44294213079d39de5a12a7ba468afe2c0f9f2c3faa95b4.jpg)

![46a1b73d995cf3fb5524daea3c6de9d9ab719db62334b6033809e11107f13c2f.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/tables/46a1b73d995cf3fb5524daea3c6de9d9ab719db62334b6033809e11107f13c2f.jpg)

![7206533e6a54c971f293c87fd2fc10e20fe5e16c71cb2fac72243062fa32ca68.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/tables/7206533e6a54c971f293c87fd2fc10e20fe5e16c71cb2fac72243062fa32ca68.jpg)

![a6aaa920aa8a2fbea7abc5dc3a89ff8280ef4364247f7bec98a82c43f6ecf2a2.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/tables/a6aaa920aa8a2fbea7abc5dc3a89ff8280ef4364247f7bec98a82c43f6ecf2a2.jpg)

![f2361a5032820bd3509532e124b7f058629768e71c17ec4565cebd0a1850c1ef.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/tables/f2361a5032820bd3509532e124b7f058629768e71c17ec4565cebd0a1850c1ef.jpg)

![f46ae9b224b55f57c2baa905cbc78bddd95efaad2df00c52969f5c2b099808f1.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/tables/f46ae9b224b55f57c2baa905cbc78bddd95efaad2df00c52969f5c2b099808f1.jpg)

![f6b587f60435bddeaffc085079a6487d8eec6fa621b6b9c2651aa90ce0ceaf10.jpg](acl_results/826_Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization/tables/f6b587f60435bddeaffc085079a6487d8eec6fa621b6b9c2651aa90ce0ceaf10.jpg)

## Global Eye: Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process

### Images

![01ddd49f511de605de4bb3411b6a2059aed5e2ed016eeef0de6eb03ab11415da.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/images/01ddd49f511de605de4bb3411b6a2059aed5e2ed016eeef0de6eb03ab11415da.jpg)

![36304ec7b5966bc8cb4482ca76adcf8113fab4aec6d9d96ce3069503c0baeb1d.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/images/36304ec7b5966bc8cb4482ca76adcf8113fab4aec6d9d96ce3069503c0baeb1d.jpg)

![47bb00296001b259410615b4a1a7ce900754549306c37d889f2247242c967205.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/images/47bb00296001b259410615b4a1a7ce900754549306c37d889f2247242c967205.jpg)

![78ba15894fbc12b48dc50ccb45909abd7930efc14a3e1f05307f13239fe53dd0.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/images/78ba15894fbc12b48dc50ccb45909abd7930efc14a3e1f05307f13239fe53dd0.jpg)

![a079edb9cfd012af89ba20a6bc7fdfd50115cb4cfb864c0f8156c16dd87d0fa6.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/images/a079edb9cfd012af89ba20a6bc7fdfd50115cb4cfb864c0f8156c16dd87d0fa6.jpg)

![b4ff733f36b521d1d21dd40a774db11e7e36d2962c5585d75ae9dbd417002c9f.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/images/b4ff733f36b521d1d21dd40a774db11e7e36d2962c5585d75ae9dbd417002c9f.jpg)

![cc3f948f1d3523adfb2e25213e438fef4bf63fa4b885ffde89dd5aa7b29241b9.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/images/cc3f948f1d3523adfb2e25213e438fef4bf63fa4b885ffde89dd5aa7b29241b9.jpg)

![e8464736e36816fb8420328e1af30c22acbbe1ba23fe6f2b2f1912c5bf57339b.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/images/e8464736e36816fb8420328e1af30c22acbbe1ba23fe6f2b2f1912c5bf57339b.jpg)

### Tables

![313f980ca5d442586ff32cce6cdf6c649c69496683079fff0ac50fe570eec65d.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/tables/313f980ca5d442586ff32cce6cdf6c649c69496683079fff0ac50fe570eec65d.jpg)

![3580a433114dc2f11724a84b426896e605c7fd413e3dd1e1ef73aa4376d20bb2.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/tables/3580a433114dc2f11724a84b426896e605c7fd413e3dd1e1ef73aa4376d20bb2.jpg)

![564dcb9ac8f3e8168f3a8318e098d1b60de0feb43ec0d1f0acb02749a97b705a.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/tables/564dcb9ac8f3e8168f3a8318e098d1b60de0feb43ec0d1f0acb02749a97b705a.jpg)

![8d907633165bb71efc8679f055aea5aae00a931d4c365844f25063308cf17a36.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/tables/8d907633165bb71efc8679f055aea5aae00a931d4c365844f25063308cf17a36.jpg)

![8f4d951ddd5b3eab8e5fa345fa7a0d4422e7676793eb234ee4a67ac9fe58d6fa.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/tables/8f4d951ddd5b3eab8e5fa345fa7a0d4422e7676793eb234ee4a67ac9fe58d6fa.jpg)

![a88983c183b912eead43bd55b5633aa8702db318b410c69b3bdbc65f492c4299.jpg](acl_results/827_Global Eye_ Breaking the “Fixed Thinking Pattern” during the Instruction Expansion Process/tables/a88983c183b912eead43bd55b5633aa8702db318b410c69b3bdbc65f492c4299.jpg)

## On Synthesizing Data for Context Attribution in Question Answering

### Images

![150c3dbdc3f015895ebfb78f47386b6674a17f96570527b432b19e149eebc90e.jpg](acl_results/828_On Synthesizing Data for Context Attribution in Question Answering/images/150c3dbdc3f015895ebfb78f47386b6674a17f96570527b432b19e149eebc90e.jpg)

![77f64d18765be1eb6ba55232557e3424c6f0ebead5a136981bc1bb2d3f32f839.jpg](acl_results/828_On Synthesizing Data for Context Attribution in Question Answering/images/77f64d18765be1eb6ba55232557e3424c6f0ebead5a136981bc1bb2d3f32f839.jpg)

![963f1a7c6c3b9031c643c5b1d3f801e81f8bf1ee8fa0dc2d610c655d89e78106.jpg](acl_results/828_On Synthesizing Data for Context Attribution in Question Answering/images/963f1a7c6c3b9031c643c5b1d3f801e81f8bf1ee8fa0dc2d610c655d89e78106.jpg)

![d4b90dfa8181cb755222e705fc91c07e81e64ca36e96428b6a076125fe108c2b.jpg](acl_results/828_On Synthesizing Data for Context Attribution in Question Answering/images/d4b90dfa8181cb755222e705fc91c07e81e64ca36e96428b6a076125fe108c2b.jpg)

![d9849586a61cae6c14f4ee00426ff6a9a0e275e8e90766585e8fb1a2497cf9e3.jpg](acl_results/828_On Synthesizing Data for Context Attribution in Question Answering/images/d9849586a61cae6c14f4ee00426ff6a9a0e275e8e90766585e8fb1a2497cf9e3.jpg)

![f1129e7c771d09a5537fcb4249aef7d47b82b019aaf53f7b2cfeb0c875424336.jpg](acl_results/828_On Synthesizing Data for Context Attribution in Question Answering/images/f1129e7c771d09a5537fcb4249aef7d47b82b019aaf53f7b2cfeb0c875424336.jpg)

### Tables

![2485936f8dfbfdc49dc46b161e0f9bcb703d0236f618686d8d4901a7a95189f3.jpg](acl_results/828_On Synthesizing Data for Context Attribution in Question Answering/tables/2485936f8dfbfdc49dc46b161e0f9bcb703d0236f618686d8d4901a7a95189f3.jpg)

![a1349168d22ccae6d866b22a6c35fb5fc0c093a9a84888e2200039c91a777afb.jpg](acl_results/828_On Synthesizing Data for Context Attribution in Question Answering/tables/a1349168d22ccae6d866b22a6c35fb5fc0c093a9a84888e2200039c91a777afb.jpg)

![b0549e61fa9ed848396f1026e2adfb1456ccbb92168ea3d7dac29c2f219f9a24.jpg](acl_results/828_On Synthesizing Data for Context Attribution in Question Answering/tables/b0549e61fa9ed848396f1026e2adfb1456ccbb92168ea3d7dac29c2f219f9a24.jpg)

![e998d1a9e761eca70c1e070c4ddb35b5c2c9b7e54d804dc3d299205228cad84e.jpg](acl_results/828_On Synthesizing Data for Context Attribution in Question Answering/tables/e998d1a9e761eca70c1e070c4ddb35b5c2c9b7e54d804dc3d299205228cad84e.jpg)

## TST: A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks

### Images

![00bd147e6b50f15348ba0af5e1cd46f315df22e434891b4cf94fce4615fbcf12.jpg](acl_results/829_TST_ A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks/images/00bd147e6b50f15348ba0af5e1cd46f315df22e434891b4cf94fce4615fbcf12.jpg)

![055c24b7058df3f09a2092a2ba3bd742b7206487404a86471fc635c6238f73f1.jpg](acl_results/829_TST_ A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks/images/055c24b7058df3f09a2092a2ba3bd742b7206487404a86471fc635c6238f73f1.jpg)

![450298257aa83cb5e4c0890d57829a00b65fa394f33a06fe5b6c52c61af9c1a6.jpg](acl_results/829_TST_ A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks/images/450298257aa83cb5e4c0890d57829a00b65fa394f33a06fe5b6c52c61af9c1a6.jpg)

![6826d5b837e892bdb46e2eae3490e70f4aaf66600f39f148195fdef3faed8279.jpg](acl_results/829_TST_ A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks/images/6826d5b837e892bdb46e2eae3490e70f4aaf66600f39f148195fdef3faed8279.jpg)

![a475d95ff4b8d1dc9dd9274ee0367b00947153de857da08330344f755c1e75c0.jpg](acl_results/829_TST_ A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks/images/a475d95ff4b8d1dc9dd9274ee0367b00947153de857da08330344f755c1e75c0.jpg)

### Tables

![41d63961ffc07462ec015b764ff3df1c753de75927246e459e8935ce9205ded3.jpg](acl_results/829_TST_ A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks/tables/41d63961ffc07462ec015b764ff3df1c753de75927246e459e8935ce9205ded3.jpg)

![771b86c5fc8808d3c6673ab27127ef196c566e95e3257eed0d5010b9886bfad1.jpg](acl_results/829_TST_ A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks/tables/771b86c5fc8808d3c6673ab27127ef196c566e95e3257eed0d5010b9886bfad1.jpg)

![b1e948378cac331cb8f6a196eb446cabe4d8354ec32a0cc03ddecf317e98eb79.jpg](acl_results/829_TST_ A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks/tables/b1e948378cac331cb8f6a196eb446cabe4d8354ec32a0cc03ddecf317e98eb79.jpg)

![b2578c5f402934c98f3e0c9b89ac7c16a0cb382e2faf7979145eedf93444c4bb.jpg](acl_results/829_TST_ A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks/tables/b2578c5f402934c98f3e0c9b89ac7c16a0cb382e2faf7979145eedf93444c4bb.jpg)

![b5019386b2c79bddd3d9cf71d424e5b6f4577c462d6f0629a5e526def08c39c1.jpg](acl_results/829_TST_ A Schema-Based Top-Down and Dynamic-Aware Agent of Text-to-Table Tasks/tables/b5019386b2c79bddd3d9cf71d424e5b6f4577c462d6f0629a5e526def08c39c1.jpg)

## EventRAG: EnhancingLLMGeneration with Event Knowledge Graphs

### Images

![0b95ddf0d53aa656474fefdffe1f454603fd46d89b0d837f45207bd6320d033d.jpg](acl_results/830_EventRAG_ EnhancingLLMGeneration with Event Knowledge Graphs/images/0b95ddf0d53aa656474fefdffe1f454603fd46d89b0d837f45207bd6320d033d.jpg)

![41d66c01d5665212a4789e34cf3cfbc3f6bca70af32ee37cf41fc7716ddc509a.jpg](acl_results/830_EventRAG_ EnhancingLLMGeneration with Event Knowledge Graphs/images/41d66c01d5665212a4789e34cf3cfbc3f6bca70af32ee37cf41fc7716ddc509a.jpg)

![63e35a7d0c78975ec11aac2db6f45a272d0c3a42f23ef37d09d4ba13806a2323.jpg](acl_results/830_EventRAG_ EnhancingLLMGeneration with Event Knowledge Graphs/images/63e35a7d0c78975ec11aac2db6f45a272d0c3a42f23ef37d09d4ba13806a2323.jpg)

![f706a7839aa97fbdd5b4a42d5c18bb149d239d3337d9ffb8ee3b371ad6bed948.jpg](acl_results/830_EventRAG_ EnhancingLLMGeneration with Event Knowledge Graphs/images/f706a7839aa97fbdd5b4a42d5c18bb149d239d3337d9ffb8ee3b371ad6bed948.jpg)

### Tables

![194d101a7347002e56c6b7a26fd01ab4d095fd9bd19e4cf6f3be084270c706c5.jpg](acl_results/830_EventRAG_ EnhancingLLMGeneration with Event Knowledge Graphs/tables/194d101a7347002e56c6b7a26fd01ab4d095fd9bd19e4cf6f3be084270c706c5.jpg)

![225e4f4ba501375ac1c6c1e2e9e2fa679ff9f474e93cbcbf924de954f348bec3.jpg](acl_results/830_EventRAG_ EnhancingLLMGeneration with Event Knowledge Graphs/tables/225e4f4ba501375ac1c6c1e2e9e2fa679ff9f474e93cbcbf924de954f348bec3.jpg)

![7d94469490fcdb85be8aee6e2bcc82a78495d3fb67a61d8a432e38172570b0e8.jpg](acl_results/830_EventRAG_ EnhancingLLMGeneration with Event Knowledge Graphs/tables/7d94469490fcdb85be8aee6e2bcc82a78495d3fb67a61d8a432e38172570b0e8.jpg)

## Analyzing the Rapid Generalization ofSFTvia the Perspective of Attention Head Activation Patterns

### Images

![0db51e5cd347ba70f8f47ef1d0173340d90a2e6fcb40b9cfffa0975e9af57bf0.jpg](acl_results/831_Analyzing the Rapid Generalization ofSFTvia the Perspective of Attention Head Activation Patterns/images/0db51e5cd347ba70f8f47ef1d0173340d90a2e6fcb40b9cfffa0975e9af57bf0.jpg)

![4607cf557e2aee84a012a797ad515f6a638f04faf3992f61310ac12340ebcdc9.jpg](acl_results/831_Analyzing the Rapid Generalization ofSFTvia the Perspective of Attention Head Activation Patterns/images/4607cf557e2aee84a012a797ad515f6a638f04faf3992f61310ac12340ebcdc9.jpg)

![67b7a5269941c911eb67104437231fb7ed36925f1cad9ad50d7efd8a6ed68e8e.jpg](acl_results/831_Analyzing the Rapid Generalization ofSFTvia the Perspective of Attention Head Activation Patterns/images/67b7a5269941c911eb67104437231fb7ed36925f1cad9ad50d7efd8a6ed68e8e.jpg)

![9b07b42e046c62e9a41741218b39acde29beac6d5c75cdedbda23da1a64ea1e0.jpg](acl_results/831_Analyzing the Rapid Generalization ofSFTvia the Perspective of Attention Head Activation Patterns/images/9b07b42e046c62e9a41741218b39acde29beac6d5c75cdedbda23da1a64ea1e0.jpg)

![c563253c4aed20e18f890535e69051b60ddd031ac7f47662205cf4a305b87326.jpg](acl_results/831_Analyzing the Rapid Generalization ofSFTvia the Perspective of Attention Head Activation Patterns/images/c563253c4aed20e18f890535e69051b60ddd031ac7f47662205cf4a305b87326.jpg)

### Tables

![6957450b0b6f382e71a485f0d09ed4002318cd483f14cc9d977ed1fc237dfc86.jpg](acl_results/831_Analyzing the Rapid Generalization ofSFTvia the Perspective of Attention Head Activation Patterns/tables/6957450b0b6f382e71a485f0d09ed4002318cd483f14cc9d977ed1fc237dfc86.jpg)

![7980600a9db7f8de6aed4554f5c55f174800bd82d574154c6b0409330c363ac9.jpg](acl_results/831_Analyzing the Rapid Generalization ofSFTvia the Perspective of Attention Head Activation Patterns/tables/7980600a9db7f8de6aed4554f5c55f174800bd82d574154c6b0409330c363ac9.jpg)

![dd99ef015b315ea6e768bad1e4095342f17feb562d9f9151e548ceabef763824.jpg](acl_results/831_Analyzing the Rapid Generalization ofSFTvia the Perspective of Attention Head Activation Patterns/tables/dd99ef015b315ea6e768bad1e4095342f17feb562d9f9151e548ceabef763824.jpg)

## Can’t See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for MultimodalLLMs

### Images

![034f1b4cf99cfb4f0603d8f9cf3e349752679cae5e0436aa3bfd2e41ec5db8fa.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/034f1b4cf99cfb4f0603d8f9cf3e349752679cae5e0436aa3bfd2e41ec5db8fa.jpg)

![0561c6bdf1f21a940a1e8f396ee9bfa89f8278237f86ede35b10eadb072e6182.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/0561c6bdf1f21a940a1e8f396ee9bfa89f8278237f86ede35b10eadb072e6182.jpg)

![0b8898505d325884fd81343d62720ba3145b92989cebaaa956ad24d3cf97b15a.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/0b8898505d325884fd81343d62720ba3145b92989cebaaa956ad24d3cf97b15a.jpg)

![119521a0ed2f8828a78c0679d3aee9006f3f3441bf0ef744b9cd48771afb8ed3.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/119521a0ed2f8828a78c0679d3aee9006f3f3441bf0ef744b9cd48771afb8ed3.jpg)

![197228ca173cd0e91133238323dd4a5476e76d0fe5c9b6401416160841657470.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/197228ca173cd0e91133238323dd4a5476e76d0fe5c9b6401416160841657470.jpg)

![26f41ae5c5856bbb1a06fcd974f067ee6fabcfbf4cc2fa87d831d55a691317b2.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/26f41ae5c5856bbb1a06fcd974f067ee6fabcfbf4cc2fa87d831d55a691317b2.jpg)

![2be84f3f62e1f63a5587ae3802cfea59754f6c2bd3746b0993a35d0a870afe70.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/2be84f3f62e1f63a5587ae3802cfea59754f6c2bd3746b0993a35d0a870afe70.jpg)

![3683f37d9383e4506c1fb06bd03abeaf919092cbe2e4bddb155c2e375c3e9141.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/3683f37d9383e4506c1fb06bd03abeaf919092cbe2e4bddb155c2e375c3e9141.jpg)

![3a080b53c13d80797199518b207c5d2adc14aed80d09edb812535138f061491c.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/3a080b53c13d80797199518b207c5d2adc14aed80d09edb812535138f061491c.jpg)

![4abd763bdd441aba381504c7c42c2ded33eadee071e2cbd1e8f3c747839a66d6.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/4abd763bdd441aba381504c7c42c2ded33eadee071e2cbd1e8f3c747839a66d6.jpg)

![5b1ef1ec26a679d28e4d3208f074276db7f5e7a693b4c5accae137f8e270135f.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/5b1ef1ec26a679d28e4d3208f074276db7f5e7a693b4c5accae137f8e270135f.jpg)

![5f18bd33cb98968b13bb10bae5d6b8aca22cb3ee3e2e965b131b6fee4eb54124.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/5f18bd33cb98968b13bb10bae5d6b8aca22cb3ee3e2e965b131b6fee4eb54124.jpg)

![60d0f4f41e73786ee7f5e1d335e0d2df57c4fb83c3500ec427f92ed7bbf41dcb.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/60d0f4f41e73786ee7f5e1d335e0d2df57c4fb83c3500ec427f92ed7bbf41dcb.jpg)

![659886b10c981ca7bc296208536d57926b70d369f16fb5611b4e965f5e551e0a.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/659886b10c981ca7bc296208536d57926b70d369f16fb5611b4e965f5e551e0a.jpg)

![72df50690d0710d3ce90759060d91c8f7bd5bd5f69c8b36d94877591b2eabdfa.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/72df50690d0710d3ce90759060d91c8f7bd5bd5f69c8b36d94877591b2eabdfa.jpg)

![87490af0ee2097a275261540da9e8693abeae000a7b2fb2891b2254109742982.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/87490af0ee2097a275261540da9e8693abeae000a7b2fb2891b2254109742982.jpg)

![87e6ff90bd7f795e7685ae037d2928ca33e1b55dce883c86c62440b6658de35d.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/87e6ff90bd7f795e7685ae037d2928ca33e1b55dce883c86c62440b6658de35d.jpg)

![89fadd2aebafd99a87c2568ab251fb38f2b2589cda6d89acd63d16ef10787ab6.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/89fadd2aebafd99a87c2568ab251fb38f2b2589cda6d89acd63d16ef10787ab6.jpg)

![925b0d1c2d06f18417ae389fd6872ceec444d4862b987713bb024c3baad50655.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/925b0d1c2d06f18417ae389fd6872ceec444d4862b987713bb024c3baad50655.jpg)

![9535cf9856c80c43d8b37515800ec0baf2a0b0664bca1fa656b10583dd4026d7.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/9535cf9856c80c43d8b37515800ec0baf2a0b0664bca1fa656b10583dd4026d7.jpg)

![9537e7456cf87bade080be038a6f8af1affb63ec6bd8f7ba35c66846d8920829.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/9537e7456cf87bade080be038a6f8af1affb63ec6bd8f7ba35c66846d8920829.jpg)

![962339de4d03b2a8b202a4564e5afa7b15db818f9367964b6d8bf1955b00adbe.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/962339de4d03b2a8b202a4564e5afa7b15db818f9367964b6d8bf1955b00adbe.jpg)

![9e08213b6be62f8b80ccd04942dc80c643e3d8be0adbf6da9922d09b2310319b.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/9e08213b6be62f8b80ccd04942dc80c643e3d8be0adbf6da9922d09b2310319b.jpg)

![b501d50d51cc7ecbcc1678c0947a10ce9f689bc57a0ffc16abbd5abc1c645cc2.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/b501d50d51cc7ecbcc1678c0947a10ce9f689bc57a0ffc16abbd5abc1c645cc2.jpg)

![b6bf88e9b2830c283369f87e1edb2198dc295596308c694b5a5018d686b87e30.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/b6bf88e9b2830c283369f87e1edb2198dc295596308c694b5a5018d686b87e30.jpg)

![bd98c1fc5d42cde524192ac3f061c1406dd50a09842b05eca3836e1061c509c5.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/bd98c1fc5d42cde524192ac3f061c1406dd50a09842b05eca3836e1061c509c5.jpg)

![ca2a7ee4b15b87f05e81fd7cafbbe3e64fa8e1f50b0fc7d0474392cb902deb03.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/ca2a7ee4b15b87f05e81fd7cafbbe3e64fa8e1f50b0fc7d0474392cb902deb03.jpg)

![d32a6f9617567c4d2c5f4062232c9deeae67be29cb7cda8ea26bf86e0efa67d9.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/d32a6f9617567c4d2c5f4062232c9deeae67be29cb7cda8ea26bf86e0efa67d9.jpg)

![d792d998eb3f3835073731f2f23e5759671ad4253ac96decc431d3477f065404.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/d792d998eb3f3835073731f2f23e5759671ad4253ac96decc431d3477f065404.jpg)

![da2f352d60811dc0d51e25b77c7282d6b4c8ad67d82c38ef8238fdb69ec75fbe.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/da2f352d60811dc0d51e25b77c7282d6b4c8ad67d82c38ef8238fdb69ec75fbe.jpg)

![da639401fef6b9e6fb2f4d6a5a645cd5ef4ecdf05642d0b2b9c6538761866339.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/da639401fef6b9e6fb2f4d6a5a645cd5ef4ecdf05642d0b2b9c6538761866339.jpg)

![de8be138e80bc7da51da4f4d70b9e20e2711bdabd410e039ebf48007192aad57.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/de8be138e80bc7da51da4f4d70b9e20e2711bdabd410e039ebf48007192aad57.jpg)

![e4204bfdaf18dd4ea04475d582c741ce6f4eacf95fe807cd33706ba619da552a.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/e4204bfdaf18dd4ea04475d582c741ce6f4eacf95fe807cd33706ba619da552a.jpg)

![e730d1b5d55872df34ea6b3776540edf551333cf4b708dd60a13ad161096bd41.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/e730d1b5d55872df34ea6b3776540edf551333cf4b708dd60a13ad161096bd41.jpg)

![f0dcc162b828bc5893fe3c310694a81b6a0070680cf55e5899218d28dd87461a.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/f0dcc162b828bc5893fe3c310694a81b6a0070680cf55e5899218d28dd87461a.jpg)

![f2638fc80cf135ee876c9b992cb731e3cf4f92b293d0c5c2bfd2baf2e0b6ef81.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/f2638fc80cf135ee876c9b992cb731e3cf4f92b293d0c5c2bfd2baf2e0b6ef81.jpg)

![fa4b032ea4c98c78e4813e029c7dde45b3c7ec2c3ba00937c761a5c4e126eff0.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/images/fa4b032ea4c98c78e4813e029c7dde45b3c7ec2c3ba00937c761a5c4e126eff0.jpg)

### Tables

![180277257044f0a98a600221f3d6940f834d17bf769edffde5bc09b3971e055a.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/tables/180277257044f0a98a600221f3d6940f834d17bf769edffde5bc09b3971e055a.jpg)

![4ae4a0b4d4d09cb5b51723cc0b8ab42acb05fd21b94ddcebf902af4286c4e5e7.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/tables/4ae4a0b4d4d09cb5b51723cc0b8ab42acb05fd21b94ddcebf902af4286c4e5e7.jpg)

![78b3d71720687cedf29954465c71f22c1fbf0162c2bc42d32691c3d4a0f30c22.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/tables/78b3d71720687cedf29954465c71f22c1fbf0162c2bc42d32691c3d4a0f30c22.jpg)

![97691019416dfd651a7d6048f05307659be6cba5cea00fba0cb98aec637c8f01.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/tables/97691019416dfd651a7d6048f05307659be6cba5cea00fba0cb98aec637c8f01.jpg)

![d377b386aa1736c197422f18deee633515146cfcfa385572e30c03ff26afba5f.jpg](acl_results/832_Can’t See the Forest for the Trees_ Benchmarking Multimodal Safety Awareness for MultimodalLLMs/tables/d377b386aa1736c197422f18deee633515146cfcfa385572e30c03ff26afba5f.jpg)

## Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling

### Images

![2cc99f09a0520d0591153500a3705f52fdc2290617fbaa8bba3d8ec5f6c552dd.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/images/2cc99f09a0520d0591153500a3705f52fdc2290617fbaa8bba3d8ec5f6c552dd.jpg)

![34027dc2dc66e5cd716eed85ac10a5801847cca8fc64c7437a4b7b949e9ebc61.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/images/34027dc2dc66e5cd716eed85ac10a5801847cca8fc64c7437a4b7b949e9ebc61.jpg)

![589b903386fa1b037ed3fad453540580fbc2a574e2130ce7f690dafb532123b3.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/images/589b903386fa1b037ed3fad453540580fbc2a574e2130ce7f690dafb532123b3.jpg)

![595c6fbc1ee4749c66054eda3f027a17fd32c8e10d8c1b2bc7138ee583b2df9a.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/images/595c6fbc1ee4749c66054eda3f027a17fd32c8e10d8c1b2bc7138ee583b2df9a.jpg)

![643f33df8a4c2777008c1574430677e9968757454df71fbb7d3c6f38472f2922.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/images/643f33df8a4c2777008c1574430677e9968757454df71fbb7d3c6f38472f2922.jpg)

![d1cd3a9b4beb7b8b015ad3bf8e0ec68b6d0577618deeff0f988b9ecfae9ceb97.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/images/d1cd3a9b4beb7b8b015ad3bf8e0ec68b6d0577618deeff0f988b9ecfae9ceb97.jpg)

![da5a3fc3f9b133574394cb51d961c2908481c52eadd4ba8326b96d9c28fe8b47.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/images/da5a3fc3f9b133574394cb51d961c2908481c52eadd4ba8326b96d9c28fe8b47.jpg)

![fb7ddb6c6e6500908018bdac67e9790853913dff54b2b49351ac3da9f6a173a8.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/images/fb7ddb6c6e6500908018bdac67e9790853913dff54b2b49351ac3da9f6a173a8.jpg)

### Tables

![09b44d1abb73be8ac97cbf5b3c1dca25962440cbe6e815548edb266bb0ca3b6d.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/tables/09b44d1abb73be8ac97cbf5b3c1dca25962440cbe6e815548edb266bb0ca3b6d.jpg)

![48822e3c89c2b73cf2c0157e8765f1571e907a63642d595a31e25c6248b056ac.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/tables/48822e3c89c2b73cf2c0157e8765f1571e907a63642d595a31e25c6248b056ac.jpg)

![56c7ddac3757dc332f7620ac96dea09c99076df6f436dd78a591b36e03c7eaeb.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/tables/56c7ddac3757dc332f7620ac96dea09c99076df6f436dd78a591b36e03c7eaeb.jpg)

![67cf07639362570a888fae1b6a8528f76a28297878bed2db33b6c2c48a39c521.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/tables/67cf07639362570a888fae1b6a8528f76a28297878bed2db33b6c2c48a39c521.jpg)

![6abc9b897afe4be8ef4ae9affbd4a3138a812bf2623afe858f4c82b4a29710b8.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/tables/6abc9b897afe4be8ef4ae9affbd4a3138a812bf2623afe858f4c82b4a29710b8.jpg)

![6bd661bbb49727bcf5cf099e6b0677d836635040e92576f1e5f8e1f6b7fa0600.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/tables/6bd661bbb49727bcf5cf099e6b0677d836635040e92576f1e5f8e1f6b7fa0600.jpg)

![76d2403be13b4b00c43bcda858c9ad9c0b7bbc47bd2a1ff7d878d4c63fd83bc1.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/tables/76d2403be13b4b00c43bcda858c9ad9c0b7bbc47bd2a1ff7d878d4c63fd83bc1.jpg)

![836c86a582209a97c7b8eda8e50f86f0af6d5c85a272cd40858609db147187b6.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/tables/836c86a582209a97c7b8eda8e50f86f0af6d5c85a272cd40858609db147187b6.jpg)

![a13cb788504786c25b87633ae2f3840fc0d7fa3f6f05ad0ff5140dedc10c97b2.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/tables/a13cb788504786c25b87633ae2f3840fc0d7fa3f6f05ad0ff5140dedc10c97b2.jpg)

![b7fcd46cb11e2bb6ed17ffff8ea0f899e97667847048f6e3c5ae945a828d830a.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/tables/b7fcd46cb11e2bb6ed17ffff8ea0f899e97667847048f6e3c5ae945a828d830a.jpg)

![e0f8ac64daa4dfab1bcdd7627b5f29bbd5c655ea84c114b08af16752deffdd4b.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/tables/e0f8ac64daa4dfab1bcdd7627b5f29bbd5c655ea84c114b08af16752deffdd4b.jpg)

![e5cdc8262c39410fc018501c1fe473c94e152791650cc8e56d365fe1db308ccb.jpg](acl_results/833_Mis-prompt_ Benchmarking Large Language Models for Proactive Error Handling/tables/e5cdc8262c39410fc018501c1fe473c94e152791650cc8e56d365fe1db308ccb.jpg)