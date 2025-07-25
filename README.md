# FlavorMiner: a machine learning platform for extracting molecular flavor profiles from structural data
Fabio Herrera‑Rocha1,2, Miguel Fernández‑Niño 2,3, Jorge Duitama4, Mónica P. Cala5, María José Chica6,
Ludger A. Wessjohann2, Mehdi D. Davari2* and Andrés Fernando González Barrios1*

1 Grupo de Diseño de Productos y Procesos (GDPP), Department of Chemical and Food Engineering, Universidad de los Andes, 111711 Bogotá, Colombia.
2 Leibniz‑Institute of Plant Biochemistry, Department of Bioorganic Chemistry, Weinberg 3, 06120 Halle, Germany. 
3 Institute of Agrochemistry and Food Technology (IATA‑CSIC), Valencia, Spain. 
4 Systems and Computing Engineering Department, Universidad de Los Andes, 111711 Bogotá, Colombia. 5 MetCore

Flavor is the main factor driving consumers acceptance of food products. However, tracking the biochemistry of flavor is a formidable challenge due to the complexity of food composition. Current methodologies for linking individual molecules to flavor in foods and beverages are expensive and time-consuming. Predictive models based on machine learning (ML) are emerging as an alternative to speed up this process. Nonetheless, the optimal approach to predict flavor features of molecules remains elusive. In this work we present FlavorMiner, an ML-based multilabel flavor predictor. FlavorMiner seamlessly integrates different combinations of algorithms and mathematical representations, augmented with class balance strategies to address the inherent class of the input dataset. Notably, Random Forest and K-Nearest Neighbors combined with Extended Connectivity Fingerprint and RDKit molecular descriptors consistently outperform other combinations in most cases. Resampling strategies surpass weight balance methods in mitigating bias associated with class imbalance. FlavorMiner exhibits remarkable accuracy, with an average ROC AUC score of 0.88. This algorithm was used to analyze cocoa metabolomics data, unveiling its profound potential to help extract valuable insights from intricate food metabolomics data. FlavorMiner can be used for flavor mining in any food product, drawing from a diverse training dataset that spans over 934 distinct food products. 

[https://doi.org/10.1186/s13321-024-00935-9](https://doi.org/10.1186/s13321-024-00935-9)

![image](https://github.com/FabioHerrera97/FlavorMiner/assets/147598169/f31b8dcb-f8bc-4cb2-8a5a-a174094a7a84)

The Google Colab Notebook linked down contains detailed instructions to Run FlavorMiner.

[FlavorMiner.ipynb](https://colab.research.google.com/github/FabioHerrera97/FlavorMiner/blob/main/FlavorMiner.ipynb)https://colab.research.google.com/github/FabioHerrera97/FlavorMiner/blob/main/FlavorMiner.ipynb

# PAPER UNDER REVIEW
