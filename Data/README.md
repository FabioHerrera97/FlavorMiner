# Data sources and data preparations

## 1. Flavor Threshold data 

The flavor threshold data was collected from the following sources:

- Zhu, Y., Chen, J., Chen, X., Chen, D., & Deng, S. (2020). Use of relative odor activity value (ROAV) to link aroma profiles to volatile compounds: application to fresh and dried eel (Muraenesox cinereus). International Journal of Food Properties, 23(1), 2257–2270. https://doi.org/10.1080/10942912.2020.1856133.
- Pullen, J. (2007) Review of odour character and thresholds. Bristol: Environment Agency. 
- Ma, R. et al. (2020) Odor-active volatile compounds profile of triploid rainbow trout with different marketable sizes. Aquaculture Reports, 17, p. 100312. doi:10.1016/j.aqrep.2020.100312.
- Ma, T.-Z., Gong, P.-F., Lu, R.-R., Zhang, B., Morata, A., & Han, S.-Y. (2020). Effect of Different Clarification Treatments on the Volatile Composition and Aromatic Attributes of ‘Italian Riesling’ Icewine. Molecules, 25(11), 2657. https://doi.org/10.3390/molecules25112657.
- Żołnierczyk, A. K., & Szumny, A. (2021). Sensory and Chemical Characteristic of Two Insect Species: Tenebrio molitor and Zophobas morio Larvae Affected by Roasting Processes. Molecules, 26(9), 2697. https://doi.org/10.3390/molecules26092697.
- Czerny, M., Christlbauer, M., Christlbauer, M. et al. Re-investigation on odour thresholds of key food aroma compounds and development of an aroma language based on odour qualities of defined aqueous odorant solutions. Eur Food Res Technol 228, 265–273 (2008). https://doi.org/10.1007/s00217-008-0931-x
- Bokowa, A.H. (2022). Odour Detection Threshold Values for Fifty-Two Selected Pure Compounds.
- Leonardos, G., Kendall, D., & Barnard, N. (1969). Odor Threshold Determinations of 53 Odorant Chemicals. Journal of the Air Pollution Control Association, 19(2), 91–95. https://doi.org/10.1080/00022470.1969.10466465.
- RUTH, J. H. (1986). Odor Thresholds and Irritation Levels of Several Chemical Substances: A Review. American Industrial Hygiene Association Journal, 47(3), A-142-A-151. https://doi.org/10.1080/1529866869138959
- Guan, Q., Meng, L.-J., Mei, Z., et al (2022). Volatile Compound Abundance Correlations Provide a New Insight into Odor Balances in Sauce-Aroma Baijiu. Foods, 11(23), 3916. https://doi.org/10.3390/foods11233916.
- Lobo-Prieto, A., Tena, N., Aparicio-Ruiz, R., Morales, M. T., & García-González, D. L. (2020). Tracking Sensory Characteristics of Virgin Olive Oils During Storage: Interpretation of Their Changes from a Multiparametric Perspective. Molecules, 25(7), 1686. https://doi.org/10.3390/molecules25071686.
- Yan, J., Alewijn, M., & van Ruth, S. M. (2020). From Extra Virgin Olive Oil to Refined Products: Intensity and Balance Shifts of the Volatile Compounds versus Odor. Molecules, 25(11), 2469. https://doi.org/10.3390/molecules25112469
- TAMURA, H. et al. (2001) The volatile constituents in the peel and pulp of a green thai mango, Khieo Sawoei cultivar(Mangifera indica L.)., Food Science and Technology Research, 7(1), pp. 72–77. doi:10.3136/fstr.7.72.
- Slaghenaufi, D., Boscaini, A., Prandi, A., Dal Cin, A., Zandonà, V., Luzzini, G., & Ugliano, M. (2020). Influence of Different Modalities of Grape Withering on Volatile Compounds of Young and Aged Corvina Wines. Molecules, 25(9), 2141. https://doi.org/10.3390/molecules25092141.
- Nagata, Y. (2011). Measurement of Odor Threshold by Triangle Odor Bag Method.
- Goel, M. et al. (2024) Flavordb2: An updated database of flavor molecules, Journal of Food Science, 89(11), pp. 7076–7082. doi:10.1111/1750-3841.17298.

The data gathered from research articles was manually extracted into an excel file (`FlavorThresholdLiterature.xlsx`). The data from FlavorDB was scrapped using the script `../Script/FlavorDB2_scrapping.py` ans stored in a csv file (`flavorDBthreshold.csv`)

### Data preprocessing steps

1. All the entries without threshold from FlavorDB were removed.
2. The respective value for the threshold and the units were extracted in separate columns manually.
3. The smiles for all the entries collected from literature were retrieved from PubChem using pubchem API rest. Those entries without a valid SMILES were manually checked in PubChem and those without resolved SMILES were removed.
4. Both datasets were merged into a single dataset (`threshold_data.xlsx`)
5. Entries with less than 2 heavy atoms were removed from the dataset. 

