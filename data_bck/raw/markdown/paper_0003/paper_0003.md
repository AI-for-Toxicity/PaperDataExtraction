<!-- image -->

Contents lists available at ScienceDirect

## NeuroToxicology

## Burst and principal components analyses of MEA data for 16 chemicals describe at least three effects classes

Cina M. Mack a , Bryant J. Lin b , James D. Turner c , Andrew F.M. Johnstone a , Lyle D. Burgoon d , Timothy J. Shafer a, *

- a NHEERL, ORD, U.S. Environmental Protection Agency, Research Triangle Park, NC, United States

b N.C. School of Science and Mathethmatics, Durham, NC, United States

- c North Carolina State University, Raleigh, NC, United States
- d NCEA, ORD, U.S. Environmental Protection Agency, Research Triangle Park, Durham, NC, United States

## A R T I C L E I N F O

Article history: Received 19 November 2012 Accepted 28 November 2013 Available online 8

December 2013

## Keywords:

Neurotoxicity screening In vitro

Chemical fingerprinting

## 1. Introduction

Microelectrode array (MEA) recordings have been widely utilized to study the effects of different drugs and chemicals on neuronal network function (Keefer et al., 2001; Gramowski et al., 2000; Gopal, 2003; for review, see Johnstone et al., 2010). More recently, it has been proposed that MEA approaches could be useful as in vitro screens to characterize potential neuroactivity/ neurotoxicity of chemicals (Johnstone et al., 2010). Recent publications have demonstrated the reproducibility of MEA data across different laboratories (Novellino et al., 2011), the ability of MEA approaches to distinguish neuroactive/neurotoxic

* Corresponding author at: Integrated Systems Toxicology Division, MD-B105-03, U.S. Environmental Protection Agency, Research Triangle Park, NC 27711, United States.

Tel.: +1 919 541 0647; fax: +1 919 541 4849.

E-mail address:

Shafer.tim@epa.gov

(T.J.

Shafer).

## A B S T R A C T

Microelectrode arrays (MEAs) can be used to detect drug and chemical induced changes in neuronal network function and have been used for neurotoxicity screening. As a proof-of-concept, the current study assessed the utility of analytical ''fingerprinting'' using principal components analysis (PCA) and chemical class prediction using support vector machines (SVMs) to classify chemical effects based on MEA data from 16 chemicals. Spontaneous firing rate in primary cortical cultures was increased by bicuculline (BIC), lindane (LND), RDX and picrotoxin (PTX); not changed by nicotine (NIC), acetaminophen (ACE), and glyphosate (GLY); and decreased by muscimol (MUS), verapamil (VER), fipronil (FIP), fluoxetine (FLU), chlorpyrifos oxon (CPO), domoic acid (DA), deltamethrin (DELT) and dimethyl phthalate (DMP). PCA was performed on mean firing rate, bursting parameters and synchrony data for concentrations above each chemical's EC50 for mean firing rate. The first three principal components accounted for 67.5, 19.7, and 6.9% of the data variability and were used to identify separation between chemical classes visually through spatial proximity. In the PCA, there was clear separation of GABAA antagonists BIC, LND, and RDX from other chemicals. For the SVM prediction model, the experiments were classified into the three chemical classes of increasing, decreasing or no change in activity with a mean accuracy of 83.8% under a radial kernel with 10-fold cross-validation. The separation of different chemical classes through PCA and high prediction accuracy in SVM of a small dataset indicates that MEA data may be useful for separating chemicals into effects classes using these or other related approaches.

Published by Elsevier Inc.

compounds from non-toxic compounds (Defranchi et al., 2011), and that these characteristics of MEAs can be applied in higher throughput platforms (McConnell et al., 2012). Thus, recordings of neural network activity using MEAs can be utilized as an in vitro assay for hazard identification of potential neuroactivity/neurotoxicity.

Development of in vitro assays that predict adverse outcomes based on toxicity pathways was a key concept in the NAS report on Toxicity testing in the 21st century (NRC, 2007). In vitro assays that measure endpoints that can be related clearly to adverse outcomes in vivo via a toxicity pathway will be more useful in making decisions about chemical safety. However, neurotoxicity can result from a wide variety of different insults to the nervous system, thus there are numerous potential toxicity pathways through which chemicals may act. Physiological function, such as measured by MEA recordings, is sensitive to alterations induced by chemical perturbations through many of these diverse toxicity pathways. For example, MEA recordings have demonstrated that neural network activity is sensitive to disruption by heavy metals (Gopal,

<!-- image -->

<!-- image -->

Fig. 1. Spike and bursting activity of cortical networks. (A) Example recording from a cortical network showing activity of the cortical network across the electrode array. Each small box represents an individual electrode. Note that the bottom row of electrodes is only partially visible. In this example, relatively synchronous bursts of activity across many of the electrodes are apparent. (B) An example of activity on a single electrode (circled) from (A) showing action potential spikes (arrow), a burst of action potentials, and the spike detection threshold (solid horizontal line). Burst duration (C) and the percentage of spikes in a burst (D) are illustrated in schematic diagrams. These parameters along with the mean firing rate, the number of active electrodes, the burst rate (bursts/min) and bursting channels (electrodes) were considered in the PCA.

<!-- image -->

2003; Gramowski et al., 2000; van Vliet et al., 2007; Parviz and Gross, 2007), pyrethroid insecticides (Meyer et al., 2008; Shafer et al., 2008), acetylcholinesterase inhibitors (Keefer et al., 2001), and GABAA and glutamate receptor agonists and antagonists (Gross et al., 1995). While the ability of MEAs to detect changes induced by chemicals acting through a broad variety of different targets makes this approach useful as a screen to detect effects of uncharacterized chemicals, approaches to classify such effects into different pathways would improve further the utility of MEA data. Therefore, the goal of the current studies was to provide proofof-concept for chemical classification, or ''fingerprinting'' using MEA data.

Spontaneous electrical activity in neural networks consists of action potentials (spikes) and organized patterns of action potential bursts (Fig. 1). Most commonly, chemical effects on network function focus on disruption of network firing (spike) rate, as this measure is sensitive to disruption by chemical exposure and can be easily and quickly extracted from the data files. However, spontaneous activity in primary cultures is further organized across time and space in the network (e.g. synchrony of activity). MEA recordings made from neural networks capture all of this information and the resultant data could be considered high content. Other investigators have demonstrated that such information can be utilized to classify compounds acting on glutamate, GABA and glycine receptors (Gramowski et al., 2004). In the present experiments, chemical effects on mean firing rate (MFR), burst rate, burst duration, potency, synchrony and other parameters were determined and then utilized to examine if principal components analysis (PCA) and other approaches could classify chemicals by their actions. To do so, concentration-dependent effects of a group of 16 chemicals on spontaneous network activity in primary cultures of cortical neurons grown on MEAs were utilized in PCA and support vector machine (SVM) analyses. These data were drawn from historical data (Novellino et al., 2011) and from other ongoing projects from the laboratory completed over a three year period from 2009 to 2011. The chemicals selected have known and different primary modes of action that result in changes in nervous system function, or are known not to cause significant changes in nervous system activity.

## 2. Materials and methods

## 2.1. Chemicals

Table 1 provides a list of chemicals tested in the present experiments, along with other information including the purity, source, solvent used and CAS number. It should be noted that MFR data for verapamil, muscimol and fluoxetine has previously been published (Novellino et al., 2011); however, other analyses (burst rate, synchrony, etc.) for these chemicals have not been published. Data for all the other chemicals are novel and have not been published previously, other than in abstract form. Other chemicals and materials used were from commercial vendors and were of reagent grade or higher.

## 2.2. Cell culture

All procedures involving animals were reviewed and approved by the NHEERL Institutional Animal Care and Use Committee and all animal welfare guidelines were observed. Timed pregnant Long-Evans rat dams (Charles River, Raleigh, NC, USA) were obtained and housed in the animal colony in individual plastic cages. They were provided food and water ad libitum , and maintained on a 12 h light/dark cycle at 37 8 C. Primary cortical cell cultures were made from postnatal day 0-1 pups as has been previously described (McConnell et al., 2012). Pups were euthanized by decapitation and their brains immediately dissected and placed in ice-cold dissection buffer (phosphate buffered saline (PBS), 15 mM HEPES, pH 7.4). In a sterile hood, the dissected cortices were minced with fine scissors in a dish of HEPES buffer.

Table 1 Chemical training set details.

| Chemical                 | CAS #       | Purity     | Vehicle     | Source               | Mechanism of action                                              | Previous MEA data                                        |
|--------------------------|-------------|------------|-------------|----------------------|------------------------------------------------------------------|----------------------------------------------------------|
| Acetaminophen (ACE)      | 103-90-2    | 99.0%      | DMSO a      | Sigma                | Cyclooxygenase (COX-2) inhibitor                                 | NE- McConnell et al., 2012                               |
| Bicuculline (BIC)        | 40709-69-1  | > 95.0%    | DMSO/EtOH b | Sigma                | GABA A receptor antagonist                                       | " MFR- McConnell et al., 2012; Rijal and Gross, 2008.    |
| Carbaryl (CAR)           | 63-25-2     | 99.8%      | DMSO/EtOH   | Chem                 | Cholinesterase inhibitor                                         | # MFR- McConnell et al., 2012; Defranchi et al., 2011.   |
| Chlorpyrifos Oxon (CPO)  | 5598-15-2   | 98.6%      | DMSO/EtOH   | Service Chem Service | Cholinesterase inhibitor                                         | # MFR-McConnell et al., 2012.                            |
| Deltamethrin (DELT)      | 52918-63-5  | 99.5%      | DMSO/EtOH   | Chem Service         | Voltage-gated sodium channel modulator                           | # MFR- McConnell et al., 2012; Scelfo et al., 2012       |
| Dimethyl Phthalate (DMP) | 131-11-3    | /C21 99.0% | DMSO        | Sigma                | Peroxisome proliferator-activated receptor- a (PPAR- a ) agonist |                                                          |
| Domoic Acid (DA)         | 14277-97-5  | 90.0%      | H 2 O       | Sigma                | Kainic acid receptor agonist                                     | # MFR- Hogberg et al., 2011; McConnell et al., 2012      |
| Fipronil (FIP)           | 120068-37-3 | 98.5%      | DMSO/EtOH   | Sigma                | GABA A receptor antagonist                                       | # MFR- Defranchi et al., 2011; McConnell et al., 2012    |
| Fluoxetine (FLU)         | 114247-09-5 | > 98.0%    | DMSO/EtOH   | Sigma                | Selective serotonin reuptake inhibitor (SSRI)                    | # MFR- Xia et al., 2003; Novellino et al., 2011          |
| Glyphosate (GLY)         | 1071-83-6   | 99.3%      | H 2 O       | Chem Service         | Amino acid metabolism inhibitor                                  | NE- McConnell et al., 2012                               |
| Lindane (LND)            | 58-89-9     | 99.9%      | DMSO/EtOH   | Aldrich              | GABA A receptor antagonist                                       | " MFR- McConnell et al., 2012                            |
| Muscimol (MUS)           | 18174-72-6  | /C21 98.0% | H 2 O       | Sigma                | GABA A receptor agonist                                          | # MFR- Rijal and Gross, 2008; Novellino et al., 2011     |
| Nicotine (NIC)           | 54-11-5     | /C21 99.0% | H 2 O       | Sigma                | Nicotinic acetylcholine receptor agonist                         | NE- McConnell et al., 2012 " MFR- Defranchi et al., 2011 |
| Picrotoxin (PTX)         | 124-87-8    | 98.0%      | DMSO/EtOH   | Sigma                | GABA A receptor antagonist                                       |                                                          |
| RDX c                    | 121-82-4    | > 99.5%    | DMSO        | Gift                 | GABA A receptor antagonist                                       | " MFR- McConnell et al., 2012                            |
| Verapamil (VER)          | 152-11-4    | 99.0%      | H 2 O       | Sigma                | Ca 2+ channel antagonist                                         | # MFR- Novellino et al., 2011                            |

Provided by Dr. Larry Williams, US Army Center for Health Promotion and Preventative Medicine; NE = No Effect; MFR = Mean Firing Rate.

a Dimethyl sulfoxide (DMSO).

b Ethanol (EtOH).

c Hexahydro-1,3,5-trinitro-1,3,5-triazine, hexogen.

Next, trituration with a plastic pipette tip was used to disperse the cells. The cortical cells were then placed into a 1.5 mL microcentrifuge tube containing Neurobasal A (NBA) media containing 10% heat-inactivated fetal bovine serum (FBS), 2.5 m M glutamate, penicillin (100 units/mL), and streptomycin (0.1 mg/mL) (pen/ strep). The cell solution was pipetted dropwise through a 100 m M filter (BD), and the filter rinsed with NBA media. The concentration of viable cells was determined using a hemacytometer and then diluted to a 2.5 /C2 10 5 cells/50 m L. The cells (50 m L drop) were then plated onto the center of glass MEA ''chips'' (MEAs; Multichannel Systems (MCS), Reutlingen, Germany). The MEAs were pre-coated prior to the culture with 1 mL poly-L-lysine (50 m g/mL, incubated 1 h removed and allowed to dry completely), and then on the day of the culture coated with 50 m L laminin (20 m g/mL) incubated at least 30 min and removed immediately prior to plating the cells. Cells were allowed 15 min to attach to the MEA substrate before 1 mL of NBA media (pen/strep) was added to each MEA. Post culture, MEAs were maintained at 37 8 C in a humidified 5% CO2 incubator. The day following (18-24 h) plating, the NBA media was replaced with 1 mL NBA media plus B27 Supplement TM . Four days later, this media was replaced with 1 mL glutamateand B27-free NBA media. Thereafter, old media was removed and MEAs received 1 mL fresh media every 7 days until electrophysiological recording began. Before the beginning of an experiment, each MEA was covered with a plastic cap fitted with a fluorinated ethylenepropylene membrane (ALA Scientific) to allow proper diffusion of air and CO2, and to reduce evaporation and to maintain sterility (Potter and DeMarse, 2001).

## 2.3. Chemical treatments

Stock solutions of each chemical were diluted with appropriate vehicle (Table 1) to working solutions. Chemicals were applied to the chips using a cumulative concentration exposure paradigm, ranging from 1 pM to 100 m M (or higher when no effect was observed). All experiments were performed in non-perfused, NBA (glutamate-free) media. Immediately prior to administration of each dose, 200 m L of culture media was carefully removed from the MEA, added to a glass vial containing the chemical and 10 m L of deionized water (d.i. H2O) to compensate for any evaporation and vortexed. This mixture was gently added back to the MEA, and the cap replaced to maintain temperature and volume until the next addition. A minimum 30 min interval separated consecutive doses of a chemical. The experiment ended when all doses had been administered or network activity ceased.

Post experiment, MEA chips were cleaned for re-use as follows. Each chip was rinsed with d.i. H2O and inspected for debris. Next, chips were sonicated for 30 s before addition of 2.5% pancreatin (Sigma)-PBS buffer solution onto the electrode array. Chips were rinsed in d.i. water 18 h later, dipped in 95% ethanol, dried and autoclaved. Lastly, chips were exposed to UV light for 30 min before being placed in sterile Petri dishes and stored at 4 8 C until used.

## 2.4. Microelectrode array (MEA) recordings

Each MEA contained an internal ground electrode and 59 recording electrodes with diameters of 30 m m, and 200 m m interelectrode spacing. Primary cortical cell cultures grown on MEAs form synaptic connections and develop spontaneous network activity (spiking and bursting) between days in vitro (DIV) 10-14 and have stable levels of activity thereafter (Robinette et al., 2011); other studies in our laboratory have demonstrated that GABAA receptor responses reach a mature state at DIV 6-8 (Inglefield and Shafer, 2000). In this study, MEA chips with mature neural networks (DIV 13-30) were placed on MCS preamplifiers with temperature maintained at 37 /C6 0.1 8 C by an MCS TC02 temperature controller. The system hardware consisted of four MCS 1060BC preamplifiers that were interfaced to a PC via MCS 2:1 switch and a

128 channel MC\_Card PCI board. This configuration allows for multiple (up to 4) chips to be recorded from simultaneously. MC\_Rack software version 3.6.8 or later (MCS) was used for data acquisition. Signals were collected with a sampling rate of 25 kHz and high pass filtered with 200 Hz cutoff. Peak to peak electrode noise levels were 8 m V or lower, well within MCS specifications for planar 30 m m Titanium Nitrite electrodes. Channels with higher levels of noise were grounded and not included in data analysis. Criteria for chip use included robust firing defined as exceeding a detection threshold level of /C6 15 m V (a software-supplied spike analyzer was used to detect spontaneous events), and a minimum of 10 active ( &gt; 5 spikes/ min) electrodes. A 30-60 min period of stable baseline (native) activity was recorded for use as an internal reference for each chip. MEAs where activity was clearly not stable (activity on all channels increasing or declining over time) were not used for experiments. Post hoc, networks where mean activity was not stable (mean activity clearly declining or increasing over time) were not included in the analysis. Two types of data files were saved from each experiment; one containing spike counts (MCS software *.dat file) binned on a per minute basis, and one that contained spike waveforms and timestamps (MCS software *.mcd file).

## 2.5. Data analysis

Each MEA chip was considered as an independent observation (''n'') for purposes of analysis. For each chemical, data was collected from at least two-three different culture dates. Spontaneous firing rate data (for MCS software, *.dat files) were exported into Microsoft Excel and averaged across electrodes on each chip for each minute of recording to produce a network-wide MFR for each MEA for baseline (control) and each concentration of chemical. The average MFR for a 10 min epoch during the baseline period (immediately prior to addition of the chemical) was calculated and compared to an equivalent 10 min period (immediately prior to addition of the next concentration of a chemical) after each chemical addition to assess effects on a percentage of control basis. Data were not utilized from a 5 min ''buffer'' period preceding and following each chemical addition, to avoid addition artifacts, as recording was not stopped during addition of chemicals. These data were then averaged over several MEAs to create concentration curves for each chemical. EC50 values (effective concentration causing decrease to 50% of control MFR or increase to 50% of maximal MFR for that chemical) for changes in MFR (Table 2) were determined by fitting these mean data to a sigmoidal (log (agonist) vs. response) function using Graphpad Prism software.

Table 2 EC50 values for effects on mean firing rate and direction of change.

| Chemical           | EC 50 ( m M)      | Direction of change   |   # Networks |
|--------------------|-------------------|-----------------------|--------------|
| Acetaminophen      | N.D. ( > 500 m M) | No change             |            4 |
| Bicuculline        | 0.41              | Increase              |           11 |
| Carbaryl           | 5.03              | Decrease              |            8 |
| Chlorpyrifos Oxon  | 0.037             | Decrease              |            7 |
| Deltamethrin       | 0.19              | Decrease              |           10 |
| Dimethyl Phthalate | 7.87              | Decrease              |            4 |
| Domoic Acid        | 0.28              | Decrease              |            7 |
| Fipronil           | 0.38              | Decrease              |            6 |
| Fluoxetine         | 5.1               | Decrease              |            7 |
| Glyphosate         | N.D. ( > 500 m M) | No change             |            5 |
| Lindane            | 1.9 m M           | Increase              |            6 |
| Muscimol           | 0.07 m M          | Decrease              |            5 |
| Nicotine           | N.D. ( > 500 m M) | No change             |           11 |
| Picrotoxin         | 0.15 m M          | Increase              |            8 |
| RDX                | 12.3 m M          | Increase              |            9 |
| Verapamil          | 5.1 m M           | Decrease              |           13 |

N.D. = Not determined because there was less than a 50% change at the highest concentration tested.

Burst analysis was conducted by importing data (for MCS software, *.mcd files) into NeuroExplorer (Nex Technologies) and then applying the burst analysis tool in this program to extract burst parameters (Section 3). In the Neuroexplorer burst definition parameters, a bin size of 0.5 s and the default values of maximum interval of starting and ending a burst of 0.01 s, minimum burst interval of 0.01 s, minimum burst duration of 0.02 s, and a minimum of four spikes per burst were selected to describe bursts. Burst parameters were determined for each individual electrode on an MEA, then data from all bursting electrodes in a network were averaged. The data were exported to Excel spreadsheets and bursting parameters at each concentration for each chip were determined by calculating the averages of the bursting channels in the same manner as MFR. The parameters from the bursting analysis included the percentage of channels bursting, number of spikes, mean spike frequency, number of bursts, bursts per minute, percentage of spikes in bursts, mean burst duration, mean spikes in bursts, mean burst interspike interval, mean spiking frequency in bursts, mean peak spiking frequency in bursts, and mean interburst interval (percentage of channels bursting was calculated from the number of channels of the NeuroExplorer output).

Synchrony of network activity : Network burst synchrony was determined in the following manner. Data (from *.mcd) were imported into NeuroExplorer, and then the timestamps of the action potentials were exported as a text file. Based on details in laboratory notebooks, additional files were written listing the intervals for analysis (times during which different concentrations of each compound were present) and the electrodes to be analyzed. Those electrodes from which data was collected were included; grounded electrodes or those that became noisy during the experiment (per notation in the lab notebook) were excluded. Using a custom program written in the Python programming language, the timestamps data were imported and analyzed using methods based on Selinger et al., 2004, as described in the following paragraphs. The code and supporting documentation are available by request (contact: shafer.tim@epa.gov).

Since a correlation coefficient can only be calculated for data from time intervals of equal length, the series of timestamps for each electrode was converted into a ''binary activity signal''. To generate this binary activity signal, the timestamps data were first divided into sequential bins of duration D t . A duration of 0.5 s was chosen for D t because this was significantly longer than the average duration of bursts and significantly shorter than the average time between bursts. Each bin in the binary activity signal was assigned a 0 if there were no timestamps in that bin or a 1 if there was at least one timestamp in that bin (Supplemental Fig. 1). This binning focused the analysis on burst synchrony and timestamps outside of bursts, preventing differences in individual timestamps within bursts (which are insignificant from a burst synchrony standpoint) from overly influencing the results while still providing sufficient resolution of individual bursts. It should be noted that this method does not account for potential shifts or jitter in firing (e.g. Wu et al., 2011), as the focus is on burst synchrony.

For each pair of electrodes in the array, a correlation coefficient was calculated for their binary activity signals using the formula

!

<!-- formula-not-decoded -->

#

where rij is the correlation coefficient, N is the number of bins in the binary activity signal, xi ( t ) is the value of the binary activity signal for the first electrode at bin t , and xj ( t ) is the value of the binary activity signal for the second electrode at bin t (Selinger et al., 2004). Electrodes with no activity or activity in every bin for a complete interval of analysis were disregarded from the analysis because correlation coefficients cannot be calculated for such data.

A histogram was plotted for the list of correlation coefficients for each interval for each MEA. These histograms could be examined for shifts to the right or left (increasing or decreasing shifts in synchrony, respectively) between applications of chemicals. In the example histograms in Supplemental Fig. 2, an increase in correlation compared to baseline activity is observed after treatment with bicuculline and Schering 50911. The arithmetic mean and standard error of the correlation coefficients were also calculated for each network (MEA). It should be noted that this method determines overall synchrony across the network, and that synchrony ''locally'' between adjacent electrodes may be higher than the network average.

Identifying chemical similarity through pattern recognition : The combined bursting, spiking, and synchrony values were compiled for the PCA. A single mean value for each parameter was obtained from each chip and chemical concentration combination and standardized to the chip control (baseline activity). For each chip and concentration pair, electrodes were removed if there was no bursting activity (usually grounded channels) or if any parameter value was 10 times larger than the mean for the chip/concentration, as such data indicated a noisy electrode. Prior to performing PCA on the dataset, groups of parameters with Pearson correlation coefficients greater than 0.95 were reduced to a single parameter. Additionally, using a random forest (randomForest package in R: Liaw et al., Liaw and Wiener, 2002) with 10,000 trees was used to identify the variables that are the most predictive in classifying the chemicals as demonstrating an increasing, decreasing, or unchanged MFR (for concentrations above the EC50/IC50). Based on the Strong Law of Large Numbers (Brieman, 2001) , the use of 10,000 trees in the random forest algorithm is robust to overfitting when using a large number of trees. From the categorization, parameters with low mean decrease-accuracy indices (less than 0.015) were not utilized for further analyses. Finally, for controls and chemicals that did not change activity enough to determine an EC50 value, data from the entire concentration range was utilized for PCA. For all other chemicals, only data from treatment concentrations greater than the mean EC50 values for MFR for each chemical were used for PCA, as data for concentrations below the EC50 may not change activity sufficiently from control to aid in separation.

Each principal component (PC) is the linear combination of the input parameters that minimizes the remaining variability in the dataset. These are readily calculated through singular vector decomposition using the FactoMineR R package (Husson et al., 2012). Once the PCs are determined, values for data points (identified by chip and chemical concentration) can be evaluated for each PC. The entire dataset can be quickly visualized by plotting these values against the PCs as the axes. Similar values indicate similar responses in activity for the particular experiment. To illustrate the overall chemical responses, data were condensed to single points for each chemical by taking an arithmetic mean for each PC. Further, 95% confidence ellipses were drawn around chemicals in the space given by the first two principal components, offering a definitive way to classify separation of classes by considering ellipse overlap.

A C-classification SVM was conducted on the same dataset used for PCA (Dimitriadou et al., 2011). The SVM used a radial-basis kernel with a gamma value of 1/7 (seven parameters were used). The SVM was used to generate a predictive model capable of using the data to identify chemicals that had a ''no change'', ''increased'' or ''decreased'' MFR (Table 2). The 10-fold cross-validation is a standard mechanism for model analysis, as is the use of receiveroperating characteristic (ROC) curves. The cross-validation is used to calculate the model's accuracy and generalizability and reproducibility. Ten-fold cross-validation means that the SVM software divided the dataset into 10 smaller equal sized subsamples. The SVM is trained on nine of the subsamples, and tested against the remaining one subsample. This process is repeated until each subsample has been used as the test set (10 times). The ROC curves are a visual way to represent the results from the SVM. Performance of the SVM was measured by the overall accuracy of prediction as well as by receiver operator characteristic (ROC) curves. Classification of the training data was compared to the true category (determined by chemical applied) to yield an overall percent accuracy. Decision values can be used to determine false and true positive rates for each pair of categories [increased/decreased, increased/no change, no change/decreased) to indicate the power of categorization in the model. All calculations were done in R v2.15.0 (R Development Core Team (2012)].

## 3. Results

The data analyzed in the present experiment were collected from a total of 114 different networks (MEA chips) over a period of 3 years (2009-2011). Overall, 46.0 /C6 2.2% (mean /C6 SEM) of electrodes available to record from on each chip were ''active'' ( &gt; 5 spikes/min) and the mean ( /C6 SEM) firing rate of these cortical networks was 128 /C6 11 spikes/min. The use of DMSO in the present experiments did not exceed a maximum of 0.7% by volume; up to a concentration of 1% by volume did not alter activity of cortical networks (data not shown), which is consistent with previously published data (Defranchi et al., 2011). Concentration-response relationships were determined for each of the compounds, and could be described by one of three general effects on the MFR (Fig. 2); no change (e.g. nicotine), increase (e.g. RDX), or decrease (e.g. carbaryl). Concentration-response relationships for the other compounds are provided in Supplemental Figs. 3 and 4. Four of the five compounds that are known GABAA receptor antagonists (BIC, PTX, RDX, LND) increased MFR in a concentration-dependent manner, although PTX was far less efficacious than the other three. At the highest concentrations of LND and PTX tested, MFR decreased. This may be due to over-stimulation of the network by increased glutamatergic input (Frega et al., 2012), or at least in the case of lindane, insolubility (E. Croom, personal communication) or potentially actions on voltage-gated calcium channels (Heusinkveld et al., 2010). Three compounds (GLY, ACE and nicotine) caused no substantial changes in MFR, while the remaining nine compounds (CARB, CFO, DA, DM, DMP, FLU, FIP, MUS, VER,) all decreased MFR. Two compounds did not have the expected effects: nicotine, which has been reported by others to alter activity on MEAs (Defranchi et al., 2011), caused only slight decreases in activity only at the highest concentrations. DMP caused concentration-dependent decreases in MFR. It was expected that this compound would not alter firing rate due to the lack of reported effects on neuronal function in the literature, as well as its use in in vitro assays as a negative control (Breier et al., 2008; Radio et al., 2008). For all other compounds, the EC50 values were in the low nanomolar to low micromolar range (Table 2), consistent with the fact that these are compounds known to be neuroactive or neurotoxic.

Analysis of network synchrony indicated that the basal level of synchrony across all networks was low but variable; with an overall correlation coefficient of 0.47 /C6 0.20 (mean /C6 s.d.; N = 114). Four compounds caused less than a 10% change in synchrony: DMP, GLY, DA and CAR. Five compounds increased synchrony, and four compounds increased network synchrony by at least 35% (Table 3).

Fig. 2. Representative effects of chemicals on mean firing rate (MFR) of cortical cultures grown on microelectrode arrays. Following collection of baseline data, concentration-response for each chemical was determined by cumulative dosing. In general, chemical effects on MFR fell into one of three categories; no change (Nicotine, top), increased (RDX, middle) or decreased (Carbaryl, bottom). Data from each experiment were normalized to the firing rate during baseline for that experiment (MEA), and each data point is the mean and SEM (the number of individual experiments are provided in Table 2 as the number of networks). SEM values for baseline points represent the SEM of the baseline values for each experiment relative to a mean baseline value for all experiments. Plots for the remaining 13 chemicals are in Supplemental Figs. 3 and 4.

<!-- image -->

Four of the five compounds that increased synchrony of activity were GABAA receptor antagonists (BIC, PTX, RDX and LND), while the fifth compound was nicotine. Changes in synchrony were concentrationdependent (Fig. 3), and in general, were similar to changes in MFR. However, this was not always the case, as evidenced by the fact that DA and CAR, which cause large decreases in MFR, did not change network synchrony. By contrast, PTX and NIC, caused similar increases in synchrony, despite producing only modest or no change in MFR, respectively.

## 3.1. Spontaneous network burst activity

Twelve different parameters were derived from the burst analysis (Section 2). Many of the parameters extracted from the data are highly correlated (Pearson correlation coefficient &gt; 0.95)

## Lindane

Fig. 3. Example concentration-response curves for chemical effects on network synchrony. Illustrated here are the effects of Lindane (top), which increased network synchrony, and Muscimol (bottom), which decreased it. Synchrony of activity was calculated for each experiment and data are the means and SEM of 6 (lindane) and 5 (muscimol) separate experiments. The highest concentration of lindane decreased both MFR and synchrony, possibly due to solubility issues. This point was not used to determine EC50 values.

<!-- image -->

with the other parameters (i.e., most likely collinear); thus they do not provide any additional information that would help separate the chemicals. Furthermore, pre-processing the dataset to remove low information variables will have the net benefit of decreasing the overall dataset size and complexity, which will decrease analysis times when using more complicated machine learning methods, and will enhance the ability to interpret the results. The parameters that were removed from the analysis were all mathematically derived from the included parameters. The removed parameters were: mean frequency (correlated with the number of spikes), number of bursts (correlated with burst per minute), and mean spikes in burst (correlated with burst duration). In order to identify those parameters that provided the most information regarding chemical separation, a random forest analysis was conducted using effects on firing rate (up, down, no change) as a discriminator (Table 4). Parameters with low mean decrease-accuracy indices (less than 0.015) removed on the basis of the random forest analysis included mean ISI in burst (inversely proportional to mean spikes in bursts), mean interburst interval (inversely proportional to number of bursts and bursts per minute), and mean peak frequency. The remaining parameters values for concentrations above the chemical EC50 were included in the PCA. The MFR and synchrony were also added to the parameters that were considered for the PCA. The first, second, and third principal components each accounted for 67.5, 19.7, and 6.9% of the variance, respectively. The first principal component accounts for frequency of bursts-such as bursts per minute and spike rate. In the second principal component, decreasing

Table 3 Synchrony of network activity.

| Chemical   | Baseline synchrony a   |   Conc. ( m M) | Change b          | Percent control c   |
|------------|------------------------|----------------|-------------------|---------------------|
| ACE        | 0.36 /C6 0.05          |          100   | /C0 0.04 /C6 0.12 | 79.4 /C6 27.7%      |
| BIC        | 0.53 /C6 0.05          |            5   | 0.15 /C6 0.06     | 141.2 /C6 17.0%     |
| DMP        | 0.26 /C6 0.05          |          100   | /C0 0.01 /C6 0.02 | 99.1 /C6 9.9%       |
| FLU        | 0.50 /C6 0.05          |           10   | /C0 0.21 /C6 0.10 | 57.5 /C6 18.4%      |
| GLY        | 0.45 /C6 0.10          |          300   | /C0 0.03 /C6 0.05 | 94.1 /C6 14.1%      |
| LND        | 0.38 /C6 0.08          |           50   | 0.26 /C6 0.06     | 257.7 /C6 90.2%     |
| MUS        | 0.65 /C6 0.07          |            1   | /C0 0.23 /C6 0.16 | 65.9 /C6 20.4%      |
| PTX        | 0.32 /C6 0.03          |            5   | 0.15 /C6 0.05     | 145.2 /C6 10.2%     |
| RDX        | 0.57 /C6 0.07          |          200   | 0.08 /C6 0.05     | 114.4 /C6 8.2%      |
| VER        | 0.64 /C6 0.05          |            3   | /C0 0.24 /C6 0.08 | 59.6 /C6 12.5%      |
| CPO        | 0.51 /C6 0.10          |            1   | /C0 0.12 /C6 0.06 | 78.4 /C6 8.2%       |
| DA         | 0.42 /C6 0.08          |            0.1 | /C0 0.07 /C6 0.15 | 90.4 /C6 29.5%      |
| DELT       | 0.30 /C6 0.07          |            0.3 | /C0 0.21 /C6 0.07 | 24.7 /C6 15.7%      |
| FIP        | 0.53 /C6 0.04          |            5   | /C0 0.40 /C6 0.11 | 22.7 /C6 20.9%      |
| CAR        | 0.43 /C6 0.05          |          300   | 0.01 /C6 0.01     | 101.5 /C6 4.3%      |
| NIC        | 0.48 /C6 0.09          |          100   | 0.07 /C6 0.07     | 138.1 /C6 26.2%     |

- a Mean values for network synchrony in the absence of treatment.

b Mean net change in network synchrony at the concentration of chemical indicated in the ''Conc.'' column.

c Mean percent control for changes in network synchrony (synchrony at the indicated concentration/synchrony baseline /C2 100). Values are the means and SEMs for 3-11 individual experiments (MEA chips). Most synchrony changes in the presence of chemical were correlated to the change in MFR. However, while DMP, and CAR decrease MFR, they did not have the same effect on network synchrony.

percentage of spikes in burst and increases in spike and spike rate account for the majority of the variance (Fig. 4).

In the plot of PCA barycenters, GABAA receptor antagonists RDX, BIC, and LND clustered together positively in the first principal second and third principal components (Fig. 5). Although the 95% confidence ellipse for LND is not as tight as those for RDX and BIC, the 95% confidence ellipses for these compounds overlap in the first and second as well as the first and third components (Supplemental Fig. 5). This clustering reflects the fact that these chemicals increase bursting at higher concentrations, while spikes within the bursts decreased (Fig. 4).

GLY, NIC and PTX also formed a small cluster (Fig. 5) where the 95% confidence ellipses (Supplemental Fig. 5) separated from all other chemicals except DELT and FLU. There were also small clusters formed by some of the remaining chemicals. However, there was also overlap in the 95% confidence ellipses for many of these chemicals (Supplemental Fig. 5). Carbaryl, MUS, and CPO were negative in the first and second principal components but near zero in the third component, thus forming a small cluster (Fig. 5). As expected, these chemicals decreased bursting, but tended to increase the percentage of spikes within bursts. Verapamil, FIP, DELT and DMP clustered negatively in PC1 and and PC3, but near the PC2 axis from their decreases in bursting and spiking activity. FLU and DA were intermediate in the space between the cluster formed by GLY, PTX and NIC and that of FIP, VER, DELT and DMP. Finally, the barycenter for ACE was very close to the control barycenter.

Table 4 Random forest analysis results a .

| Parameter                       | Mean decrease accuracy ( /C6 st. dev)   |
|---------------------------------|-----------------------------------------|
| % of channels with bursts       | 0.023 /C6 0.004                         |
| Spikes                          | 0.050 /C6 0.008                         |
| Bursts per minute               | 0.060 /C6 0.009                         |
| '% of All Spikes within a Burst | 0.0180 /C6 0.004                        |
| Mean burst duration             | 0.0120 /C6 0.004                        |
| Mean ISI in burst               | 0.010 /C6 0.003                         |
| Mean Peak Freq in bursts        | 0.011 /C6 0.003                         |
| Mean interburst interval        | 0.011 /C6 0.003                         |
| Spike rate                      | 0.115 /C6 0.0151                        |
| Synchrony                       | 0.060 /C6 0.0081                        |

a Mean decrease accuracy index with standard deviation from random forest analysis conducted on bursting parameters. RFA was used to classify chemicals as causing increases (up) no change or decreases (down), in MFR.

## 3.2. Support vector machine (SVM) results

On the input dataset, SVM correctly predicted chemical class of a particular concentration on a chip using 10-fold cross-validation, with a mean accuracy of 83.8 /C6 7.4% (mean accuracy /C6 s.d., N = 10) and overall correct classification rate of 86.6%. A mean accuracy of 33.33% is expected due to random chance, thus the SVM model is predicting outcomes with an accuracy much greater than chance. A total of 258 support vectors were used (104 in ''Decrease'', 76 in ''Neutral'', and 78 in ''Increase''). The table of numbers of predicted categories compared to the original classes is included in Table 5, while the accuracy for prediction on individual chemicals is provided in Table 6. While a large majority of the current dataset was predicted

Table 5 Support vector machine results.

| Predicted   | Actual   | Actual   | Actual   |
|-------------|----------|----------|----------|
|             | Decrease | Neutral  | Increase |
| Decrease    | 102      | 10       | 6        |
| Neutral     | 24       | 211      | 10       |
| Increase    | 5        | 13       | 126      |

Table 6 Accuracy of SVM for individual chemicals.

|      |   Accuracy (%) |   N |
|------|----------------|-----|
| ACE  |           92.3 |  26 |
| BIC  |           80.8 |  52 |
| CAR  |           87.5 |  16 |
| CPO  |           88.2 |  17 |
| DA   |          100   |   3 |
| DELT |           93.3 |  15 |
| DMP  |           62.5 |   8 |
| FIP  |           86.7 |  15 |
| FLU  |           50   |  30 |
| GLY  |           40   |  25 |
| LND  |          100   |  26 |
| MUS  |           76.5 |  17 |
| NIC  |           91.3 |  69 |
| PTX  |           85.7 |  28 |
| RDX  |           94.4 |  36 |
| VER  |          100   |  10 |

Table of chemical performance in SVM. Accuracy is measured as a percentage of the number of experiments predicted correctly. N is the number of experiments, or the number of unique chips and concentration pairs for the chemical above the IC50 value.

## Variables factor map (PCA)

Fig. 4. Variables factor map for PCA. Contributions of firing and bursting parameters to the first three principal components. The horizontal and vertical axes represent the first and second (top) or first and third (bottom) principal components, respectively. Each principal component represents the amount of the variation in the dataset that is noted in the parentheses. Each variable is represented by a vector in the map, and the length and direction (positive/negative) of the vector represents the degree to which the axes explain the variable. For example, bursts/minute contributes strongly to the variability in PC1, but much less so in PC2, while spikes, spike rate and % of spikes in burst all contribute to variability in PC1 and PC2, the % of spikes in bursts contributes in the opposite direction in PC2 from the other variables. Spatially, coordinates of the individuals can be visualized by the value of the parameters plotted against the parameter vector in the figure.

<!-- image -->

correctly, misclassified experiments were not distributed equally across chemicals. DA, LND, and VER were predicted with 100% accuracy, but there was some difficulty predicting FLU and GLY, which had the lowest percentage of experiments predicted correctly. Fluoxetine had primarily been predicted incorrectly as neutral for concentrations between 100 nM and 5 m M. Because the EC50 values were based on mean concentration-response curves, it is possible that individual experiments had varying EC50 values, allowing the inclusion of data that had not begun decreasing MFR at the particular concentration yet. This is consistent with the prediction errors mostly in the lower concentrations above the EC50, but no error at the highest two concentrations at 10 and 100 m M. Glyphosate was predicted as increasing in experiments from only one of the three chips used, thus additional data is needed to determine the overall performance of GLY in SVM. Interestingly, in the class of increasing MFR, it seemed that PTX performed as well or better than BIC, despite being closer to the controls in PCA. PTX was only misclassified at the highest two concentrations (5, 10 m M), while concentrations of misclassified BIC ranged from 500 nM to 50 m M.

Fig. 5. Plot of all chemical barycenters in PC1, PC2 and PC3. BIC, RDX and LND clearly separate from the other chemicals in the ''back'' left side of the figure. While the confidence limits of remaining chemicals overlap, CPO, CAR and MUS form a group of chemicals that is located in the ''front'' right side of the figure, and VER, DLT, DMP and FLU form a group of chemicals in the ''front'' lower portion of the figure. Overall, 94.1% of the total variance is accounted for by the first three components (PC1, PC2 and PC3) of the PCA. In this figure, ''control'' (solid square) is the baseline data for all experiments used in the PCA across all chemicals. Chemicals that increased MFR are shown as solid circles, while those that decreased MFR are open circles. Those that were without effects on MFR (EC50 &gt; 500 m M) are shown as solid triangles.

<!-- image -->

The receiver operating characteristic (ROC) curves show the pairwise true positive rate (the number of true positives that are correctly classified divided by the total number of samples) compared to the false positive rate (the number of incorrectly classified divided by the total number of samples) for each pair of classes (Fig. 6).

The ROC curve shows how a model performs in correctly classifying items. Random chance classifications would be depicted as a straight line through the origin, with equal true positive and false positive rates, and an area under the curve of 0.50. AUC values for each pair of classes were 0.96, 0.96, and 0.97 for Neutral/Decrease, Neutral/Increase, and Increase/Decrease, respectively. In all cases, the model performs better than random when discriminating chemicals that have increased or decreased activity (''up'' or ''down'', respectively) from neutral chemicals, with only minimal differences in being able to discriminate between the different pairs of classes.

## 4. Discussion

The present studies utilized a ''data mining'' approach to test proof-of-concept for chemical classification approaches for MEA data. Concentration-response relationships for 16 chemicals on spontaneous activity in cortical networks demonstrated three distinct classes of effect on the network MFR; increased, decreased and no change. These results confirm previous studies demonstrating that MEA approaches be utilized to detect neuroactive/ neurotoxic substances (Johnstone et al., 2010; Defranchi et al., 2011; McConnell et al., 2012) as effects on MFR in the current set of experiments were consistent with previous data (Table 2). Analysis of network synchronization and burst characteristics in the presence of these chemicals demonstrated further differences between groups of chemicals, such as increased synchronization observed in the presence of GABAA receptor antagonists. When all of these data were considered together using a PCA and SVM

## Neutral vs Increase

Fig. 6. Receiver-operator curves (ROC) from support vector machine analysis. ROC curves are provided for each pair of classifiers. The true positive rate is the number of true positives that are correctly classified (e.g., classifying as ''up'' when it should be ''up'') divided by the total number of samples. The false positive rate is the number of false positives (up or down) that are incorrectly classified (e.g., classifying as ''up'' when it should be ''down'') divided by the total number of samples. Random chance classifications would be depicted as a straight line through the origin, with equal true positive and false positive rates, and an area under the curve of 0.50. The corresponding areas under the curve (AUC) measures are (a) 0.96, (b) 0.96, and (c) 0.97 for Neutral/Decrease, Neutral/Increase, and Increase/Decrease, respectively. These values demonstrate that in all cases the model performs much better than random chance.

<!-- image -->

approaches, the results demonstrated that some GABAA antagonists clearly separate from other compounds in the PCA, and that SVM reliably predicts three effects classes determined on the basis of the data. In addition, there are trends toward separation of other compounds in the PCA. Considering the small dataset, these results provide proof-of-concept that, with improvements, such approaches may be useful to categorize previously uncharacterized compounds (''unknowns'') into effect classes.

Overall, the PCA identified four general clusters of chemicals. The first, which included the GABAA antagonists LND, BIC and RDX, consisted of compounds that caused clear increases in MFR and bursts per minute. This cluster separated most clearly from the others based on 95% confidence limits. The second cluster consisted of GLY, NIC and PTX, which separate completely from all chemicals except for DELT and FLU. These chemicals were less effective at altering MFR, but altered other burst parameters. The third cluster was formed by VER, DELT, DMP and FIP; all chemicals that decreased most of the parameters used in this study. The fourth cluster consists of two cholinesterase inhibitors, CPO, and CAR, as well as the GABA receptor agonist, MUS. Separation of these last two clusters is not clear, as there is overlap between the confidence intervals. The barycenter for ACE was very close to the control barycenter, indicating that this chemical was without effect on most parameters and is a good negative control chemical for both MFR and burst characteristics.

Overall the SVM was able to correctly classify experiments into the three effects classes based on MFR with 86.6% accuracy. This also supports the proof-of-concept that MEA data can be used to separate chemicals by effects class. Because of the small size of the dataset, only three classes were used for the current SVM. As larger datasets become available, the number of classes in the SVM can be expanded, for example, by including potency estimates. An advantage of SVM models is that new data can be incorporated into the model and the category of response can be predicted, such that effects of a previously uncharacterized chemical can be classified with other chemicals that have similar effects.

GABAA receptor antagonists were the largest single class of compounds in this set of 16 chemicals, yet only three (BIC, RDX and LND) of these compounds clustered together. These GABAA antagonists typically cause seizures in vivo , and the present data indicate that MEAs recordings may be able to reliably detect and separate seizurigenic compounds from those with other types of effects. Four of five GABAA antagonists (BIC, RDX, LND, PTX) caused concentration-dependent increases in synchrony, while FIP decreased it. Picrotoxin-induced increases in synchrony in neuronal networks were comparable to those induced by BIC (Table 3), therefore differences in other parameters must account for its lack of clustering with BIC, LND and RDX. Although other differences may also exist, one clear difference is the overall efficacy of PTX to increase firing rate, which was much lower ( /C24 120% of control) than for BIC, LND and RDX ( /C24 250-400% of control). Fipronil, which also binds to (Kamijima and Casida, 2000) and inhibits mammalian (Ikeda et al., 2001) GABAA receptors, decreased firing rate, bursting rate, and synchrony. The results with fipronil are consistent with previous results wherein it decreased weighted MFR in a single concentration screen (McConnell et al., 2012). This lack of excitatory action of fipronil appears to be inconsistent with the other GABAA antagonists. However, fipronil binding to vertebrate GABAA receptors is several orders of magnitude lower than for insect receptors, and metabolites of fipronil have higher affinities for binding to (Hainzl et al., 1998) and inhibiting (Zhao et al., 2005) the receptor than does the parent compound. Furthermore, in vivo studies indicate that fipronil toxicity in vertebrates is not well correlated with inhibition of specific binding of the GABAA antagonist ethynylbicycloorthobenzoate (EBOB), even at seizureinducing doses (Kamijima and Casida, 2000) . Fipronil is a noncompetitive antagonist of EBOB binding (Cole et al., 1993) and does not share a common site with picrotoxinin (Ikeda et al., 2001). Thus, it does not interact with the GABAA receptor in the same manner as do the other compounds. It is possible that the effects of FIP in the present study are not the result of interactions of this compound with the GABAA receptor but are instead due to non-specific effects of this compound. This hypothesis is supported by the fact that the PCA separated fipronil from the other GABAA antagonists.

The chemical barycenters of other classes of compounds also showed clustering in the PCA, although the overlaps of 95% confidence limits indicted that they were not completely separated. The acetylcholinesterase inhibitors chlorpyrifos oxon and carbaryl grouped close to each other and away from other classes of compounds. Interestingly, muscimol, a GABAA receptor agonist also clusters closely with these two compounds, a reflection of the ability of all three of these compounds to decrease bursting, but to increase the percentage of spikes within bursts.

Nicotine was without effect on MFR in the current experiments. This compound has been reported by others (Defranchi et al., 2011) to alter MFR, but its lack of effect here is consistent with other data published (McConnell et al., 2012) and preliminary data (Valdivia et al., 2013) from our laboratory. Interestingly, in the PCA nicotine's barycenter along with GLY and PTX, separates from the other compounds despite their having little effect on MFR. Two of these compounds (NIC and PTX) increased synchrony, while GLY was without effect on this parameter (Table 3). The variable factors map indicated that an increase in the percentage of spikes occurring in a burst and a decrease in spikes and spike rate are primarily responsible for compounds locating in the same quadrant of the PCA space as nicotine. Indeed, nicotine, even as low as10 m M, slightly increased the percent of spikes in a burst from 34.2 /C6 8.3 (mean /C6 SEM, N = 9) in control to 35.9 /C6 5.2 ( N = 9) and decreased the number of spikes and spike rate from 972 /C6 283 to 646 /C6 140 and 84 /C6 15 to 66 /C6 11, respectively. Similar changes also occurred at higher concentrations of nicotine. Confirmation of this result with other nicotinic compounds is needed, but one advantage of considering other endpoints besides MFR may be that compounds like nicotine which are without effects on MFR, can be detected.

While the present study is promising and provides a proof-ofconcept for compound ''fingerprinting'' using MEA data, it is not without limitations and will require improvements to reliably separate classes of chemicals beyond GABAA antagonists. This set of 16 drugs and chemicals contained representative chemicals from many different classes, but the number of compounds from any given class were limited. GABAA antagonists were the most abundant class with five compounds tested, and two different cholinesterase inhibitors were tested, but these were comprised of one each of an organophosphate (chlorpyrifos) and carbamate (carbaryl). Otherwise, the list of compounds tested consisted of a single representative from nine additional classes. One class of chemical lacking from this set of chemicals was a compound that increased MFR but was not a GABAA antagonist (for example, glutamate or NMDA; see Frega et al., 2012), which could have helped to determine how well the GABAA antagonists separated from other compounds that increase firing. Furthermore, while separation of three of the GABAA antagonists is clear, separation of the other classes of compounds is not as robust, as the 95% confidence ellipses overlap for these compounds (Supplemental Fig. 5). In order for these approaches to provide useful classification of unknown chemicals, they will need to produce clear separation of different classes of known chemicals. The overlap is driven by the variability of the data, which may be influenced by several different parameters. Concentration-response curves were collected in a cumulative manner, thus issues such as intracellular accumulation of chemicals and/or receptor desensitization may contribute to variability or differences between chemicals. This variability could be decreased either by including more replicates, or preferably, by having a better understanding of and control over experimental conditions that contribute to variability. For example, chip to chip, day-to-day and culture-to-culture differences may all contribute to variability. This may in part be improved through the use of multi-well chips (McConnell et al., 2012). Additional data, both in terms of more chemicals from each class, as well as defining other parameters (for example, temporal regularity and network synchronization (CVtime and CVnet) such as used by Keefer et al., 2001) that discriminate between chemical classes will also improve this approach. With respect to the latter, there are several potential metrics that could be informative, including stimulus induced activity, measurement of plasticity (Arnold et al., 2005), and/or assessment of changes to the amplitude or duration of action potentials. Approaches to separate chemicals may also benefit from recent improvements in the throughput of MEA systems to include 12 (McConnell et al., 2012) and 48 well MEA plates. The data for the present study were collected over a prolonged period of time and from numerous different cultures. Higher throughput MEA systems will allow for the simultaneous testing of many chemicals within the same plate and/or culture, which will reduce one source of variability for these types of measurements and allow for many more chemicals to be tested. Finally, additional data to help define compound classes need not come solely from MEA recordings, as combination of MEA data with, for example, assays from the EPA ToxCast dataset may also improve classification schemes.

There are also other data analysis approaches that could also be investigated to determine whether they might provide better separation of different chemical classes based on their effects on spontaneous network activity. Spike train analysis and features selection (Gramowski et al., 2004) have previously been used to classify pharmaceutical compounds based on their effects on MEAs. Additional approaches include other types of pattern recognition methods, such as k -means, naı ¨ve Bayes, and marketbasket approaches. Applying these approaches to datasets such as present here will be necessary to evaluate their utility and determine the best approaches.

Utilization of PCA and SVM approaches to analyze data may also have other applications. For example, individual or particular groups of experiments that were predicted incorrectly by the model can be identified and studied to improve data collection methods. Incorrect prediction in SVM or differences identified in PCA may also identify data curation issues for a particular experiment, providing a quality control check. Finally, it should also be noted that PCA and SVM could be applied to data from other toxicity testing approaches. This includes high-content imaging (Harrill et al., 2011) and other assays where multiple parameters are evaluated, such as developmental toxicity studies using zebrafish (Padilla et al., 2011, 2012).

In conclusion, the present results demonstrate proof-ofconcept that chemicals that may alter neuronal function can not only be detected using MEA recording approaches, but that more comprehensive consideration of the data from such recordings can reveal differences in chemicals effects that can be separated and classified. Upon refinement of this approach, it will be possible to build effects databases for chemicals with known mechanisms of action, which will then help to identify potential modes of action for uncharacterized chemicals. Future studies will explore this further by considering additional chemicals, endpoints for analysis, and analysis approaches.

## Conflicts of interest

None.

## Acknowledgments

The authors gratefully acknowledge the assistance of Mr. Brian Robinette in preparing cortical cultures on the MEAs. In addition, we thank Drs William LeFew and David Herr of the US EPA for constructive comments on previous versions of this manuscript. Preparation of this document has been funded by the U.S. Environmental Protection Agency. This document has been reviewed by the National Health and Environmental Effects

Research Laboratory and approved for publication. Approval does not signify that the contents reflect the views of the Agency, nor does mention of trade names or commercial products constitute endorsement or recommendation for use. BJL and JDT were supported in part through the EPA-Shaw University Research Apprenticeship Program for High School Students (Cooperative agreement number CR-83414001).

## Appendix A. Supplementary data

Supplementary data associated with this article can be found, in the online version, at http://dx.doi.org/10.1016/j.neuro.2013. 11.008.

## References

Arnold FJ, Hofmann F, Bengtson CP, Wittmann M, Vanhoutte P, Bading H. Microelectrode array recordings of cultured hippocampal networks reveal a simple model for transcription and protein synthesis-dependent plasticity. J Physiol 2005;564:3-19.

Breier J, Radio N, Mundy WR, Shafer TJ. Development of a high-throughput assay for assessing antiproliferative compounds using human neuroprogenitor cells. Tox Sci 2008;105:119-33.

Brieman L. Random Forests. Mach Learn 2001;45:5-32.

Defranchi E, Novellino A, Whelan M, Vogel S, Ramirez T, van Ravenzwaay B, Landsiedel R. Feasibility assessment of mirco-electrode chip assay as a method of detecting neurotoxicity in vitro . Front Neuroeng 2011;4(6):1-12.

Dimitriadou E, Hornik K, Leisch F, Meyer D, Weingessel A. e1071: Misc Functions of the Department of Statistics (e1071), TU Wien. R package version 1.6. http://CRAN.Rproject.org/package=e1071; 2011.

Frega M, Pasquale V, Tedesco M, Marcoli M, Contestabile A, Nanni M, Bonzano L, Maura G, Chippalone M. Cortical cultures coupled to micro-electrode arrays: a novel approach to perform in vitro excitotoxicity testing. Neurotoxicol Teratol 2012;34:116-27.

Gopal K. Neurotoxic effects of mercury on auditory cortex networks growing on microelectrode arrays: a preliminary analysis. Neurotoxicol Teratol 2003;25:69-76.

Gramowski A, Schiffmann D, Gross GW. Quantification of acute neurotoxic effects of trimethyltin using neuronal networks cultures on microelectrode arrays Neurotoxicology 2000;21:331-42.

Gramowski A, Ju ¨ gelt K, Weiss DG, Gross GW. Substance identification by quantitative characterization of oscillatory activity in murine spinal cord networks on microelectrode arrays. Eur J Neurosci 2004;19:2815-25.

Gross GW, Rhoades BK, Azzazy HME, Wu M-C. The use of neuronal networks on multielectrode arrays as biosensors. Biosens Bioelectron 1995;10:553-67.

Hainzl D, Cole LM, Casida JE. Mechanisms for selective toxicity of fipronil insecticide and its sulfone metabolite and desulfinyl photoproduct. Chem Res Toxicol 1998;11:1529-35.

Harrill JA, Robinette BL, Mundy WR. Use of high content image analysis to detect chemical-induced changes in synaptogenesis in vitro . Toxicol In Vitro 2011;25:368-87.

Heusinkveld HJ, Thomas GO, Lamot I, van den Berg M, Kroese AB, Westerink RH. Dual actions of lindane ( g -hexachlorocyclohexane) on calcium homeostasis and exocytosis in rat PC12 cells. Toxicol Appl Pharmacol 2010;248:12-9.

Hogberg HT, Sobanski T, Novellino A, Whelan M, Weiss DG, Bal-Price AK. Application of micro-electrode arrays (MEAs) as an emerging technology for developmental neurotoxicity: evaluation of domoic acid-induced effects in primary cultures of rat cortical neurons. Neurotoxicology 2011;32:158-68.

Husson F, Josse J, Le S, Mazet J. FactoMineR: Multivariate Exploratory Data Analysis and Data Mining with R. R package version 1.18. http://CRAN.R-project.org/package=FactoMineR; 2012.

Ikeda T, Zhao X, Nagata K, Kono Y, Shono T, Yeh JZ, Narahashi T. Fipronil modulation of l -aminobutyric acidA receptors in rat dorsal root ganglion neurons. J Pharmacol Exp Ther 2001;296:914-21.

Inglefield JR, Shafer TJ. Perturbation by the PCB mixture aroclor 1254 of GABA(A) receptor-mediated calcium and chloride responses during maturation in vitro of rat neocortical cells. Toxicol Appl Pharmacol 2000;164:184-95.

Kamijima M, Casida JE. Regional modification of [ 3 H]ethynylbicycloorthobenzoate binding of mouse brain GABAA receptor by endosulfan, fipronil and avermectin B1a. Toxicol Appl Pharmacol 2000;163:188-94.

Keefer E, Norton S, Boyle N, Talesa V, Gross G. Acute toxicity screening of novel AChE inhibitors using neuronal networks on microelectrode arrays. NeuroToxicol 2001;22:3-12.

Johnstone AFM, Gross GW, Weiss D, Schroeder O, Shafer TJ. Use of microelectrode arrays for neurotoxicity testing in the 21st century. Neurotoxicology 2010;31:331-50.

Liaw A, Wiener M. Classification and regression by randomforest. R News 2002;2:1822.

McConnell ER, McClain MA, Ross J, LeFew WR, Shafer TJ. Evaluation of multi-well microelectrode arrays for neurotoxicity screening using a chemical training set. Neurotoxicology 2012;33:1048-57.

Meyer DA, Carter JM, Johnstone AFM, Shafer TJ. Pyrethroid modulation of spontaneous neuronal excitability and neurotransmission in hippocampal neurons in culture. Neurotoxicology 2008;29:213-25.

Novellino A, Scelfo B, Palosaari T, Price A, Sobanski T, Shafer T, Johnstone A, Gross G, Gramowski A, Schroeder O, Jugelt K, Chiappalone M, Benfenati F, Martinoia S, Tedesco M, Defranchi E, D'Angelo P, Whelan M. Development of micro-electrode array based tests for neurotoxicity: assessment of interlaboratory reproducibility with neuroactive chemicals. Front Neuroeng 2011;4(4):1-14.

NRC. Toxicity Testing in the Twenty-First Century: A Vision and a Strategy. Washington, D.C.: The National Academies Press, 2007.

Padilla S, Hunter DL, Padnos B, Frady S, MacPhail RC. Assessing locomotor activity in larval zebrafish: influence of extrinsic and intrinsic variables. Neurotoxicol Teratol 2011;33:624-30.

Padilla S, Corum D, Padnos B, Hunter DL, Beam A, Houck KA, Sipes N, Kleinstreuer N, Knudsen T, Dix DJ, Reif DM. Zebrafish developmental screening of the ToxCast TM Phase I chemical library. Reprod Toxicol 2012;33:174-87.

Parviz M, Gross GW. Quantification of zinc toxicity using neuronal networks on microelectrode arrays. Neurotoxicology 2007;28:520-31.

Potter SM, DeMarse TB. A new approach to neural cell culture for long-term studies. J Neurosci Methods 2001;110(1-2):17-24.

Radio N, Breier J, Shafer TJ, Mundy WR. Development of a high-throughput assay for assessing neurite outgrowth. Tox Sci 2008;105:106-18.

R Development Core Team. R: A language and environment for statistical computing. R Foundation for Statistical Computing, Vienna, Austria. ISBN 3-900051-07-0, http:// www.R-project.org/; 2012.

Rijal SO, Gross GW. Dissociation constants for GABA(A) receptor antagonists determined with neuronal networs on microelectrode arrays. J Neurosci Methods 2008;173(2):183-92.

Robinette BL, Harrill JA, Mundy WR, Shafer TJ. In vitro assessment of developmental neurotoxicity: use of microelectrode arrays to measure functional changes in neuronal network ontogeny. Front Neuroeng 2011;4:1.

Scelfo B, Politi M, Reniero F, Palosaari T, Whelan M, Zaldı ´var JM. Application of multielectrode array (MEA) chips for the evaluation of mixtures neurotoxicity. Toxicology 2012;299(2-3):172-83.

Selinger JV, Pancrazio JJ, Gross GW. Measuring synchronization in neuronal networks for biosensor applications. Biosens Bioelectron 2004;19:675-83.

Shafer TJ, Rijal SO, Gross GW. Complete inhibition of spontaneous activity in neuronal networks in vitro by deltamethrin and permethrin. Neurotoxicology 2008;29:20312.

Valdivia P, Martin M, LeFew W, Ross J, Houck K, Shafer TJ. Evaluation of the neuroactivity of ToxCast compounds using multiwell microelectrode array recordings of primary cortical neurons. Abstract #900. The Toxicol 2013;132(1) [Society of Toxicology Annual Meeting Abstract Available online: www.toxicology.org]..

van Vliet E, Stoppini L, Balestrino M, Eskes C, Griesinger C, Sobanski T, Whelan M, Hartung T, Coecke S. Electrophysiological recording of re-aggregating brain cell cultures on multi-electrode arrays to detect acute neurotoxic effects. Neurotoxicology 2007;28:1136-46.

Wu W, Wheeler DW, Pipa G. Bivariate and multivariate NeuroXidence: a robust and reliable method to detect modulations of spike-spike synchronization across experimental conditions. Front Neuroinformatics 2011;5:14.

Zhao X, Yeh JZ, Salgado VL, Narahashi T. Sulfone metabolite of fipronil blocks l -aminobutyric acidand glutamate-activated chloride channels in mammalian and insect neurons. J Pharmacol Exp Therap 2005;314:316-73.