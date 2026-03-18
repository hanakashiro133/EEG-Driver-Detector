# EEG-Driver-Detector
A driver Fatigue and Alcohol State Warning System Based on EEG Detection Equipment. College Student Innovation Training Program Project

Data acquisition device：Macrotellect brainlinkpro

Device Information:
    Using electrodes: 10-20 International Federation of EEG System Standard Fp1
    EEG data sampling frequency: 512Hz

Core algorithm:
    RF(Using the scikit-learn library)
    Adopt features: θ/β Relative energy,α/β Relative energy,Hjorth motive,(α+θ)/β Relative energy，δ energy，Shannon entropy，Hjorth active，α/(δ+θ+α+β) Relative energy，δ/(δ+θ+α+β) Relative energy

python version:3.11

Development Log:
2026/2/24 Add feature: Relative energy,θ/β Relative energy,α/β Relative energy,(α+θ)/β Relative energy,δ energy
2026/3/2 Add frature: α/(δ+θ+α+β) Relative energy && Add feature matrix extraction code
2026/3/8 Add frature: δ/(δ+θ+α+β) Relative energy && Download dataset
2026/3/15 Add frature: entropy, Hjorth motive, Hjorth active && Preliminary simple SVM classification
