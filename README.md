# Master's Thesis: Impact of Data Inconsistency Removal on Classification

This repository contains the source code and experimental environment for the Master's thesis titled:

**_“Selected methods of removing data inconsistencies and their impact on the classification process”_**

The project investigates how removing inconsistent data blocks (based on **Rough Set Theory**) affects the performance and complexity of **Decision Trees** and **Random Forests**.

---

## 📂 Repository Contents

### `skrypt_badawczy.py`
The main research script performing the following steps:

- **Loads datasets** from the UCI Machine Learning Repository:  
  *Adult, Mushroom, Breast Cancer, Car Evaluation*
- **Preprocesses data**, including:  
  - One-Hot Encoding  
  - Handling missing values
- **Detects inconsistencies** by identifying conflicting data blocks  
  (objects with identical condition attributes but different decision classes)
- **Removes inconsistencies** using two strategies:  
  - **Quantitative Method** – keeps objects from the majority class  
  - **Drastic Method** – removes the entire conflicting block
- **Trains and evaluates models** using **10-fold cross-validation**

---

### `skrypt_wykres.py`
A visualization script used to generate the charts presented in the thesis, including:

- Accuracy comparison plots  
- Decision tree node reduction analysis

---

