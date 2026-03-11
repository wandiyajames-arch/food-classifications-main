# 🍲 African Food Dataset Examples

This page presents some examples of food categories used in the **African Food Image Recognition Challenge**.

Each category corresponds to a label used in the dataset for image classification.

---

## 🍛 Thieboudienne (Senegal)

![Thieboudienne](https://upload.wikimedia.org/wikipedia/commons/3/3f/Thieboudienne.JPG)

**Label:** `thieboudienne`

Description:  
Thieboudienne (Ceebu Jën) is the **national dish of Senegal**, made with rice, fish, and vegetables cooked in tomato sauce. :contentReference[oaicite:0]{index=0}

---

## 🍚 Jollof Rice (West Africa)

![Jollof Rice](https://upload.wikimedia.org/wikipedia/commons/0/0b/Jollof_rice_with_fried_chicken.jpg)

**Label:** `jollof_rice`

Description:  
Jollof rice is a popular West African dish made with rice cooked in tomato sauce, spices, and vegetables.

---

## 🍗 Yassa Poulet (Senegal)

![Yassa Chicken](https://upload.wikimedia.org/wikipedia/commons/5/58/Yassa_poulet.jpg)

**Label:** `yassa_poulet`

Description:  
Yassa is a Senegalese dish made with marinated chicken, onions, mustard, and lemon.

---

## 🥜 Mafé (Peanut Stew)

![Mafe](https://upload.wikimedia.org/wikipedia/commons/4/4e/Maafe_peanut_stew.jpg)

**Label:** `mafe`

Description:  
Mafé is a traditional West African peanut stew made with meat or chicken and vegetables in a rich peanut sauce.

---

## 🐟 Attiéké with Fish (Ivory Coast)

![Attieke Poisson](https://upload.wikimedia.org/wikipedia/commons/b/b0/Attieke_poisson.jpg)

**Label:** `attieke_poisson`

Description:  
Attiéké is a cassava-based dish from Côte d’Ivoire often served with grilled fish and vegetables.

---

# 🧾 Dataset Labels

Example label list used in the competition:
    thieboudienne
    jollof_rice
    yassa_poulet
    mafe
    attieke_poisson

data/
│
├── train/
│ ├── thieboudienne
│ ├── jollof_rice
│ ├── yassa_poulet
│ ├── mafe
│ └── attieke_poisson
│
├── test/
│ └── images
│
├── train.csv
└── test.csv



---

# 🧠 Goal

Students must train a **deep learning model capable of recognizing African dishes from images**.

Possible approaches:

- CNN from scratch
- Transfer learning (ResNet, EfficientNet)
- Vision Transformers


🍲 Dataset Preview
+-------------+-------------+-------------+
| Thieboudienne | Jollof Rice | Yassa |
| image         | image       | image |
+-------------+-------------+-------------+