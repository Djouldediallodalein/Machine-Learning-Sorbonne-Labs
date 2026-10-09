# Machine Learning & Deep Learning – Travaux pratiques

Série de 7 TP d'apprentissage automatique réalisés dans le cadre de l'UE *Apprentissage* (UM4RBR11), **M1 Systèmes Communicants – Sorbonne Université**.

Chaque dossier contient le **compte rendu sous forme de notebook Jupyter** (code, résultats, figures et analyses rédigées). GitHub affiche les notebooks directement, sorties comprises : il suffit de cliquer.

| TP | Sujet | Notions et outils |
|----|-------|-------------------|
| [TP1](TP1_kNN_Classification_Visages) | Classification de visages par k-NN | Base LFW, partition apprentissage/test, choix de k, validation croisée (scikit-learn) |
| [TP2](TP2_ACP_Reconstruction) | ACP, classification et reconstruction | Réduction de dimension, compression et reconstruction d'images, classification dans l'espace réduit |
| [TP3](TP3_Performance_Classifieur) | Performance d'un classifieur | Détection de pixels de peau, densités gaussiennes 2D, courbes ROC, optimisation de seuil |
| [TP4](TP4_Random_Forest) | Arbres de décision et forêts aléatoires | Profondeur d'arbre, nombre d'arbres, comparaison arbre seul / Random Forest |
| [TP5](TP5_Biais_Variance) | Compromis biais / variance | Complexité du modèle, sur-apprentissage, sous-apprentissage |
| [TP6](TP6_SVM) | Machines à vecteurs de support | Hyperplan séparateur, marge souple, noyaux, résolution du dual (cvxopt) |
| [TP7](TP7_CNN) | Réseaux de neurones convolutifs | MLP et CNN (TensorFlow / Keras), SGD, Dropout, Max-Pooling |

## Exécuter les notebooks

```bash
git clone https://github.com/Djouldediallodalein/Machine-Learning-Sorbonne-Labs.git
cd Machine-Learning-Sorbonne-Labs
pip install -r requirements.txt
cd TP1_kNN_Classification_Visages   # ou n'importe quel autre TP
jupyter notebook
```

Chaque dossier est autonome : il contient le notebook, les données qu'il charge (`.npy`, `.npz`, images) et, pour les TP2 à TP4, le fichier `TPx_ETU.py` qu'il importe. Il faut lancer Jupyter **depuis le dossier du TP** (les fichiers sont lus avec des chemins relatifs). Le TP7 télécharge MNIST au premier lancement (connexion Internet requise).

Les 7 notebooks ont été exécutés de bout en bout dans un environnement Python propre (aucune erreur).

## Remarques

- Les notebooks sont fournis avec leurs sorties (figures, courbes, métriques).
- Les données proviennent des sujets de TP (base de visages LFW, images de test pour la détection de peau, jeux de points 2D) ; les énoncés ne sont pas inclus.
