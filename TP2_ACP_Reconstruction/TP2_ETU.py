# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:55:33 2025

@author: cathe
"""

sans_notebook = 1 # 0 si vous avez le notebook sinon 1 pour tester le code sur un terminal

"""
TP2 - 

Etudiant :
Nom : DIALLO
Prénom : Mamadou djouldé 
cursus : M1 Syscom

"""
# Importation des BUs: 
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.decomposition import PCA # Pour l'ACP

def plotGallery(images, n=16, title=None):
    # Affiche les n premières images contenues dans images
    # images est de taille Nb image*Ny*Nx
    n = min(n, images.shape[0])
    nSubplots = int(np.ceil(np.sqrt(n)))
    fig, axs = plt.subplots(nSubplots, nSubplots)
    for i in range(n):
        axs[i // nSubplots, i % nSubplots].imshow(images[i], cmap=plt.cm.gray)
        axs[i // nSubplots, i % nSubplots].set_xticks([])
        axs[i // nSubplots, i % nSubplots].set_yticks([])
    if title:
        plt.suptitle(title)
    plt.show()

def plotHistoClasses(lbls):
    # Affiche le nombre d'exemples par classe
    nLbls = np.array([[i, np.where(lbls == i)[0].shape[0]] for i in np.unique(lbls)])
    plt.figure()
    plt.bar(nLbls[:, 0], nLbls[:, 1])
    plt.title("Nombre d'exemples par classe")
    plt.grid(axis='y')
    plt.show()
    




# ---------------------------------------------------------------------------------------------
# I Chargement des données

# I.a. Chargement de la base
[X, y, name]=np.load("TP2.npy",allow_pickle=True)

if (sans_notebook) :
    
    plotGallery(X)
    
    plotHistoClasses(y)

# Partitionnement de la base d’apprentissage

def partitionner_et_afficher(X, y, test_size=0.25, random_state=52):
    # Partitionne la base d'apprentissage et affiche tailles et dimensions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\nDonnées en apprentissage et en test :")
    print("Nombre d'images en apprentissage :", X_train.shape[0])
    print("Nombre d'images en test :", X_test.shape[0])

    return X_train, X_test, y_train, y_test

if (sans_notebook) :
    X_train, X_test, y_train, y_test = partitionner_et_afficher(X, y)


# Redimensionnement des données pour le codage rétinien (kppv)
# Chaque image est vectorisée en un tableau 2D (N, P)
def redimensionnement(X_train, X_test):
    # Chaque image doit être représentée par un vecteur de 2914 caractéristiques (62*47)
    # On redimensionne X_train et X_test pour obtenir (N, 2914) où N est le nombre d'exemples
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print("\nAprès redimensionnement :")
    print("Dimensions de X_train :", X_train.shape) 
    print("Dimensions de X_test :", X_test.shape)

    return X_train, X_test

if (sans_notebook) :
    X_train, X_test = redimensionnement(X_train, X_test)



#---------------------------------------------------------------------------------------

# II. Analyse en composantes principales et classification 
# I.1 ACP en gardant le maximum de composantes, ajustement et tracé des variances

def acp_max_composantes(X_train, afficher=True):
   
    """
    Cette fonction réalise une ACP en gardant le maximum de composantes.
    """
    # ACP avec nombre maximum de composantes soit 966 pour notre tp
    pca = PCA(n_components= 966)
    pca.fit(X_train)  # on fit simpleement  le X_train

    if afficher:
        tracer = pca.explained_variance_ratio_
        plt.figure()
        plt.plot(range(0,966), tracer)
        plt.ylabel('variance expliquée')
        plt.xlabel('Composante principale')
        plt.title('Tracer des variances (ACP max)')
        plt.show()

    return pca


if(sans_notebook):
    pca = acp_max_composantes(X_train, afficher=True)

# II.2 ACP avec 100 composantes, ajustement de X_train et transformation de X_train/X_test
    
def ACP_n_composants(X_train, X_test, n_components, tracer=False):
    """
        # ACP avec n composantes, retourne les données transformées
    """
    pca_n = PCA(n_components=n_components)
    pca_n.fit(X_train)
    X_train_n = pca_n.transform(X_train)
    X_test_n = pca_n.transform(X_test)
    if tracer:
        plt.figure()
        plt.plot(range(0,n_components), pca_n.explained_variance_ratio_)
        plt.ylabel('variance expliquée')
        plt.xlabel('Composante principale')
        plt.title('Tracer des variances (ACP)')
        plt.show()
    return X_train_n, X_test_n


if (sans_notebook) :
    
    X_train1, X_test1 = ACP_n_composants(X_train, X_test, n_components=100, tracer=True)
    print('\nForme de l ACP en concervant 100 composantes : \nX_train1 : ', X_train1.shape, '\nX_test1 : ', X_test1.shape)



# II.3 Classification avec kppv (k=5) et distance de Manhattan

# Je crée une fonction permettant de faire la classification kppv en fonction de la valeur de k qu'on veut avec la distance de manhattan
def classification_5ppv(X_train, y_train, X_test, y_test, k=5):
    """
         kppv (K voisins), distance = Manhattan
    """

    # Création du classifieur kppv avec distance de Manhattan
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) # taux de reconaissance
    
    print("\nMatrice de confusion :\n", cm)
    print("Taux de reconnaissance : {:.2f}%".format(accuracy * 100))


if (sans_notebook) :
        
    # Classification sur les données de départ : 
    tps1 = time.time() 
    classification_5ppv(X_train, y_train, X_test, y_test, k=5) 
    tps2 = time.time() 
    print("Durée de classification sur les données de départ",tps2 - tps1) 
    
    # Classification sur les nouvelles données : 
    tps1 = time.time() 
    classification_5ppv(X_train1, y_train, X_test1, y_test, k=5)
    tps2 = time.time() 
    print("Durée de classification sur les nouvelles données",tps2 - tps1) 




#----------------------------------------------------------------------------------------------------------------------------------

# III. Analyse en composantes principales et reconstruction 

# a-) Définissons la décomposition en composantes principales en utilisant la fonction PCA() en conservant 50 composantes et ajustez le modèle sur X_train. 

if (sans_notebook) : # On reutilise la fonction ACP_n_components que j'ai definie dans la partie precedente
    
    X_train50, X_test50 = ACP_n_composants(X_train, X_test, n_components=50, tracer=True)
    print('\nForme de l ACP en concervant 50 composantes : \nX_train50 : ', X_train50.shape, '\nX_test50 : ', X_test50.shape)

# b-) Récupérons les vecteurs propres en utilisant un attribut de PCA(). Redimensionnons les vecteurs propres en images propres (np.reshape()) de manièrà #          pourvoir les visualiser sous forme d’images (array de taille 50x62x47). On utilisera la fonction plot_gallery() pour la visualisation.

def vecteur_propre_en_image_propre(pca):
    """
        vecteurs propres en images propres (50 x 62 x 47).
    """
    return pca.components_.reshape(-1, 62, 47)

if (sans_notebook) :
    pca50 = PCA(n_components=50) # on choisi d'utiliser siplement 50 composantes
    pca50.fit(X_train)
    X_train50_redi = vecteur_propre_en_image_propre(pca50)
    plotGallery(X_train50_redi, n=16, title="Images propres (50x62x47)")



# c-) Appliquons l’ACP sur les images de X_test. On appelera (X_test_comp) les nouvelles données. Cette étape constituera la compression des images.

X_test_comp = pca50.transform(X_test) 
if (sans_notebook) : 
    print("X_test_comp :", X_test_comp.shape)
    
# d-) Reconstruisons (ou décompressons) les images contenues dans X_test_comp pour obtenir les images X_test_reconst à partir d’une des méthodes de PCA() 
#     Afficons ensuitez les images reconstruiteen les comparantser visuellement aux images de départ.

X_test_reconst = pca50.inverse_transform(X_test_comp) # On utilise pca.inverse_transforme()

if (sans_notebook) : 
    plotGallery(X_test_reconst.reshape(-1, 62, 47), n=16, title="Images reconstruites (50x62x47)")


# e-) Comparons les images initiales et reconstruites de manière quantitative en faisant la moyenne des distances euclidiennes : 

E= (X_test_reconst -X_test)**2 
E = np.mean(np.sqrt(np.sum(E,axis=0)))

if (sans_notebook) :
    print("Moyenne des distances ecludiennes :", E)



# Determinons les tailles X_test et X_test_comp et deduisons-en le taux de compression
taille_X_test =  X_test.size
taille_X_test_comp = X_test_comp.size 
taux_de_compression = taille_X_test / taille_X_test_comp

# Observons la taille de X_test_reconst
taille_X_test_reconst = X_test_reconst.size 

if(sans_notebook) : 
    
    print("Taille de depart :", taille_X_test, "\nTaille compressées :",taille_X_test_comp, "\ntaux de compression",taux_de_compression)
    print("Taille des images reconstruites : ",taille_X_test_reconst)



# f-) On Fait varier le nombre de composantes conservées de 10 à 950 par pas de 50 puis on calcul l’erreur de reconstruction 
#     et on Affiche à la fin l’erreur de reconstruction en fonction du nombre de composantes.

# On fait varier le nombre de composantes de 10 à 950 par pas de 50
composantes = list(range(10, 951, 50))
erreurs = []

for n in composantes:
    pca_n = PCA(n_components=n)
    pca_n.fit(X_train)
    
    # Compression et reconstruction des images de test
    X_test_comp = pca_n.transform(X_test)
    X_test_reconst = pca_n.inverse_transform(X_test_comp)
    
    # Erreur de reconstruction
    E = (X_test_reconst - X_test)**2
    erreur = np.mean(np.sqrt(np.sum(E, axis=0)))  # moyenne des distances euclidiennes
    erreurs.append(erreur)

if (sans_notebook) : 
    plt.figure()
    plt.plot(composantes, erreurs, marker='o')
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Erreur de reconstruction")
    plt.title("Erreur de reconstruction en fonction du nombre de composantes")
    plt.show()



# Comparons visuellement les images initiales et reconstruites à partir de 950 composantes.
# Code pour repondre au 3 eme point de la question III.3
pca_950 = PCA(n_components=950)
pca_950.fit(X_train)

# Compression et reconstruction des images de test
X_test_comp_950 = pca_950.transform(X_test)
X_test_reconst_950 = pca_950.inverse_transform(X_test_comp_950)
if(sans_notebook) :
    plotGallery(X_test.reshape(-1,62,47), n=16, title="Images de depart")
    plotGallery(X_test_reconst_950.reshape(-1, 62, 47), n=16, title="Images reconstruites pour 950 composantes")



