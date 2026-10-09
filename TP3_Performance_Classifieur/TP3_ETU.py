
import numpy as np
import matplotlib.pyplot as plt
# from fn import norm1, norm2
import os
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal, norm
from PIL import Image

sans_notebook = 1 # A garder à 1 si vous n'avez pas de notebook pour visualiser les graphes et lire les resultats.

DATA_PATH = 'DATA/'

image_ext = ['.jpg']
images =  [os.path.join(DATA_PATH, f) for f in sorted(os.listdir(DATA_PATH)) if f.endswith(".jpg")]

X_train=np.empty((0, 2))
y_train=[]
for name in images[0:-4]:
    I= Image.open(name)
    I = I.convert('YCbCr')
    I = np.array(I)
    I=np.reshape(I[:,:,1:3],[I.shape[0]*I.shape[1],2])
    X_train = np.concatenate((X_train, I), axis=0)
    fichier_png= os.path.splitext(name)[0] + ".png"
    GT = Image.open(fichier_png)
    GT = np.array(GT)
    GT=np.reshape(GT[:,:,1]/255,[GT.shape[0]*GT.shape[1]])
    y_train = np.concatenate((y_train, GT), axis=0)


X_test=np.empty((0, 2))
y_test=[]
for name in images[-4:]:
    I= Image.open(name)
    I = I.convert('YCbCr')
    I = np.array(I)
    I=np.reshape(I[:,:,1:3],[I.shape[0]*I.shape[1],2])
    X_test = np.concatenate((X_test, I), axis=0)
    fichier_png= os.path.splitext(name)[0] + ".png"
    GT = Image.open(fichier_png)
    GT = np.array(GT)
    GT=np.reshape(GT[:,:,1]/255,[GT.shape[0]*GT.shape[1]])
    y_test = np.concatenate((y_test, GT), axis=0)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

X_train, pipo, y_train, pipo = train_test_split(X_train, y_train, train_size=1/1000, random_state=52)
X_test, pipo, y_test, pipo = train_test_split(X_test, y_test, train_size=1/1000, random_state=52)

# #############################################################################
# I.	Chargement et visualisation des données


#Pixel peau
Peau_Train = X_train[np.where(y_train==1),:]
Peau_Train = np.reshape(Peau_Train,(Peau_Train.shape[1],Peau_Train.shape[2] ))
#Pixel non peau
Nonpeau_Train = X_train[np.where(y_train==0),:]
Nonpeau_Train = np.reshape(Nonpeau_Train,(Nonpeau_Train.shape[1],Nonpeau_Train.shape[2] ))


plt.plot(Nonpeau_Train[:,0], Nonpeau_Train[:,1], '.b', label='Non peau')
plt.plot(Peau_Train[:,0], Peau_Train[:,1], '.r', label='Peau')
plt.show()


# Determinons comment sont constituées les bases d’apprentissage et de test ?
if (sans_notebook) : 
    print("Dimension de la base d'apprentissage pour 1/1000 de tous les pixels : ",X_train.shape)
    
    # Affichage du nombre d'exemple par classe 
    print("\nNombre d'exemple de la classe peau : ", Peau_Train.shape[0])
    print("\n Nombre d'exemple de la classe non peau",Nonpeau_Train.shape[0])

# ###############################################################################
# II. Modélisation de la vraisemblance des observations par une loi normale 2D avec des dimensions décorrélées

# a. Estimation de la vraisemblance des observations des pixels de teinte chaire


"""Faute d'une mauvaise lecture du sujet j'avais trouvé une autre methode pour calculer la vraisemblance avant de voir la methode imposée,
    Néanmoins j'ai aussi appliqué la méthode imposée par le sujet qui aboutit au même résultat."""

#---------------------------------------------------------------------------------------------------
# Moyennes et écarts-types des composantes Cb et Cr pour les pixels peau
def calcul_moyenn_ecartype(Peau_Train):
    # On retorne les moyennes et écarts types des composantes Cb et Cr 

    mCb, mCr = np.mean(Peau_Train, axis=0) 
    ecart_typeCb, ecart_typeCr = np.std(Peau_Train, axis=0)
    
    return mCb, mCr,ecart_typeCb, ecart_typeCr
    

# Fonction pour Calculer la vraisemblance 
def vraisemblance(X, mCb, mCr, ecart_typeCb, ecart_typeCr):
    """
    X : tableau de forme (N,2) contenant N pixels
    On retourne p(x/chair)
    """
    Cb = X[:,0]
    Cr = X[:,1]
    constante = 1 / (2 * np.pi * ecart_typeCb * ecart_typeCr)
    exposant = - ( (Cb - mCb)**2 / (2 * ecart_typeCb**2) + (Cr - mCr)**2 / (2 * ecart_typeCr**2) )
    return constante * np.exp(exposant)
#----------------------------------------------------------------------------------------------------


def methode_du_sujet(Peau_Train):
    """
    Implémente la méthode imposée par le sujet :
    Estimation de la vraisemblance p(x/chair) sur la base d'apprentissage.
    """
    # Moyennes et écarts-types
    mCb, mCr = np.mean(Peau_Train, axis=0)
    ecart_typeCb, ecart_typeCr = np.std(Peau_Train, axis=0)

    # Lois normales indépendantes pour Cb et Cr
    p1 = norm(mCb, ecart_typeCb)
    p2 = norm(mCr, ecart_typeCr)

    # Calcul de la vraisemblance p(x/chair)
    p_train = p1.pdf(Peau_Train[:, 0]) * p2.pdf(Peau_Train[:, 1])

    return p_train, mCb, mCr, ecart_typeCb, ecart_typeCr




if (sans_notebook): 
    
    methode_sujet = 1
    methode_djoulde = 0
    
    if(methode_sujet) :
        
        # Affichage du resultat des moyennes et écarts types des composantes Cb et Cr 
        p_train, mCb, mCr, ecart_typeCb, ecart_typeCr = methode_du_sujet(Peau_Train)
        
        # Affichage de la vraisemblance    
        print("Moyenne Cb :", mCb)
        print("Moyenne Cr :", mCr)
        print("Écart-type Cb :", ecart_typeCb)
        print("Écart-type Cr :", ecart_typeCr)
        print("Vraisemblance des 5 premiers pixels :", p_train[:5])
        print('Affichage des dimensions : \nmCb :',mCb.shape, '\nmCr :', mCr.shape,'\necart_typeCb :',ecart_typeCb.shape, '\necart_typeCr :', ecart_typeCr.shape, '\nP(𝒙/𝑐ℎ𝑎𝑖𝑟) :', p_train.shape, '\nP_train :', p_train.shape)
#--------------------------------------------------------------------------------------------------------   
    if (methode_djoulde) : 
        # Affichage du resultat des moyennes et écarts types des composantes Cb et Cr 
        mCb, mCr,ecart_typeCb, ecart_typeCr = calcul_moyenn_ecartype(Peau_Train)
        print("Moyenne Cb (mCb) :", mCb)
        print("Moyenne Cr (mCr) :", mCr)
        print("Écart-type Cb (σCb) :", ecart_typeCb)
        print("Écart-type Cr (σCr) :", ecart_typeCr)
    
        # Affichage de la vraisemblance
        p_train = vraisemblance(Peau_Train, mCb, mCr, ecart_typeCb, ecart_typeCr)
        print("vraisemblance des observations des 5 premiers pixels de teinte chaire 𝑝(𝒙/𝑐ℎ𝑎𝑖𝑟) :", p_train[:5])
        print('Affichage des dimensions : \nmCb :',mCb.shape, '\nmCr :', mCr.shape,'\necart_typeCb :',ecart_typeCb.shape, '\necart_typeCr :', ecart_typeCr.shape, '\nP(𝒙/𝑐ℎ𝑎𝑖𝑟) :', p_train.shape, '\nP_train :', p_train.shape)
#---------------------------------------------------------------------------------------------------------    
    

# b. Mise en place du classifieur :


# Determinons le seuil qui correspond à la moyenne de p_train
seuil = np.mean(p_train)
p_tous_pixels = methode_du_sujet(X_train)[0]  # fonction methode_du_sujet


def classifieur(y_train, p_tous_pixels, seuil) :
    TP = TN = FP = FN = 0
    
    # Boucle sur chaque pixel
    for i in range(len(y_train)):
        vrai = y_train[i]
        p_nouveau = p_tous_pixels[i]
        
        # Classification selon le seuil
        if p_nouveau >= seuil:
            prediction = 1  # peau
        else:
            prediction = 0  # non-peau
        
        if vrai == 1 and prediction == 1:
            TP += 1
        elif vrai == 0 and prediction == 0:
            TN += 1
        elif vrai == 0 and prediction == 1:
            FP += 1
        elif vrai == 1 and prediction == 0:
            FN += 1
    
    # Affichage des résultats
    print("TP =", TP, "TN =", TN, "FP =", FP, "FN =", FN)
    
    sensibilite = TP / (TP + FN)
    specificite = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    print("Sensibilité :", sensibilite)
    print("Spécificité :", specificite)
    print("Taux de bonne reconnaissance :", accuracy)



if (sans_notebook) : 
    print("\nValeur de seuil qu'on va utiliser :", seuil)
    resultats = classifieur(y_train, p_tous_pixels, seuil)

# c. Courbe de ROC : 

# Determination des seuils 
NB = 20 
SEUILS = np.linspace(np.min(p_train), np.max(p_train), NB)


def courbe_ROC(y_train, p_tous_pixels, SEUILS):
    rappels = []
    faux_pos = []  # Qui correspond au Taux de faux positifs (1 - spécificité) pour chaque seuil

    for seuil in SEUILS:
        TP = TN = FP = FN = 0
        for i in range(len(y_train)):
            vrai = y_train[i]
            p_nouveau = p_tous_pixels[i]
            prediction = 1 if p_nouveau >= seuil else 0

            if vrai == 1 and prediction == 1:
                TP += 1
            elif vrai == 0 and prediction == 0:
                TN += 1
            elif vrai == 0 and prediction == 1:
                FP += 1
            elif vrai == 1 and prediction == 0:
                FN += 1

        # Calcul du rappel
        if (TP + FN) != 0 :                  
            rappel = TP / (TP + FN)
        else :
            rappel = 0

        # Calcul du taux de faux positifs (1 - spécificité)
        if (FP + TN) != 0 : 
            taux_fp = FP / (FP + TN)
        else : 
            taux_fp = 0

        
        rappels.append(rappel)
        faux_pos.append(taux_fp)

    # Tracé de la courbe de ROC
    plt.figure()
    plt.plot(faux_pos, rappels, marker='o', color='red')
    plt.xlabel('Taux de faux positifs (1 - spécificité)')
    plt.ylabel('Rappel (sensibilité)')
    plt.title('Courbe ROC')
    plt.grid(True)
    plt.show()

    return faux_pos, rappels


faux_pos, rappels = courbe_ROC(y_train, p_tous_pixels, SEUILS)

# Calcul de la spécificité
specificites = 1 - np.array(faux_pos)

# Chercher le seuil où rappel ≈ spécificité
seuil_equilibre = SEUILS[0]
plus_petite_diff = abs(rappels[0] - specificites[0])

for i in range(1, len(SEUILS)):
    diff = abs(rappels[i] - specificites[i])
    if diff < plus_petite_diff:
        plus_petite_diff = diff
        seuil_equilibre = SEUILS[i]

if (sans_notebook) :
    print("\nLes seuils que nous allons utiliser pour la suite sont :\n ", SEUILS)
    #faux_pos, rappels = courbe_ROC(y_train, p_tous_pixels, SEUILS) 
    print("Seuil choisi pour avoir rappel ≈ spécificité :", seuil_equilibre)

# Appel de la fonction classifieur pour calculer le taux de bonne classification 
    classifieur(y_train, p_tous_pixels, seuil_equilibre)




# d. Classification des pixels de test 

p_test = methode_du_sujet(X_test)[0]  # vraisemblances des pixels de test
if (sans_notebook) :

# Appel de la fonction classifieur pour calculer le taux de bonne classification des pixels de test
    classifieur(y_test, p_test, seuil_equilibre)



# ########################################################################################

# III. Modélisation de la vraisemblance des observations par une loi normale 2D 

def Loi_normale2D(Peau_Train, X):
    """
    Implémente la modélisation de la vraisemblance p(x/chair)
    avec une loi normale 2D complète (corrélation entre Cb et Cr prise en compte).
    """
    # Moyenne et matrice de covariance
    m = np.mean(Peau_Train, axis=0)
    cov = np.cov(Peau_Train.T)

    # Loi normale 2D
    p = multivariate_normal(mean=m, cov=cov)

    # Calcul de la vraisemblance sur X (ex: X_train ou X_test)
    p_train = p.pdf(X)

    return p_train, m, cov

# Appel de la méthode 2D complète
p_train_2D, m, cov = Loi_normale2D(Peau_Train, X_train)

if (sans_notebook):
    print("Moyenne :\n", m)
    print("Matrice de covariance :\n", cov)
    print("Vraisemblance des 5 premiers pixels :", p_train_2D[:5])

# Seuil : moyenne des vraisemblances sur la base d'apprentissage
seuil_2D = np.mean(p_train_2D)

if sans_notebook:
    print("\nSeuil pour la loi normale 2D complète :", seuil_2D)


# Classification sur la base d'apprentissage
if sans_notebook:
    print("\nRésultats sur la base d'apprentissage (2D complète) :")
    classifieur(y_train, p_train_2D, seuil_2D)

# Détermination des seuils
SEUILS_2D = np.linspace(np.min(p_train_2D), np.max(p_train_2D), NB)

# Tracé de la courbe ROC
faux_pos_2D, rappels_2D = courbe_ROC(y_train, p_train_2D, SEUILS_2D)


specificites_2D = 1 - np.array(faux_pos_2D)

seuil_equilibre_2D = SEUILS_2D[0]
plus_petite_diff = abs(rappels_2D[0] - specificites_2D[0])

for i in range(1, len(SEUILS_2D)):
    diff = abs(rappels_2D[i] - specificites_2D[i])
    if diff < plus_petite_diff:
        plus_petite_diff = diff
        seuil_equilibre_2D = SEUILS_2D[i]

if sans_notebook:
    print("\nSeuil choisi pour rappel ≈ spécificité (2D complète) :", seuil_equilibre_2D)

# Vraisemblances des pixels de test
p_test_2D, _, _ = Loi_normale2D(Peau_Train, X_test)

if sans_notebook:
    print("\nRésultats sur la base de test (2D complète) :")
    classifieur(y_test, p_test_2D, seuil_equilibre_2D)


# ##########################################################################################

# IV. Test sur une nouvelle image : 

# Chargeons l'image
img = Image.open('image.jpg').convert('YCbCr')
img_array = np.array(img)
X_img = img_array[:, :, 1:3].reshape(-1, 2).astype('float64')

# Calculons les vraisemblances avec la loi normale 2D complète
p_img, _, _ = Loi_normale2D(Peau_Train, X_img)

# Classifions les pixels selon le seuil choisi (seuil_equilibre_2D)
mask = (p_img >= seuil_equilibre_2D).astype(np.uint8)
# Remettons sous forme image
mask_img = mask.reshape(img_array.shape[0], img_array.shape[1])


if sans_notebook :
        
    # Affichage du résultat
    plt.figure(figsize=(8, 6))
    plt.imshow(mask_img, cmap='gray')
    plt.title("Détection des pixels de teinte chair")
    plt.axis('off')
    plt.show()
