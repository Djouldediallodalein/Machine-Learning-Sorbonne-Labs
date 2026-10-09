

"""
Created on Thu Mar 27 10:37:24 2025

@author: cathe
"""


sans_notebook = 1 # 1 si vous n'avez pas le notebook sinon 0

"""

ETUDIANT :                DIALLO Mamadou Djouldé
PARCOURS :                SysCom

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV


def visualize_classifier(model, X, y):
    ax = plt.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=1, cmap='rainbow',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Create a color plot with the results

    n_classes = len(np.unique(y))
    plt.scatter(xx.ravel(), yy.ravel(), c=Z, s=0.1, cmap='rainbow');
    ax.set(xlim=xlim, ylim=ylim)
    plt.show()

# ##################################################################################################
# I. Chargement et visualisation

data = np.load("TP4.npz")
X_train, y_train, X_test, y_test = (data[key] for key in ["X_train", "y_train", "X_test", "y_test"])
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=1, cmap='rainbow');
plt.show()


# Determinons combien y'a-t-il de points dans la base d'apprentissage, de test en determinant la dimension des données
if sans_notebook :
    print("\nLe nombre de points dans la base d'apprentissage est :", X_train.shape[0])
    print("\nLe nombre de points dans la base d'apprentissage est :", X_test.shape[0])
    print("\nDimension des données :", "\nteste :", X_test.shape[1], "\nApprentissage :",X_train.shape[1] )


# ####################################################################################################
# II. Arbre de décision

# II.a Principe des arbres de décision
tree1 = DecisionTreeClassifier(criterion='entropy', max_depth = 3, random_state= 52) # Il faut importer DecisionTreeClassifier depuis sklearn.tree 
tree1.fit(X_train, y_train) 
visualize_classifier(tree1, X_train, y_train) 
tree.plot_tree(tree1)                             # Pour utiliser plot_tree il faut importer tree dans sklearn
plt.show() 
text_representation = tree.export_text(tree1) 
print(text_representation) 


# II.b Performance d’un classifieur muti-classes 
y_pred = tree1.predict(X_test)                      # test sur tree1 
C = confusion_matrix(y_test, y_pred) 
print(classification_report(y_test, y_pred)) 
print('Accuracy=',accuracy_score(y_test, y_pred)) 

# Retrouvons par le calcul la première ligne (classe 1) renvoyée par classification_report à partir de la matrice de confusion.

# On va verifier si la matrice n'est pas vide
if C.size:
    
    # Pour la première classe (index 0)
    TP = C[0, 0]  

    # Calcul de FN
    FN = 0
    for j in range(len(C[0])): 
        if j != 0:
            FN += C[0, j]

    # Calcul de FP
    FP = 0
    for i in range(len(C)):
        if i != 0:
            FP += C[i, 0]

    # Application des forumules vue en cours pour le calcul de la precision et du rappel 
    precision_classe1 = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rappel_classe1 = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    if sans_notebook : 
        # Resultat
        print("Pour la première classe (index 0) :")
        print(f"   TP={TP}, FP={FP}, FN={FN}")
        print(f"   precision ≈ {precision_classe1:.2f}, recall ≈ {rappel_classe1:.2f}")


        
# II.c Optimisation de la profondeur de l’arbre 

def test_profondeur(X_train, y_train, X_test, y_test, profondeurs, criterion='entropy', random_state=52, affichage=False, resultats_ou_precimax = 1):
   
    resultats = {}
    liste_des_profondeurs = []
    liste_des_taux_de_precision = []
    
    for d in range(1,profondeurs+1):
        arbre = DecisionTreeClassifier(criterion=criterion, max_depth=d, random_state=random_state)
        arbre.fit(X_train, y_train)
        y_pred = arbre.predict(X_test)
        C = confusion_matrix(y_test, y_pred)

        acc = accuracy_score(y_test, y_pred)
        resultats[d] = acc

        liste_des_taux_de_precision.append(acc)

        liste_des_profondeurs.append(d)
        
        if resultats_ou_precimax == 1 :
            print(classification_report(y_test, y_pred))
            print(f"\n--- Profondeur = {d} ---")
            print(f"Accuracy : {acc:.3f}")
        else : 
            liste = liste_des_taux_de_precision
        
        if affichage:  # Si on veut afficher les arbres
            tree.plot_tree(arbre)
            plt.title(f"Arbre de décision (max_depth={d})")
            plt.show()
            
    if resultats_ou_precimax == 1 :
        # Affichage du taux d'evolution du taux de precision
        plt.plot(liste_des_profondeurs, liste_des_taux_de_precision, marker='o')
        plt.xlabel("Profondeurs (max_depth)")
        plt.ylabel("Taux de reconnaissance)")
        plt.title("Taux de reconnaissance en fonction de la profondeur (max_depth)")
        plt.show()
        return resultats
        
    elif resultats_ou_precimax == 0: 
        precision_maximale = np.max(liste_des_taux_de_precision) 
        return precision_maximale
    

# trouvons le meilleur arbre pour afficher l'arbre avec le meilleur taux de classification 
def meilleur_arbre(profondeurs, precision_maximale, random_state=52) :
    meilleur_arbre = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=random_state)
    for i in range(1,profondeurs +1) :
        arbre = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=random_state)
        arbre.fit(X_train, y_train)
        y_pred = arbre.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        if acc >= precision_maximale : 
            #meilleure_acc = acc
            meilleure_profondeur = i 
            meilleur_arbre = arbre
            print("Le meilleur arbre a une profondeur de ",meilleure_profondeur)


    
    return meilleur_arbre


if sans_notebook : 
    
    profondeurs = 7      # On peut modifier la profondeur selon le nombre qu'on veut, par defaut je la laissek à 7
    scores = test_profondeur(X_train, y_train, X_test, y_test, profondeurs, affichage =0, resultats_ou_precimax = 1)
    precision_maximale = test_profondeur(X_train, y_train, X_test, y_test, profondeurs, affichage =0,resultats_ou_precimax = 0)
    print("\nLa precision maximale est : ",precision_maximale) 

    # Visualisation du partitionnement de l'espace optenu avec l'arbre avec le meilleur taux de classification
    meilleur_optimale = meilleur_arbre(profondeurs, precision_maximale, random_state=52)
    visualize_classifier(meilleur_optimale, X_train, y_train)


# II.d Arbre de décision sur des données de grande dimension 

# Chargeons maintenant les données TP4b.npz et utilisons la question précédente pour optimiser la profondeur de l’arbre.  

data = np.load("TP4b.npz") 
X_train, y_train, X_test, y_test = (data[key] for key in ["X_train", "y_train", "X_test", "y_test"]) 
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=1, cmap='rainbow'); 
plt.show() 

if sans_notebook : 
    print("\nLe nombre de points dans la base d'apprentissage est :", X_train.shape[0])
    print("\nLe nombre de points dans la base d'apprentissage est :", X_test.shape[0])
    print("\nDimension des données :", "\nteste :", X_test.shape[1], "\nApprentissage :",X_train.shape[1] )
    precision_maximale1 = test_profondeur(X_train, y_train, X_test, y_test, profondeurs, affichage =0,resultats_ou_precimax = 0)
    print("\nLa precision maximale est : ",precision_maximale1)



# ################################################################################################################################
# III. Forêt d’arbres aléatoires 

# a. Test d’une forêt d’arbres aléatoires 
RF = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=52) 
RF.fit(X_train, y_train) 

if sans_notebook : 
    print(classification_report(y_test, RF.predict(X_test)))
    print('Accuracy =', accuracy_score(y_test, RF.predict(X_test)))


# b. Influence des paramètres 
liste_de_nombre_arbre = [1, 5, 10, 20, 50, 100]
taux_de_bonne_classification = []

for i in liste_de_nombre_arbre:
    RF = RandomForestClassifier(criterion='entropy', n_estimators=i, random_state=52)
    RF.fit(X_train, y_train)
    acc = accuracy_score(y_test, RF.predict(X_test))
    taux_de_bonne_classification.append(acc)

    if sans_notebook :
        print(f"liste_de_nombre_arbre={i}, taux_de_bonne_classification={acc:.3f}")        

if sans_notebook :
    plt.plot(liste_de_nombre_arbre, taux_de_bonne_classification, marker='o')
    plt.xlabel("Nombre d'arbres")
    plt.ylabel("taux_de_bonne_classification")
    plt.title("Influence du nombre d'arbres sur le taux de bonne classification")
    plt.show()

#  En conservant le nombre d’arbres optimal, observons l’évolution du taux de bonne classification en fonction de la profondeur de l’arbre.
profondeurs = [1, 2, 3, 5, 7, 10, 15, 20,30,40,50,60,70,80,90,100]
n_optimal = 100
taux_par_profondeur = []

for p in profondeurs:
    RF = RandomForestClassifier(criterion='entropy', n_estimators=n_optimal, max_depth=p, random_state=52)
    
    RF.fit(X_train, y_train)
    acc = accuracy_score(y_test, RF.predict(X_test))
    taux_par_profondeur.append(acc)

    if sans_notebook :
        
        print(f"Profondeur={p}, Taux de bonne classification={acc:.3f}")

if sans_notebook :
    plt.plot(profondeurs, taux_par_profondeur, marker='o')
    plt.xlabel("Profondeur maximale des arbres (max_depth)")
    plt.ylabel("Taux de bonne classification")
    plt.title(f"Influence de la profondeur (n_estimators={n_optimal})")
    plt.show()
    

# En conservant le nombre d’arbres et la profondeur, observons l’évolution du taux de bonne classification en fonction du nombre de caractéristique utilisées 
# lors du bagging.
features = [1, 2, 3, 4, 5,6,7,8,9,10,15,17,25,30,50, X_train.shape[1]]
n_optimal = 100
p_optimal = 7
taux_par_feature = []

for f in features:
    RF = RandomForestClassifier(criterion='entropy', n_estimators=n_optimal, max_depth=p_optimal, max_features=f, random_state=52)
    RF.fit(X_train, y_train)
    acc = accuracy_score(y_test, RF.predict(X_test))
    taux_par_feature.append(acc)

    print(f"max_features={f}, Taux de bonne classification={acc:.3f}")


plt.plot(features, taux_par_feature, marker='o')
plt.xlabel("Nombre de caractéristiques (max_features)")
plt.ylabel("Taux de bonne classification")
plt.title(f"Influence du nombre de caractéristiques (n_estimators={n_optimal}, max_depth={p_optimal})")
plt.show()



# c. Choix des paramètres optimaux

# Définition de la grille de paramètres autour des valeurs trouvées précédemment
grille_parametres = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 7, 10],
    'max_features': [20, 25, 30]
}

foret = RandomForestClassifier(criterion='entropy', random_state=52)

# GridSearchCV 5-fold
recherche_grille = GridSearchCV(estimator=foret, param_grid=grille_parametres, scoring='accuracy', cv=5, n_jobs=-1)
recherche_grille.fit(X_train, y_train)

# Récupération des meilleurs paramètres et du score moyen
meilleurs_parametres = recherche_grille.best_params_
score_moyen_cv = recherche_grille.best_score_

if sans_notebook :
    print("Meilleurs paramètres trouvés :", meilleurs_parametres)
    print("Taux de reconnaissance moyen :", score_moyen_cv)

# Évaluation
meilleure_foret = recherche_grille.best_estimator_
predictions_test = meilleure_foret.predict(X_test)
taux_reconnaissance_test = accuracy_score(y_test, predictions_test)

if sans_notebook :
    print("Taux de reconnaissance sur le jeu de test :", taux_reconnaissance_test)
