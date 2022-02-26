
# Table des matières

-   [Introduction](#orgf04bc0c)
-   [Objectifs pédagogiques](#orgea72e5b)
-   [Cours](#org9fc7d77)
    -   [Semaine 1 : Introduction et notions fondamentales](#org2564cfb)
-   [Évaluation du cours](#org56c8aaa)
-   [Ressources pédagogiques](#orgfbabbcd)
    -   [Références](#org3675afc)
    -   [L'environnement `Python`](#orge3c3a74)
    -   [Principaux modules `Python` utilisés](#orgffcd2e5)



<a id="orgf04bc0c"></a>

# Introduction

Ce site présente l'ensemble des ressources pédagogiques relatives au cours de classification non
supervisée pour la promotion de STID2 à l'IUT de Vannes.


<a id="orgea72e5b"></a>

# Objectifs pédagogiques

Les principaux objectifs pédagogiques du cours sont :

-   Comprendre la problématique de classification non supervisée.
-   Replacer cette problématique dans le contexte méthodologique de l'analyse de données.
-   Introduire la notion d'inertie et son utilisation.
-   Présenter différentes approches de classification non supervisée pour l'analyse des données
    quantitatives, à savoir :
    -   deux méthodes de partitionnement (moyennes mobiles et mélange gaussien) ;
    -   la classification ascendante hiérarchique.
-   Savoir mettre en oeuvre ces méthodes avec la langage `Python`.


<a id="org9fc7d77"></a>

# Cours

Les supports de cours sont disponibles en ligne sous forme de présentation HTML (une connexion internet est
donc requise).

**Note 1 : Il est recommandé d'utiliser le navigateur `Firefox` pour visualiser les slides de cours.**

**Note 2 : Pour reproduire les exemples du cours, n'oubliez pas de télécharger le module
[`clust_util.py`](https://github.com/roland-donat/cours-class-non-sup/tree/main/python) en le plaçant dans le même répertoire que les scripts.**


<a id="org2564cfb"></a>

## Semaine 1 : Introduction et notions fondamentales

**Cours :**

-   [Slides du cours HTML](https://roland-donat.github.io/cours-class-non-sup/cours/c1/c1.html)
-   [Scripts `Python` du cours](https://github.com/roland-donat/cours-class-non-sup/tree/main/cours/c1/)

**TD :**

-   [Sujet du TD](https://roland-donat.github.io/cours-class-non-sup/td/td1/td1.html)


<a id="org56c8aaa"></a>

# TODO Évaluation du cours

L'objectif est d'évaluer votre compréhension des notions et méthodes abordées en cours et
en TD. Pour ce faire, nous vérifierons votre aptitude à mettre en oeuvre les traitements
informatiques adéquats face à une problématique de classification non supervisée. Les évaluations se
présenteront sous la forme de quiz sur la plateforme Moodle. 

Planning des évaluations :

**Note :** 
Le petit quiz suivant vous aidera afin de préparer votre environnement de programmation pour
les devoirs : <https://moodle.univ-ubs.fr/mod/quiz/view.php?id=271762>


<a id="orgfbabbcd"></a>

# Ressources pédagogiques


<a id="org3675afc"></a>

## Références

L'élaboration de ce cours s'est appuyée sur de nombreuses références. Voici mes principales sources
d'inspiration :

-   [Cours de classification non supervisée](http://www2.agroparistech.fr/IMG/pdf/ClassificationNonSupervisee-AgroParisTech.pdf) de E. Lebarbier, T. Mary-Huard (Agro ParisTech).
-   [Cours de classification non supervisée](https://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-m-explo-classif.pdf) de Philippe Besse (INSA de Toulouse).
-   [Cours de classification automatique de données quantitatives](http://www.math.u-bordeaux.fr/~mchave100p/wordpress/wp-content/uploads/2013/10/cours_classif_quanti.pdf) de Marie Chavent (Université de Bordeaux).
-   [Série de vidéos sur la classification non supervisée](https://www.youtube.com/watch?v=SE_4dLh5vXY) de François Husson (Agrocampus Rennes).
-   [Plateforme `EduClust`](https://educlust.dbvis.de) de l'Universität Konstanz qui permet de expérimenter certains algorithmes de
    *clustering* et visualiser les résultats.
-   [Précédent cours de *Clustering* de l'IUT de Vannes](https://moodle.univ-ubs.fr/course/view.php?id=3596) de Arlette ANTONI (IUT de Vannes) qui avait en
    charge le cours de classification non supervisée sur l'année 2019-2020.


<a id="orge3c3a74"></a>

## L'environnement `Python`

Les exemples du cours et les travaux dirigés utilisent le logiciel `Python`. Si vous pensez ne pas
être à l'aise avec `Python`, je vous encourage vivement à faire une petite mise à niveau.

Il existe énormément de très bons cours/didacticiels sur internet qui traitent de l'apprentissage de ce
langage en statistiques. Voici quelques liens à titre indicatif :

-   [Python 3 : des fondamentaux aux concepts avancés du langage](https://www.fun-mooc.fr/fr/cours/python-3-des-fondamentaux-aux-concepts-avances-du-langage/)
-   [Apprendre à coder avec Python](https://www.fun-mooc.fr/fr/cours/apprendre-a-coder-avec-python/)
-   [Introduction to Data Science in Python](https://www.coursera.org/learn/python-data-analysis?specialization=data-science-python)

Par ailleurs, si vous disposez d'une connexion internet fiable, je vous recommande d'utiliser une
plateforme de *Notebooks* en ligne telle que [`DeepNote`](https://deepnote.com). L'inscription est gratuite et évite de
devoir gérer l'installation de l'environnement `Python` sur votre poste. 
Sinon vous pouvez bien évidement utiliser `Python` en local en installant la distribution
[`Anaconda`](https://www.anaconda.com/products/individual). 

**L'important est de vous assurer que vous utilisez bien la version `3.8` ou supérieure de `Python`.**


<a id="orgffcd2e5"></a>

## Principaux modules `Python` utilisés

Dans ce cours, nous utiliserons principalement les modules `Python` suivants :

-   `pandas`, pour la manipulation des données ;
-   `plotly`, pour les représentations graphiques ;
-   `numpy`, pour utiliser des fonctions de calculs numériques "bas niveau", e.g. génération de
    nombres aléatoires ;
-   `scipy`, pour utiliser d'autres fonctions de calculs numériques plus "haut niveau", e.g. calcul de
    distances ;
-   `sklearn`, pour avoir accès aux algorithmes de classification.

Ces modules ne seront pas forcément installés par défaut dans votre environnement logiciel. Si vous
utilisez un *notebook* (local ou distant de type `Deepnote`), vous pouvez utiliser la commande `!pip
install <nom_module>` pour les installer : 

    !pip install pandas
    !pip install plotly
    !pip install scipy
    !pip install sklearn

Vous pourrez ainsi les importer dans vos scripts de la manière suivante :

    import pandas as pd
    import numpy as np
    import plotly.express as px
    from scipy.spatial.distance import pdist, cdist, squareform

