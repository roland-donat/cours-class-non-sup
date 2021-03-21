
# Table des matières

-   [Introduction](#orgb8a7a4d)
-   [Objectifs Pédagogiques](#orga1cacc7)
-   [Crédit](#org5014f10)
-   [Organisation pratique](#orgc373c09)
-   [Cours](#orgb1724b5)
    -   [Semaine 1 : Introduction et notions fondamentales](#orgd60fb26)
    -   [Semaine 2 : Méthodes de partitionnement](#org1074b3c)
    -   [Semaine 3 : Classification hiérarchique](#org16d8fe5)
-   [Évaluation du cours](#org8c62333)
-   [Références](#org1900ba0)
    -   [L'environnement `Python`](#org5e727bc)
    -   [Principaux modules `Python` utilisés](#orga48856e)



<a id="orgb8a7a4d"></a>

# Introduction

Ce site présente l'ensemble des ressources pédagogiques relatives au cours de classification non
supervisée pour la promotion de STID2 à l'IUT de Vannes.


<a id="orga1cacc7"></a>

# Objectifs Pédagogiques

Les principaux objectifs pédagogiques du cours sont :

-   Comprendre la problématique de classification non supervisée.
-   Replacer cette problématique dans le contexte méthodologique de l'analyse de données.
-   Introduire la notion d'inertie et son utilisation.
-   Présenter différentes approches de classification non supervisée pour l'analyse des données
    quantitatives, à savoir :
    -   deux méthodes de partitionnement (moyennes mobiles et mélange gaussien) ;
    -   la classification ascendante hiérarchique.
-   Savoir mettre en oeuvre ces méthodes avec la langage `Python`.


<a id="org5014f10"></a>

# Crédit

Les cours et TD proposés s'inspirent largement des supports pédagogiques élaborées par Mme. Arlette
ANTONI qui avait en charge le cours de classification non supervisée sur l'année 2019-2020. Vous
pouvez consulter ces ressources pédagogiques en vous rendant dans [l'espace *Clustering* sur Moodle](https://moodle.univ-ubs.fr/course/view.php?id=3596).


<a id="orgc373c09"></a>

# Organisation pratique

Compte tenu du contexte sanitaire actuel, les cours et TD sont prévus pour se tenir en 100%
distanciel. Mais nous ne sommes pas à l'abri de changements de dernières minutes, restez donc sur
vos gardes&#x2026;

Pour suivre les cours, les TD et avoir de passionnants débats sur la classification non supervisée, merci de
faire une demande d'inscription à l'équipe Teams [STID2 - Classification non
supervisée](https://teams.microsoft.com/l/team/19%3a541fb9397ced490aab1776de0de9202f%40thread.tacv2/conversations?groupId=775ce021-bec5-4bc8-9892-4854cd178be3&tenantId=2fbd12a9-cbb9-49a2-9612-7af4096a6529).


<a id="orgb1724b5"></a>

# Cours

Les supports de cours sont disponibles en ligne sous forme de présentation HTML (une connexion internet est
donc requise).

**Note 1 : Il est recommandé d'utiliser le navigateur `Firefox` pour visualiser les slides de cours.**

**Note 2 : Pour reproduire les exemples du cours, n'oubliez pas de télécharger le module
[`clust_util.py`](https://github.com/roland-donat/cours-class-non-sup/tree/main/python) en le plaçant dans le même répertoire que les scripts.**


<a id="orgd60fb26"></a>

## Semaine 1 : Introduction et notions fondamentales

**Cours :**

-   [Slides du cours HTML](https://roland-donat.github.io/cours-class-non-sup/cours/C1%20-%20Introduction%20g%C3%A9n%C3%A9rale/c1_intro.html)
-   [Scripts `Python` du cours](https://github.com/roland-donat/cours-class-non-sup/tree/main/cours/C1%20-%20Introduction%20g%C3%A9n%C3%A9rale)

**TD :**

-   [Sujet du TD](https://roland-donat.github.io/cours-class-non-sup/td/td1/td1.html)
-   [Notebook de la séance](https://colab.research.google.com/drive/1OxeIT0Kj5nimaCdbpbwHqgiEuXo-HQvn?usp=sharing) (sur Google Colab)


<a id="org1074b3c"></a>

## Semaine 2 : Méthodes de partitionnement

**Cours :**

-   [Slides du cours HTML](https://roland-donat.github.io/cours-class-non-sup/cours/C2%20-%20Méthodes%20de%20partitionnement/c2_meth_part.html)
-   [Scripts `Python` du cours](https://github.com/roland-donat/cours-class-non-sup/tree/main/cours/C2%20-%20Méthodes%20de%20partitionnement)

**TD :**

-   [Sujet du TD](https://roland-donat.github.io/cours-class-non-sup/td/td2/td2.html)
-   [Notebook de la séance](https://colab.research.google.com/drive/1hbkkCzxpxK4I68eKdGmj5N2by3gYQjZS?usp=sharing) (sur Google Colab)


<a id="org16d8fe5"></a>

## Semaine 3 : Classification hiérarchique

**Cours :**

-   [Slides du cours HTML](https://roland-donat.github.io/cours-class-non-sup/cours/c3/c3.html)
-   [Scripts `Python` du cours](https://github.com/roland-donat/cours-class-non-sup/tree/main/cours/c3/)

**TD :**

-   [Sujet du TD](https://roland-donat.github.io/cours-class-non-sup/td/td3/td3.html)
-   [Notebook de la séance](https://colab.research.google.com/drive/1eqrltVAsk2_v9Lcvy4zMdK7mvuGqQcco?usp=sharing) (sur Google Colab)


<a id="org8c62333"></a>

# Évaluation du cours

L'objectif est d'évaluer votre compréhension des notions et méthodes abordées en cours et
en TD. Pour ce faire, nous vérifierons votre aptitude à mettre en oeuvre les traitements
informatiques adéquats face à une problématique de classification non supervisée. Les évaluations se
présenteront sous la forme de quiz sur la plateforme Moodle. 

Planning des évaluations :

-   [Devoir médian](https://moodle.univ-ubs.fr/mod/quiz/view.php?id=271431&forceview=1) : mardi 9 mars 2021 entre 9h00 et 9h30
-   [Devoir final](https://moodle.univ-ubs.fr/mod/quiz/view.php?id=277121) : mercredi 24 mars 2021 entre 15h00 et 16h00

**Note :** 
Le petit quiz suivant vous aidera afin de préparer votre environnement de programmation pour
les devoirs : <https://moodle.univ-ubs.fr/mod/quiz/view.php?id=271762>


<a id="org1900ba0"></a>

# Références

L'élaboration de ce cours s'est appuyée sur de nombreuses références. Voici mes principales sources
d'inspiration :

-   [Cours de classification non supervisée](http://www2.agroparistech.fr/IMG/pdf/ClassificationNonSupervisee-AgroParisTech.pdf) de E. Lebarbier, T. Mary-Huard (Agro ParisTech).
-   [Cours de classification non supervisée](https://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-m-explo-classif.pdf) de Philippe Besse (INSA de Toulouse).
-   [Cours de classification automatique de données quantitatives](http://www.math.u-bordeaux.fr/~mchave100p/wordpress/wp-content/uploads/2013/10/cours_classif_quanti.pdf) de Marie Chavent (Université de Bordeaux).
-   [Série de vidéos sur la classification non supervisée](https://www.youtube.com/watch?v=SE_4dLh5vXY) de François Husson (Agrocampus Rennes).
-   [Plateforme `EduClust`](https://educlust.dbvis.de) de l'Universität Konstanz qui permet de expérimenter certains algorithmes de
    *clustering* et visualiser les résultats.


<a id="org5e727bc"></a>

## L'environnement `Python`

Les exemples du cours et les travaux dirigés utilisent le logiciel `Python`. Si vous pensez ne pas
être à l'aise avec `Python`, je vous encourage vivement à faire une petite mise à niveau.

Il existe énormément de très bon didacticiels sur internet qui traitent de l'apprentissage de ce
langage en statistiques. À titre indicatif, je vous ai mis à disposition une série de TD sur
`Python` dans l'[équipe Teams du cours](https://teams.microsoft.com/_#/school/files/G%C3%A9n%C3%A9ral?threadId=19%3A541fb9397ced490aab1776de0de9202f%40thread.tacv2&ctx=channel&context=Python%2520-%2520les%2520bases&rootfolder=%252Fsites%252FSTID2-Classificationnonsupervise%252FSupports%2520de%2520cours%252FPython%2520-%2520les%2520bases). Ces TD reprennent la base de la base sur le langage
`Python`. Autrement dit, si vous avez déjà programmé, vous risquez de vous ennuyer. Vous pouvez
toujours jeter un oeil aux TD6, TD7 et TD8 qui portent sur le traitement des données.

Par ailleurs, si vous disposez d'une connexion internet fiable, je vous recommande d'utiliser une
plateforme de *Notebooks* en ligne telle que [`DeepNote`](https://deepnote.com). L'inscription est gratuite et évite de
devoir gérer l'installation de l'environnement `Python` sur votre poste. 
Sinon vous pouvez bien évidement utiliser `Python` en local en installant la distribution
[`Anaconda`](https://www.anaconda.com/products/individual). 

**L'important est de vous assurer que vous utilisez bien la version `3.8` ou supérieure de `Python`.**


<a id="orga48856e"></a>

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

