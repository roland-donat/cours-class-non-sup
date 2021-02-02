
# Table des matières

1.  [Introduction](#orgb5d1ffa)
2.  [Objectifs Pédagogiques](#org39a384c)
3.  [Crédit](#org3226cc2)
4.  [Organisation pratique](#org53e52a7)
5.  [Cours](#org762bf6f)
    1.  [Cours 1 : introduction générale](#org4bc8bc4)
6.  [Travaux dirigés](#org0b3aa89)
    1.  [TD1 : distance et inertie](#org9ae3112)
7.  [Évaluation du cours](#orgd5d1443)
8.  [Bibliographie](#org4949b42)
    1.  [L'environnement `Python`](#org9989fb1)
    2.  [Principaux modules `Python` utilisés](#orgf4db8c8)



<a id="orgb5d1ffa"></a>

# Introduction

Ce site présente l'ensemble des ressources pédagogiques relatives au cours de classification non
supervisée pour la promotion de STID2 à l'IUT de Vannes.


<a id="org39a384c"></a>

# Objectifs Pédagogiques

Les principaux objectifs pédagogiques du cours sont :

-   Comprendre la problématique de classification non supervisée.
-   Replacer cette problématique dans le contexte méthodologique de l'analyse de données.
-   Introduire la notion d'inertie et son utilisation.
-   Présenter différentes approches de classification non supervisée pour l'analyse des données
    quantitatives, à savoir les méthodes de partitionnement et les méthodes de construction d'arbres hiérarchiques
    le modèle de mélange gaussien.
-   Savoir mettre en oeuvre ces méthodes avec la langage `Python`.


<a id="org3226cc2"></a>

# Crédit

Les cours et TD proposés s'inspirent largement des supports pédagogiques élaborées par Mme. Arlette
ANTONI qui avait en charge le cours de classification non supervisée sur l'année 2019-2020. Vous
pouvez consulter ces ressources pédagogiques en vous rendant dans [l'espace *Clustering* sur Moodle](https://moodle.univ-ubs.fr/course/view.php?id=3596).


<a id="org53e52a7"></a>

# Organisation pratique

Compte tenu du contexte sanitaire actuel, les cours et TD sont prévus pour se tenir en 100%
distanciel. Mais nous ne sommes pas à l'abri de changements de dernières minutes, restez donc sur
vos gardes&#x2026;

Pour suivre les cours, les TD et avoir de passionnants débats sur la classification non supervisée, merci de
faire une demande d'inscription à l'équipe Teams [STID2 - Classification non
supervisée](https://teams.microsoft.com/l/team/19%3a541fb9397ced490aab1776de0de9202f%40thread.tacv2/conversations?groupId=775ce021-bec5-4bc8-9892-4854cd178be3&tenantId=2fbd12a9-cbb9-49a2-9612-7af4096a6529).


<a id="org762bf6f"></a>

# Cours

Les supports de cours sont disponibles en ligne sous forme de présentation HTML (une connexion internet est
donc requise)


<a id="org4bc8bc4"></a>

## Cours 1 : introduction générale

-   [Slides HTML](https://roland-donat.github.io/cours-class-non-sup/cours/C1%20-%20Introduction%20g%C3%A9n%C3%A9rale/c1_intro.html)
-   Scripts `Python` : à venir


<a id="org0b3aa89"></a>

# Travaux dirigés


<a id="org9ae3112"></a>

## TD1 : distance et inertie

-   Sujet de TD
-   Scripts `Python` : à venir


<a id="orgd5d1443"></a>

# Évaluation du cours

L'objectif est d'évaluer votre compréhension des notions et méthodes abordées en cours et
en TD. Pour ce faire, nous vérifierons votre aptitude à mettre en oeuvre les traitements
informatiques adéquats face à une problématique de classification non supervisée. Les évaluations se
présenteront sous la forme de quiz sur la plateforme Moodle. 

Planning des évaluations (à venir)


<a id="org4949b42"></a>

# Bibliographie

L'élaboration de ce cours s'est appuyée sur de nombreuses références. Voici mes principales sources
d'inspiration :

-   [Cours de classification non supervisée](http://www2.agroparistech.fr/IMG/pdf/ClassificationNonSupervisee-AgroParisTech.pdf) de E. Lebarbier, T. Mary-Huard (Agro ParisTech).
-   [Cours de classification non supervisée](https://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-m-explo-classif.pdf) de Philippe Besse (INSA de Toulouse).
-   [Cours de classification automatique de données quantitatives](http://www.math.u-bordeaux.fr/~mchave100p/wordpress/wp-content/uploads/2013/10/cours_classif_quanti.pdf) de Marie Chavent (Université de Bordeaux).
-   [Série de vidéos sur la classification non supervisée](https://www.youtube.com/watch?v=SE_4dLh5vXY) de François Husson (Agrocampus Rennes).


<a id="org9989fb1"></a>

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


<a id="orgf4db8c8"></a>

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

