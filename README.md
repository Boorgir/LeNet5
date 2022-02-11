# **Hardware for Signal Processing : Implémentation d'un CNN - LeNet5 sur GPU**

## Objectifs des TP

Les objectif de ces 4 séances de TP de HSP sont :
- Apprendre à utiliser CUDA,
- Etudier la complexité de vos algorithmes et l'accélération obtenue sur GPU par rapport à une éxécution sur CPU,
- Observer les limites de l'utilisation d'un GPU,
- Implémenter "from scratch" un CNN : juste la partie inférence et non l'entrainement,
- Exporter des données depuis un notebook python et les réimporter dans un projet cuda,
- Faire un suivi de votre projet et du versionning à l'outil git,

### LeNet5

L'architecture de notre modèle sera la suivante :
![alt text](https://www.datasciencecentral.com/wp-content/uploads/2021/10/1lvvWF48t7cyRWqct13eU0w.jpeg)

### Quel IDE?

Jupyter-lab

## Partie 1 : Prise en main de CUDA : multiplication de matrices

On crée dans cette partie plusieurs fonctions qui nous permettrons de faire certaines opérations matricielles sur CPU et sur GPU.
Toutes les fonctions prennent les mêmes paramètres :
- n le nombre de lignes de la matrice
- p le nombre de colonnes de la matrice
- M le pointeur de la matrice

Les fonctions créées pour les opérations sur CPU sont : création d'une matrice, affichage d'une matrice, addition de 2 matrices et multiplication de 2 matrices

Les fonctions créées pour les opérations sur GPU sont : addition de 2 matrices et multiplication de 2 matrices


### Paramétrage du programme

Pour pouvoir entrer en argument du programme les dimensions de la matrice, donc qu'elles ne soient pas fixées à la compilation, il suffit pour chaque argument  d'ajouter dans la fonction main : 
```
int x = atoi(argv[i])
```

## Partie 2 : Premières couches du réseau de neurone : Conv2D et subsampling

### Layer 1

On ne travaille pas tout de suite sur la base de données MNIST mais on va plutôt générer des matrices qui correspondent à ce qu'on a besoin :
- Une matrice float raw_data de taille 32x32 initialisé avec des valeurs comprises entre 0 et 1, correspondant à nos données d'entrée.
- Une matrice float C1_data de taille 6x28x28 initialisé à 0 qui prendra les valeurs de sortie de la convolution 2D. C1 correspond aux données après la première Convolution.
- Une matrice float S1_data de taille 6x14x14 intialisé à 0 qui prendra les valeurs de sortie du sous-échantillonnage. S1 correspond aux données après le premier Sous-échantillonnage.
- Une matrice float C1_kernel de taille 6x5x5 initialisé à des valeurs comprises entre 0 et 1 correspondant à nos premiers noyaux de convolution.

On va créer pour chacune de ces matrices des tableaux à 1 dimension, donc chaque matrice sera de dimension 32x32, 6x28x28, 6x14x14 et 6x5x5.

#### Layer 2

C'est une convolution avec 6 noyaux de taille 5x5. Le vecteur résultant est donc de taille 6x28x28.

### Layer 3

C'est un sous-échantillonnage par moyennage de 2x2 vers 1 pixel. Le vecteur résultant est donc de taille 6x14x14.

### Tests

On vérifie ici que notre fonction de convolution fonctionne bien en convolutant une matrice générée par le noyau 5x5 idendité :
```
[[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0]]
```

Enfin, on rajoute la fonction d'activation tanh qui est adaptée à nos données.

```
__device__ float activation_tanh(float M)
```

## Partie 3 : Un peu de python

On importe le notebook Python dans Jupyter-lab.

### Couches manquantes

Plusieurs couches manquaient dans le modèle proposé, entre autres :
- Les couches Dense
- Les fonctions d'activation (relu et softmax)
- Le 0 padding puisque les images MNIST sont de taille 28x28 et on a en entrées des images 32x32

### Importation du dataset MNIST

Le script mis à disposition nous permet d'afficher une image contenue dans le dataset également disponible sur Moodle.

<img src="https://zupimages.net/up/22/02/buqw.png" alt="alt text" width="200" />

### Exportation des poids du modèle

A partir du modèle du notebook Python, on exporte les poids et les biais associés aux couches dans le fichier *'weights_n_bias.h'* en les stockant dans des variable, et on les importe dans notre programme C. Ces poids pourront être utilisés en utilisant *#include*.

### Test de notre réseau de neurones

On évalue maintenant notre modèle et on essaie de reconnaitre des chiffres.

<img src="https://zupimages.net/up/22/02/h8ta.png" alt="alt text" width="200" /> <img src="https://zupimages.net/up/22/02/m8xd.png" alt="alt text" width="200" />

On parvient à obtenir quelques bonnes estimations, mais le modèle se trompe assez fréquemment (voir exemple ci-dessous).

<img src="https://zupimages.net/up/22/02/agej.png" alt="alt text" width="200" />

C'est compréhensible cependant, puisque ce 5 est assez proche d'un 6.


## Conclusion

Nous avons réussi quelques uns des objectifs du TP, à savoir :
- Apprendre à utiliser CUDA,
- Implémenter "from scratch" un CNN : juste la partie inférence et non l'entrainement,
- Exporter des données depuis un notebook python et les réimporter dans un projet cuda,
- Faire un suivi de votre projet et du versionning à l'outil git (à peu près).

Concernant le CNN, les causes d'erreurs d'estimation pourraient être dues à un manque de précision, on pourrait convertir toutes les variables en double, ou à des problèmes d'importation de poids.
