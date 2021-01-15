# Introduction à MLFlow

# 1 - Installation
Vous aurez besoin de :
- conda
- un environnement créé a partir du fichier Yaml présent dans le dossier src avec la commande suivante :
```conda env create -f src/conda.yaml```

# 2 - Utilisation

Ouvrez un terminal et activez votre environnement. 

Déplacez vous dans le dossier src et utilisez la commande suivante : ```python model.py {taille de kernel1} {kernel2} {kernel3} {nombre d'epoch}``` pour chaque configuration que vous souhaitez tester.

Pour afficher les résultats, dans ce même terminal : ```mlflow ui``` (certains antivirus peuvent bloquer le bon fonctionnement de cette commande) puis rendez vous [ici](http://localhost:5000/#/).
