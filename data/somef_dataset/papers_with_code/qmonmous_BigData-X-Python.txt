# Big Data *X* Python

*Introduction to pySpark by building a very simple recommender system.*

![](img/pyspark.png)

## Rappels sur le Big Data 

Lorsqu'on souhaite travailler avec un volume important de données, des outils spécifiques sont nécessaires pour scaler les processus. Des modèles distribués comme celui d'*Hadoop MapReduce* sont donc apparus pour répondre à ces problématiques de scale. 

Avec MapReduce, le traîtement des données est réalisé en deux phases distinctes, *Map* et *Reduce*. Pour se familiariser avec le concept, nous allons voir l'exemple du compteur de mots.

![Test Image](img/mapreduce.png)

On pourrait, de prime abord, vouloir compter manuellement le nombre de fois qu’un mot apparaît en input, mais cela prendrait potentiellement beaucoup de temps.  
Si l’on répartit cette tâche entre une vingtaine de personnes, les choses peuvent aller beaucoup plus vite. En effet, chaque personne prend une page du roman et écrit le nombre de fois que le mot apparaît sur la page. Il s’agit de la partie Map de MapReduce. Si une personne s’en va, une autre prend sa place. Cet exemple illustre la tolérance aux erreurs de MapReduce. 
Lorsque toutes les pages sont traitées, les utilisateurs répartissent tous les mots dans 26 boîtes en fonction de la première lettre de chaque mot. Chaque utilisateur prend une boîte, et classe les mots par ordre alphabétique. Le nombre de pages avec le même mot est un exemple de la partie Reduce de MapReduce.

> The main issue with MapReduce is that at each step, the file system (called HDFS) read and write and loose a lot of performance by saving each intermediary outputs. Building a better algorithm would require a primitive to share the data efficiently. Here comes Spark with his ability to store iterations in your cache and computer memory to avoid moving data.

## Introduction à pySpark

En distribuant le travail en différentes partitions et sur différents noeuds, avec ce qu'on appelle le *Resilient Distributed Dataset (RDD)*, Spark est jusqu'à 30 fois plus rapide que Hadoop MapReduce pour exécuter un tri par exemple.

Spark fonctionne en 4 grandes étapes :
- on crée un RDD à partir de notre jeu de données,
- on applique différentes transformations pour en créer de nouveaux ; résultants de fonctions dites 'immutables' telles que `.map` ou `.filter`,
- on décide quels RDDs garder en mémoire avec les fonctions `.persist` ou `.unpersist`,
- et on peut ensuite appliquer des fonctions plus classiques à nos RDDs comme `.count` ou `.collect` qui modifie le RDD directement, sans en créer un nouveau.

Essayons de reproduire l'algorithme de MapReduce pour compter les mots.

```python
from pyspark import SparkContext

sc = pyspark.SparkContext()
file = sc.textfile("data/count.txt")

            #split words on each line
count = file.flatMap(lambda line: line.split(" "))
            #add 1 for each occurence of a word
            .map(lambda word: (word, 1))
            #aggregate the number of occurences of each word
            .reduceByKey(lambda a, b: a + b)
            
count.persist()
count.saveAsTextFile("data/count.txt")
```


## Build a recommender system

Implémentation dans [ce Notebook](https://github.com/qmonmous/BigData-X-Python/blob/master/Intro-pySpark.ipynb).

## Ressources:
- https://arxiv.org/pdf/1606.07792.pdf%20(https://arxiv.org/pdf/1606.07792.pdf)
- http://www.3leafnodes.com/apache-spark-introduction-recommender-system
- https://towardsdatascience.com/pyspark-in-google-colab-6821c2faf41c
- https://github.com/asifahmed90/pyspark-ML-in-Colab/blob/master/PySpark_Regression_Analysis.ipynb

## Run Spark in Google Colab