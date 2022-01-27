from os import sep
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

if __name__ == "__main__":

    conf = SparkConf(True)
    conf.set("spark.executor.memory", "32g")
    conf.set("spark.executor.cores", "4")

    sc = SparkContext(appName="MNIST_CLF", conf=conf)

    sql = SQLContext(sc)

    df = sql.read.format("com.databricks.spark.csv").options(
        header=True, inferSchema=True, sep=',').load("mnist_data/train.csv")
