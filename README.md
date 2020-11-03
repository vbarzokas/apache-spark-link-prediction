# Link Prediction in Citation Networks

A set of methods and model evaluation metrics for predicting links in an academic citation network.

### Description
In this experimental study we develop methods and try to evaluate models for predicting links in an academic citation network, by taking two different aspects into consideration: 

1. Having an insight about the existing network and some of its links and trying to restore a portion of it that has been deliberately removed
2. Having no information about the existing network and rely only on the information of the scientific papers in order to predict the structure of the whole network.

For the first aspect we used supervised binary classification and more specifically the method of Logistic Regression which had a very good result, with ***F1*** score close to *86%* against the testing
set. For the second aspect we relied mainly on [Jaccard
Similarity](https://en.wikipedia.org/wiki/Jaccard_index) of the [MinHash LSH](https://en.wikipedia.org/wiki/MinHash) of each paper’s abstract which
had being vectorized using [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

For more detailed information check the [draft paper](https://github.com/vbarzokas/apache-spark-link-prediction/blob/main/Draft%20Paper%20-%20Barzokas%20-%20Link%20Prediction.pdf). 

#### Prerequisites
* [Apache Spark 2.4.0](https://spark.apache.org/releases/spark-release-2-4-0.html)
* [Scala 2.11.8](https://www.scala-lang.org/download/2.11.8.html)

#### Dataset
Our dataset contains _27,770_ academic papers that are
associated with the following information:

    1. unique ID
    2. publication year (between 1993 and 2003)
    3. title
    4. authors
    5. name of journal
    6. abstract

And exists under `src/main/resources`.
