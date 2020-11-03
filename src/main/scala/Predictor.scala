import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object Predictor {
  // making this variable global in order to allow "spark.implicits._" be used on every function without re-importing
  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Link Prediction")
    .getOrCreate()

  import spark.implicits._

  // constants
  object Configuration {
    val NODE_INFO_FILENAME = "src/main/resources/node_information.csv"
    val TRAINING_SET_FILENAME = "src/main/resources/training_set.txt"
    val TESTING_SET_FILENAME = "src/main/resources/testing_set.txt"
    val GROUND_TRUTH_FILENAME = "src/main/resources/Cit-HepTh.txt"

    // For the first problem (p1) it is ok to run with value 1, but for the second problem (p2) this value should be
    // maximum 0.2 in order to run under a single machine within a reasonable amount of time.
    val INFO_DATAFRAME_PORTION = 1
    val TF_SIZE = 10000
    val LOGISTIC_REGRESSION_ITERATIONS = 100
    val SIMILARITY_THRESHOLD = 0.97
  }

  /**
   * Calculates the difference between two publication years.
   *
   * @param yearFrom the starting year
   * @param yearTo the end year
   * @return an integer indicating the difference.
   */
  def getPublicationYearDifference(yearFrom: Int, yearTo: Int): Int = {
    Math.abs(yearFrom - yearTo)
  }

  /**
   * Checks whether two journal names are the same.
   *
   * @param journalA the first journal
   * @param journalB the second journal
   * @return an integer with value 1 if they are the same or 0 otherwise.
   */
  def isPublishedOnSameJournal(journalA: String, journalB: String): Int = {
    if (journalA == journalB) {
      1
    }
    else {
      0
    }
  }

  /**
   * Counts the common words between two sentences.
   *
   * @param textA the first sentence
   * @param textB the second sentence
   * @return an integer indicating the number of common words.
   */
  def countCommonWords(textA: Seq[String], textB: Seq[String]): Int = {
    if (textA == null || textB == null) {
      0
    }
    else {
      textA.intersect(textB).length
    }
  }

  /**
   * Reads the contents of a CSV file and puts them into a Spark DataFrame.
   *
   * @param sparkSession the currently active Spark session.
   * @return the Spark DataFrame.
   */
  def getInfoDataFrame(sparkSession: SparkSession, filename: String): DataFrame = {
    val columnNames = Seq(
      "srcId",
      "year",
      "title",
      "authors",
      "journal",
      "abstract")

    sparkSession
      .read
      .option("header", "false")
      .csv(filename)
      .toDF(columnNames: _*)
  }

  /**
   * Performs several transformations on the columns of the incoming DataFrame, such as stopwords removal, tokenization,
   * tf-idf calculation
   *
   * @param dataFrame the target DataFrame to process
   * @return the transformed DataFrame
   */
  def preProcess(dataFrame: DataFrame): DataFrame = {
    val abstractTokenizer = new Tokenizer()
      .setInputCol("abstract")
      .setOutputCol("abstract_tokens_raw")

    val abstractStopWordsRemover = new StopWordsRemover()
      .setInputCol("abstract_tokens_raw")
      .setOutputCol("abstract_tokens_clean")

    val titleTokenizer = new Tokenizer()
      .setInputCol("title")
      .setOutputCol("title_tokens_raw")

    val titleStopWordsRemover = new StopWordsRemover()
      .setInputCol("title_tokens_raw")
      .setOutputCol("title_tokens_clean")

    val tf = new HashingTF()
      .setInputCol("abstract_tokens_clean")
      .setOutputCol("tf")
      .setNumFeatures(Configuration.TF_SIZE)

    val idf = new IDF()
      .setInputCol("tf")
      .setOutputCol("tf_idf")

    val transformedDataFrame = dataFrame
      .na
      .fill(Map("abstract" -> "", "title" -> "", "authors" -> "", "journal" -> ""))
      .withColumn("authors_tokens_raw", functions.split(col("authors"), ","))

    val stages = Array(
      abstractTokenizer,
      abstractStopWordsRemover,
      titleTokenizer,
      titleStopWordsRemover,
      tf,
      idf)

    new Pipeline()
      .setStages(stages)
      .fit(transformedDataFrame)
      .transform(transformedDataFrame)
  }

  /**
   * Retrieves the DataFrame that contains the training data.
   *
   * @param sparkContext the current Spark Context
   * @param filename the target filename to load data from
   * @return
   */
  def getTrainingDataFrame(sparkContext: SparkContext, filename: String): DataFrame = {
    sparkContext
      .textFile(filename)
      .map(line => {
        val fields = line.split(" ")

        (fields(0), fields(1), fields(2).toInt)
      })
      .toDF("srcId", "dstId", "label")
  }

  /**
   * Retrieves the DataFrame that contains the testing data.
   *
   * @param sparkContext the current Spark Context
   * @param filename the target filename to load data from
   * @return
   */
  def getTestingDataFrame(sparkContext: SparkContext, filename: String): DataFrame = {
    sparkContext
      .textFile(filename)
      .map(line => {
        val fields = line.split(" ")

        (fields(0), fields(1))
      })
      .toDF("srcId", "dstId")
  }

  /**
   * Retrieves the DataFrame that contains the ground truth data.
   *
   * @param sparkContext the current Spark Context
   * @param filename the target filename to load data from
   * @return
   */
  def getGroundTruthDataFrame(sparkContext: SparkContext, filename: String): DataFrame = {
    sparkContext
      .textFile(filename)
      .map(line => {
        val fields = line.split("\t")

        (fields(0), fields(1))
      })
      .toDF("srcId", "dstId")
  }

  /**
   * Joins the training and information DataFrames into one, so each row contains the "from" and "to" information about
   * two nodes among with their label.
   *
   * @param trainingDataFrame the training DataFrame
   * @param infoDataFrame the information DataFrame
   * @return the joined DataFrame
   */
  def joinDataFrames(trainingDataFrame: DataFrame, infoDataFrame: DataFrame): DataFrame = {
    val joinedDataFrame = trainingDataFrame
      .as("a")
      // the <=> operator means "equality test that is safe for null values"
      .join(infoDataFrame.as("b"), $"a.srcId" <=> $"b.srcId")
      .select($"a.srcId",
        $"a.dstId",
        $"a.label",
        $"b.year",
        $"b.title_tokens_clean",
        $"b.authors_tokens_raw",
        $"b.journal",
        $"b.abstract_tokens_clean")
      .withColumnRenamed("srcId", "id_from")
      .withColumnRenamed("dstId", "id_to")
      .withColumnRenamed("year", "year_from")
      .withColumnRenamed("title_tokens_clean", "title_from")
      .withColumnRenamed("authors_tokens_raw", "authors_from")
      .withColumnRenamed("journal", "journal_from")
      .withColumnRenamed("abstract_tokens_clean", "abstract_from")
      .as("a")
      .join(infoDataFrame.as("b"), $"a.id_to" <=> $"b.srcId")
      .withColumnRenamed("year", "year_to")
      .withColumnRenamed("title_tokens_clean", "title_to")
      .withColumnRenamed("authors_tokens_raw", "authors_to")
      .withColumnRenamed("journal", "journal_to")
      .withColumnRenamed("abstract_tokens_clean", "abstract_to")
      .drop("srcId")

    joinedDataFrame
  }

  /**
   * Prepares the incoming DataFrame for binary classification by combining the required feature columns into one.
   *
   * @param joinedDataFrame the join of training and information DataFrames.
   * @return the final DataFrame with the additional column "features".
   */
  def getFinalDataFrame(joinedDataFrame: DataFrame): DataFrame = {
    val commonTitleWords = udf(countCommonWords(_: Seq[String], _: Seq[String]))
    val commonAuthors = udf(countCommonWords(_: Seq[String], _: Seq[String]))
    val commonAbstractWords = udf(countCommonWords(_: Seq[String], _: Seq[String]))
    val isSameJournal = udf(isPublishedOnSameJournal(_: String, _: String))
    val publicationYearDifference = udf(getPublicationYearDifference(_: Int, _: Int))
    val toDouble = udf((i: Int) => if (i == 1) 1.0 else 0.0)

    val finalDataFrame = joinedDataFrame
    .withColumn("common_title_words", commonTitleWords(joinedDataFrame("title_from"), joinedDataFrame("title_to")))
    .withColumn("common_authors", commonAuthors(joinedDataFrame("authors_from"), joinedDataFrame("authors_to")))
    .withColumn("common_abstract_words", commonAbstractWords(joinedDataFrame("abstract_from"), joinedDataFrame("abstract_to")))
    .withColumn("publication_year_difference", publicationYearDifference(joinedDataFrame("year_from"), joinedDataFrame("year_to")))
    .withColumn("is_same_journal", isSameJournal(joinedDataFrame("journal_from"), joinedDataFrame("journal_to")))
    .withColumn("label", toDouble(joinedDataFrame("label")))
    .select("label",
      "common_title_words",
      "common_authors",
      "common_abstract_words",
      "publication_year_difference",
      "is_same_journal",
      "tf_idf")

    val assembler = new VectorAssembler()
      .setInputCols(Array("common_title_words",
        "common_authors",
        "common_abstract_words",
        "publication_year_difference",
        "is_same_journal",
        "tf_idf"))
      .setOutputCol("features")

    assembler
      .transform(finalDataFrame)
      .na
      .drop()
  }

  /**
   * Retrieves the testing DataFrame and after joining with the ground truth DataFrame, adds the values 0 or 1 to it as
   * labels in order to be easily evaluated.
   *
   * @param testingDataFrame the testing DataFrame
   * @param groundTruthDataFrame the ground truth DataFrame
   * @return
   */
  def addLabelsToTestDataFrame(testingDataFrame: DataFrame, groundTruthDataFrame: DataFrame): DataFrame = {
    val labeledTestingDataFrame = testingDataFrame
      .as("a")
      .join(
        groundTruthDataFrame
          .as("b"),
          $"a.srcId" <=> $"b.srcId" &&
          $"a.dstId" <=> $"b.dstId",
          "left"
      )
      .withColumn("label", when($"b.srcId".isNull, 0).otherwise(1))
      .drop($"b.srcId")
      .drop($"b.dstId")

    labeledTestingDataFrame
  }

  /**
   * Calculates and prints the metrics for the predictions of a binary classification model.
   *
   * @param predictions the DataFrame containing the predictions.
   */
  def calculateMetrics(predictions: DataFrame): Unit = {
    val predictionAndLabels = predictions.select("label", "prediction")
      .rdd
      .map(row =>
        (row.getAs[Double]("prediction"), row.getAs[Double]("label"))
      )

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    val precisionX = metrics.precisionByThreshold
    precisionX.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    val recallX = metrics.recallByThreshold
    recallX.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }
  }

  /**
   * Problem 1:
   * Given the network and a list of possible links, provide predictions if the links exist or not.
   *
   * @param sparkContext the current Spark Context
   */
  def p1(sparkContext: SparkContext): Unit = {
    println("Retrieving DataFrames...")
    val infoDataFrame = preProcess(getInfoDataFrame(spark, Configuration.NODE_INFO_FILENAME)
      .sample(Configuration.INFO_DATAFRAME_PORTION, 12345L))
    val trainingDataFrame = getTrainingDataFrame(sparkContext, Configuration.TRAINING_SET_FILENAME)
    val testingDataFrame = getTestingDataFrame(sparkContext, Configuration.TESTING_SET_FILENAME)
    val groundTruthDataFrame = getGroundTruthDataFrame(sparkContext, Configuration.GROUND_TRUTH_FILENAME)
    val labeledTestingDataFrame = addLabelsToTestDataFrame(testingDataFrame, groundTruthDataFrame)

    println("Joining DataFrames...")
    val joinedTrainDataFrame = joinDataFrames(trainingDataFrame, infoDataFrame)
    val joinedTestDataFrame = joinDataFrames(labeledTestingDataFrame, infoDataFrame)

    val finalTrainDataFrame = getFinalDataFrame(joinedTrainDataFrame)
    val finalTestDataFrame = getFinalDataFrame(joinedTestDataFrame)

    println("Running Logistic Regression classification...\n")
    val model = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setRawPredictionCol("prediction_raw")
      .setMaxIter(Configuration.LOGISTIC_REGRESSION_ITERATIONS)

    val predictions = model
      .fit(finalTrainDataFrame)
      .transform(finalTestDataFrame)

    println("Calculating metrics...\n")
    calculateMetrics(predictions)
  }

  /**
   * Problem 2:
   * Given only the papers you are asked to discover all possible links that may appear in the network.
   *
   * @param sparkContext the current Spark Context
   */
  def p2(sparkContext: SparkContext): Unit = {
    println("Retrieving DataFrames...")
    val infoDataFrame = preProcess(getInfoDataFrame(spark, Configuration.NODE_INFO_FILENAME)
      .sample(Configuration.INFO_DATAFRAME_PORTION, 12345L))
    val groundTruthDataFrame = getGroundTruthDataFrame(sparkContext, Configuration.GROUND_TRUTH_FILENAME)

    val minHashModel = new MinHashLSH()
      .setNumHashTables(3)
      .setInputCol("tf_idf")
      .setOutputCol("minhash_lsh")
      .fit(infoDataFrame)

    val transformedDataframe = minHashModel
      .transform(infoDataFrame)

    println("Joining DataFrames...")
    var predictionsDataFrame = minHashModel
      .approxSimilarityJoin(transformedDataframe, transformedDataframe, 1)
      .select("datasetA.srcId", "datasetB.srcId", "distCol")
      .filter("distCol >= " + Configuration.SIMILARITY_THRESHOLD)

    predictionsDataFrame = predictionsDataFrame.toDF("srcId", "dstId", "jaccardSimilarity")

    val crossValidatedDataFrame = predictionsDataFrame
      .as("a")
      .join(groundTruthDataFrame
        .as("b"),
        $"a.srcId" <=> $"b.srcId" &&
          $"a.dstId" <=> $"b.dstId")
      .drop($"b.srcId")
      .drop($"b.dstId")

    println("Total edges created: " + predictionsDataFrame.count())
    println("Correct edges detected: " + crossValidatedDataFrame.count())
  }

  def main(args: Array[String]): Unit = {
    val sparkContext = spark.sparkContext
    sparkContext.setLogLevel("ERROR")

    p1(sparkContext)
    // p2(sparkContext)

    spark.stop()
  }
}