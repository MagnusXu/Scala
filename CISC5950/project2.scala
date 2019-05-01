import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{Dataset, Row, SparkSession}


object LogisticRegression {
  val spark: SparkSession = SparkSession.builder().master("local[*]").appName("ML project").getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  def load(filePath: String): Dataset[Row] = {
    val columns = Seq("age", "workclass", "fnlwgt", "education",
      "education_num", "marital_status", "occupation", "relationship",
      "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
      "native_country", "income")
    spark.read.
      format("csv").
      option("inferSchema", true).
      load(filePath).
      toDF(columns: _*)
  }

  def main(args: Array[String]): Unit = {
    val fileName = "adult.data.txt"
    val filePath = getClass.getResource(s"/$fileName").getPath
    val adultIncomeDs = load(filePath).na.drop()
    //adultIncomeDs.show(10, false)
    //adultIncomeDs.printSchema()


    import spark.implicits._

    val categoricalColumns = Seq("workclass", "education", "marital_status", "occupation",
      "relationship", "race", "sex", "native_country")

    val indexers = categoricalColumns.
      map(colName =>
        new StringIndexer().
          setInputCol(colName).
          setOutputCol(colName + "_indexed"))
      .toArray

    val labelStringIdx = new StringIndexer().
      setInputCol("income").
      setOutputCol("label")

    val AllIndexer = indexers.toList ++ List(labelStringIdx)


    // OneHotEncoderEstimator:: OneHotEncoderEstimator to convert the categorical variables. The OneHotEncoderEstimator will return a SparseVector.
    val encoder = new OneHotEncoderEstimator().
      setInputCols(categoricalColumns.map(_ + "_indexed").toArray)
      .setOutputCols(categoricalColumns.map(name => s"${name}_vec").toArray)

    val numericCols = Array("age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week")
    val combined = categoricalColumns.map(name => s"${name}_vec") ++ numericCols

    val vectorAssembler = new VectorAssembler().
      setInputCols(combined.toArray).
      setOutputCol("features")


    val lr = new LogisticRegression()
    lr.setFeaturesCol("features")
    lr.setLabelCol("label")
    lr.setMaxIter(10)

    val allStage = AllIndexer ++ List(encoder, vectorAssembler, lr)
    val pipeline = new Pipeline().setStages(allStage.toArray)

    // prepare training and testing data
    val Array(trainingData, testData) = adultIncomeDs.randomSplit(Array(0.7, 0.3), seed = 100)

    // train model
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)
    predictions.printSchema()


    val results = predictions.select("label", "prediction", "probability", "age", "occupation")
    results.show(10, false)
    results.select("label").distinct().show()


    val evaluator = new BinaryClassificationEvaluator().setRawPredictionCol("rawPrediction")
    val accuracy = evaluator.evaluate(predictions)

    println(s"Model accuracy : $accuracy")


    // For Metrics and Evaluation
    import org.apache.spark.mllib.evaluation.MulticlassMetrics

    // Need to convert to RDD to use this
    val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)
    println(metrics.accuracy)


  }
}
