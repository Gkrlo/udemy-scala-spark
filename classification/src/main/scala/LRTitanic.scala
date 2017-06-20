
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object LRTitanic extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().master("local[2]").getOrCreate()

  val dataset = "datasets/titanic.csv"

  val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load(dataset)

  //Show columns
  data.columns.foreach(println)

  //Show some rows
  data.take(5).foreach(println)

  import spark.sqlContext.implicits._
  val logRegDataAll = data.select(data("Survived").as("label"), $"Pclass", $"Name", $"Sex", $"Age", $"SibSp", $"Parch", $"Fare", $"Embarked")

  //Drop missing values
  val logRegData = logRegDataAll.na.drop()

  //**** Dealing with categorical features (OneHotEncoder) ****
  //Converting Strings inot numerical values
  val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
  val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")

  //Converting numerical values into One Hot Encoder 0 or 1
  val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
  val embarkedEncoder = new OneHotEncoder().setInputCol("EmbarkedIndex").setOutputCol("EmbarkedVec")

  // (label, features)
  val assembler = new VectorAssembler().setInputCols(Array("Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare", "EmbarkedVec"))
                                       .setOutputCol("features")

  //Building model
  val Array(training, test) = logRegData.randomSplit(Array(0.7, 0.3), seed=12345)

  val lr = new LogisticRegression()

  val pipeline = new Pipeline().setStages(Array(genderIndexer, embarkedIndexer, genderEncoder, embarkedEncoder, assembler, lr))

  val model = pipeline.fit(training)

  val results = model.transform(test)

  //Model Evaluation
  val predictionsAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

  val metrics = new MulticlassMetrics(predictionsAndLabels)

  println("CONFUSION MATRIX: ")
  println(metrics.confusionMatrix)

}