import org.apache.log4j._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object Ecommerce extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("Linear Regression Project")
                                    .master("local[2]").getOrCreate()

  val filePath = "datasets/ecommerce.csv"

  val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load(filePath)

  //Show columns
  data.columns.foreach(println)

  //show 5 rows
  data.take(5).foreach(println)

  val features = Array("Avg Session Length","Time on App","Time on Website", "Length of Membership")
  val label = "Yearly Amount Spent"

  val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")

  val output = assembler.transform(data).select(label, "features")

//  println("\nOUTPUT DATAFRAME:" )
//  output.columns.foreach(println)
//  output.take(5).foreach(println)

  val lr = new LinearRegression().setLabelCol(label)

  val lrModel = lr.fit(output)

  val trainingSummary = lrModel.summary

  println(s"numIterations: ${trainingSummary.totalIterations}")
  println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
  trainingSummary.residuals.show()
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")




}
