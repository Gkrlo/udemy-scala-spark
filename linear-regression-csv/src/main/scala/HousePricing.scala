import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.log4j._

object HousePricing extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().master("local[2]").getOrCreate()

  val data = spark.read.option("header","true")
                       .option("inferSchema", "true")
                       .format("csv").load("datasets/USA_Housing.csv")

  // Show columns
  data.columns.foreach(println)

  //Showing 5 rows
  data.take(5).foreach(println)

  //("label", "features")
  import spark.implicits._

  val df = data.select(data("Price").as("label"), $"Avg Area Income",$"Avg Area House Age",
                        $"Avg Area Number of Rooms",$"Avg Area Number of Bedrooms",
                        $"Area Population")

  val assembler = new VectorAssembler().setInputCols(Array("Avg Area Income","Avg Area House Age",
                                        "Avg Area Number of Rooms","Avg Area Number of Bedrooms",
                                        "Area Population")).setOutputCol("features")

  val output = assembler.transform(df).select("label", "features")

  val lr = new LinearRegression()

  val lrModel = lr.fit(output)

  val trainingSummary = lrModel.summary

  println(trainingSummary.residuals.show())

  println(trainingSummary.predictions.show())

  println("R2-> " + trainingSummary.r2)

}