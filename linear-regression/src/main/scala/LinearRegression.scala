import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

object LinearRegression extends App {

  val spark = SparkSession.builder().appName("LinearRegression")
                                    .master("local[2]")
                                    .config("spark.sql.warehouse.dir", "datasets").getOrCreate()

  val txtpath = "datasets/sample_linear_regression_data.txt"

  // Training data
  val training = spark.read.format("libsvm").load(txtpath)

  //Create new LinearRegression Object & Fit the model
  val lr = new LinearRegression().setMaxIter(100).setRegParam(0.3).setElasticNetParam(0.8)
  val lrModel = lr.fit(training)

  //Print the coefficients and intercepts fro linear regression
  println(s"Coefficients: ${lrModel.coefficients}  Intercepts: ${lrModel.intercept}")

  //Summarize the model
  val trainingSummary = lrModel.summary
  println(s"numIterations: ${trainingSummary.totalIterations}")
  println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
  trainingSummary.residuals.show()
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")

}