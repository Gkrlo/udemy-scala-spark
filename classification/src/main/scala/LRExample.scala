import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession


object LRExample extends App {

  val spark = SparkSession.builder().appName("LogisticRegression").master("local[2]").getOrCreate()

  val training = spark.read.format("libsvm").load("datasets/sample_libsvm_data.txt")

  val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

  //Fit the model
  val lrModel = lr.fit(training)

  // Print the coefficients and intercept for logistic regression
  println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

  spark.stop()

}