import org.apache.log4j._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans

object KmeansExample extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().master("local[2]").getOrCreate()

  //Load data
  val dataset = spark.read.format("libsvm").load("datasets/sample_kmeans_data.txt")

  dataset.printSchema()
  dataset.take(5).foreach(println)

  //Train a k-mean model
  val kmeans = new KMeans().setK(2).setSeed(1L)
  val model = kmeans.fit(dataset)

  //Evaluate clustering by computing within set Sum of Squared errors
  val WSSSE = model.computeCost(dataset)
  println(s"Within Set Sum of Squared Errors = $WSSSE")

  //Shows the result
  println("Cluster Centers: ")
  model.clusterCenters.foreach(println)


}