import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.VectorAssembler

object KmeansProject extends App {

  /////////////////////////////////
  // K MEANS PROJECT EXERCISE ////
  ///////////////////////////////

  // Your task will be to try to cluster clients of a Wholesale Distributor
  // based off of the sales of some product categories

  // Source of the Data
  //http://archive.ics.uci.edu/ml/datasets/Wholesale+customers

  // Here is the info on the data:
  // 1)	FRESH: annual spending (m.u.) on fresh products (Continuous);
  // 2)	MILK: annual spending (m.u.) on milk products (Continuous);
  // 3)	GROCERY: annual spending (m.u.)on grocery products (Continuous);
  // 4)	FROZEN: annual spending (m.u.)on frozen products (Continuous)
  // 5)	DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
  // 6)	DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
  // 7)	CHANNEL: customers Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal)
  // 8)	REGION: customers Region- Lisnon, Oporto or Other (Nominal)

  ////////////////////////////////////
  // COMPLETE THE TASKS BELOW! //////
  //////////////////////////////////

  Logger.getLogger("org").setLevel(Level.ERROR)

  // Import SparkSession
  // Optional: Use the following code below to set the Error reporting
  // Create a Spark Session Instance
  val spark = SparkSession.builder().master("local[2]").getOrCreate()

  // Import Kmeans clustering Algorithm
  // Load the Wholesale Customers Data
  val data = spark.read.option("header", "true").option("inferSchema", "true").csv("datasets/wholesale_customers_data.csv")
  data.printSchema()
  data.take(5).foreach(println)

  // Select the following columns for the training set:
  // Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
  // Cal this new subset feature_data
  val featureData = data.select("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")
  featureData.printSchema()

  // Import VectorAssembler and Vectors
  // Create a new VectorAssembler object called assembler for the feature
  // columns as the input Set the output column to be called features
  // Remember there is no Label column
  val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"))
                                       .setOutputCol("features")

  // Use the assembler object to transform the feature_data
  // Call this new data training_data
  val trainingData = assembler.transform(featureData).select("features")
  trainingData.printSchema()

  // Create a Kmeans Model with K=3
  val kmeans = new KMeans().setK(3)

  // Fit that model to the training_data
  val model = kmeans.fit(trainingData)

  // Evaluate clustering by computing Within Set Sum of Squared Errors.
  val WSSSE = model.computeCost(trainingData)
  println(s"Within Set Sum of Squared Errors = $WSSSE")

  // Shows the result.
  model.clusterCenters.foreach(println)

}