import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler

object LRAdvertising extends App {

  //////////////////////////////////////////////
  // LOGISTIC REGRESSION PROJECT //////////////
  ////////////////////////////////////////////

  //  In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
  //  This data set contains the following features:
  //    'Daily Time Spent on Site': consumer time on site in minutes
  //    'Age': customer age in years
  //    'Area Income': Avg. Income of geographical area of consumer
  //    'Daily Internet Usage': Avg. minutes a day consumer is on the internet
  //    'Ad Topic Line': Headline of the advertisement
  //    'City': City of consumer
  //    'Male': Whether or not consumer was male
  //    'Country': Country of consumer
  //    'Timestamp': Time at which consumer clicked on Ad or closed window
  //    'Clicked on Ad': 0 or 1 indicated clicking on Ad

  Logger.getLogger("org").setLevel(Level.ERROR)

  // Create a Spark Session
  val spark = SparkSession.builder().master("local[2]").appName("LRAdvertising").getOrCreate()

  // Use Spark to read in the Advertising csv file.
  val ads = spark.read.format("csv").option("inferSchema","true").option("header","true").load("datasets/advertising.csv")

  // Print the Schema of the DataFrame
  ads.printSchema()

  ///////////////////////
  /// Display Data /////
  /////////////////////

  // Print out a sample row of the data (multiple ways to do this)
  ads.take(5).foreach(println)

  ////////////////////////////////////////////////////
  //// Setting Up DataFrame for Machine Learning ////
  //////////////////////////////////////////////////

  //   Do the Following:
  //    - Rename the Clicked on Ad column to "label"
  //    - Grab the following columns "Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Timestamp","Male"
  //    - Create a new column called Hour from the Timestamp containing the Hour of the click
  import spark.sqlContext.implicits._
  val sqlFunctions = org.apache.spark.sql.functions
  val df1 = ads.select(ads("Clicked on Ad").as("label"),$"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Timestamp",$"Male")
  val df2 = df1.withColumn("Hour", sqlFunctions.hour(df1("Timestamp")))
  df2.printSchema()

  // Import VectorAssembler and Vectors
  // Create a new VectorAssembler object called assembler for the feature
  // columns as the input Set the output column to be called features
  val assembler = new VectorAssembler().setInputCols(Array("Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Hour","Male"))
                                       .setOutputCol("features")

  // Use randomSplit to create a train test split of 70/30
  val Array(training, test) = df2.randomSplit(Array(0.7,0.3))

  ///////////////////////////////
  // Set Up the Pipeline ///////
  /////////////////////////////

  // Import Pipeline
  // Create a new LogisticRegression object called lr
  val lr = new LogisticRegression()

  // Create a new pipeline with the stages: assembler, lr
  val pipeline = new Pipeline().setStages(Array(assembler, lr))

  // Fit the pipeline to training set.
  val model = pipeline.fit(training)

  // Get Results on Test Set with transform
  val results = model.transform(test)
  results.printSchema()

  ////////////////////////////////////
  //// MODEL EVALUATION /////////////
  //////////////////////////////////

  // For Metrics and Evaluation import MulticlassMetrics
  // Convert the test results to an RDD using .as and .rdd
  val predictionsAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

  // Instantiate a new MulticlassMetrics object
  val metrics = new MulticlassMetrics(predictionsAndLabels)

  // Print out the Confusion matrix
  //Model Evaluation
  println("CONFUSION MATRIX: ")
  println(metrics.confusionMatrix)


}