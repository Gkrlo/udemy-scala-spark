import org.apache.log4j._
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object PCAExample extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  //Create some data
  val data = Array(
    Vectors.sparse(5, Seq((1,1.0), (3, 7.0))),
    Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
    Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
  )

  val spark = SparkSession.builder().master("local[*]").getOrCreate()

  val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
  val pca = new PCA().setInputCol("features")
                     .setOutputCol("pcaFeatures")
                     .setK(3).fit(df)

  val pcaDF = pca.transform(df)
  val result = pcaDF.select("pcaFeatures")
  result.show()

}