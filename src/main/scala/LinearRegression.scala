package com.gkrlo.udemy.scalasparkml

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

object LinearRegression extends App {

  val spark = SparkSession.builder().appName("LinearRegression")
    .master("local[2]")
    .config("spark.sql.warehouse.dir", "datasets").getOrCreate()

  val txtpath = "datasets/sample_linear_regression_data.txt"

  val training = spark.read.format("libsvm").load(txtpath)


}