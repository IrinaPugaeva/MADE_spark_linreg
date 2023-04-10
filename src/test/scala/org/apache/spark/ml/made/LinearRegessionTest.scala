package org.apache.spark.ml.made

import org.scalatest._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should
import org.apache.spark.sql.functions._
import scala.util.Random

class LinearRegressionTest
    extends AnyFlatSpec
    with should.Matchers
    with WithSpark {

  val lr = 1e-4
  val maxIter = 100
  lazy val w: linalg.DenseVector = LinearRegressionTest._w
  lazy val b: Double = LinearRegressionTest._b
  lazy val data = LinearRegressionTest._data

  "math" should "work" in {
    (1 + 1) should be(2)
  }

  "LinearRegression model" should "have the expected number of weights" in {
    val estimator: LinearRegression = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("prediction")
      .setLabelCol("target")
      .setLR(lr)
      .setMaxIter(maxIter)

    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y", "z"))
      .setOutputCol("features")

    val vectorized = assembler.transform(data)

    val model = estimator.fit(vectorized)

    model.weights.size should be(w.size) // (10)
  }

  "Prediction" should "have the same number of elements as test data" in {
    val estimator: LinearRegression = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("prediction")
      .setLabelCol("target")
      .setLR(lr)
      .setMaxIter(maxIter)

    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y", "z"))
      .setOutputCol("features")

    val vectorized = assembler.transform(data)

    val model = estimator.fit(vectorized)

    val predictions = model
      .transform(vectorized)
      .select("prediction")
      .collect()
      .map(_.getDouble(0))

    predictions.size should be(data.count())
  }

  "LinearRegression model" should "converge in a reasonable number of iterations" in {
    val estimator: LinearRegression = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("prediction")
      .setLabelCol("target")
      .setLR(lr)
      .setMaxIter(maxIter)

    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y", "z"))
      .setOutputCol("features")

    val vectorized = assembler.transform(data)

    val model = estimator.fit(vectorized)

    model.getMaxIter should be <= maxIter
  }

//   "LinearRegressionModel" should "produce correct predictions" in {

//     val estimator: LinearRegression = new LinearRegression()
//       .setInputCol("features")
//       .setOutputCol("prediction")
//       .setLabelCol("target")
//       .setLR(lr)
//       .setMaxIter(maxIter)

//     val assembler = new VectorAssembler()
//       .setInputCols(Array("x", "y", "z"))
//       .setOutputCol("features")

//     val vectorized = assembler.transform(learnData)

//     val model = estimator.fit(vectorized)

//     val predictions = model
//       .transform(vectorized)
//       .select("prediction")
//       .collect()
//       .map(_.getDouble(0))

//     predictions(0) should be(7.0 +- 0.1)
//     predictions(1) should be(17.0 +- 0.1)
//     predictions(2) should be(27.0 +- 0.1)
//     predictions(3) should be(37.0 +- 0.1)
//     predictions(4) should be(47.0 +- 0.1)
//   }

//   "Estimator" should "produce functional model" in {
//     val estimator: LinearRegression = new LinearRegression()
//       .setInputCol("features")
//       .setOutputCol("prediction")
//       .setLabelCol("target")
//       .setLR(lr)
//       .setMaxIter(maxIter)

//     val assembler = new VectorAssembler()
//       .setInputCols(Array("x", "y", "z"))
//       .setOutputCol("features")

//     val vectorized = assembler.transform(data)

//     val model = estimator.fit(vectorized)
//     model.weights(0) should be(w(0) +- 0.1)
//     model.weights(1) should be(w(1) +- 0.1)
//     model.weights(2) should be(w(2) +- 0.1)

//     model.bias should be(b +- 0.1)
//   }
}

object LinearRegressionTest extends WithSpark {

  private val numRecords = 100000
  private val featureCols = Seq("x", "y", "z")
  private val labelCol = "target"
  private lazy val _w = Vectors.dense(1.5, 0.3, -0.7).toDense
  private val _b = 1.0

  lazy val _rdd = spark.sparkContext.parallelize(
    Seq.fill(numRecords) {
      (Random.nextDouble, Random.nextDouble, Random.nextDouble)
    }
  )

  lazy val _data: DataFrame = spark
    .createDataFrame(_rdd)
    .toDF(featureCols: _*)
    .withColumn(
      "target",
      lit(_w(0)) * col(featureCols(0))
        + lit(_w(1)) * col(featureCols(1))
        + lit(_w(2)) * col(featureCols(2)) + _b + rand()
    )

}
