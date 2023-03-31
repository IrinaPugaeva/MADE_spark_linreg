package org.apache.spark.ml.made

import breeze.linalg
import breeze.linalg.{DenseMatrix, DenseVector => BDV}
import org.apache.spark.ml.linalg.BLAS.dot
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.param.{
  DoubleParam,
  ParamMap,
  ParamValidators,
  Params,
  Param
}
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.types.StructType
import scala.util.control.Breaks.{break, breakable}
import org.apache.spark.mllib

trait HasLR extends Params {
  final val lr: DoubleParam =
    new DoubleParam(this, "lr", "gd learning rate", ParamValidators.gtEq(0))
  final def getLR: Double = $(lr)
}

trait LinearRegressionParams
    extends HasInputCol
    with HasOutputCol
    with HasLabelCol
    with HasMaxIter
    with HasLR {
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(
        schema,
        schema(getInputCol).copy(name = getOutputCol)
      )
    }
  }

  setDefault(lr -> 1e-4, maxIter -> 10)
}

// Класс для описания модели линейной регрессии.
class LinearRegression(override val uid: String)
    extends Estimator[LinearRegressionModel]
    with LinearRegressionParams
    with DefaultParamsWritable {

  // Конструктор для идентификации модели.
  def this() = this(Identifiable.randomUID("LinReg"))

  def setLR(value: Double): this.type = set(lr, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()
    implicit val doubleEncoder: Encoder[Double] = ExpressionEncoder()

    // Конвертируем входные данные в пары вектор-метка
    val vectors: Dataset[(Vector, Double)] = dataset.select(
      dataset($(inputCol)).as[Vector],
      dataset($(labelCol)).as[Double]
    )

    val numFeatures: Int = MetadataUtils.getNumFeatures(dataset, $(inputCol))

    val weights: BDV[Double] = BDV.zeros[Double](numFeatures)
    var bias: Double = 0.0
    var error: Double = Double.MaxValue

    breakable {
      for (i <- 1 to getMaxIter) {
        val (gradSum, count) = vectors.rdd
          .mapPartitions((data: Iterator[(Vector, Double)]) => {
            val weightsSummarizer = new MultivariateOnlineSummarizer()
            val biasSummarizer = new MultivariateOnlineSummarizer()

            data.foreach(r => {
              val x: linalg.Vector[Double] = r._1.asBreeze
              val y: Double = r._2
              val preds: Double = (x dot weights) + bias
              val res: Double = y - preds

              weightsSummarizer.add(mllib.linalg.Vectors.fromBreeze(x * res))
              biasSummarizer.add(mllib.linalg.Vectors.dense(res))
            })

            Iterator((weightsSummarizer, biasSummarizer))
          })
          .reduce((x, y) => {
            (x._1 merge y._1, x._2 merge y._2)
          })

        error = count.mean(0)

        val meanGrad: BDV[Double] =
          gradSum.mean.asBreeze.toDenseVector
        meanGrad :*= (-2.0) * getLR
        weights -= meanGrad

        var meanGradBias = (-2.0) * getLR * error
        bias -= meanGradBias
      }
    }

    // Return fitted model
    val lrModel = copyValues(
      new LinearRegressionModel(uid, new DenseVector(weights.toArray), bias)
    )
    lrModel
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] =
    defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema)

}

class LinearRegressionModel private[made] (
    override val uid: String,
    val weights: DenseVector,
    val bias: Double
) extends RegressionModel[Vector, LinearRegressionModel]
    with LinearRegressionParams
    with MLWritable {

  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("LinReg"), weights.toDense, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = {
    copyValues(new LinearRegressionModel(weights, bias), extra)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf =
      dataset.sqlContext.udf.register(
        uid + "_transform",
        (x: Vector) => {
          (x.asBreeze dot weights.asBreeze) + bias
        }
      )

    dataset.withColumn($(labelCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors: (Vector, Vector) =
        weights.asInstanceOf[Vector] -> Vectors.fromBreeze(
          BDV(bias)
        )

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }

  override def predict(features: Vector): Double = {
    // Breeze prediction
    (features.asBreeze dot weights.asBreeze) + bias
  }

}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] =
    new MLReader[LinearRegressionModel] {
      override def load(path: String): LinearRegressionModel = {
        val metadata = DefaultParamsReader.loadMetadata(path, sc)

        val vectors = sqlContext.read.parquet(path + "/vectors")

        implicit val encoder: Encoder[Vector] = ExpressionEncoder()

        val (weights, bias) = vectors
          .select(vectors("_1").as[Vector], vectors("_2").as[Vector])
          .first()

        val model = new LinearRegressionModel(weights.toDense, bias(0))
        metadata.getAndSetParams(model)
        model
      }
    }
}
