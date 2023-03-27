package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector => BreezeDenseVector}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.types.StructType

trait LinearRegressionParams
    extends HasInputCol
    with HasOutputCol
    with HasLabelCol {
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  // Определяем параметры модели.
  // Параметр lr - скорость обучения.
  val lr = new DoubleParam(this, "lr", "Learning rate")
  def getLr: Double = $(lr)
  def setLr(value: Double): this.type = set(lr, value)
  setDefault(lr -> 1e-4)

  // Параметр epochs - количество эпох (итераций).
  val epochs = new IntParam(this, "numEpochs", "Number of epochs")
  def getEpochs: Int = $(epochs)
  def setEpochs(value: Int): this.type = set(epochs, value)
  setDefault(epochs -> 30)

  // Проверяем и преобразуем схему входных данных.
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
}

// Класс для описания модели линейной регрессии.
class LinearRegression(override val uid: String)
    extends Estimator[LinearRegressionModel]
    with LinearRegressionParams
    with DefaultParamsWritable {

  // Конструктор для идентификации модели.
  def this() = this(Identifiable.randomUID("LinReg"))

  // Обучение модели линейной регрессии.
  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()
    implicit val doubleEncoder: Encoder[Double] = ExpressionEncoder()

    // Конвертируем входные данные в пары вектор-метка
    val vectors: Dataset[(Vector, Double)] = dataset.select(
      dataset($(inputCol)).as[Vector],
      dataset($(labelCol)).as[Double]
    )

    val numFeatures: Int = MetadataUtils.getNumFeatures(dataset, $(inputCol))
    var weights: BreezeDenseVector[Double] =
      BreezeDenseVector.rand[Double](numFeatures + 1)

    (0 until $(epochs)).foreach { _ =>
      val summary = vectors.rdd
        .mapPartitions { data =>
          val summarizer = new MultivariateOnlineSummarizer()
          data.foreach { v =>
            val x = v.asBreeze(0 until weights.size).toDenseVector
            val y = v.asBreeze(weights.size)
            val loss = sum(x * weights) - y
            val grad = x * loss
            summarizer.add(fromBreeze(grad))
          }
          Iterator(summarizer)
        }
        .reduce(_ merge _)

      weights = weights - $(lr) * summary.mean.asBreeze
    }

    copyValues(
      new LinearRegressionModel(
        Vectors.fromBreeze(weights(0 until weights.size - 1)).toDense,
        weights(weights.size - 1)
      )
    ).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] =
    defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType =
    validateTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made] (
    override val uid: String,
    val weights: DenseVector,
    val bias: Double
) extends Model[LinearRegressionModel]
    with LinearRegressionParameters
    with MLWritable {

  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("LinearRegressionModel"), weights.toDense, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = {
    copyValues(new LinearRegressionModel(weights, bias), extra)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(
      uid + "_transform",
      (x: Vector) => {
        sum(x.asBreeze *:* weights.asBreeze) + bias
      }
    )
    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType =
    validateTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors: (Vector, Vector) =
        weights.asInstanceOf[Vector] -> Vectors.fromBreeze(
          BreezeDenseVector(bias)
        )

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
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
