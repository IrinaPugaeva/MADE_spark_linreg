package org.apache.spark.ml.made
import breeze.linalg
import breeze.linalg.{DenseMatrix, DenseVector => BDV}
import breeze.stats.mean
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{
  DoubleParam,
  ParamMap,
  ParamValidators,
  Params,
  Param
}
import org.apache.spark.ml.param.shared.{
  HasInputCol,
  HasLabelCol,
  HasOutputCol,
  HasMaxIter
}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.StructType
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks.{break, breakable}
import org.apache.spark.mllib

trait LinearRegressionParams
    extends HasInputCol // Трейт, который добавляет параметр для задания названия столбца с признаками
    with HasOutputCol // Трейт, который добавляет параметр для задания названия столбца с предсказанными значениями
    with HasLabelCol // Трейт, который добавляет параметр для задания названия столбца с метками классов
    with HasMaxIter { // Трейт, который добавляет параметр для задания максимального количества итераций алгоритма

  // Определяем методы для задания значений параметров
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  // Определяем параметр для задания скорости обучения (learning rate) с дефолтным значением 1e-4
  final val lr: DoubleParam =
    new DoubleParam(this, "lr", "learning rate")
  final def getLR: Double = $(lr)
  setDefault(lr -> 1e-4)

  // Определяем метод для проверки корректности входной схемы и преобразования ее, если нужно
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    // Проверяем тип столбца с признаками (должен быть VectorUDT)
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) { // Если задан столбец с предсказанными значениями
      // Проверяем тип столбца с предсказанными значениями (должен быть VectorUDT)
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema // Возвращаем исходную схему
    } else {
      // Добавляем в схему новый столбец с названием outputCol и типом VectorUDT, аналогичный столбцу с признаками
      SchemaUtils.appendColumn(
        schema,
        schema(getInputCol).copy(name = getOutputCol)
      )
    }
  }

  setDefault(
    maxIter -> 10
  ) // Задаем значение по умолчанию для максимального количества итераций
}

// Класс для описания модели линейной регрессии.
class LinearRegression(override val uid: String)
    extends Estimator[LinearRegressionModel]
    with LinearRegressionParams
    with DefaultParamsWritable {

  // Конструктор для идентификации модели.
  def this() = this(Identifiable.randomUID("LinReg"))

// Метод для установки значения learning rate.
  def setLR(value: Double): this.type = set(lr, value)

// Метод для установки максимального количества итераций.
  def setMaxIter(value: Int): this.type = set(maxIter, value)

// Метод для обучения модели линейной регрессии.
  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()
    implicit val doubleEncoder: Encoder[Double] = ExpressionEncoder()

    // Преобразование векторов и меток в формат Dataset[(Vector, Double)].
    val vectors: Dataset[(Vector, Double)] = dataset.select(
      dataset($(inputCol)).as[Vector],
      dataset($(labelCol)).as[Double]
    )

    val numFeatures: Int = AttributeGroup
      .fromStructField(dataset.schema($(inputCol)))
      .numAttributes
      .getOrElse(
        vectors.first()._1.size
      )

    // Инициализация весов и bias.
    var weights: BDV[Double] = BDV.rand[Double](numFeatures)
    var bias: Double = scala.util.Random.nextDouble() * 2 - 1

    val iters: Int = getMaxIter
    val lr: Double = getLR

    var error: Double = Double.MaxValue

    // Цикл обучения модели.
    for (_ <- 0 until iters) {
      // Вычисление градиента на каждой партиции данных.
      val (lossSum, biasSum) = vectors.rdd
        .mapPartitions((data: Iterator[(Vector, Double)]) => {
          val weightsSummarizer = new MultivariateOnlineSummarizer()
          val biasSummarizer = new MultivariateOnlineSummarizer()

          data.foreach { case (features, label) =>
            val x = features.asBreeze
            val preds = x.dot(weights) + bias
            val error = preds - label

            weightsSummarizer.add(mllib.linalg.Vectors.fromBreeze(x * error))
            biasSummarizer.add(mllib.linalg.Vectors.dense(error))
          }
          Iterator((weightsSummarizer, biasSummarizer))
        })
        .reduce((x, y) => {
          (x._1.merge(y._1), x._2.merge(y._2))
        })

      // Вычисление среднего градиента и обновление весов и bias.
      val meanGrad: BDV[Double] =
        lossSum.mean.asBreeze.toDenseVector *:* (-2.0) * lr
      weights -= meanGrad

      var meanGradBias = (-2.0) * lr * biasSum.mean(0)
      bias -= meanGradBias
    }

    // Возвращение обученной модели.
    copyValues(
      new LinearRegressionModel(uid, Vectors.fromBreeze(weights).toDense, bias)
    )

  }
  // Копируем параметры этапа обучения в новый объект модели с помощью функции defaultCopy.
  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] =
    defaultCopy(extra)
  // Проверяем и трансформируем схему входного DataFrame на основе заданных параметров для этапа обучения с помощью функции validateAndTransformSchema.
  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema)

}

class LinearRegressionModel private[made] (
    override val uid: String, // уникальный идентификатор модели
    val weights: DenseVector, // вектор весов модели
    val bias: Double // смещение модели
) extends RegressionModel[Vector, LinearRegressionModel]
    with LinearRegressionParams // параметры линейной регрессии
    with MLWritable {

  // Конструктор
  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("LinReg"), weights.toDense, bias)

  // метод для создания копии модели
  override def copy(extra: ParamMap): LinearRegressionModel = {
    copyValues(new LinearRegressionModel(weights, bias), extra)
  }

  // метод для применения модели к данным
  override def transform(dataset: Dataset[_]): DataFrame = {
    // создаем UDF, используемый для применения модели к каждому вектору признаков
    val transformUdf =
      dataset.sqlContext.udf.register(
        uid + "_transform",
        (x: Vector) => {
          (x.asBreeze dot weights.asBreeze) + bias // Breeze-операции для умножения векторов
        }
      )

    dataset.withColumn(
      $(outputCol),
      transformUdf(dataset($(inputCol)))
    ) // применяем модель к данным и возвращаем результат
  }

  // метод для проверки соответствия схемы входных данных схеме, используемой моделью
  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema)

  // метод для сохранения модели
  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors: (Vector, Vector) =
        weights.asInstanceOf[Vector] -> Vectors.fromBreeze(
          BDV(bias)
        )

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
      // сохраняем вектор весов и смещение в формате Parquet
    }
  }

  // Метод для предсказания
  override def predict(features: Vector): Double = {
    // Breeze-операции для умножения векторов
    (features.asBreeze dot weights.asBreeze) + bias
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {

  // Переопределение метода `read` для чтения объекта `LinearRegressionModel`
  override def read: MLReader[LinearRegressionModel] =
    new MLReader[LinearRegressionModel] {

      override def load(path: String): LinearRegressionModel = {
        // Загрузка метаданных модели
        val metadata = DefaultParamsReader.loadMetadata(path, sc)

        // Чтение векторов модели из Parquet
        val vectors = sqlContext.read.parquet(path + "/vectors")

        // Кодировщик
        implicit val encoder: Encoder[Vector] = ExpressionEncoder()

        // Веса и баес
        val (weights, bias) = vectors
          .select(vectors("_1").as[Vector], vectors("_2").as[Vector])
          .first()

        // Создание новой модели LinearRegressionModel
        val model = new LinearRegressionModel(weights.toDense, bias(0))

        // Метаданные
        metadata.getAndSetParams(model)

        // Возврат загруженной модели
        model
      }
    }
}
