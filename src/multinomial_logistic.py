import sys
import time
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import sin, cos, radians, col, row_number
from pyspark.sql.types import FloatType
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    StandardScaler,
    PCA,
    VectorAssembler,
    StringIndexer,
)

# 3 arguments: script name, input data, output directory
if len(sys.argv) != 4:
    sys.exit(2)

spark = SparkSession.builder.appName(
    "Covertype Multinomial Logistic Regression"
).getOrCreate()

seed = sys.argv[3]

# load in csv file and use column names from info file
df = spark.read.load(
    sys.argv[1], format="csv", sep=",", inferSchema="true", header="false"
).toDF(
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area_1",
    "Wilderness_Area_2",
    "Wilderness_Area_3",
    "Wilderness_Area_4",
    "Soil_Type_1",
    "Soil_Type_2",
    "Soil_Type_3",
    "Soil_Type_4",
    "Soil_Type_5",
    "Soil_Type_6",
    "Soil_Type_7",
    "Soil_Type_8",
    "Soil_Type_9",
    "Soil_Type_10",
    "Soil_Type_11",
    "Soil_Type_12",
    "Soil_Type_13",
    "Soil_Type_14",
    "Soil_Type_15",
    "Soil_Type_16",
    "Soil_Type_17",
    "Soil_Type_18",
    "Soil_Type_19",
    "Soil_Type_20",
    "Soil_Type_21",
    "Soil_Type_22",
    "Soil_Type_23",
    "Soil_Type_24",
    "Soil_Type_25",
    "Soil_Type_26",
    "Soil_Type_27",
    "Soil_Type_28",
    "Soil_Type_29",
    "Soil_Type_30",
    "Soil_Type_31",
    "Soil_Type_32",
    "Soil_Type_33",
    "Soil_Type_34",
    "Soil_Type_35",
    "Soil_Type_36",
    "Soil_Type_37",
    "Soil_Type_38",
    "Soil_Type_39",
    "Soil_Type_40",
    "Cover_Type",
)

# need to drop one of Wilderness_Area and Soil_Type to use as a reference
# level. Using first level of each as reference level
df = df.drop("Wilderness_Area_1", "Soil_Type_1")

# transformations and interaction terms
df = df.select(
    "*",
    (df.Elevation * df.Slope).alias("Elevation:Slope"),
    (cos(radians("Aspect"))).alias("cos_aspect"),
    (sin(radians("Aspect"))).alias("sin_aspect"),
    (df.Slope * cos(radians("Aspect"))).alias("Slope:cos_aspect"),
    (df.Slope * sin(radians("Aspect"))).alias("Slope:sin_aspect"),
    (df.Hillshade_9am * df.Hillshade_Noon).alias(
        "Hillshade_9am:Hillshade_Noon"
    ),
    (df.Hillshade_9am * df.Hillshade_3pm).alias("Hillshade_9am:Hillshade_3pm"),
    (df.Hillshade_Noon * df.Hillshade_3pm).alias(
        "Hillshade_Noon:Hillshade_3pm"
    ),
).drop("Aspect")

numeric_features = [
    "Elevation",
    "Slope",
    "Elevation:Slope",
    "cos_aspect",
    "sin_aspect",
    "Slope:cos_aspect",
    "Slope:sin_aspect",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Horizontal_Distance_To_Fire_Points",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Hillshade_9am:Hillshade_Noon",
    "Hillshade_9am:Hillshade_3pm",
    "Hillshade_Noon:Hillshade_3pm",
]
categorical_features = [
    "Wilderness_Area_2",
    "Wilderness_Area_3",
    "Wilderness_Area_4",
    "Soil_Type_2",
    "Soil_Type_3",
    "Soil_Type_4",
    "Soil_Type_5",
    "Soil_Type_6",
    "Soil_Type_7",
    "Soil_Type_8",
    "Soil_Type_9",
    "Soil_Type_10",
    "Soil_Type_11",
    "Soil_Type_12",
    "Soil_Type_13",
    "Soil_Type_14",
    "Soil_Type_15",
    "Soil_Type_16",
    "Soil_Type_17",
    "Soil_Type_18",
    "Soil_Type_19",
    "Soil_Type_20",
    "Soil_Type_21",
    "Soil_Type_22",
    "Soil_Type_23",
    "Soil_Type_24",
    "Soil_Type_25",
    "Soil_Type_26",
    "Soil_Type_27",
    "Soil_Type_28",
    "Soil_Type_29",
    "Soil_Type_30",
    "Soil_Type_31",
    "Soil_Type_32",
    "Soil_Type_33",
    "Soil_Type_34",
    "Soil_Type_35",
    "Soil_Type_36",
    "Soil_Type_37",
    "Soil_Type_38",
    "Soil_Type_39",
    "Soil_Type_40",
]
feature_cols = numeric_features + categorical_features

# combine numeric features into a list
numericAssembler = (
    VectorAssembler().setInputCols(numeric_features).setOutputCol("numFeatures")
)
df = numericAssembler.transform(df)
# then combine with categorical features to form our unscaled features
Assembler = (
    VectorAssembler()
    .setInputCols(["numFeatures"] + categorical_features)
    .setOutputCol("unscaledFeatures")
)
df = Assembler.transform(df)

# index class labels to start from zero
labelIndexer = StringIndexer(
    inputCol="Cover_Type",
    outputCol="Cover_Type_ind",
    stringOrderType="alphabetAsc",
).fit(df)
df = labelIndexer.transform(df)

# split the data into training and test sets (20% held out for testing)
train, test = df.randomSplit(weights=[0.8, 0.2], seed=seed)

# standardization of numeric features for comparison
scaler = StandardScaler(
    inputCol="numFeatures",
    outputCol="scaledNumFeatures",
    withStd=True,
    withMean=True,
)
scalerModel = scaler.fit(train)
# combining scaled numeric features and binary features into a list
scalerAssembler = (
    VectorAssembler()
    .setInputCols(["scaledNumFeatures"] + categorical_features)
    .setOutputCol("scaledFeatures")
)
train = scalerAssembler.transform(scalerModel.transform(train))
test = scalerAssembler.transform(scalerModel.transform(test))

# using principal components instead of features for comparison
pca = PCA(k=10, inputCol="scaledFeatures", outputCol="pcaFeatures")
pcaModel = pca.fit(train)
train = pcaModel.transform(train)
test = pcaModel.transform(test)

# Logistic Regression Model - 3 different versions
# untransformed features
lr = LogisticRegression(
    labelCol="Cover_Type_ind",
    featuresCol="unscaledFeatures",
    maxIter=500,
    standardization=False,
)
# standardized features
scaledLr = LogisticRegression(
    labelCol="Cover_Type_ind",
    featuresCol="scaledFeatures",
    maxIter=500,
    standardization=False,
)
# pca transformed features
pcaLr = LogisticRegression(
    labelCol="Cover_Type_ind",
    featuresCol="pcaFeatures",
    maxIter=500,
    standardization=False,
)

# tune and fit 3 models, calculate execution time
lr_start = time.time()
lrModel = lr.fit(train)
lr_end = time.time()

scaledLr_start = time.time()
scaledLrModel = scaledLr.fit(train)
scaledLr_end = time.time()

pcaLr_start = time.time()
pcaLrModel = pcaLr.fit(train)
pcaLr_end = time.time()

# evaluate model performance with defalt F_1 score
evaluator = MulticlassClassificationEvaluator(labelCol="Cover_Type_ind")

# make predictions
lr_train_preds = lrModel.transform(train)
lr_test_preds = lrModel.transform(test)

scaledLr_train_preds = scaledLrModel.transform(train)
scaledLr_test_preds = scaledLrModel.transform(test)

pcaLr_train_preds = pcaLrModel.transform(train)
pcaLr_test_preds = pcaLrModel.transform(test)

# calculate training and test F_1 scores
lr_train_f1 = evaluator.evaluate(lr_train_preds)
lr_test_f1 = evaluator.evaluate(lr_test_preds)

scaledLr_train_f1 = evaluator.evaluate(scaledLr_train_preds)
scaledLr_test_f1 = evaluator.evaluate(scaledLr_test_preds)

pcaLr_train_f1 = evaluator.evaluate(pcaLr_train_preds)
pcaLr_test_f1 = evaluator.evaluate(pcaLr_test_preds)

# store model scores and running time to disk for analysis
columns = ["Measure", "Value", "Value (Standardized)", "Value (PCA)"]
values = [
    ("Training f1", (lr_train_f1), (scaledLr_train_f1), (pcaLr_train_f1)),
    ("Test f1", (scaledLr_test_f1), (scaledLr_test_f1), (pcaLr_test_f1)),
    (
        "Run time",
        (lr_end - lr_start) / 60,
        (scaledLr_end - scaledLr_start) / 60,
        (pcaLr_end - pcaLr_start) / 60,
    ),
]
results = spark.createDataFrame(values, columns)
results.coalesce(1).write.csv(sys.argv[2] + "/results", header=True)

# original model parameters
weights = lrModel.coefficientMatrix
intercept = lrModel.interceptVector
weights_list = weights.toArray().tolist()
intercept_list = intercept.toArray().tolist()

for i in range(len(intercept_list)):
    weights_list[i] = [intercept_list[i]] + weights_list[i]

coeffs = spark.createDataFrame(weights_list, ["Intercept"] + feature_cols)
coeffs.coalesce(1).write.csv(sys.argv[2] + "/orig_model", header=True)

# scaled model parameters
scaledWeights = scaledLrModel.coefficientMatrix
scaledIntercept = scaledLrModel.interceptVector
scaledWeights_list = scaledWeights.toArray().tolist()
scaledIntercept_list = scaledIntercept.toArray().tolist()

for i in range(len(scaledIntercept_list)):
    scaledWeights_list[i] = [scaledIntercept_list[i]] + scaledWeights_list[i]

scaledCoeffs = spark.createDataFrame(
    scaledWeights_list, ["Intercept"] + feature_cols
)
scaledCoeffs.coalesce(1).write.csv(sys.argv[2] + "/scaled_model", header=True)

# PCA model parameters
pcaWeights = pcaLrModel.coefficientMatrix
pcaIntercept = pcaLrModel.interceptVector
pcaWeights_list = pcaWeights.toArray().tolist()
pcaIntercept_list = pcaIntercept.toArray().tolist()

for i in range(len(pcaIntercept_list)):
    pcaWeights_list[i] = [pcaIntercept_list[i]] + pcaWeights_list[i]

pcaCoeffs = spark.createDataFrame(
    pcaWeights_list,
    ["Intercept"] + ["PC" + str(X) for X in range(1, pcaModel.getK() + 1)],
)
pcaCoeffs.coalesce(1).write.csv(sys.argv[2] + "/pca_model", header=True)

# PCA Explained Variance
explained_var_list = (pcaModel.explainedVariance).toArray().tolist()
explained_var_df = spark.createDataFrame(explained_var_list, FloatType())

value_desc = Window.orderBy(col("value").desc())
explained_var_df = explained_var_df.select(
    (row_number().over(value_desc)).alias("PC"),
    col("value").alias("pct_var"),
)
explained_var_df.coalesce(1).write.csv(
    sys.argv[2] + "/pca_explained_var", header=True
)

# PCA Components
pc_list = (pcaModel.pc).toArray().tolist()
pc_df = spark.createDataFrame(
    pc_list, ["PC" + str(X) for X in range(1, pcaModel.getK() + 1)]
)
pc_df.coalesce(1).write.csv(sys.argv[2] + "/pca_components", header=True)

# Confusion matrices
lr_preds_and_labels = (
    lr_test_preds.select(["prediction", "Cover_Type_ind"])
    .withColumn("label", col("Cover_Type_ind").cast(FloatType()))
    .orderBy("prediction")
)
scaled_preds_and_labels = (
    scaledLr_test_preds.select(["prediction", "Cover_Type_ind"])
    .withColumn("label", col("Cover_Type_ind").cast(FloatType()))
    .orderBy("prediction")
)
pca_preds_and_labels = (
    pcaLr_test_preds.select(["prediction", "Cover_Type_ind"])
    .withColumn("label", col("Cover_Type_ind").cast(FloatType()))
    .orderBy("prediction")
)

# select only prediction and label columns
lr_preds_and_labels = lr_preds_and_labels.select(["prediction", "label"])
scaled_preds_and_labels = scaled_preds_and_labels.select(
    ["prediction", "label"]
)
pca_preds_and_labels = pca_preds_and_labels.select(["prediction", "label"])

# create confusion matrices
lr_metrics = MulticlassMetrics(lr_preds_and_labels.rdd.map(tuple))
scaled_metrics = MulticlassMetrics(scaled_preds_and_labels.rdd.map(tuple))
pca_metrics = MulticlassMetrics(pca_preds_and_labels.rdd.map(tuple))

# Create dataframe and write to disk
# Logistic Regression (unscaled)
lr_confusion_df = spark.createDataFrame(
    lr_metrics.confusionMatrix().toArray().tolist(),
    [str(x) for x in range(1, 8)],
)
lr_confusion_df.coalesce(1).write.csv(
    sys.argv[2] + "/lr_confusion", header=True
)

# Logistic Regression (scaled)
scaled_confusion_df = spark.createDataFrame(
    scaled_metrics.confusionMatrix().toArray().tolist(),
    [str(x) for x in range(1, 8)],
)
scaled_confusion_df.coalesce(1).write.csv(
    sys.argv[2] + "/scaled_confusion", header=True
)

pca_confusion_df = spark.createDataFrame(
    pca_metrics.confusionMatrix().toArray().tolist(),
    [str(x) for x in range(1, 8)],
)
pca_confusion_df.coalesce(1).write.csv(
    sys.argv[2] + "/pca_confusion", header=True
)

spark.stop()
