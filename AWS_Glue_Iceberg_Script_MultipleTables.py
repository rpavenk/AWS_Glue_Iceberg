import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, to_timestamp

# Get job arguments
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'iceberg_job_catalog_warehouse', 'source_db', 'source_table', 'destination_db', 'destination_table', 'primary_key'])

# Initialize Spark session and Glue context
spark = SparkSession.builder.config("spark.sql.catalog.job_catalog.warehouse", args['iceberg_job_catalog_warehouse']) \
    .config("spark.sql.catalog.job_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.job_catalog.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog") \
    .config("spark.sql.catalog.job_catalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
    .config("spark.sql.iceberg.handle-timestamp-without-timezone", "true") \
    .getOrCreate()

# Set the timestamp parsing policy
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

glueContext = GlueContext(spark.sparkContext)
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Read input data from Glue catalog into DataFrame
input_dyf = glueContext.create_dynamic_frame.from_catalog(database=args['source_db'], table_name=args['source_table'], transformation_ctx="input_dyf")
input_df = input_dyf.toDF()

# Apply deduplication logic
if not input_df.rdd.isEmpty():
    # Convert 'lastupdatedon' column from string to timestamp format
    input_df = input_df.withColumn("lastupdatedon", to_timestamp("lastupdatedon", "yyyy-MM-dd HH:mm:ss"))

    # Define window specification to partition by primary key and order by lastupdatedon descending
    window_spec = Window.partitionBy(input_df[args['primary_key']]).orderBy(input_df['lastupdatedon'].desc())

    # Use row_number function to find the latest record for each primary key
    deduplicated_df = input_df.withColumn("row_number", row_number().over(window_spec)) \
                              .filter(col("row_number") == 1) \
                              .drop("row_number")

    # Derive 'createdonyear' and 'createdonmonth' from 'createdon' column
    deduplicated_df = deduplicated_df.withColumn('createdonyear', F.year('createdon')) \
                                   .withColumn('createdonmonth', F.month('createdon'))

    # Register deduplicated DataFrame as temporary view for Spark SQL operations
    deduplicated_df.createOrReplaceTempView("deduplicated_data")

    # Show deduplicated data (for debugging purposes)
    deduplicated_df.show()

    # Perform merge operation into Iceberg table
    iceberg_merge_output_df = spark.sql("""
        MERGE INTO job_catalog.{destination_db}.{destination_table} t
        USING deduplicated_data s
        ON t.{primary_key} = s.{primary_key}
        WHEN MATCHED AND s.op = 'D' THEN DELETE
        WHEN MATCHED THEN UPDATE SET 
            {}
        WHEN NOT MATCHED THEN INSERT ({})
        VALUES ({})
    """.format(
        ", ".join(["t.{0} = s.{0}".format(col) for col in deduplicated_df.columns if col != args['primary_key']]),
        ", ".join(deduplicated_df.columns),
        ", ".join(["s.{0}".format(col) for col in deduplicated_df.columns])
    ))

    # Commit the job
    job.commit()

# Stop Spark session
spark.stop()
