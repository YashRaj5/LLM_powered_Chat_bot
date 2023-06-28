# Databricks notebook source
if 'config' not in locals():
    config = {}

# COMMAND ----------

config['kb_documents_path'] = "s3://db-gtm-industry-solutions/data/rcg/diy_llm_qa_bot/"
config['vector_store_path'] = "/dbfs/tmp/qabot/vector_store" # /dbfs/... is a local file system representation

# COMMAND ----------

# create database
config['database_name'] = 'qabot'
# create database if not exists
_ = spark.sql(f"create database if not exists {config['database_name']}")

# set current database context
_ = spark.catalog.setCurrentDatabase(config['database_name'])

# COMMAND ----------

# DBTITLE 1,Setting Environmental Variables for tokens
import os
# for using OpenAI model
os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("qacb", "openai_key")
# for using Databricks Dolly
os.environ['HUGGINGFACEHUB_API_TOKEN'] = dbutils.secrets.get("qacb", "hf_key")

# COMMAND ----------

# DBTITLE 1,MLflow Settings
import mlflow
config['registered_model_name'] = 'databricks_llm_qabot_solution_accelerator'
config['model_uri'] = f"models:/{config['registered_model_name']}/production"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
_ = mlflow.set_experiment('/Users/{}/{}'.format(username, config['registered_model_name']))

# COMMAND ----------

# DBTITLE 1,Set OpenAI model configs
config['openai_embedding_model'] = 'text-embedding-ada-002'
config['openai_chat_model'] = "gpt-3.5-turbo"
config['system_message_template'] = """You are a helpful assistant built by Databricks, you are good at helping to answer a question based on the context provided, the context is a document. If the context does not provide enough relevant information to determine the answer, just say I don't know. If the context is irrelevant to the question, just say I don't know. If you did not find a good answer from the context, just say I don't know. If the query doesn't form a complete question, just say I don't know. If there is a good answer from the context, try to summarize the context to answer the question."""
config['human_message_template'] = """Given the context: {context}. Answer the question {question}."""
config['temperature'] = 0.15

# COMMAND ----------

# DBTITLE 1,Set Dolly Model configs
config['hf_embedding_model'] = 'all-MiniLM-L12-v2'
config['hf_chat_model'] = 'databricks/dolly-v2-12b'
config['hf_embedding_model_loc'] = "/dbfs/tmp/qabot/embedding_model"

# COMMAND ----------

# DBTITLE 1,Set evaluation config
config["eval_dataset_path"] = "./data/eval_data.tsv"

# COMMAND ----------

# DBTITLE 1,Set deployment config
# for OpenAI
config['openai_key_secret_scope'] = "qacb"
config['openai_key_secret_key'] = "openai_key"
# for Dolly
config['hf_key_secret_scope'] = "qacb"
config['hf_key_secret_key'] = "hf_key"

config['serving_endpoint_name'] = "llm-qabot-endpoint"

# COMMAND ----------

config
