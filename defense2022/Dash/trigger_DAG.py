from airflow.api.client.local_client import Client

c = Client(None, None)
c.trigger_dag(dag_id='Spacenews', run_id='Spacenews', conf={})