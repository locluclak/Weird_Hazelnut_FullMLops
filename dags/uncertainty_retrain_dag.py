from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="uncertainty_retrain_dag",
    start_date=datetime(2026, 1, 1),
    schedule_interval=timedelta(hours=1),  # test mỗi 5 phút
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "weird-hazelnut",
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
    },
    tags=["mlops", "retrain", "human-in-the-loop"],
):
    sync_labels = BashOperator(
        task_id="sync_labels",
        bash_command="docker exec weird-hazelnut-app python sync_data.py",
    )

    check_gate = BashOperator(
        task_id="check_retrain_gate",
        bash_command="docker exec weird-hazelnut-app python scripts/check_retrain_gate.py",
    )

    retrain_models = BashOperator(
        task_id="retrain_models",
        bash_command="docker exec weird-hazelnut-app python retrain.py",
    )

    evaluate_candidate = BashOperator(
        task_id="evaluate_candidate",
        bash_command="docker exec weird-hazelnut-app python scripts/evaluate_candidate.py",
    )

    promote_if_better = BashOperator(
        task_id="promote_if_better",
        bash_command="docker exec weird-hazelnut-app python scripts/promote_if_better.py",
    )

    sync_labels >> check_gate >> retrain_models >> evaluate_candidate >> promote_if_better