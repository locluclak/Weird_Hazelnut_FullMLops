from src.weird_hazelnut.data import DataSyncWorker


if __name__ == "__main__":
    print("Starting Data Sync from Label Studio...")
    worker = DataSyncWorker()
    worker.sync()
    print("Sync process finished.")
