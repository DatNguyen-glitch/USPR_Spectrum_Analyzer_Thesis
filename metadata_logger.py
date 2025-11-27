import sqlite3
import os

class MetadataLogger:
    def __init__(self, db_path="signals_metadata.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp DATETIME,
            frequency_hz REAL,
            bandwidth_hz REAL,
            peak_db REAL,
            snr_db REAL,
            duration_sec REAL
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def log_capture(self, filename, timestamp, freq, bw, peak, snr, duration):
        query = """
        INSERT INTO captures (filename, timestamp, frequency_hz, bandwidth_hz, peak_db, snr_db, duration_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        try:
            self.conn.execute(query, (filename, timestamp, freq, bw, peak, snr, duration))
            self.conn.commit()
            print(f"[Metadata] Logged: {os.path.basename(filename)}")
        except Exception as e:
            print(f"[Metadata] Error: {e}")

    def close(self):
        self.conn.close()