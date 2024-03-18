import sqlite3
from sqlite3 import Error
from datetime import datetime
import pytz
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.StreamHandler()])


def create_connection():
    """Create a database connection to the SQLite database."""
    try:
        conn = sqlite3.connect('events.db')
        return conn
    except Error as e:
        logging.error(e)


def create_table():
    """Create a table if it doesn't exist already."""
    conn = create_connection()
    create_table_sql = """CREATE TABLE IF NOT EXISTS events (
                            id integer PRIMARY KEY,
                            eventtimestamputc text NOT NULL,
                            eventlevel text NOT NULL,
                            detail text
                         );"""
    if conn:
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
        except Error as e:
            logging.error(e)
        finally:
            conn.close()


def log_event(event_level: str, detail: str = ""):
    """Log an event with details to the database and log the event name to the terminal."""
    eventtimestamputc = datetime.now(pytz.utc).isoformat()
    conn = create_connection()
    if conn:
        try:
            sql = '''INSERT INTO events(eventtimestamputc, eventlevel, detail)
                     VALUES(?,?,?)'''
            cur = conn.cursor()
            cur.execute(sql, (eventtimestamputc, event_level, detail))
            conn.commit()
            logging.info(f"{event_level} event logged: {detail}")
        except Error as e:
            logging.error(f"Database error: {e}")
        finally:
            conn.close()


# Ensure the table is created on startup
create_table()

# Example usage
if __name__ == "__main__":
    log_event("test_event", "This is a test detail")
