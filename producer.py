# producer.py
import redis
import time
import json
import datetime
import pandas as pd
import os
import sys
import signal
import argparse # For command-line arguments

# --- Default Configuration (can be overridden by args) ---
DEFAULT_REDIS_HOST = 'localhost'
DEFAULT_REDIS_PORT = 6379
DEFAULT_STREAM_NAME = 'generic_csv_stream'
DEFAULT_DELAY_SECONDS = 0.1
DEFAULT_TIMESTAMP_COL = 'timestamp' # Default expected name for timestamp
# --- End Configuration ---

# --- Global flag for shutdown ---
shutdown_flag = False

def handle_shutdown(signum, frame):
    """Sets the shutdown flag on receiving SIGINT or SIGTERM."""
    global shutdown_flag
    print("\nShutdown signal received. Stopping producer...")
    shutdown_flag = True

def read_csv_and_produce(redis_conn, args):
    """Reads specified columns from CSV and sends data to the Redis stream."""
    file_path = args.csv_file
    stream_name = args.stream_name
    feature_columns = args.feature_columns # List of columns to include as features
    timestamp_col = args.timestamp_column

    try:
        print(f"Attempting to read data from CSV: {file_path}")
        # Explicitly handle comments if necessary (e.g., comment='#')
        df = pd.read_csv(file_path, comment='#', skipinitialspace=True)
        print(f"Read {len(df)} rows from CSV.")

        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        actual_columns = list(df.columns)
        print(f"Actual columns found: {actual_columns}")

    except FileNotFoundError:
        print(f"ERROR: CSV file not found at '{os.path.abspath(file_path)}'")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"ERROR: CSV file '{file_path}' is empty or contains only comments.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read or parse CSV '{file_path}': {e}")
        sys.exit(1)

    # Validate specified columns exist
    all_needed_cols = feature_columns + ([timestamp_col] if timestamp_col else [])
    missing_cols = [col for col in all_needed_cols if col not in actual_columns]
    if missing_cols:
         print(f"\nERROR: The following specified columns were not found in the CSV header:")
         for col in missing_cols:
             print(f"  - '{col}'")
         print(f"Actual columns detected were: {actual_columns}")
         sys.exit(1)

    print(f"\nStarting producer for stream '{stream_name}' on {args.redis_host}:{args.redis_port}...")
    print(f"Sending data from feature columns: {feature_columns}")
    if timestamp_col:
        print(f"Using timestamp from column: '{timestamp_col}'")
    else:
        print("Using generated timestamps.")
    print(f"Press Ctrl+C to stop early.")

    produced_count = 0
    start_time = time.time()

    for index, row in df.iterrows():
        if shutdown_flag:
            print(f"\nStopping production loop due to shutdown signal after {produced_count} messages.")
            break

        # Extract timestamp
        timestamp_str = row.get(timestamp_col) if timestamp_col else None
        if timestamp_str is None:
             timestamp_str = datetime.datetime.utcnow().isoformat() + "Z"

        # Extract features and build payload
        features = {}
        valid_row = True
        for col in feature_columns:
            try:
                # Convert each specified feature column to float
                features[col] = float(row[col])
            except (ValueError, TypeError):
                print(f"\nWarning: Skipping row {index+1} due to non-numeric value in feature column '{col}': {row[col]}")
                valid_row = False
                break # Skip entire row if one feature is invalid
        if not valid_row:
            continue

        # Construct the message payload
        data_point = {"timestamp": timestamp_str, "features": features}
        # Send the entire data_point as a JSON string under a generic key
        message_data = { "data": json.dumps(data_point) }

        # Send to Redis with reconnect logic
        connected = False
        while not connected and not shutdown_flag:
            try:
                message_id = redis_conn.xadd(stream_name, message_data)
                produced_count +=1
                # print(f"  Sent ID: {message_id} | TS: {timestamp_str} | Features: {features}") # Verbose
                connected = True
            except redis.exceptions.ConnectionError as e:
                print(f"\nERROR: Redis connection lost: {e}. Attempting reconnect...")
                while not shutdown_flag:
                    time.sleep(5)
                    try:
                        redis_conn = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
                        redis_conn.ping()
                        print("Reconnected to Redis.")
                        break
                    except redis.exceptions.ConnectionError:
                        print("Reconnect failed. Retrying...")
                if not connected and not shutdown_flag: print("\nWarning: Failed reconnect.")
            except Exception as e:
                print(f"\nERROR: Unexpected error sending to Redis: {e}")
                time.sleep(1)

        if shutdown_flag: break
        time.sleep(args.delay)
    # End row loop

    end_time = time.time()
    duration = end_time - start_time
    print(f"\nProducer finished or stopped. Sent {produced_count} messages in {duration:.2f} seconds.")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV Data Producer for Redis Streams")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument("-f", "--feature-columns", required=True, nargs='+',
                        help="List of column names in the CSV containing numerical features to send.")
    parser.add_argument("-t", "--timestamp-column", default=DEFAULT_TIMESTAMP_COL,
                        help=f"Column name for timestamps (optional, defaults to '{DEFAULT_TIMESTAMP_COL}', generates if missing).")
    parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST, help="Redis server host.")
    parser.add_argument("--redis-port", type=int, default=DEFAULT_REDIS_PORT, help="Redis server port.")
    parser.add_argument("--stream-name", default=DEFAULT_STREAM_NAME, help="Name of the Redis stream to publish to.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="Delay in seconds between sending rows.")

    args = parser.parse_args()

    # Check if timestamp column exists if specified as default but not required
    if args.timestamp_column == DEFAULT_TIMESTAMP_COL:
         try:
             # Peek at header without reading whole file if large
             header_df = pd.read_csv(args.csv_file, comment='#', skipinitialspace=True, nrows=0)
             if DEFAULT_TIMESTAMP_COL not in [col.strip() for col in header_df.columns]:
                 print(f"Note: Default timestamp column '{DEFAULT_TIMESTAMP_COL}' not found. Timestamps will be generated.")
                 args.timestamp_column = None # Clear it so it's not expected later
         except Exception:
              # Ignore errors here, validation will happen during full read
              pass

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Establish initial Redis connection
    try:
        r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
        r.ping()
        print(f"Successfully connected to Redis at {args.redis_host}:{args.redis_port}")
    except redis.exceptions.ConnectionError as e:
        print(f"ERROR: Could not connect to Redis: {e}")
        sys.exit(1)

    # Start the process
    read_csv_and_produce(r, args)