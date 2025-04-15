# consumer.py
import redis
import time
import json
import datetime
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import sys
import logging
import signal
import argparse

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Default Configuration ---
DEFAULT_REDIS_HOST = 'localhost'
DEFAULT_REDIS_PORT = 6379
DEFAULT_STREAM_NAME = 'generic_csv_stream'
DEFAULT_CONSUMER_GROUP = 'generic_anomaly_detectors'
DEFAULT_DATA_FIELD = 'data'
DEFAULT_TRAIN_POINTS = 500
DEFAULT_WINDOW_SIZE = 20
DEFAULT_ANOMALIES_FILE = 'detected_anomalies.csv'
# --- End Configuration ---

# --- Global State ---
scaler = StandardScaler()
lof_model = None
data_window = []
points_processed = 0
feature_names = None
shutdown_flag = False
redis_conn = None
anomalies_data = []
# --- End State ---

def handle_shutdown(signum, frame):
    global shutdown_flag
    logger.info("Shutdown signal received. Attempting graceful stop...")
    shutdown_flag = True
    save_anomalies_to_csv()

def connect_redis(host, port):
    global redis_conn
    while not shutdown_flag:
        try:
            logger.info(f"Connecting to Redis at {host}:{port}...")
            r = redis.Redis(host=host, port=port, decode_responses=True)
            r.ping()
            redis_conn = r
            logger.info(f"Successfully connected to Redis.")
            return True
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
             logger.error(f"Unexpected error connecting to Redis: {e}. Retrying...")
             time.sleep(5)
    logger.info("Shutdown initiated during Redis connection attempt.")
    return False

def ensure_consumer_group(stream_name, group_name):
    if not redis_conn: return False
    try:
        redis_conn.xgroup_create(stream_name, group_name, id='0', mkstream=True)
        logger.info(f"Consumer group '{group_name}' ensured on stream '{stream_name}'.")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e): logger.info(f"Consumer group '{group_name}' already exists.")
        else: logger.error(f"Failed group creation: {e}"); return False
    except redis.exceptions.ConnectionError: logger.error("Redis conn lost ensure group."); return False
    return True

def save_anomalies_to_csv():
    """Saves detected anomalies to a CSV file."""
    if not anomalies_data:
        logger.info("No anomalies detected to save.")
        return

    try:
        df = pd.DataFrame(anomalies_data)
        df.to_csv(DEFAULT_ANOMALIES_FILE, index=False)
        logger.info(f"Saved {len(anomalies_data)} anomalies to {DEFAULT_ANOMALIES_FILE}")
    except Exception as e:
        logger.error(f"Failed to save anomalies to CSV: {e}")

def calculate_moving_stats(values, window_size):
    """Calculate moving average and standard deviation."""
    if len(values) < window_size:
        return None, None
    
    window = values[-window_size:]
    return np.mean(window), np.std(window)

def detect_anomaly(message_data_str, args):
    """Detects anomalies using LOF and time series analysis."""
    global lof_model, data_window, points_processed, feature_names

    try:
        data_point = json.loads(message_data_str)
        timestamp = data_point.get("timestamp", datetime.datetime.utcnow().isoformat() + "Z")
        features_dict = data_point.get("features", {})
        
        if not features_dict:
            logger.warning("Message received with no 'features' field. Skipping.")
            return False

        # Feature name discovery
        if feature_names is None:
            feature_names = sorted(features_dict.keys())
            logger.info(f"Discovered feature names: {feature_names}")

        # Prepare feature vector
        try:
            current_features = [float(features_dict[name]) for name in feature_names]
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error processing features: {e}")
            return False

        # Add to data window
        data_window.append(current_features)
        points_processed += 1

        # Training phase
        if lof_model is None and points_processed >= args.train_points:
            logger.info(f"Training LOF model with {len(data_window)} points...")
            try:
                # Scale the data
                scaled_data = scaler.fit_transform(data_window)
                
                # Train LOF model
                lof_model = LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=0.01,
                    novelty=True
                )
                lof_model.fit(scaled_data)
                logger.info("LOF model training complete.")
            except Exception as e:
                logger.error(f"Model training failed: {e}")
                return False

        # Prediction phase
        if lof_model is not None:
            try:
                # Scale the current point
                scaled_point = scaler.transform([current_features])
                
                # Get LOF score (negative scores are anomalies)
                lof_score = lof_model.score_samples(scaled_point)[0]
                
                # Calculate moving statistics
                values = [point[0] for point in data_window]  # Assuming first feature is the main one
                mean, std = calculate_moving_stats(values, args.window_size)
                
                # Combined anomaly detection
                is_anomaly = False
                if mean is not None and std is not None:
                    # Check if value is significantly different from moving average
                    z_score = abs((current_features[0] - mean) / std) if std > 0 else 0
                    is_anomaly = lof_score < -1.5 or z_score > 3.0  # Combined threshold
                
                if is_anomaly:
                    alert(timestamp, features_dict, lof_score, args.stream_name, args.consumer_name)
                    return True
                
            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
        
        return False

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return False

def alert(timestamp, features, score, stream_name, consumer_name):
    """Logs anomaly and stores it for later CSV export."""
    logger.warning(f"ANOMALY DETECTED: ts={timestamp}, score={score:.4f}, features={features}")
    
    anomaly_record = {
        'timestamp': timestamp,
        'score': score,
        'stream_name': stream_name,
        'consumer_name': consumer_name
    }
    
    for feature_name, value in features.items():
        anomaly_record[f'feature_{feature_name}'] = value
    
    anomalies_data.append(anomaly_record)

def process_stream(args):
    """Main consumer loop: reads messages, processes them using LOF."""
    global redis_conn # Need global connection object

    if not redis_conn: logger.error("No Redis connection."); return

    consumer_name = f"{args.consumer_prefix}_{os.getpid()}"
    args.consumer_name = consumer_name # Store for use in alerting

    logger.info(f"Consumer '{consumer_name}' starting processing loop for stream '{args.stream_name}'...")
    stream_read_id = '>'

    while not shutdown_flag:
        try:
            # Check connection & group
            if not redis_conn or not redis_conn.ping():
                 logger.warning("Redis connection lost. Reconnecting...")
                 if not connect_redis(args.redis_host, args.redis_port): logger.error("Stop: reconnect failed."); break
            if not ensure_consumer_group(args.stream_name, args.group_name):
                 logger.warning("Group ensure failed. Retrying setup."); time.sleep(5); continue

            # Read messages
            response = redis_conn.xreadgroup(
                groupname=args.group_name,
                consumername=consumer_name,
                streams={args.stream_name: stream_read_id},
                count=10, block=5000
            )
            if not response: continue

            # Process messages
            for stream, messages in response:
                for msg_id, msg_data in messages:
                    if shutdown_flag: break
                    if args.data_field in msg_data:
                        # Pass args to detection function
                        detect_anomaly(msg_data[args.data_field], args)
                    else:
                         logger.warning(f"Field '{args.data_field}' not in msg '{msg_id}'. Skipping.")

                    if not shutdown_flag: # Acknowledge
                         try: redis_conn.xack(args.stream_name, args.group_name, msg_id)
                         except redis.exceptions.ConnectionError as e: logger.error(f"Redis conn lost ACK: {e}"); redis_conn=None
                         except Exception as e: logger.error(f"ACK error: {e}")
                if shutdown_flag: break
        except redis.exceptions.ConnectionError as e: logger.error(f"Redis loop error: {e}."); redis_conn = None; time.sleep(2)
        except Exception as e: logger.exception(f"Unexpected consumer loop error: {e}"); time.sleep(5)

    logger.info("Consumer processing loop finished.")

# === Main Execution Block ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Anomaly Detector Consumer")
    parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST, help="Redis server host.")
    parser.add_argument("--redis-port", type=int, default=DEFAULT_REDIS_PORT, help="Redis server port.")
    parser.add_argument("--stream-name", default=DEFAULT_STREAM_NAME, help="Redis stream to consume from.")
    parser.add_argument("--group-name", default=DEFAULT_CONSUMER_GROUP, help="Consumer group name.")
    parser.add_argument("--consumer-prefix", default="consumer", help="Prefix for unique consumer name.")
    parser.add_argument("--data-field", default=DEFAULT_DATA_FIELD, help="Field in Redis message containing JSON payload.")
    parser.add_argument("--train-points", type=int, default=DEFAULT_TRAIN_POINTS, help="Number of points to collect before training.")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="Window size for moving statistics.")
    parser.add_argument("--anomalies-file", default=DEFAULT_ANOMALIES_FILE, help="Path to save detected anomalies CSV file.")

    args = parser.parse_args()

    logger.info(f"Initializing consumer process {os.getpid()}...")
    logger.info(f"Consuming Stream: {args.stream_name}, Group: {args.group_name}")
    logger.info(f"Using LOF + Time Series anomaly detection.")
    logger.info(f"Training points: {args.train_points}, Window size: {args.window_size}")
    logger.info(f"Anomalies will be saved to: {args.anomalies_file}")

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    if not connect_redis(args.redis_host, args.redis_port): sys.exit(1)
    process_stream(args)
    save_anomalies_to_csv()
    logger.info(f"Consumer process {os.getpid()} exiting.")