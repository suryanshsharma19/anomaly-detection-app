# Real-time Anomaly Detection System

A sophisticated real-time anomaly detection system built with Python, Streamlit, and Isolation Forest algorithm. This application provides an interactive interface for detecting anomalies in time-series data with real-time visualization capabilities.

## 🚀 Features

- **Real-time Anomaly Detection**: Uses Isolation Forest algorithm for efficient anomaly detection
- **Interactive Dashboard**: Built with Streamlit for an intuitive user interface
- **Data Visualization**: Interactive plots using Plotly for better insights
- **Multiple Data Formats**: Supports CSV, Excel, and Parquet file formats
- **Configurable Parameters**: Adjustable model parameters for optimal performance
- **Performance Metrics**: Detailed evaluation metrics for model performance
- **Export Capabilities**: Export detected anomalies and results

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/suryanshsharma19/anomaly-detection-app.git
cd anomaly-detection-app
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8501
```

## 📊 Usage

1. Upload your dataset (CSV, Excel, or Parquet format)
2. Select the feature columns for analysis
3. Configure the model parameters:
   - Training points
   - Contamination rate
   - Timestamp column
4. Click "Start Analysis" to begin anomaly detection
5. View the results and export if needed

## 🚀 Deployment

This application is deployed on Streamlit Cloud and can be accessed at:
[Anomaly Detection App](https://anomaly-detection-app-y2pxmrdvbonqmp2xfvwihh.streamlit.app/)

## 📝 Project Structure

```
anomaly-detection-app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── data/                 # Sample datasets
├── producer.py           # Data producer script
└── consumer.py           # Data consumer script
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

- **Suryansh Sharma** - [suryanshsharma19](https://github.com/suryanshsharma19)

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Scikit-learn for the Isolation Forest implementation
- Plotly for interactive visualizations

