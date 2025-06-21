# 🏅 Olympic Data Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing Olympic athlete data with industry-level features including data cleaning, exploratory data analysis, feature engineering, and predictive modeling.

## 🚀 Features

### 📊 Data Analysis
- **Data Quality Report**: Missing values analysis, outlier detection, and duplicate identification
- **Interactive EDA**: Age distribution, gender participation trends, medal tallies by country
- **Child-Friendly Explanations**: Simple explanations for complex visualizations
- **Download Capabilities**: Export cleaned and engineered datasets

### 🔧 Technical Features
- **Feature Engineering**: Age groups, BMI calculation, country strength metrics
- **Predictive Model**: Logistic regression model to predict medal probability
- **Interactive Tools**: Medal probability predictor and benchmark explorer
- **Business Value Analysis**: Use cases and deployment strategies

### 🎯 Key Insights
- Athlete performance patterns across different age groups
- Gender participation trends over Olympic history
- Country-wise medal distribution analysis
- Optimal athlete profiles for different sports

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Masum58/olympic-data-analysis.git
   cd olympic-data-analysis/Olmpic data Analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data:**
   - Download the Olympic dataset from [Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)
   - Place `athlete_events.csv` and `noc_regions.csv` in your working directory
   - Update the file paths in `app.py` if needed

## 🚀 Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
Olmpic data Analysis/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── .gitignore         # Git ignore file
```

## 🎮 Dashboard Sections

### 1. **Data Cleaning**
- Missing values analysis with visualizations
- Outlier detection for age, height, and weight
- Duplicate identification
- Data quality metrics

### 2. **Exploratory Data Analysis (EDA)**
- Age distribution by season
- Gender participation trends over time
- Medal tally by country
- Participation trends analysis

### 3. **Feature Engineering**
- Age group categorization
- BMI calculation
- Decade classification
- Country strength metrics
- Olympic experience tracking

### 4. **Model Demo**
- Logistic regression for medal prediction
- Classification report and confusion matrix
- Model performance metrics

### 5. **Value Proposition**
- Business use cases and applications
- Talent identification strategies
- Performance benchmarking tools
- Interactive medal probability predictor

## 🔬 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning models

### Data Sources
- Olympic athlete events data (1896-2016)
- NOC (National Olympic Committee) regions data

## 💼 Business Applications

### For Sports Organizations
- **Talent Identification**: Predict medal chances based on athlete profiles
- **Strategic Planning**: Allocate training budgets to promising athletes
- **Performance Benchmarks**: Establish target thresholds for different sports

### For Data Scientists
- **Robust Pipeline**: Scalable data processing and analysis framework
- **Interpretable Models**: Clear insights from predictive analytics
- **Export Capabilities**: Download processed data for further analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Masum Abedin**
- GitHub: [@Masum58](https://github.com/Masum58)
- LinkedIn: [Masum Abedin](https://www.linkedin.com/in/masum-abedin/)

## 🙏 Acknowledgments

- Olympic data provided by [Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)
- Streamlit community for the excellent web framework
- Open source community for various Python libraries

---

⭐ **Star this repository if you find it helpful!** 