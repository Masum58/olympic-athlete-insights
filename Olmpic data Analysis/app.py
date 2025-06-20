import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- App Config ---
st.set_page_config(page_title="Olympic Data Insights", layout="wide", initial_sidebar_state="expanded")

# --- Load Data ---
@st.cache_data

def load_data():
    athlete_df = pd.read_csv('/Users/masumabedin/Downloads/archive (3)/athlete_events.csv')
    noc_df = pd.read_csv('/Users/masumabedin/Downloads/archive (3)/noc_regions.csv')
    df = athlete_df.merge(noc_df, on='NOC', how='left')
    return df, athlete_df, noc_df

df, athlete_df, noc_df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Olympic Data Dashboard")
section = st.sidebar.radio(
    "Select Section",
    ('Data Cleaning', 'EDA', 'Feature Engineering', 'Model Demo', 'Value Proposition')
)

# --- Data Cleaning ---
def data_cleaning_section():
    st.header("Data Cleaning & Quality Report")
    st.subheader("Missing Values")
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Column', 'Missing Values']
    st.dataframe(missing)
    fig = px.bar(missing, x='Column', y='Missing Values', title='Missing Values per Column')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Outlier Detection (Age, Height, Weight)")
    outlier_cols = ['Age', 'Height', 'Weight']
    for col in outlier_cols:
        st.write(f"**{col}**")
        fig = px.box(df, y=col, points="outliers")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Duplicates")
    dups = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {dups}")

    st.subheader("Download Cleaned Data")
    st.download_button("Download Cleaned Data as CSV", df.to_csv(index=False), "cleaned_olympic_data.csv")

# --- EDA ---
def eda_section():
    st.header("Exploratory Data Analysis (EDA)")
    st.subheader("Age Distribution by Season")
    fig = px.histogram(df, x='Age', color='Season', nbins=40, barmode='overlay',
                       title='Age Distribution by Season')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        **What does this graph mean?**
        
        This picture shows how old the athletes are. Each bar is like a group of kids or grown-ups. Some are young, some are older! The colors show if they played in the summer or winter games.
        """
    )

    st.subheader("Gender Ratio Over Time")
    gender_year = df.groupby(['Year', 'Sex']).size().reset_index(name='Count')
    fig = px.line(gender_year, x='Year', y='Count', color='Sex', markers=True,
                  title='Gender Participation Over Time')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        **What does this graph mean?**
        
        This line shows how many boys and girls played in the Olympics each year. If the line goes up, more kids joined! The blue line is for boys, the pink line is for girls.
        """
    )

    st.subheader("Medal Tally by Country")
    medals = df.dropna(subset=['Medal'])
    country_medals = medals.groupby('region').size().sort_values(ascending=False).head(20)
    fig = px.bar(country_medals, x=country_medals.index, y=country_medals.values,
                 labels={'x':'Country', 'y':'Medal Count'}, title='Top 20 Countries by Medal Count')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        **What does this graph mean?**
        
        This picture shows which countries won the most medals. The taller the bar, the more medals that country won! It's like counting who got the most gold stars in class.
        """
    )

    st.subheader("Participation Trends")
    part = df.groupby(['Year', 'Season'])['ID'].nunique().reset_index()
    fig = px.line(part, x='Year', y='ID', color='Season', markers=True,
                  title='Number of Athletes by Year and Season')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        **What does this graph mean?**
        
        This line shows how many people played in the Olympics each year. If the line goes up, more people joined the games! The colors show if it was summer or winter.
        """
    )

# --- Feature Engineering ---
def feature_engineering_section():
    st.header("Feature Engineering")
    st.write("We create new features to enhance model performance and insights.")
    fe_df = df.copy()
    # Age bins
    fe_df['AgeGroup'] = pd.cut(fe_df['Age'], bins=[0,20,30,40,100], labels=['10-20','21-30','31-40','40+'])
    # BMI
    fe_df['BMI'] = fe_df['Weight'] / ((fe_df['Height']/100)**2)
    # Decade
    fe_df['Decade'] = (fe_df['Year']//10)*10
    # Country strength (historical medal count)
    country_strength = fe_df.groupby('region')['Medal'].count()
    fe_df['CountryStrength'] = fe_df['region'].map(country_strength)
    # Olympic experience
    fe_df['OlympicExperience'] = fe_df.groupby('ID')['Year'].transform('nunique')

    st.write("**Sample of Engineered Features:**")
    st.dataframe(fe_df[['Name','Age','AgeGroup','BMI','Decade','CountryStrength','OlympicExperience']].head(20))

    st.subheader("Download Engineered Dataset")
    st.download_button("Download Engineered Data as CSV", fe_df.to_csv(index=False), "engineered_olympic_data.csv")

# --- Model Demo ---
def model_demo_section():
    st.header("Model Demo: Predicting Medal Win")
    st.write("This is a simple demonstration using Logistic Regression to predict if an athlete wins a medal.")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    fe_df = df.copy()
    fe_df['AgeGroup'] = pd.cut(fe_df['Age'], bins=[0,20,30,40,100], labels=['10-20','21-30','31-40','40+'])
    fe_df['BMI'] = fe_df['Weight'] / ((fe_df['Height']/100)**2)
    fe_df['Decade'] = (fe_df['Year']//10)*10
    fe_df['CountryStrength'] = fe_df['region'].map(fe_df.groupby('region')['Medal'].count())
    fe_df['OlympicExperience'] = fe_df.groupby('ID')['Year'].transform('nunique')
    fe_df['MedalWin'] = fe_df['Medal'].notnull().astype(int)

    # Simple features for demo
    model_df = fe_df[['Age','BMI','CountryStrength','OlympicExperience','MedalWin']].dropna()
    X = model_df.drop('MedalWin', axis=1)
    y = model_df['MedalWin']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                      x=['No Medal','Medal'], y=['No Medal','Medal'], title="Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True)

# --- Value Proposition ---
def value_proposition_section():
    st.header("Business Value & Use Cases")
    st.markdown("""
    - **Talent Identification:** Predict medal chances based on athlete profile.
    - **Strategic Planning:** Allocate training budgets to athletes in bands shown to succeed historically.
    - **Performance Benchmarks:** Establish new target thresholds (e.g., ideal BMI or experience years).
    - **Technical Value:** Robust, interpretable, and scalable pipeline for future Olympics.
    """)
    st.subheader("Deployment Strategy")
    st.markdown("""
    - **Streamlit Dashboard:** For interactive analytics and insights.
    - **API/Batch Inference:** (Future) Deploy model as API for real-time or batch predictions.
    - **Export Options:** Download cleaned and engineered data for further use.
    """)

    st.markdown("---")
    st.header("🎯 Talent Finder: Medal Probability Predictor")
    st.write("Enter an athlete's details to see their chance of winning a medal (demo model):")
    # Prepare features
    fe_df = df.copy()
    fe_df['BMI'] = fe_df['Weight'] / ((fe_df['Height']/100)**2)
    fe_df['CountryStrength'] = fe_df['region'].map(fe_df.groupby('region')['Medal'].count())
    fe_df['OlympicExperience'] = fe_df.groupby('ID')['Year'].transform('nunique')
    fe_df['MedalWin'] = fe_df['Medal'].notnull().astype(int)
    model_df = fe_df[['Age','BMI','CountryStrength','OlympicExperience','MedalWin']].dropna()
    X = model_df.drop('MedalWin', axis=1)
    y = model_df['MedalWin']
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    # User input
    age = st.number_input('Age', min_value=10, max_value=60, value=22)
    bmi = st.number_input('BMI', min_value=10.0, max_value=40.0, value=22.0)
    country_strength = st.number_input('Country Medal Count', min_value=0, max_value=1000, value=100)
    olympic_exp = st.number_input('Olympic Experience (Games attended)', min_value=1, max_value=10, value=2)
    if st.button('Predict Medal Probability'):
        prob = model.predict_proba([[age, bmi, country_strength, olympic_exp]])[0][1]
        st.success(f"This athlete has a {prob*100:.1f}% chance of winning a medal (demo model).")

    st.markdown("---")
    st.header("🏅 Benchmark Explorer: Medalist Averages by Sport")
    st.write("Select a sport to see the average BMI and Olympic experience of medal winners.")
    fe_df['MedalWin'] = fe_df['Medal'].notnull().astype(int)
    benchmarks = fe_df[fe_df['MedalWin'] == 1].groupby('Sport')[['BMI', 'OlympicExperience']].mean().reset_index()
    sport = st.selectbox('Choose a sport', sorted(benchmarks['Sport'].dropna().unique()))
    row = benchmarks[benchmarks['Sport'] == sport]
    if not row.empty:
        st.info(f"In {sport}, medal winners have an average BMI of {row['BMI'].values[0]:.1f} and {row['OlympicExperience'].values[0]:.1f} Olympic Games attended.")

# --- Section Routing ---
if section == 'Data Cleaning':
    data_cleaning_section()
elif section == 'EDA':
    eda_section()
elif section == 'Feature Engineering':
    feature_engineering_section()
elif section == 'Model Demo':
    model_demo_section()
elif section == 'Value Proposition':
    value_proposition_section()