from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import pickle
import time 
from main import genai_engine


st.set_page_config(
    page_title="Sleep Health and Stress Level Prediction",
    page_icon="💤",
    initial_sidebar_state="expanded"
)


st.title('Sleep Health and Stress Level Prediction 💤')

st.markdown('<span style="color:gray">The app predicts the stress level of a person based on the data provided.</span>', unsafe_allow_html=True)

homepage, knowledge, prediction = st.columns(3)


selected = option_menu(
    menu_title=None,
    options=["Home", "Dataset", "Prediction", "Contact","Chatbot"],
    icons=["window", "table", "cpu", "phone","robot"],
    orientation="horizontal",
    default_index=0,
    styles={
        # "container": {"background-color": "white"},
        # "icon": {"color": "white"}, 
        # "nav-link": {"text-align": "center", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#176397"},
    }
)

df = pd.read_csv('C:\\Users\\sound\\OneDrive\\Desktop\\Byte Crafter\\Sleep Health Prediction\\Sleep Health Prediction\\Dataset\\clean_dataset.csv')

df.drop(['Person ID', 'Sick'], axis=1, inplace=True)
df = df[['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
       'Physical Activity Level', 'BMI Category', 'Heart Rate',
       'Daily Steps', 'Sleep Disorder', 'BP High', 'BP Low', 'Stress Level']]

if selected == "Home":
    with st.container():
        
        # Define the image URL
        image_url = "C:\\Users\\sound\\OneDrive\\Desktop\\Byte Crafter\\Sleep Health Prediction\\Images\\IntroImage.png"
        
        # Display the image without hyperlink
        st.image(image_url, caption="Homepage Image", width=None)
    

        st.markdown(
            """

            <h2 style="color:#176397">Overview</h2>

            <p style="color:#1D4665; text-align: justify;">
                Our platform provides a user-friendly interface designed to seamlessly collect your personal health and lifestyle data. Through intuitive input fields, you can easily share essential information such as demographics, sleep patterns, physical activity levels, and stress indicators. This streamlined data collection process ensures that you can effortlessly contribute to the creation of your personalized health profile without any hassle.
            </p>

            <h2 style="color:#176397">Personalized Stress Assessment</h2>

            <p style="color:#1D4665; text-align: justify;">
                Our predictive model analyzes your data to generate a personalized stress level assessment tailored to your unique profile. By understanding the intricate relationships between various factors influencing sleep quality, lifestyle choices, and stress management, our platform empowers you to gain deeper insights into your overall wellbeing. This personalized approach ensures that you receive accurate and relevant information that is directly applicable to your individual health journey.


            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <h2 style="color:#176397">Proactive Health Management</h2>

            <p style="color:#1D4665; text-align: justify;">
                With easy access to personalized health insights and early detection capabilities, you can take proactive steps towards better health and wellbeing, all from the comfort of your own home. Don't wait until symptoms arise – choose our app today and prioritize your health journey with confidence. By leveraging advanced analytics and intuitive design, our platform enables you to stay ahead of potential health concerns and make informed decisions that positively impact your overall quality of life.

            </p>
            <hr>
            <br>
            <p align='right'>
                <a href="https://github.com/Aishwarya-HM/BYTE_CRAFTERS_JITHACK" target="_blank">View on GitHub</a>
                <br>
                
            </p>
            """,
            unsafe_allow_html=True
        )
    

if selected == "Dataset":

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='#176397',
            align='left',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='white',
            align='left',
            font=dict(color='#1D4665', size=12)
        )
    )])

    fig.update_layout(
        height=800
    )


    st.markdown(
        """
        <h2 style="color:#176397">Dataset Preview</h2>

        <p style="color:#1D4665">
            The dataset consists of 400 rows and 13 columns, encompassing various demographic, health, and lifestyle variables. The dataset is divided into two parts: the first part contains demographic, health, and lifestyle variables, and the second part contains sleep health variables. The dataset contains 13 columns, out of which 12 are features and 1 is the target variable.
        </p>

        <p style="color:#1D4665">
            <a href="https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset" target="_blank">Dataset Source</a>
        </p>
        """,
        unsafe_allow_html=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme='streamlit',
        config={
            'displayModeBar': False
        }
    )

    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <a href="data:file/csv;base64,{data}" download="{file_name}" style="padding: 10px; background-color: #176397; color: white; border-radius: 5px; text-decoration: none;">
                {label}
            </a>
        </div>
        """.format(
            data=df.to_csv(index=False).encode().decode('utf-8').replace('\n', '%0A'),
            file_name='Sleep_dataset.csv',
            label="Download Dataset"
        ),
        unsafe_allow_html=True
    )


if selected == "Prediction":

    accuracies = dict(
        mape=0.04,
        rmse=0.37,
        r2=0.97
    )

    st.markdown(
    f"""
    <h2 style="color:#176397">Prediction Model Accuracy Metrics</h2>

    A machine learning study was conducted, and as a result, the Linear Regression model was chosen as the main model. 
    The accuracy of the model was evaluated using **MAPE** (Mean Absolute Percentage Error), **RMSE** (Root Mean Square Error), and **R2** (R Squared) values. 
    Based on the test results, the accuracy of the model was calculated as follows.

    <div class="kpi-container">
        <div class="kpi-box">
            <span class="kpi-title">MAPE</span>
            <span class="kpi-value">{accuracies['mape']*100}%</span>
        </div>
        <div class="kpi-box">
            <span class="kpi-title">RMSE</span>
            <span class="kpi-value">{accuracies['rmse']}</span>
        </div>
        <div class="kpi-box">
            <span class="kpi-title">R2</span>
            <span class="kpi-value">{accuracies['r2']*100}%</span>
        </div>
    </div>

    <style>
    .kpi-container {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
    }}

    .kpi-box {{
    flex-grow: 0.25;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0px;
    border-radius: 100px;
    background-color: #F2F2F2;
    transition: box-shadow 0.3s;
    }}

    .kpi-box:hover {{
    box-shadow: 0 0 10px #1D4665;
    }}

    .kpi-title {{
    color: #176397;
    font-size: 20px;
    font-weight: bold;
    font-family: 'Poppins', sans-serif;
    margin-bottom: 0px;
    }}

    .kpi-value {{
    color: #1D4665;
    font-size: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True)


    st.write("---")

    features, result = st.columns((4,2))

    with features:
        st.markdown(
            """
                <h3 style="color:#176397">Set Features</h3>
            """,unsafe_allow_html=True
        )

        def load_model():
            """
            The function `load_model()` loads a linear regression model from a pickle file.
            
            Returns:
              a loaded linear regression model.
            """
            with open('C:\\Users\\sound\\OneDrive\\Desktop\\Byte Crafter\\Sleep Health Prediction\\Sleep Health Prediction\\Models_pkl\\model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model

        gender_le = joblib.load('C:\\Users\\sound\\OneDrive\\Desktop\\Byte Crafter\\Sleep Health Prediction\\Sleep Health Prediction\\Models_pkl\\gender_encoder.pkl')
        occupation_le = joblib.load('C:\\Users\\sound\\OneDrive\\Desktop\\Byte Crafter\\Sleep Health Prediction\\Sleep Health Prediction\\Models_pkl\\occupation_encoder.pkl')
        bmiCategory_le = joblib.load('C:\\Users\\sound\\OneDrive\\Desktop\\Byte Crafter\\Sleep Health Prediction\\Sleep Health Prediction\\Models_pkl\\bmi_category_encoder.pkl')
        sleepDisorder_le = joblib.load('C:\\Users\\sound\\OneDrive\\Desktop\\Byte Crafter\\Sleep Health Prediction\\Sleep Health Prediction\\Models_pkl\\sleep_disorder_encoder.pkl')
        scaler = joblib.load('C:\\Users\\sound\\OneDrive\\Desktop\\Byte Crafter\\Sleep Health Prediction\\Sleep Health Prediction\\Models_pkl\\scaler.pkl')


        col1, col2 = st.columns(2)

        with col1:
            Gender = st.selectbox(
                "Gender",
                ("Male", "Female"),
                label_visibility='collapsed',
            )

            age = st.slider(
                "Age",
                min_value=18,
                max_value=65,
                value=30,
                step=1
            )

            Occupation = st.selectbox(
                "Occupation",
                (df['Occupation'].unique()),
                label_visibility='collapsed',
            )

            sleepDuration = st.slider(
                "Sleep Duration (Hours)",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.1,
                format="%.1f"
            )

            sleepQuality = st.slider(
                "Sleep Quality (1-10)",
                min_value=0,
                max_value=10,
                value=3,
                step=1
            )

            physicalActivity = st.slider(
                "Physical Activity Level (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                format="%.1f"
            )

        with col2:
            bmi = st.selectbox(
                "BMI Category",
                (df['BMI Category'].unique()),
                label_visibility='collapsed',
            )
            restingHeartRate = st.slider(
                "Resting Heart Rate",
                min_value=60.0,
                max_value=120.0,
                value=60.0,
                step=0.1,
                format="%.1f"
            )

            sleepDisorder = st.selectbox(
                "Sleep Disorder",
                (df['Sleep Disorder'].unique()),
                label_visibility='collapsed',
            )

            dailySteps = st.slider(
                "Daily Steps",
                min_value=0,
                max_value=10000,
                value=5000,
                step=1
            )

            bloodPressureHigh = st.slider(
                "High Blood Pressure",
                min_value=90.0,
                max_value=180.0,
                value=120.0,
                step=0.1,
                format="%.1f"
            )

            bloodPressureLow = st.slider(
                "Low Blood Pressure",
                min_value=50.0,
                max_value=120.0,
                value=80.0,
                step=0.1,
                format="%.1f"
            )


    with result:
        st.markdown(
            """
                <h3 style="color:#176397" align="center">Prediction</h3>
            """,unsafe_allow_html=True
        )

        def get_user_input():
            """
            The function `get_user_input()` collects various user inputs related to health and returns
            them as a dictionary.
            """

            prediction = {
            'Gender': Gender,
            'Age': age,
            'Occupation': Occupation,
            'Sleep Duration': sleepDuration,
            'Quality of Sleep': sleepQuality,
            'Physical Activity Level': physicalActivity,
            'BMI Category': bmi,
            'Heart Rate': restingHeartRate,
            'Daily Steps': dailySteps,
            'Sleep Disorder': sleepDisorder,
            'BP High': bloodPressureHigh,
            'BP Low': bloodPressureLow
            }

            prediction = pd.DataFrame(prediction, index=[0])
            
            return prediction

        prediction = get_user_input()

        prediction['Gender'] = gender_le.transform(prediction['Gender'])
        prediction['Occupation'] = occupation_le.transform(prediction['Occupation'])
        prediction['BMI Category'] = bmiCategory_le.transform(prediction['BMI Category'])
        prediction['Sleep Disorder'] = sleepDisorder_le.transform(prediction['Sleep Disorder'])

        numerical_features = ['Age',
                            'Sleep Duration',
                            'Quality of Sleep',
                            'Physical Activity Level',
                            'Heart Rate',
                            'Daily Steps',
                            'BP High',
                            'BP Low']

        prediction[numerical_features] = scaler.transform(prediction[numerical_features])

        with st.spinner('Wait for prediction...'):
            model = load_model()
            y_pred = model.predict(prediction)
            time.sleep(1)
            
        st.markdown(
        f"""
        <p align="center">
            The predicted stress level based on your selections on the left-hand side is as follows.
        </p>
        <div class="kpi-container1">
            <div class="kpi-box1">
                <span class="kpi-title1">Stress Level</span>
                <span class="kpi-value1">{np.round(y_pred[0],2)}/10</span>
            </div>
        </div>

        <style>
        .kpi-container1 {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 10vh;
        }}

        .kpi-box1 {{
        flex-grow: 0.25;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 0px;
        border-radius: 10px;
        background-color: #F2F2F2;
        transition: box-shadow 0.3s;
        }}

        .kpi-box:hover1 {{
        box-shadow: 0 0 10px #1D4665;
        }}

        .kpi-title1 {{
        color: #176397;
        font-size: 14px;
        font-weight: bold;
        font-family: 'Poppins', sans-serif;
        margin-bottom: 0px;
        }}

        .kpi-value1 {{
        color: #1D4665;
        font-size: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True)       

    st.write('---') 


if selected == "Contact":

    html_path = 'C:\\Users\\sound\\OneDrive\\Desktop\\Byte Crafter\\Sleep Health Prediction\\Sleep Health Prediction\\Contact_page\\index.html'
    css_path ='C:\\Users\\sound\\OneDrive\\Desktop\\Byte Crafter\\Sleep Health Prediction\\Sleep Health Prediction\\Contact_page\\style.css' 

    def get_contact_page(css_path, html_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            contact_page_css = f.read()
        
        with open(html_path, 'r', encoding='utf-8') as f:
            contact_page = f.read()

        return contact_page_css, contact_page

    contact_page, contact_css_file = get_contact_page(html_path, css_path)

    st.markdown(
        f"""
        <style>
            {contact_css_file}
        </style>
        {contact_page}
        """,
        unsafe_allow_html=True
    )
if selected == "Chatbot":
    st.markdown(
        """
        <h2 style="color:#176397">Chatbot Assistant</h2>
        <p style="color:#1D4665">Chat with our assistant to get more information about sleep health and stress level prediction.</p>
        """,
        unsafe_allow_html=True
    )

    user_input = st.text_input("Ask the chatbot:")
    if st.button("Send"):
        response = genai_engine(user_input)
        st.text_area("Chatbot response:", value=response, height=200)
