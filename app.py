import streamlit as st
import pandas as pd
import numpy as np
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from pycaret.regression import *






route = np.random.choice(['Munich East', 'Munich North', 'FTMRoute (2x)', 'FTMRoute',
       'Munich Northeast', 'Highway', 'Munich South', 'FTMRoute reverse'])
Weather = np.random.choice(['sunny', 'slightly cloudy', 'rainy', 'cloudy',
       'dark, little rainy', 'dark', 'sunrise', 'sunset'])
route_int= ['FTMRoute','FTMRoute (2x)','FTMRoute reverse','Highway','Munich East','Munich North','Munich Northeast','Munich South'].index(route)
Weather_int = ['cloudy','dark','dark, little rainy', 'rainy', 'slightly cloudy', 'sunny','sunrise','sunset'].index(Weather)

#current_model = load_model('current_model')
voltage_model = load_model('voltate_model')
trottle_model = load_model('trottle_model')
torque_model = load_model('torque_model')
Distance_model = load_model('Distance_model')
bat_drain_model = load_model('battery_drained_model')
temp_model = load_model('temperature_model')

charge = 100


st.set_page_config(
    page_title="EV Real-Time Monitoring Dashboard",
    layout="wide",
)

st.title('EV Real-Time Monitoring Dashboard')

df = px.data.gapminder()

st.markdown(
    """
    <style>
    .stButton>button {
        float: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)


tab1, tab2 = st.tabs(["Distance Predictor", "Battery Temperature Maintanance"])

def get_chart(x):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = x,
        title = {'text': "Speed"},
        gauge = {
            'axis': {'range': [None, 300]},
            'steps' : [
                {'range': [0, 150], 'color': "lightgray"},
                {'range': [150, 250], 'color': "gray"}],
            'threshold' : {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 250}}))
    fig.update_layout(height=400, width=400)

    # Display the chart in Streamlit
    return fig

with tab1:

    st.header('Range of Vechicle with different Speed')
    col1, col2, col3 = st.columns(3)
    with col1:
        
        txt = 'Route: '+ route
        txt = f"<div style='text-align: center; font-size: 24px;'>{txt}</div>"
        st.write(txt, unsafe_allow_html=True)
        fig = get_chart(60)
        st.plotly_chart(fig)
        data=pd.DataFrame.from_dict({'Route/Area': [route_int],'Weather': [Weather_int],'Battery Drained': [1],'Speed [km/h]': [60]})
        dis = predict_model(Distance_model, data=data)
        txt = 'Distance: '+ str(dis['prediction_label'][0])
        txt = f"<div style='text-align: center; font-size: 30px;'>{txt}</div>"
        st.write(txt, unsafe_allow_html=True)

    with col2:
        txt = 'Weather: '+ Weather
        txt = f"<div style='text-align: center; font-size: 24px;'>{txt}</div>"
        st.write(txt, unsafe_allow_html=True)
        fig = get_chart(150)
        st.plotly_chart(fig)
        data=pd.DataFrame.from_dict({'Route/Area': [route_int],'Weather': [Weather_int],'Battery Drained': [1],'Speed [km/h]': [150]})
        dis = predict_model(Distance_model, data=data)
        txt = 'Distance: '+ str(dis['prediction_label'][0])
        txt = f"<div style='text-align: center; font-size: 30px;'>{txt}</div>"
        st.write(txt, unsafe_allow_html=True)

    with col3: 
        txt = 'Charge: ' + str(charge)
        txt = f"<div style='text-align: center; font-size: 24px;'>{txt}</div>"
        st.write(txt, unsafe_allow_html=True)
        fig = get_chart(230)
        st.plotly_chart(fig)
        data=pd.DataFrame.from_dict({'Route/Area': [route_int],'Weather': [Weather_int],'Battery Drained': [1],'Speed [km/h]': [230]})
        dis = predict_model(Distance_model, data=data)
        txt = 'Distance: '+ str(dis['prediction_label'][0])
        txt = f"<div style='text-align: center; font-size: 30px;'>{txt}</div>"
        st.write(txt, unsafe_allow_html=True)


    col1, col2 = st.columns(2)

    with col1:
        speed1 = st.slider('Speed', 0, 300, 50)
        route1 = st.selectbox('Route', ['Munich East', 'Munich North', 'FTMRoute (2x)', 'FTMRoute',
        'Munich Northeast', 'Highway', 'Munich South', 'FTMRoute reverse'])
    with col2:
        charge1 = st.slider('Charge', 0, 100, 100)
        Weather1 = st.selectbox('Weather', ['sunny', 'slightly cloudy', 'rainy', 'cloudy',
        'dark, little rainy', 'dark', 'sunrise', 'sunset'])
    
    route_int1= ['FTMRoute','FTMRoute (2x)','FTMRoute reverse','Highway','Munich East','Munich North','Munich Northeast','Munich South'].index(route1)
    Weather_int1 = ['cloudy','dark','dark, little rainy', 'rainy', 'slightly cloudy', 'sunny','sunrise','sunset'].index(Weather1)

    
    data=pd.DataFrame.from_dict({'Route/Area': [route_int1],'Weather': [Weather_int1],'Battery Drained': [charge1/100],'Speed [km/h]': [speed1]})
    dis = predict_model(Distance_model, data=data)
    txt = 'Distance: '+ str(dis['prediction_label'][0])
    txt = f"<div style='text-align: center; font-size: 40px;'>{txt}</div>"
    st.write(txt, unsafe_allow_html=True)



with tab2:

    # Create an empty dataframe
    df1 = pd.DataFrame(columns=['Time [s]', 'Velocity [km/h]', 'Elevation [m]', 'Throttle [%]',
       'Motor Torque [Nm]', 'Battery Voltage [V]', 'Battery Current [A]',
       'Battery Temperature [°C]', 'Route/Area', 'Weather', 'Battery Drained',  'Acceleration [m/s^2]', 'Distance [m]'])
    ele_fact = 0
    e = 0

    def get_elevation():
        global e
        global ele_fact
        ele = 0
        if df1.iloc[-1]['Time [s]'] - e > np.random.randint(50,200):
            e = df1.iloc[-1]['Time [s]']
            ele_fact = np.random.randint(-30, 50)
        if ele_fact != 0:
            if ele_fact > 0:
                ele = df1.iloc[-1]['Elevation [m]'] + 1
                ele_fac = ele_fac - 1
            else:
                ele = df1.iloc[-1]['Elevation [m]'] - 1
                ele_fac = ele_fac + 1
        return ele

    def get_acceleration():
        if df1.iloc[-1]['Acceleration [m/s^2]'] <= 30:
            Acc = np.random.randint(2, 9)

        elif df1.iloc[-1]['Acceleration [m/s^2]'] > 30 and df1.iloc[-1]['Acceleration [m/s^2]'] <= 60:
            Acc = np.random.randint(-2,6)

        elif df1.iloc[-1]['Acceleration [m/s^2]'] > 60 and df1.iloc[-1]['Acceleration [m/s^2]'] <= 90:
            Acc = np.random.randint(-4, 2) 

        else:
            Acc = np.random.randint(-6, 0)
        return Acc
    
    # Define a function to generate data every second
    def generate_data():
        global df1
        if df1.shape[0] == 0:
            time_s = 0
            Velocity_kmh = 0
            elevation_m = 530
            Throttle = 0
            Motor_Torque = 11.000503	
            Battery_Voltage = 376.866195
            Battery_Current = -20.379267
            bat_temp = list(predict_model(temp_model,
                                        data = pd.DataFrame({'Time [s]': [time_s],
                                                        'Velocity [km/h]': [Velocity_kmh],
                                                        'Elevation [m]': [elevation_m],
                                                        'Throttle [%]': [Throttle],
                                                        'Motor Torque [Nm]': [Motor_Torque],
                                                        'Battery Voltage [V]': [Battery_Voltage],
                                                        'Battery Current [A]': [Battery_Current],
                                                        'Route/Area': [route],
                                                        'Weather': [Weather],
                                                        'Battery Drained': [0],
                                                        'Distance [m]': [0],
                                                        'Acceleration [m/s^2]': [0]}))['prediction_label'])[0] + 12

            df1.loc[len(df1)] = [time_s, Velocity_kmh, elevation_m, Throttle, Motor_Torque, Battery_Voltage, Battery_Current, bat_temp, route, Weather, 0, 0, 0]
        
        else:
            time_s = df1.iloc[-1]['Time [s]'] + 1
            print(df1.iloc[-1]['Battery Temperature [°C]'])
            if df1.iloc[-1]['Battery Temperature [°C]'] > 31:
                Acceleration = np.random.randint(-2,0)
            else:
                Acceleration = get_acceleration()
            Velocity_kmh = df1.iloc[-1]['Velocity [km/h]'] + Acceleration
            distance_m = df1.iloc[-1]['Velocity [km/h]'] + Velocity_kmh*0.27777778
            elevation_m = get_elevation()
            Battery_drained = list(predict_model(bat_drain_model,
                                            data = pd.DataFrame({'Route/Area':[route_int], 
                                                                    'Weather': [Weather_int], 
                                                                    'Distance [km]': [distance_m/1000],
                                                                    'Duration [min]': [time_s/60], 
                                                                    'Speed [km/h]': [Velocity_kmh]}))['prediction_label'])[0]
            Throttle = list(predict_model(trottle_model,data = pd.DataFrame({'Time [s]': [time_s], 
                                                                            'Velocity [km/h]': [Velocity_kmh], 
                                                                            'Elevation [m]': [elevation_m],
                                                                            'Route/Area': [route], 
                                                                            'Weather': [Weather], 
                                                                            'Battery Drained':[Battery_drained], 
                                                                            'Distance [m]': [distance_m],
                                                                            'Acceleration [m/s^2]': [Acceleration]}))['prediction_label'])[0]
            Motor_Torque = 	list(predict_model(torque_model,data = pd.DataFrame({'Time [s]': [time_s], 
                                                                            'Velocity [km/h]': [Velocity_kmh], 
                                                                            'Elevation [m]': [elevation_m],
                                                                            'Route/Area': [route], 
                                                                            'Weather': [Weather], 
                                                                            'Battery Drained':[Battery_drained], 
                                                                            'Distance [m]': [distance_m],
                                                                            'Acceleration [m/s^2]': [Acceleration]}))['prediction_label'])[0]

            Battery_Voltage = list(predict_model(voltage_model,data = pd.DataFrame({'Time [s]': [time_s], 
                                                                            'Velocity [km/h]': [Velocity_kmh], 
                                                                            'Elevation [m]': [elevation_m],
                                                                            'Route/Area': [route], 
                                                                            'Weather': [Weather], 
                                                                            'Battery Drained':[Battery_drained], 
                                                                            'Distance [m]': [distance_m],
                                                                            'Acceleration [m/s^2]': [Acceleration]}))['prediction_label'])[0]

            Battery_Current = np.random.randint(-397,30)

            bat_temp = list(predict_model(temp_model,
                                        data = pd.DataFrame({'Time [s]': [time_s],
                                                        'Velocity [km/h]': [Velocity_kmh],
                                                        'Elevation [m]': [elevation_m],
                                                        'Throttle [%]': [Throttle],
                                                        'Motor Torque [Nm]': [Motor_Torque],
                                                        'Battery Voltage [V]': [Battery_Voltage],
                                                        'Battery Current [A]': [Battery_Current],
                                                        'Route/Area': [route],
                                                        'Weather': [Weather],
                                                        'Battery Drained': [0],
                                                        'Distance [m]': [0],
                                                        'Acceleration [m/s^2]': [Acceleration]}))['prediction_label'])[0]+12
            df1.loc[len(df1)] = [time_s, Velocity_kmh, elevation_m, Throttle, Motor_Torque, Battery_Voltage, Battery_Current, bat_temp, route, Weather, Battery_drained, Acceleration, distance_m]
        # Wait for 1 second before generating new data

    # Define a function to display the dataframe and char


    # Create a button to start generating data
    
        # Display the dataframe and chart in real-time
        
        

    
    if st.button('Start '): 
        col1, col2 = st.columns(2)
        with col1:
            fig1 = st.line_chart(df1['Velocity [km/h]'])
            fig2 = st.line_chart(df1['Distance [m]'])
        with col2:
            fig3 = st.line_chart(df1['Acceleration [m/s^2]'])
            fig4 = st.line_chart(df1['Battery Current [A]'])

        fig5 = st.line_chart(df1['Battery Temperature [°C]'])
        
        if "min_temp" not in st.session_state:
            st.session_state["min_temp"] = 0
        if "max_temp" not in st.session_state:
            st.session_state["max_temp"] = 0
        if "mean_temp" not in st.session_state:
            st.session_state["mean_temp"] = 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Minimum Temperature:", st.session_state["min_temp"])
        with col2:
            st.write("Average Temperature:", st.session_state["mean_temp"])
        with col3:
            st.write("Maximum Temperature:", st.session_state["max_temp"])
        # Start generating data in a separate thread
        
        

        while True:
            generate_data()
            fig1.add_rows(pd.DataFrame({
                'Velocity [km/h]': [df1.iloc[-1]['Velocity [km/h]']]
            }))
            fig2.add_rows(pd.DataFrame({
                'Distance [m]': [int(df1.iloc[-1]['Distance [m]'])]
            }))
            fig3.add_rows(pd.DataFrame({
                'Acceleration [m/s^2]': [df1.iloc[-1]['Acceleration [m/s^2]']]
            }))
            fig4.add_rows(pd.DataFrame({
                'Battery Current [A]': [df1.iloc[-1]['Battery Current [A]']]
            }))
            fig5.add_rows(pd.DataFrame({
                'Battery Temperature [°C]': [df1.iloc[-1]['Battery Temperature [°C]']]
            }))
            st.session_state["min_temp"] = min(df1['Battery Temperature [°C]'])
            st.session_state["max_temp"] = max(df1['Battery Temperature [°C]'])
            st.session_state["mean_temp"] = df1['Battery Temperature [°C]'].mean()
            time.sleep(2)
            if df1.shape[0] > 40:
                break
            