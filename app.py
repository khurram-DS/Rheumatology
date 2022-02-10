import streamlit as st
# Eda packages

import pandas as pd
import numpy as np

#Data viz packages

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

#function

def main():
    
    title_container1 = st.container()
    col1, col2 ,  = st.columns([6,12])
    from PIL import Image
    image = Image.open('static/asia.jpeg')
    with title_container1:
        with col1:
            st.image(image, width=200)
        with col2:
            st.markdown('<h1 style="color: tomato;">ASIA Consulting</h1>',
                           unsafe_allow_html=True)
    
    st.subheader('Rheumatology Project - Al-Amiri Hospital - Kuwait ')
    st.subheader("ACTEMRA")
    st.markdown("""**Dr. Adeeba Al-Herz**""")
    new_title4 = '<p style="font-family:sans-serif; color:black; font-size: 14 px;">Rheumatology Unit, Department of Internal Medicine, Al-Amiri Hospital - Kuwait <a  href=" mailto:adeebaalherz@yahoo.com"> Email </a> </p> '            
    st.markdown(new_title4, unsafe_allow_html=True)
    st.markdown("""**Ahmad Al-Saber (Statistician)**""")
    new_title5 = '<p style="font-family:sans-serif; color:black; font-size: 14 px;">University of Strathclyde/UK <a  href="admin@acs-kw.com"> Email </a> </p> ' 
    st.markdown(new_title5, unsafe_allow_html=True)
    
    
    
    st.sidebar.image("static/rhe.jpg", use_column_width=True)
    activites = ["About","Data Analysis"]
    choice =st.sidebar.selectbox("Select Activity",activites)
    
    @st.cache(allow_output_mutation=True)
    def get_df(file):
      # get extension and read file
      extension = file.name.split('.')[1]
      if extension.upper() == 'CSV':
        df = pd.read_csv(file,parse_dates = ["Entry Date"],error_bad_lines=False)
      elif extension.upper() == 'XLSX':
        df = pd.read_excel(file,parse_dates = ["Entry Date"],error_bad_lines=False)
      
      return df
    file = st.file_uploader("Upload file", type=['csv' 
                                             ,'xlsx'])
    if not file:
        st.write("Upload a .csv or .xlsx file to get started")
        return
      
    df = get_df(file)
    st.write("**Data has been loaded Successfully**")


    if choice == 'About':
        st.subheader("About Al-Amiri Hospital")
        title_container1 = st.container()
        col1, col2 ,  = st.columns([4,12])
        from PIL import Image
        image = Image.open('static/Logo.png')
        with title_container1:
            
            with col2:
                st.image(image, width=350, caption='Al-Amiri Hospital')
            
        st.markdown("""**❓ What is rheumatology?**
                    
Rheumatic diseases are autoimmune and inflammatory diseases that cause your immune system to attack your joints, muscles, bones and organs. Rheumatic diseases are often grouped under the term “arthritis” — which is used to describe over 100 diseases and conditions. This does not include the most common form of arthritis, known as osteoarthritis, which results in a breakdown of bone and cartilage in joints rather than inflammation.
Rheumatic diseases can cause damage to your vital organs, including the lungs, heart, nervous system, kidneys, skin and eyes. Rheumatic diseases may result in conditions so severe that those who suffer from them cannot bathe or dress themselves. Additionally, a simple task such as walking can cause pain and be difficult or even impossible.""")
        
        title_container1 = st.container()
        col1, col2 ,  = st.columns([12,12])
        from PIL import Image
        image = Image.open('static/rheu.jpg')
        with title_container1:
            with col1:
                st.image(image, width=350, caption='Rheumatic Disease')
                
            with col2:
                st.image("https://ml3avapoifb3.i.optimole.com/be2GvoQ-GrjsfN49/w:auto/h:auto/q:100/https://goodhealthwecare.org/wp-content/uploads/2020/03/stagesofra_gif.gif", width=385, caption='Rheumatic Disease gif')

        st.markdown("Osteoarthritis, the most common form of arthritis, involves the wearing away of the cartilage that caps the bones in your joints. With rheumatoid arthritis, the synovial membrane that protects and lubricates joints becomes inflamed, causing pain and swelling. Joint erosion may follow.")
        
        st.markdown("""**❓ Why would you need to see a rheumatologist?**

Muscle and joint pain are common, but see a primary care physician if you have pain that lasts for more than a few days.

A doctor can evaluate whether you’re experiencing temporary pain from an injury or other inflammatory causes. They can also refer you to a rheumatologist if needed.

If your pain worsens over a short time, you should see a rheumatologist.

Also, if your symptoms decrease with initial treatment, such as with pain medication, but return once the treatment stops, you might need a specialist.
""")
        st.markdown("""**➡️➡️ You may want to see a rheumatologist if you:**

- experience pain in multiple joints
- have new joint pain not related to a known injury
- have joint or muscle pain accompanied by fever, fatigue, rashes, morning stiffness, or chest pain
- have muscle pain with or without other symptoms
- are over 50 years old and have recurring headaches or muscle aches
- have a chronic illness without a unifying diagnosis
Many rheumatic conditions are hereditary, so let your doctor and rheumatologist know if you have any family history of:

- autoimmune disease
- rheumatic disease
- cancer
Don’t delay seeking treatment if you have persistent joint, bone, or muscle pain. A doctor should evaluate joint stiffness that lasts more than 30 minutes, especially if it’s worse in the morning after long periods of inactivity or any joint swelling.

Rheumatic diseases can lead to permanent damage over time if not addressed promptly. Outcomes improve when these conditions are treated earlier, even for chronic and progressive diseases.                    
        """)
        st.markdown("""**🕮🕮  What’s the difference between a rheumatologist and an orthopedist?**
Rheumatologists and orthopedists both treat rheumatic diseases, but in different ways.

Generally, rheumatologists treat rheumatic diseases with nonsurgical interventions, whereas orthopedists perform surgeries to improve function and quality of life.

You may want to see an orthopedist if you need a joint replacement or have:

- joint or musculoskeletal pain related to an injury
- hip or knee pain that gets worse when you put weight on these joints
- severe joint pain that interferes with your daily life
- moderate or advanced arthritis in your hips or knees
- joint pain that hasn’t responded to previous treatment
- A good rule of thumb: Unless you’ve experienced a traumatic injury that requires surgery, see a rheumatologist before you consult an orthopedist.                    
                    

                    """)

        st.subheader("About this Research Analysis")
        title_container1 = st.container()
        col1, col2 ,  = st.columns([4,12])
        from PIL import Image
        #image = Image.open('static/Logo.png')
        with title_container1:
            
            with col2:
                st.image("https://www.gilead.com/-/media/gilead-corporate/images/rsp/gilead_rsp_landingpage_cta.gif?h=400&w=425&la=en&hash=5DF912BA0A01DAB09E7B75B866DCF987", width=350, caption='Data Analysis')

        st.markdown("""For this research we have recorded the data of our patient visit to Al-Amiri Hospital and there problems regarding this disease,
                    we have recorded down all the data of visit and did the analysis on that, currently we have 12024 patient data available for this research. """)

        st.text('© ASIA Consulting 2022')

    if choice == 'Data Analysis':
        st.subheader("Data analysis")
        title_container1 = st.container()
        col1, col2 ,  = st.columns([4,12])
        from PIL import Image
        #image = Image.open('static/Logo.png')
        with title_container1:
            
            with col2:
                
                st.image("https://i.pinimg.com/originals/b8/23/e3/b823e38cc01fdb9278b6f7faa2feda6d.gif", width=350, caption='Data Analysis')
        if st.checkbox('Show Raw Data'):
            st.subheader('Raw Data')
            st.write(df.sample(5))
            
            if st.checkbox("Select Columns to see frequency table"):
                all_columns=df.columns.to_list()
                selected_columns= st.multiselect("Select Columns to see Counts and frequency", all_columns)
                cnt=df[selected_columns].value_counts()
                per=df[selected_columns].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            
                freq=pd.DataFrame({'counts': cnt,'Frequency %': per})
                freq.reset_index(inplace = True)
                #freq.rename(columns={'index':'Course'},inplace=True)
                freq['Total_data']=freq['counts'].sum()
                st.dataframe(freq)
            
                st.text("Download the Above Data table by clicking on Download CSV")
                
                st.download_button(label='Download CSV',data=freq.to_csv(),mime='text/csv')
                st.markdown("*****************************************************************")
            
        if st.checkbox("Show Patients Analysis"):
            st.subheader("Patients Analysis")
            st.markdown("**1- IDD Analysis**")
            IDD = df.IDD
            counts = IDD.value_counts()
            percent = IDD.value_counts(normalize=True)
            percent100 = IDD.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            IDD=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            IDD.reset_index(inplace = True)
            IDD.rename(columns={'index':'Visit type'},inplace=True)
            IDD['No_of_total_data']=IDD['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : #f95f5f "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:1,:] = r
                temp_df.iloc[1:2,:] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(IDD.style.apply(highlight_col, axis=None))
            st.markdown("- The above table showing the number and the percentages of patients splitted into first (initial) visit and his remaining visits:")

            import plotly.express as px
            fig = px.bar(IDD, x="Visit type", y="counts", color="Frequency")
            st.write(fig)
            
            remain=IDD['counts'][:1].to_string(index=False)
            initial=IDD['counts'][1:].to_string(index=False)
            remain_per=IDD['Frequency'][:1].to_string(index=False)
            initial_per=IDD['Frequency'][1:].to_string(index=False)
            count=IDD['No_of_total_data'][:1].to_string(index=False)
            
            
            st.markdown("""
                                                  **IDD_Statistics**
                   
                   1. Here we can observe that the Initial visit with the counts of [{}] out of [{}] counts
                      With percentage of [{}]
                      
                   2. Here we can observe that the Remaining visit with the counts of [{}] out of [{}] counts
                      With percentage of [{}]
                   
            
                   """.format(initial,count,initial_per,remain,count,remain_per))
            st.markdown("*****************************************************************")
            
            
            st.markdown("**2- Gender Analysis**")
            gen = df.Gender
            counts = gen.value_counts()
            percent = gen.value_counts(normalize=True)
            percent100 = gen.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            gen=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            gen.reset_index(inplace = True)
            gen.rename(columns={'index':'Gender'},inplace=True)
            gen['No_of_total_data']=gen['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : skyblue "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:1,:] = r
                temp_df.iloc[1:2,:] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(gen.style.apply(highlight_col, axis=None))
            
            st.markdown("- The above table showing the number and the percentage of Gender")
            import plotly.express as px
            fig = px.bar(gen, x="Gender", y="counts", color="Frequency")
            st.write(fig)
            
            female=gen['counts'][:1].to_string(index=False)
            male=gen['counts'][1:].to_string(index=False)
            female_per=gen['Frequency'][:1].to_string(index=False)
            male_per=gen['Frequency'][1:].to_string(index=False)
            count=gen['No_of_total_data'][:1].to_string(index=False)
            
            
            st.markdown("""
                                                  **Gender_Statistics**
                   
                   1. Here we can observe that the Female with the counts of [{}] out of [{}] counts
                      With percentage of [{}]
                      
                   2. Here we can observe that the Male with the counts of [{}] out of [{}] counts
                      With percentage of [{}]
                   
            
                   """.format(female,count,female_per,male,count,male_per))
            st.markdown("*****************************************************************")
            
        if st.checkbox("Show Age Based Analysis"):
            st.markdown("**Age_years Analysis**")
            st.markdown('**Summary Statistics**')
            minimum= df.age_years.min().round(2)
            Average= df.age_years.mean().round(2)
            maximum= df.age_years.max().round(2)
            std=df.age_years.std().round(2)
            counts = df.age_years.value_counts().sum()
            st.write(df.age_years.describe())
            
            
            import plotly.express as px
            
            fig2 = px.histogram(df, x="age_years",
                   
                   labels={'age_years':'age_years'}, # can specify one label per df column
                   opacity=0.8,
                   
                   color_discrete_sequence=['indianred'] # color of histogram bars
                   )
            fig2.update_layout(
                        title={
                                    'text': """Age_years_histogram with mean= 53.950877, std=12.454308""",
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'})
            st.markdown("**1- Histogram of age_years**")
            st.write(fig2)
            
            st.markdown("**2- Distplot of age_years**")
            

            import plotly.figure_factory as ff
            import numpy as np
            from plotly.offline import init_notebook_mode, iplot
            import plotly.figure_factory as ff
            import cufflinks
            cufflinks.go_offline()
            cufflinks.set_config_file(world_readable=True, theme='pearl')
            import plotly.graph_objs as go
            from chart_studio import plotly as py
            import plotly
            from plotly import tools
            x = df['age_years'].dropna(how='any',axis=0)
            hist_data = [x]
            group_labels = ['Distplot for age_years'] # name of the dataset
            colors = ['orangered']
            fig3 = ff.create_distplot(hist_data, group_labels,bin_size=1, colors=colors)
            fig3.update_layout(title_text='Distplot for age_years')
            fig3.update_layout(
            autosize=False,
            width=850,
            height=550
            )
            fig3.update_xaxes(title_text='age_years')
            fig3.update_yaxes(title_text='Density')
            
            st.write(fig3)
            
            
            
            
            
            
            st.markdown('**3- Box plot**')
            st.text('Box Plot with Displaying Underlying Data')
            import plotly.graph_objects as go

            fig5 = go.Figure(data=[go.Box(y=df['age_years'],
            boxpoints='all', # can also be outliers, or suspectedoutliers, or False
            jitter=0.3, # add some jitter for a better separation between points
            pointpos=-1.8, # relative position of points wrt box
            name='Age in years',
            fillcolor='pink',
            
              )])
    
            fig5.update_layout(title_text='Boxplot for Age in years')
            st.write(fig5)
            #cdf plot
            
            st.markdown('**4- CDF plot**')
            import plotly.express as px
            fig9 = px.ecdf(df, x="age_years",
                  title='CDF of age_years',
                   labels={'age_years':'age_years'}, # can specify one label per df column
                   opacity=0.8,
                   
                   color_discrete_sequence=['seagreen'] # color of histogram bars
                   )
            st.write(fig9)
            
            # plotting q-q plot
            st.markdown('**5- Q-Q plot**')
            st.text('plot for checking the distribution using Q-Q plot')
            st.text('We can also observe the distribution difference by clicking')
            from statsmodels.graphics.gofplots import qqplot

            qqplot_data = qqplot(df.age_years, line='s').gca().lines
            fig = go.Figure()

            fig.add_trace({
                'type': 'scatter',
                'x': qqplot_data[0].get_xdata(),
                'y': qqplot_data[0].get_ydata(),
                'mode': 'markers',
                'marker': {
                    'color': '#19d3f3'
                }
            })
            
            fig.add_trace({
                'type': 'scatter',
                'x': qqplot_data[1].get_xdata(),
                'y': qqplot_data[1].get_ydata(),
                'mode': 'lines',
                'line': {
                    'color': '#636efa'
                }
            
            })
            
            
            fig['layout'].update({
                'title': 'Quantile-Quantile Plot',
                'xaxis': {
                    'title': 'Theoritical Quantities',
                    'zeroline': False
                },
                'yaxis': {
                    'title': 'Sample Quantities'
                },
                'showlegend': False,
                'width': 600,
                'height': 500,
            })
            
            st.write(fig)
            
           
            st.markdown('**Checking the Normality of Data**')
            st.text(' Checking normality of data for age_years using shapiro test')
            from scipy.stats import shapiro
            stat, p = shapiro(df.age_years)
                
                # interpret
            alpha = 0.05
            if p > alpha:
                msg = 'Sample looks Gaussian (fail to reject H0)'
            else:
                msg = 'Sample does not look Gaussian (reject H0)'
                
            result_mat = [
                ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                [len(df.age_years), stat, p, msg]
            ]
            import plotly.figure_factory as ff
            swt_table = ff.create_table(result_mat)
            swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
            swt_table['layout']['height']=200
            swt_table['layout']['margin']['t']=50
            swt_table['layout']['margin']['b']=50
                
            #py.iplot(swt_table, filename='shapiro-wilk-table')
            st.write(swt_table)
            
            st.markdown("""
                                                  **Age_years_Statistics**
                   
                                       
                           
                           1. Here we can see that the minimum Age of Patient is [{}]
                           2. where as the Average Age of Patient is [{}]
                           3. And the maximum Age of Patient is [{}]
                           4. out of the total count of Patient [{}]
                              
                                      with Standard deviation of [{}]
                           
                           Shapirowilks statistics  [{}]  &  p-value  [{}]
                    
                           """.format(minimum,Average,maximum,counts,std,stat,p))
            st.markdown("*****************************************************************")

            st.markdown("**🔍 Spliited the Data based on Year gap of 20 years**")
            import numpy as np
            def age_year(x):
                if x <= 20:
                    return "Age Below 20"
                elif x<= 40:
                    return "Age Below 40"
                elif x <= 60:
                    return "Age Below 60"
                elif x<=80:
                    return "Age Below 80"
                elif x<= 100:
                    return "Age Below 100"
                elif x> 100:
                    return "Age Above 100"
                else:
                    return np.NaN 
            
            age_data=pd.DataFrame()
            
            age_data['age_year'] = round(df["age_years"])
            
            age_data['Age_Gap']= df['age_years'].apply(lambda x: age_year(x))
            
            age = age_data.Age_Gap
            counts = age.value_counts()
            percent = age.value_counts(normalize=True)
            percent100 = age.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            age=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            age.reset_index(inplace = True)
            age.rename(columns={'index':'Age'},inplace=True)
            age['No_of_total_data']=age['counts'].sum()
            st.markdown("Different age group analysis")
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : skyblue "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:,0] = r
                temp_df.iloc[:,2] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(age.style.apply(highlight_col, axis=None))
            
            import plotly.express as px
            fig = px.bar(age, x="Age", y="counts", color="Frequency")
            st.write(fig)
            
            
            st.markdown("*****************************************************************")
            
        if st.checkbox("Show Age Diagnosis of patient"):
            st.markdown("**Age_diagnosis Analysis**")
            st.markdown("**Summary Statistics**")
            minimum= df.age_Diagnosis.min().round(2)
            Average= df.age_Diagnosis.mean().round(2)
            maximum= df.age_Diagnosis.max().round(2)
            std=df.age_Diagnosis.std().round(2)
            counts = df.age_Diagnosis.value_counts().sum()
            st.write(df.age_Diagnosis.describe())
            
            import plotly.express as px
            
            fig5 = px.histogram(df, x="age_Diagnosis",
                   
                   labels={'age_Diagnosis':'age_Diagnosis'}, # can specify one label per df column
                   opacity=0.8,
                   
                   color_discrete_sequence=['indianred'] # color of histogram bars
                   )
            fig5.update_layout(
                        title={
                                    'text': """Age_Diagnosis_histogram with mean= 11.21, std=6.71""",
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'})
            st.markdown("**1- Histogram of age_Diagnosis**")
            st.write(fig5)
            
            st.markdown("**2- Distplot of age_Diagnosis**")
            

            import plotly.figure_factory as ff
            import numpy as np
            from plotly.offline import init_notebook_mode, iplot
            import plotly.figure_factory as ff
            import cufflinks
            cufflinks.go_offline()
            cufflinks.set_config_file(world_readable=True, theme='pearl')
            import plotly.graph_objs as go
            from chart_studio import plotly as py
            import plotly
            from plotly import tools
            x = df['age_Diagnosis'].dropna(how='any',axis=0)
            hist_data = [x]
            group_labels = ['Distplot for age_Diagnosis'] # name of the dataset
            colors = ['orangered']
            fig6 = ff.create_distplot(hist_data, group_labels,bin_size=1, colors=colors)
            fig6.update_layout(title_text='Distplot for age_Diagnosis')
            fig6.update_layout(
            autosize=False,
            width=850,
            height=550
            )
            fig6.update_xaxes(title_text='age_Diagnosis')
            fig6.update_yaxes(title_text='Density')
            
            st.write(fig6)
            
            st.markdown('**3- Box plot**')
            st.text('Box Plot with Displaying Underlying Data')
            import plotly.graph_objects as go

            fig5 = go.Figure(data=[go.Box(y=df['age_Diagnosis'],
            boxpoints='all', # can also be outliers, or suspectedoutliers, or False
            jitter=0.3, # add some jitter for a better separation between points
            pointpos=-1.8, # relative position of points wrt box
            name='age_Diagnosis',
            fillcolor='pink',
            
              )])
    
            fig5.update_layout(title_text='Boxplot for age_Diagnosis in years')
            st.write(fig5)
            #cdf plot
            st.markdown('**4- CDF plot**')
            import plotly.express as px
            fig9 = px.ecdf(df, x="age_Diagnosis",
                  title='CDF of age_Diagnosis',
                   labels={'age_Diagnosis':'age_Diagnosis'}, # can specify one label per df column
                   opacity=0.8,
                   
                   color_discrete_sequence=['seagreen'] # color of histogram bars
                   )
            st.write(fig9)
            
            # plotting q-q plot
            st.markdown('**5- Q-Q plot**')
            st.text('plot for checking the distribution using Q-Q plot')
            st.text('We can also observe the distribution difference by clicking')
            from statsmodels.graphics.gofplots import qqplot

            qqplot_data = qqplot(df.age_Diagnosis, line='s').gca().lines
            fig = go.Figure()

            fig.add_trace({
                'type': 'scatter',
                'x': qqplot_data[0].get_xdata(),
                'y': qqplot_data[0].get_ydata(),
                'mode': 'markers',
                'marker': {
                    'color': '#19d3f3'
                }
            })
            
            fig.add_trace({
                'type': 'scatter',
                'x': qqplot_data[1].get_xdata(),
                'y': qqplot_data[1].get_ydata(),
                'mode': 'lines',
                'line': {
                    'color': '#636efa'
                }
            
            })
            
            
            fig['layout'].update({
                'title': 'Quantile-Quantile Plot',
                'xaxis': {
                    'title': 'Theoritical Quantities',
                    'zeroline': False
                },
                'yaxis': {
                    'title': 'Sample Quantities'
                },
                'showlegend': False,
                'width': 600,
                'height': 500,
            })
            
            st.write(fig)
            
            
            
            
            st.markdown('**Checking the Normality of Data**')
            st.text(' Checking normality of data for age_years using shapiro test')
            from scipy.stats import shapiro
            stat, p = shapiro(df.age_Diagnosis)
                
                # interpret
            alpha = 0.05
            if p > alpha:
                msg = 'Sample looks Gaussian (fail to reject H0)'
            else:
                msg = 'Sample does not look Gaussian (reject H0)'
                
            result_mat = [
                ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                [len(df.age_Diagnosis), stat, p, msg]
            ]
            import plotly.figure_factory as ff
            swt_table = ff.create_table(result_mat)
            swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
            swt_table['layout']['height']=200
            swt_table['layout']['margin']['t']=50
            swt_table['layout']['margin']['b']=50
                
            #py.iplot(swt_table, filename='shapiro-wilk-table')
            st.write(swt_table)
            
            st.markdown("""
                                                  **age_Diagnosis_Statistics**
                   
                                       
                           
                           1. Here we can see that the minimum Age of Patient is [{}]
                           2. where as the Average Age of Patient is [{}]
                           3. And the maximum Age of Patient is [{}]
                           4. out of the total count of Patient [{}]
                              
                                       with Standard deviation of [{}]
                           
                           Shapirowilks statistics  [{}]  &  p-value  [{}]
                    
                           """.format(minimum,Average,maximum,counts,std,stat,p))
            st.markdown("*****************************************************************")
            
            
            st.subheader("lets categorised the Age_Diagnosis Data with 2 years of Gap each")
            import numpy as np
            def age_year(x):
                if x <=2:
                    return "Age_diagnosis_Below 2 years"
                elif x<=4:
                    return "Age_diagnosis_Below 4 years"
                elif x <= 6:
                    return "Age_diagnosis_Below 6 years"
                elif x<=8:
                    return "Age_diagnosis_Below 8 years"
                elif x<= 10:
                    return "Age_diagnosis_Below 10 years"
                elif x > 11:
                    return "Age_diagnosis_above 11 years"
                else:
                    return np.NaN 
            
            Age_dia=pd.DataFrame()
            
            Age_dia['age_year'] = df["age_Diagnosis"].round(2)
            
            Age_dia['year_till_dignosed']= Age_dia['age_year'].apply(lambda x: age_year(x))
            
            age = Age_dia.year_till_dignosed
            counts = age.value_counts()
            percent = age.value_counts(normalize=True)
            percent100 = age.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            age=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            age.reset_index(inplace = True)
            age.rename(columns={'index':'Age_Diagnosis'},inplace=True)
            age['No_of_total_data']=age['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : skyblue "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:,0] = r
                temp_df.iloc[:,2] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(age.style.apply(highlight_col, axis=None))            
            
            import plotly.express as px
            fig = px.bar(age, x="Age_Diagnosis", y="counts", color="Frequency")
            st.write(fig)
            
            
            
            st.markdown("*****************************************************************")
            
        if st.checkbox("Show RF and Anti CCP Analysis"):
            st.markdown("**RF Data Analysis**")
            rf = df.RF
            counts = rf.value_counts()
            percent = rf.value_counts(normalize=True)
            percent100 = rf.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            rf=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            rf.reset_index(inplace = True)
            rf.rename(columns={'index':'RF type'},inplace=True)

            rf['No_of_total_data']=rf['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : #f95f5f "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:1,:] = r
                temp_df.iloc[1:2,:] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(rf.style.apply(highlight_col, axis=None))
            st.markdown("- The above table showing the number and the percentage of RF Data")
            import plotly.express as px
            fig = px.bar(rf, x="RF type", y="counts", color="Frequency")
            st.write(fig)
            
            positive=rf['counts'][:1].to_string(index=False)
            negative=rf['counts'][1:].to_string(index=False)
            positive_per=rf['Frequency'][:1].to_string(index=False)
            negative_per=rf['Frequency'][1:].to_string(index=False)
            count=rf['No_of_total_data'][:1].to_string(index=False)


            st.markdown("""
                                      **RF_Statistics**
       
               1. Here we can observe that the Positive case with the counts of [{}] out of [{}] counts
                  With percetage of [{}]
                  
               2. Here we can observe that the Negative case with the counts of [{}] out of [{}] counts
                  With percetage of [{}]
               
        
               """.format(positive,count,positive_per,negative,count,negative_per))
            st.markdown("*****************************************************************")
            
            st.markdown("**ANTI- CCP analysis**")
            
            ccp = df['ANTI CCP']
            counts = ccp.value_counts()
            percent = ccp.value_counts(normalize=True)
            percent100 = ccp.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            ccp=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            ccp.reset_index(inplace = True)
            ccp.rename(columns={'index':'ANTI CCP type'},inplace=True)
            ccp['No_of_total_data']=ccp['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : #f95f5f "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:1,:] = r
                temp_df.iloc[1:2,:] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(ccp.style.apply(highlight_col, axis=None))
            st.markdown("- The above table showing the number and the percentage of ANTI CCP Data")
            import plotly.express as px
            fig = px.bar(ccp, x="ANTI CCP type", y="counts", color="Frequency")
            st.write(fig)
            
            positive=ccp['counts'][:1].to_string(index=False)
            negative=ccp['counts'][1:].to_string(index=False)
            positive_per=ccp['Frequency'][:1].to_string(index=False)
            negative_per=ccp['Frequency'][1:].to_string(index=False)
            count=ccp['No_of_total_data'][:1].to_string(index=False)
            
            
            st.markdown("""
                                                  **CCP_Statistics**
                   
                   1. Here we can observe that the Positive case with the counts of [{}] out of [{}] counts
                      With percetage of [{}]
                      
                   2. Here we can observe that the Negative case with the counts of [{}] out of [{}] counts
                      With percetage of [{}]
                   
            
                   """.format(positive,count,positive_per,negative,count,negative_per))
            st.markdown("*****************************************************************")
            
        if st.checkbox("Show Hospital stats"):
            
            st.markdown("**Lets see Hospital Data**")
            
            hospital = df['Hospital']
            counts = hospital.value_counts()
            percent = hospital.value_counts(normalize=True)
            percent100 = hospital.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            hospital=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            hospital.reset_index(inplace = True)
            hospital.rename(columns={'index':'Name'},inplace=True)
            hospital['No_of_total_data']=hospital['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : #f95f5f "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:,0] = r
                temp_df.iloc[:,2] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(hospital.style.apply(highlight_col, axis=None))
            
            st.markdown("- The above table showing the number and the percentage of Data related to Hospital")
            import plotly.express as px
            fig = px.bar(hospital, x="Name", y="counts", color="Frequency")
            st.write(fig)
            
            st.markdown("*****************************************************************")
            
            st.markdown("**🔍 Finding which Employee has entered the most cases**")
            
            data_entry = df['Data Entry By']
            counts = data_entry.value_counts()
            percent = data_entry.value_counts(normalize=True)
            percent100 = data_entry.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            data_entry=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            data_entry.reset_index(inplace = True)
            data_entry.rename(columns={'index':'Name'},inplace=True)
            data_entry['No_of_total_data']=data_entry['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : #f95f5f "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:,0] = r
                temp_df.iloc[:,2] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(data_entry.style.apply(highlight_col, axis=None))
            st.markdown("- The above table showing the number and the percentage of Data entered by employee")
            fig = px.bar(data_entry, x="Name", y="counts", color="Frequency")
            st.write(fig)
            st.markdown("*****************************************************************")
            
            st.markdown("**🔍 Finding Rheumatologist Attended most no of cases**")
            
            rheu = df['Rheumatologist']
            counts = rheu.value_counts()
            percent = rheu.value_counts(normalize=True)
            percent100 = rheu.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            rheu=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            rheu.reset_index(inplace = True)
            rheu.rename(columns={'index':'Doc_name'},inplace=True)
            rheu['No_of_total_data']=rheu['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : #f95f5f "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:,0] = r
                temp_df.iloc[:,2] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(rheu.style.apply(highlight_col, axis=None))
            
            st.markdown("- The above table showing the number and the percentage of Doctor who attanded number of cases")
            fig = px.bar(rheu, x="Doc_name", y="counts", color="Frequency")
            st.write(fig)
            
            st.markdown("*****************************************************************")
            
        if st.checkbox("Show Governorate and Nationality based Analysis"):
            
            st.markdown("**Different cases Based on Governorate**")
            
            Gov = df['Governorate']
            counts = Gov.value_counts()
            percent = Gov.value_counts(normalize=True)
            percent100 = Gov.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            Gov=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            Gov.reset_index(inplace = True)
            Gov.rename(columns={'index':'Governorate'},inplace=True)
            Gov['No_of_total_data']=Gov['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : #f95f5f "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:,0] = r
                temp_df.iloc[:,2] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(Gov.style.apply(highlight_col, axis=None))
            
            st.markdown("- The above table showing the number and the percentage of Governorate ")
            fig = px.bar(Gov, x="Governorate", y="counts", color="Frequency")
            st.write(fig)
            
            st.markdown("*****************************************************************")
                        
            st.markdown("**Different cases Based on Nationality**")
            
            national = df['nationality.G']
            counts = national.value_counts()
            percent = national.value_counts(normalize=True)
            percent100 = national.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            national=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            national.reset_index(inplace = True)
            national.rename(columns={'index':'Nationality'},inplace=True)
            national['No_of_total_data']=national['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : #f95f5f "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:1,:] = r
                temp_df.iloc[1:2,:] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(national.style.apply(highlight_col, axis=None))
            
            st.markdown("- The above table showing the number and the percentage of Nationality ")
            fig = px.bar(national, x="Nationality", y="counts", color="Frequency")
            st.write(fig)
            
            st.markdown("*****************************************************************")
            
            st.markdown("**Indepth analysis for different Nationality**")
            
            diff_national = df['Nationality']
            counts = diff_national.value_counts()
            percent = diff_national.value_counts(normalize=True)
            percent100 = diff_national.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            diff_national=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            diff_national.reset_index(inplace = True)
            diff_national.rename(columns={'index':'Nationality'},inplace=True)
            diff_national['No_of_total_data']=diff_national['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : #f95f5f "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:,0] = r
                temp_df.iloc[:,2] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(diff_national.style.apply(highlight_col, axis=None))
            
            st.markdown("- The above table showing the number and the percentage of different Nationality ")
            fig = px.bar(diff_national, x="Nationality", y="counts", color="Frequency")
            st.write(fig)
            
            st.markdown("*****************************************************************")
            
        if st.checkbox(" Show DMARDS Analysis"):
            st.markdown('**DMARDS Analysis**')
            dmard = df['DMARDS_PN']
            counts = dmard.value_counts()
            percent = dmard.value_counts(normalize=True)
            percent100 = dmard.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            dmard=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            dmard.reset_index(inplace = True)
            dmard.rename(columns={'index':'Type'},inplace=True)
            dmard['No_of_total_data']=dmard['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : #f95f5f "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:,0] = r
                temp_df.iloc[:,2] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(dmard.style.apply(highlight_col, axis=None))
            st.markdown("- The above table showing the number and the percentage of DMARDS Analysis ")
            fig = px.bar(dmard, x="Type", y="counts", color="Frequency")
            st.write(fig)
            
            st.markdown("*****************************************************************")
            
            st.markdown("**Lets check the Distribution of DMARDS using BAR plot**")
            
            dmard_final=[]
            dmard = df['MTX']
            counts = dmard.value_counts()
            percent = dmard.value_counts(normalize=True)
            percent100 = dmard.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            dmard=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            dmard.reset_index(inplace = True)
            
            dmard['No_of_total_data']=dmard['counts'].sum()
            dmard_final=dmard[0:1]
            dmard_final.replace('Yes','MTX', inplace =True)
            
            dmard = df['SSZ']
            counts = dmard.value_counts()
            percent = dmard.value_counts(normalize=True)
            percent100 = dmard.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            dmard=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            dmard.reset_index(inplace = True)
            
            dmard['No_of_total_data']=dmard['counts'].sum()
            dmard_final=dmard_final.append(dmard[1:2])
            dmard_final.replace('Yes','SSZ', inplace =True)
            
            dmard = df['LEF']
            counts = dmard.value_counts()
            percent = dmard.value_counts(normalize=True)
            percent100 = dmard.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            dmard=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            dmard.reset_index(inplace = True)
            
            dmard['No_of_total_data']=dmard['counts'].sum()
            dmard_final=dmard_final.append(dmard[1:2])
            dmard_final.replace('Yes','LEF', inplace =True)
            
            dmard = df['HCQ']
            counts = dmard.value_counts()
            percent = dmard.value_counts(normalize=True)
            percent100 = dmard.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            dmard=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            dmard.reset_index(inplace = True)
            
            dmard['No_of_total_data']=dmard['counts'].sum()
            dmard_final=dmard_final.append(dmard[1:2])
            dmard_final.replace('Yes','HCQ', inplace =True)
            
            dmard = df['IMUR']
            counts = dmard.value_counts()
            percent = dmard.value_counts(normalize=True)
            percent100 = dmard.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            dmard=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            dmard.reset_index(inplace = True)
            
            dmard['No_of_total_data']=dmard['counts'].sum()
            dmard_final=dmard_final.append(dmard[1:2])
            dmard_final.replace('Yes','IMUR', inplace =True)
            
            dmard = df['CYC']
            counts = dmard.value_counts()
            percent = dmard.value_counts(normalize=True)
            percent100 = dmard.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            dmard=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            dmard.reset_index(inplace = True)
            
            dmard['No_of_total_data']=dmard['counts'].sum()
            dmard_final=dmard_final.append(dmard[1:2])
            dmard_final.replace('Yes','CYC', inplace =True)
            
            dmar=dmard_final.sort_values('counts',ascending = False)
            dmar.reset_index(drop=True)
            dmar.rename(columns={'index':'Type'},inplace=True)
            
            st.write(dmar)
            
            st.markdown("- The above table showing the number and the percentage of Distribution of DMARDS")
            fig = px.bar(dmar, x="Type", y="counts", color="Frequency")
            st.write(fig)
            
            st.markdown("*****************************************************************")
            
        if st.checkbox('Show Biologics Analysis'):
            st.markdown('**Biologics Analysis**')

            bio = df['onbio']
            counts = bio.value_counts()
            percent = bio.value_counts(normalize=True)
            percent100 = bio.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            bio=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            bio.reset_index(inplace = True)
            bio.rename(columns={'index':'yes/no'},inplace=True)
            bio['No_of_total_data']=bio['counts'].sum()
            
            def highlight_col(x):
                r = "background-color : pink "
                g = "background-color : grey "
                y = "background-color : #f95f5f "
                temp_df= pd.DataFrame(" ", index= x.index, columns=x.columns)
                temp_df.iloc[:,0] = r
                temp_df.iloc[:,2] = g
                temp_df.iloc[:,4] = y
                return temp_df
            st.write(bio.style.apply(highlight_col, axis=None))           
            
            st.markdown("- The above table showing the number and the percentage of Biologics Analysis")
            fig = px.bar(bio, x="yes/no", y="counts", color="Frequency")
            st.write(fig)
            
            st.markdown("*****************************************************************")
            
            st.markdown("**Biologics Frequancies from Last visit Table**")
            biologics=[]
            bio = df['RIT']
            counts = bio.value_counts()
            percent = bio.value_counts(normalize=True)
            percent100 = bio.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            bio=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            bio.reset_index(inplace = True)
            
            bio['No_of_total_data']=bio['counts'].sum()
            biologics=bio[1:2]
            biologics.replace('Yes','RIT', inplace =True)
            
            
            bio = df['ADA']
            counts = bio.value_counts()
            percent = bio.value_counts(normalize=True)
            percent100 = bio.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            bio=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            bio.reset_index(inplace = True)
            
            bio['No_of_total_data']=bio['counts'].sum()
            biologics=biologics.append(bio[1:2])
            biologics.replace('Yes','ADA', inplace =True)
            
            bio = df['TOC']
            counts = bio.value_counts()
            percent = bio.value_counts(normalize=True)
            percent100 = bio.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            bio=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            bio.reset_index(inplace = True)
            
            bio['No_of_total_data']=bio['counts'].sum()
            biologics=biologics.append(bio[1:2])
            biologics.replace('Yes','TOC', inplace =True)
            
            bio = df['ETA']
            counts = bio.value_counts()
            percent = bio.value_counts(normalize=True)
            percent100 = bio.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            bio=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            bio.reset_index(inplace = True)
            
            bio['No_of_total_data']=bio['counts'].sum()
            biologics=biologics.append(bio[1:2])
            biologics.replace('Yes','ETA', inplace =True)
            
            bio = df['ABA']
            counts = bio.value_counts()
            percent = bio.value_counts(normalize=True)
            percent100 = bio.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            bio=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            bio.reset_index(inplace = True)
            
            bio['No_of_total_data']=bio['counts'].sum()
            biologics=biologics.append(bio[1:2])
            biologics.replace('Yes','ABA', inplace =True)
            
            bio = df['INF']
            counts = bio.value_counts()
            percent = bio.value_counts(normalize=True)
            percent100 = bio.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            bio=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            bio.reset_index(inplace = True)
            
            bio['No_of_total_data']=bio['counts'].sum()
            biologics=biologics.append(bio[1:2])
            biologics.replace('Yes','INF', inplace =True)
            
            bio = df['TOF']
            counts = bio.value_counts()
            percent = bio.value_counts(normalize=True)
            percent100 = bio.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            bio=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            bio.reset_index(inplace = True)
            
            bio['No_of_total_data']=bio['counts'].sum()
            biologics=biologics.append(bio[1:2])
            biologics.replace('Yes','TOF', inplace =True)
            
            bio = df['CER']
            counts = bio.value_counts()
            percent = bio.value_counts(normalize=True)
            percent100 = bio.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            bio=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            bio.reset_index(inplace = True)
            
            bio['No_of_total_data']=bio['counts'].sum()
            biologics=biologics.append(bio[1:2])
            biologics.replace('Yes','CER', inplace =True)
            
            bio = df['GOL']
            counts = bio.value_counts()
            percent = bio.value_counts(normalize=True)
            percent100 = bio.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            bio=pd.DataFrame({'counts': counts, 'each': percent, 'Frequency': percent100})
            bio.reset_index(inplace = True)
            
            bio['No_of_total_data']=bio['counts'].sum()
            biologics=biologics.append(bio[1:2])
            biologics.replace('Yes','GOL', inplace =True)

            biog=biologics.sort_values('counts',ascending = False)
            biog.reset_index(drop=True)
            biog.rename(columns={'index':'Type'},inplace=True)
            st.write(biog)
            st.markdown("- The above table showing the number and the percentage of Biologics Frequancies from Last visit")
            fig = px.bar(biog, x="Type", y="counts", color="Frequency")
            st.write(fig)
            
            st.markdown("*****************************************************************")
            
            st.markdown("**Mean Value of Different disease activity**")
            
            data=[]
            mean = df['DAS28'].mean()
            count= df['DAS28'].value_counts().shape[0]
            data=pd.DataFrame({'Mean': mean, 'Counts': count},index=['DAS28'])
            
            mean = df['SDAI'].mean()
            count= df['SDAI'].value_counts().shape[0]
            b=pd.DataFrame({'Mean': mean, 'Counts': count},index=['SDAI'])
            
            data=data.append(b)
            
            mean = df['CDAI'].mean()
            count= df['CDAI'].value_counts().shape[0]
            c=pd.DataFrame({'Mean': mean, 'Counts': count},index=['CDAI'])
            
            data=data.append(c)
            
            mean = df['VAS'].mean()
            count= df['VAS'].value_counts().shape[0]
            d=pd.DataFrame({'Mean': mean, 'Counts': count},index=['VAS'])
            
            data=data.append(d)
            
            mean = df['CRP'].mean()
            count= df['CRP'].value_counts().shape[0]
            e=pd.DataFrame({'Mean': mean, 'Counts': count},index=['CRP'])
            
            data=data.append(e)
            
            mean = df['ESR'].mean()
            count= df['ESR'].value_counts().shape[0]
            f=pd.DataFrame({'Mean': mean, 'Counts': count},index=['ESR'])
            
            data=data.append(f)
            
            mean = df['Hgb'].mean()
            count= df['Hgb'].value_counts().shape[0]
            g=pd.DataFrame({'Mean': mean, 'Counts': count},index=['Hgb'])
            
            data=data.append(g)
            
            #mean = df['Morning Stiffness (mg)'].mean()
            #count= df['Morning Stiffness (mg)'].value_counts().shape[0]
            #h=pd.DataFrame({'Mean': mean, 'Counts': count},index=['Morning S'])
            
            #data=data.append(h)
            
            mean = df['Tender'].mean()
            count= df['Tender'].value_counts().shape[0]
            i=pd.DataFrame({'Mean': mean, 'Counts': count},index=['Tender'])
            
            data=data.append(i)
            
            mean = df['Swollen'].mean()
            count= df['Swollen'].value_counts().shape[0]
            j=pd.DataFrame({'Mean': mean, 'Counts': count},index=['Swollen'])
            
            data=data.append(j)
            
            mean = df['HAQ'].mean()
            count= df['HAQ'].value_counts().shape[0]
            k=pd.DataFrame({'Mean': mean, 'Counts': count},index=['HAQ'])
            
            data=data.append(k)
            
            st.write(data.T)
            st.markdown("- The above table showing the Mean Value of Different disease activity")
            st.markdown("*****************************************************************")
            
        if st.checkbox("Show Biologics usage by Nationality level"):
            st.markdown("**Biologics usage by Nationality level**")
            
            bio_nationality=pd.DataFrame(df.groupby('nationality.G')['BioMono'].value_counts())
            bio_nationality['percentage']=pd.DataFrame(df.groupby('nationality.G')['BioMono'].value_counts(normalize=True).mul(100).round(2).astype(str)) + '%'
            bio_nationality.columns = ['{}{}'.format(col[1], col[0]) for col in bio_nationality.columns]
            bio_nationality.reset_index(inplace=True)
            bio_nationality.rename(columns = {'nationality.G':'Nationality','iB':'Counts','ep':'Frequency'}, inplace = True)
            
            st.write(bio_nationality) 
            st.markdown("**cross table**")
            biom = pd.pivot_table(bio_nationality, index=['Nationality'], columns=["BioMono"], values=['Counts'])
            st.write(biom)
            
            st.markdown("- The above table showing the Biologics usage by Nationality level")
            import plotly.express as px
            fig = px.bar(bio_nationality, x="Nationality", y="Counts", color="BioMono")
            fig.update_layout(barmode='group')
            st.write(fig)
            
            st.markdown("*****************************************************************")
            
            st.markdown("**Biomono2 Analysis Based on Nationality level**")
            bio2_nationality=pd.DataFrame(df.groupby('nationality.G')['BioMono2'].value_counts())
            bio2_nationality['percentage']=pd.DataFrame(df.groupby('nationality.G')['BioMono2'].value_counts(normalize=True).mul(100).round(2).astype(str)) + '%'
            bio2_nationality.columns = ['{}{}'.format(col[1], col[0]) for col in bio2_nationality.columns]
            bio2_nationality.reset_index(inplace=True)
            bio2_nationality.rename(columns = {'nationality.G':'Nationality','iB':'Counts','ep':'Frequency'}, inplace = True)
            st.write(bio2_nationality)
            st.markdown("**cross table**")
            biom2 = pd.pivot_table(bio2_nationality, index=['Nationality'], columns=["BioMono2"], values=['Counts'])
            st.write(biom2)
            
            st.markdown("- The above table showing the Biomono2 Analysis Based on Nationality level")
            import plotly.express as px
            fig = px.bar(bio2_nationality, x="Nationality", y="Counts", color="BioMono2")
            fig.update_layout(barmode='group')
            st.write(fig)
            
            st.markdown("*****************************************************************")            
            
            st.markdown("**Mean Value Based on Different disease activity among nationality**")
            
            bio_d28=pd.DataFrame(df.groupby('nationality.G')['DAS28'].mean())
            bio_SDAI=pd.DataFrame(df.groupby('nationality.G')['SDAI'].mean())
            step1=pd.merge(bio_d28,bio_SDAI,on='nationality.G')
            
            bio_CDAI=pd.DataFrame(df.groupby('nationality.G')['CDAI'].mean())
            step2=pd.merge(step1,bio_CDAI,on='nationality.G')
            
            bio_VAS=pd.DataFrame(df.groupby('nationality.G')['VAS'].mean())
            step3=pd.merge(step2,bio_VAS,on='nationality.G')
            
            bio_CRP=pd.DataFrame(df.groupby('nationality.G')['CRP'].mean())
            step4=pd.merge(step3,bio_CRP,on='nationality.G')
            
            bio_ESR=pd.DataFrame(df.groupby('nationality.G')['ESR'].mean())
            step5=pd.merge(step4,bio_ESR,on='nationality.G')
            
            bio_Hgb=pd.DataFrame(df.groupby('nationality.G')['Hgb'].mean())
            step6=pd.merge(step5,bio_Hgb,on='nationality.G')
            
            bio_Tender=pd.DataFrame(df.groupby('nationality.G')['Tender'].mean())
            step7=pd.merge(step6,bio_Tender,on='nationality.G')
            
            bio_Swollen=pd.DataFrame(df.groupby('nationality.G')['Swollen'].mean())
            step8=pd.merge(step7,bio_Swollen,on='nationality.G')
            
            bio_HAQ=pd.DataFrame(df.groupby('nationality.G')['HAQ'].mean())
            data_nat=pd.merge(step8,bio_HAQ,on='nationality.G')
            st.write(data_nat)
            st.markdown("*****************************************************************")            
            
        if st.checkbox("Show BioMono Type's Analysis"):
            st.markdown("**BioMono Type's Analysis**")
            bio_mono_type=[]
            
            bio_rit=pd.DataFrame(df.groupby('BioMono')['RIT'].value_counts())
            bio_rit['percentage']=pd.DataFrame(df.groupby('BioMono')['RIT'].value_counts(normalize=True).mul(100).round(2))
            bio_rit.columns = ['{}{}'.format(col[1], col[0]) for col in bio_rit.columns]
            bio_rit.reset_index(inplace=True)
            bio_rit.rename(columns = {'RIT':'Type','IR':'Counts','ep':'Percentage'}, inplace = True)
            bio1=bio_rit.loc[bio_rit['Type'] == 'Yes']
            bio1.replace('Yes','RIT', inplace =True)
            bio_mono_type=bio1
            
            bio_ada=pd.DataFrame(df.groupby('BioMono')['ADA'].value_counts())
            bio_ada['percentage']=pd.DataFrame(df.groupby('BioMono')['ADA'].value_counts(normalize=True).mul(100).round(2))
            bio_ada.columns = ['{}{}'.format(col[1], col[0]) for col in bio_ada.columns]
            bio_ada.reset_index(inplace=True)
            bio_ada.rename(columns = {'ADA':'Type','DA':'Counts','ep':'Percentage'}, inplace = True)
            bio2=bio_ada.loc[bio_ada['Type'] == 'Yes']
            bio2.replace('Yes','ADA', inplace =True)
            
            
            bio_mono_type=bio_mono_type.append(bio2)
            
            bio_toc=pd.DataFrame(df.groupby('BioMono')['TOC'].value_counts())
            bio_toc['percentage']=pd.DataFrame(df.groupby('BioMono')['TOC'].value_counts(normalize=True).mul(100).round(2))
            bio_toc.columns = ['{}{}'.format(col[1], col[0]) for col in bio_toc.columns]
            bio_toc.reset_index(inplace=True)
            bio_toc.rename(columns = {'TOC':'Type','OT':'Counts','ep':'Percentage'}, inplace = True)
            bio3=bio_toc.loc[bio_toc['Type'] == 'Yes']
            bio3.replace('Yes','TOC', inplace =True)
            
            
            bio_mono_type=bio_mono_type.append(bio3)
            
            bio_eta=pd.DataFrame(df.groupby('BioMono')['ETA'].value_counts())
            bio_eta['percentage']=pd.DataFrame(df.groupby('BioMono')['ETA'].value_counts(normalize=True).mul(100).round(2))
            bio_eta.columns = ['{}{}'.format(col[1], col[0]) for col in bio_eta.columns]
            bio_eta.reset_index(inplace=True)
            bio_eta.rename(columns = {'ETA':'Type','TE':'Counts','ep':'Percentage'}, inplace = True)
            bio4=bio_eta.loc[bio_eta['Type'] == 'Yes']
            bio4.replace('Yes','ETA', inplace =True)
            
            
            bio_mono_type=bio_mono_type.append(bio4)
            
            bio_aba=pd.DataFrame(df.groupby('BioMono')['ABA'].value_counts())
            bio_aba['percentage']=pd.DataFrame(df.groupby('BioMono')['ABA'].value_counts(normalize=True).mul(100).round(2))
            bio_aba.columns = ['{}{}'.format(col[1], col[0]) for col in bio_aba.columns]
            bio_aba.reset_index(inplace=True)
            bio_aba.rename(columns = {'ABA':'Type','BA':'Counts','ep':'Percentage'}, inplace = True)
            bio5=bio_aba.loc[bio_aba['Type'] == 'Yes']
            bio5.replace('Yes','ABA', inplace =True)
            
            bio_mono_type=bio_mono_type.append(bio5)
            
            bio_inf=pd.DataFrame(df.groupby('BioMono')['INF'].value_counts())
            bio_inf['percentage']=pd.DataFrame(df.groupby('BioMono')['INF'].value_counts(normalize=True).mul(100).round(2))
            bio_inf.columns = ['{}{}'.format(col[1], col[0]) for col in bio_inf.columns]
            bio_inf.reset_index(inplace=True)
            bio_inf.rename(columns = {'INF':'Type','NI':'Counts','ep':'Percentage'}, inplace = True)
            bio6=bio_inf.loc[bio_inf['Type'] == 'Yes']
            bio6.replace('Yes','INF', inplace =True)
            
            bio_mono_type=bio_mono_type.append(bio6)
            
            bio_tof=pd.DataFrame(df.groupby('BioMono')['TOF'].value_counts())
            bio_tof['percentage']=pd.DataFrame(df.groupby('BioMono')['TOF'].value_counts(normalize=True).mul(100).round(2))
            bio_tof.columns = ['{}{}'.format(col[1], col[0]) for col in bio_tof.columns]
            bio_tof.reset_index(inplace=True)
            bio_tof.rename(columns = {'TOF':'Type','OT':'Counts','ep':'Percentage'}, inplace = True)
            bio7=bio_tof.loc[bio_tof['Type'] == 'Yes']
            bio7.replace('Yes','TOF', inplace =True)
            
            bio_mono_type=bio_mono_type.append(bio7)
            
            bio_cer=pd.DataFrame(df.groupby('BioMono')['CER'].value_counts())
            bio_cer['percentage']=pd.DataFrame(df.groupby('BioMono')['CER'].value_counts(normalize=True).mul(100).round(2))
            bio_cer.columns = ['{}{}'.format(col[1], col[0]) for col in bio_cer.columns]
            bio_cer.reset_index(inplace=True)
            bio_cer.rename(columns = {'CER':'Type','EC':'Counts','ep':'Percentage'}, inplace = True)
            bio8=bio_cer.loc[bio_cer['Type'] == 'Yes']
            bio8.replace('Yes','CER', inplace =True)
            
            bio_mono_type=bio_mono_type.append(bio8)
            
            bio_gol=pd.DataFrame(df.groupby('BioMono')['GOL'].value_counts())
            bio_gol['percentage']=pd.DataFrame(df.groupby('BioMono')['GOL'].value_counts(normalize=True).mul(100).round(2))
            bio_gol.columns = ['{}{}'.format(col[1], col[0]) for col in bio_gol.columns]
            bio_gol.reset_index(inplace=True)
            bio_gol.rename(columns = {'GOL':'Type','OG':'Counts','ep':'Percentage'}, inplace = True)
            bio9=bio_gol.loc[bio_gol['Type'] == 'Yes']
            bio9.replace('Yes','GOL', inplace =True)
            
            bio_mono_type=bio_mono_type.append(bio9)
            
            bio_mono_type=bio_mono_type.sort_values('Counts',ascending = False)
            st.write(bio_mono_type.reset_index(drop=True))
            st.markdown("**cross table**")
            biom = pd.pivot_table(bio_mono_type, index=['BioMono'], columns=["Type"], values=['Counts','Percentage'])
            st.write(biom)
            st.markdown("- The above table showing counts and percentage of BIOMONO types with cross table")
            import plotly.express as px
            fig = px.bar(bio_mono_type, x="Type", y="Counts", color="BioMono")
            fig.update_layout(barmode='group')
            st.write(fig)
            
            st.markdown("*****************************************************************")            
            
            st.markdown("**BioMono2 Type's Analysis**")
            bio_mono_type2=[]
            bio_rit=pd.DataFrame(df.groupby('BioMono2')['RIT'].value_counts())
            bio_rit['percentage']=pd.DataFrame(df.groupby('BioMono2')['RIT'].value_counts(normalize=True).mul(100).round(2))
            bio_rit.columns = ['{}{}'.format(col[1], col[0]) for col in bio_rit.columns]
            bio_rit.reset_index(inplace=True)
            bio_rit.rename(columns = {'RIT':'Type','IR':'Counts','ep':'Percentage'}, inplace = True)
            bio1=bio_rit.loc[bio_rit['Type'] == 'Yes']
            bio1.replace('Yes','RIT', inplace =True)
            bio_mono_type2=bio1
            
            bio_ada=pd.DataFrame(df.groupby('BioMono2')['ADA'].value_counts())
            bio_ada['percentage']=pd.DataFrame(df.groupby('BioMono2')['ADA'].value_counts(normalize=True).mul(100).round(2))
            bio_ada.columns = ['{}{}'.format(col[1], col[0]) for col in bio_ada.columns]
            bio_ada.reset_index(inplace=True)
            bio_ada.rename(columns = {'ADA':'Type','DA':'Counts','ep':'Percentage'}, inplace = True)
            bio2=bio_ada.loc[bio_ada['Type'] == 'Yes']
            bio2.replace('Yes','ADA', inplace =True)
            
            
            bio_mono_type2=bio_mono_type2.append(bio2)
            
            bio_toc=pd.DataFrame(df.groupby('BioMono2')['TOC'].value_counts())
            bio_toc['percentage']=pd.DataFrame(df.groupby('BioMono2')['TOC'].value_counts(normalize=True).mul(100).round(2))
            bio_toc.columns = ['{}{}'.format(col[1], col[0]) for col in bio_toc.columns]
            bio_toc.reset_index(inplace=True)
            bio_toc.rename(columns = {'TOC':'Type','OT':'Counts','ep':'Percentage'}, inplace = True)
            bio3=bio_toc.loc[bio_toc['Type'] == 'Yes']
            bio3.replace('Yes','TOC', inplace =True)
            
            
            bio_mono_type2=bio_mono_type2.append(bio3)
            
            bio_eta=pd.DataFrame(df.groupby('BioMono2')['ETA'].value_counts())
            bio_eta['percentage']=pd.DataFrame(df.groupby('BioMono2')['ETA'].value_counts(normalize=True).mul(100).round(2))
            bio_eta.columns = ['{}{}'.format(col[1], col[0]) for col in bio_eta.columns]
            bio_eta.reset_index(inplace=True)
            bio_eta.rename(columns = {'ETA':'Type','TE':'Counts','ep':'Percentage'}, inplace = True)
            bio4=bio_eta.loc[bio_eta['Type'] == 'Yes']
            bio4.replace('Yes','ETA', inplace =True)
            
            
            bio_mono_type2=bio_mono_type2.append(bio4)
            
            bio_aba=pd.DataFrame(df.groupby('BioMono2')['ABA'].value_counts())
            bio_aba['percentage']=pd.DataFrame(df.groupby('BioMono2')['ABA'].value_counts(normalize=True).mul(100).round(2))
            bio_aba.columns = ['{}{}'.format(col[1], col[0]) for col in bio_aba.columns]
            bio_aba.reset_index(inplace=True)
            bio_aba.rename(columns = {'ABA':'Type','BA':'Counts','ep':'Percentage'}, inplace = True)
            bio5=bio_aba.loc[bio_aba['Type'] == 'Yes']
            bio5.replace('Yes','ABA', inplace =True)
            
            bio_mono_type2=bio_mono_type2.append(bio5)
            
            bio_inf=pd.DataFrame(df.groupby('BioMono2')['INF'].value_counts())
            bio_inf['percentage']=pd.DataFrame(df.groupby('BioMono2')['INF'].value_counts(normalize=True).mul(100).round(2))
            bio_inf.columns = ['{}{}'.format(col[1], col[0]) for col in bio_inf.columns]
            bio_inf.reset_index(inplace=True)
            bio_inf.rename(columns = {'INF':'Type','NI':'Counts','ep':'Percentage'}, inplace = True)
            bio6=bio_inf.loc[bio_inf['Type'] == 'Yes']
            bio6.replace('Yes','INF', inplace =True)
            
            bio_mono_type2=bio_mono_type2.append(bio6)
            
            bio_tof=pd.DataFrame(df.groupby('BioMono2')['TOF'].value_counts())
            bio_tof['percentage']=pd.DataFrame(df.groupby('BioMono2')['TOF'].value_counts(normalize=True).mul(100).round(2))
            bio_tof.columns = ['{}{}'.format(col[1], col[0]) for col in bio_tof.columns]
            bio_tof.reset_index(inplace=True)
            bio_tof.rename(columns = {'TOF':'Type','OT':'Counts','ep':'Percentage'}, inplace = True)
            bio7=bio_tof.loc[bio_tof['Type'] == 'Yes']
            bio7.replace('Yes','TOF', inplace =True)
            
            bio_mono_type2=bio_mono_type2.append(bio7)
            
            bio_cer=pd.DataFrame(df.groupby('BioMono2')['CER'].value_counts())
            bio_cer['percentage']=pd.DataFrame(df.groupby('BioMono2')['CER'].value_counts(normalize=True).mul(100).round(2))
            bio_cer.columns = ['{}{}'.format(col[1], col[0]) for col in bio_cer.columns]
            bio_cer.reset_index(inplace=True)
            bio_cer.rename(columns = {'CER':'Type','EC':'Counts','ep':'Percentage'}, inplace = True)
            bio8=bio_cer.loc[bio_cer['Type'] == 'Yes']
            bio8.replace('Yes','CER', inplace =True)
            
            bio_mono_type2=bio_mono_type2.append(bio8)
            
            bio_gol=pd.DataFrame(df.groupby('BioMono2')['GOL'].value_counts())
            bio_gol['percentage']=pd.DataFrame(df.groupby('BioMono2')['GOL'].value_counts(normalize=True).mul(100).round(2))
            bio_gol.columns = ['{}{}'.format(col[1], col[0]) for col in bio_gol.columns]
            bio_gol.reset_index(inplace=True)
            bio_gol.rename(columns = {'GOL':'Type','OG':'Counts','ep':'Percentage'}, inplace = True)
            bio9=bio_gol.loc[bio_gol['Type'] == 'Yes']
            bio9.replace('Yes','GOL', inplace =True)
            
            bio_mono_type2=bio_mono_type2.append(bio9)
            bio_mono_type2=bio_mono_type2.sort_values('Counts',ascending = False)
            st.write(bio_mono_type2.reset_index(drop=True))
            st.markdown("**cross table**")
            biom2 = pd.pivot_table(bio_mono_type2, index=['BioMono2'], columns=["Type"], values=['Counts','Percentage'])
            st.write(biom2)
            
            st.markdown("- The above table showing counts and percentage of BIOMONO2 types with cross table")
            import plotly.express as px
            fig = px.bar(bio_mono_type2, x="Type", y="Counts", color="BioMono2")
            fig.update_layout(barmode='group')
            st.write(fig)
            
            st.markdown("*****************************************************************")            
            
            st.markdown("**Mean Value Based on Different disease activity among BIOMONO2**")
            
            bio_d28=pd.DataFrame(df.groupby('BioMono2')['DAS28'].mean())
            bio_SDAI=pd.DataFrame(df.groupby('BioMono2')['SDAI'].mean())
            step1=pd.merge(bio_d28,bio_SDAI,on='BioMono2')
            
            bio_CDAI=pd.DataFrame(df.groupby('BioMono2')['CDAI'].mean())
            step2=pd.merge(step1,bio_CDAI,on='BioMono2')
            
            bio_VAS=pd.DataFrame(df.groupby('BioMono2')['VAS'].mean())
            step3=pd.merge(step2,bio_VAS,on='BioMono2')
            
            bio_CRP=pd.DataFrame(df.groupby('BioMono2')['CRP'].mean())
            step4=pd.merge(step3,bio_CRP,on='BioMono2')
            
            bio_ESR=pd.DataFrame(df.groupby('BioMono2')['ESR'].mean())
            step5=pd.merge(step4,bio_ESR,on='BioMono2')
            
            bio_Hgb=pd.DataFrame(df.groupby('BioMono2')['Hgb'].mean())
            step6=pd.merge(step5,bio_Hgb,on='BioMono2')
            
            bio_Tender=pd.DataFrame(df.groupby('BioMono2')['Tender'].mean())
            step7=pd.merge(step6,bio_Tender,on='BioMono2')
            
            bio_Swollen=pd.DataFrame(df.groupby('BioMono2')['Swollen'].mean())
            step8=pd.merge(step7,bio_Swollen,on='BioMono2')
            
            bio_HAQ=pd.DataFrame(df.groupby('BioMono2')['HAQ'].mean())
            data_bio=pd.merge(step8,bio_HAQ,on='BioMono2')
            
            
            st.write(data_bio)
            
            st.markdown("*****************************************************************")            
        
        st.text('© ASIA Consulting 2022')
            




if __name__=='__main__':
    main()