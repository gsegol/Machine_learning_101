import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify

from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
database_filepath = 'disasterdb.db'
engine = create_engine('sqlite:///../data/{}'.format(database_filepath))
tablenames = engine.table_names()
df = pd.read_sql_table(tablenames[0], engine)
print("Reading data base. . .")
print("dimensions:", df.shape)
print(df.head(5))

def group_msg_types(df):
    
    #This function groups the messages into six categories of need
    '''
    INPUT:
        the dataframe of cleaned data
    OUTPUT:
        df_group: the dataframe with columns arranged according to the subgroups
        df_group_total: a new dataframe containing the sum of messages in each subgrop
        group_size: number of variables in each group 
    '''
    group_size = []
    
    df_basic = df[['food', 'water', 'shelter', 'clothing', 'cold', 'money']]
    group_size.append(df_basic.shape[1])
                      
    df_medical = df[['medical_help', 'medical_products', 'hospitals', 'death']]
    group_size.append(df_medical.shape[1])
                      
    df_weather = df[['storm', 'fire', 'earthquake', 'floods', 'weather_related', 'other_weather']]
    group_size.append(df_weather.shape[1])
                      
    df_utilities = df[['infrastructure_related', 'buildings', 'electricity', 'tools', 'transport', 'shops','other_infrastructure']]
    group_size.append(df_utilities.shape[1])
        
    df_people = df[['missing_people', 'child_alone', 'refugees', 'search_and_rescue', 'security', 'military']]
    group_size.append(df_people.shape[1])
        
    df_other = df[['aid_related', 'other_aid', 'aid_centers', 'request', 'offer', 'direct_report']]
    group_size.append(df_other.shape[1])
        
    df_group = pd.concat([df_basic, df_medical, df_weather, df_utilities, df_people, df_other], axis=1)
    df_group_total = pd.DataFrame({'basic': df_basic.sum(axis=1),
                                     'medical': df_medical.sum(axis=1),
                                     'weather': df_weather.sum(axis=1),
                                     'utilities': df_utilities.sum(axis=1),
                                     'people': df_people.sum(axis=1),
                                     'other': df_other.sum(axis=1),
                                    })
                                                          
    
    return df_group, df_group_total, group_size
      
df_group, df_group_total, group_size = group_msg_types(df)
print(df_group.columns)
print(df_group_total.columns)
print(group_size)
print(df_group_total.sum())
print(df_group_total.sum().sum(axis=0))
total_requests = df_group_total.sum().sum(axis=0)

# load model
print("Loading pkl file")
model = joblib.load("../models/tuned_model.pkl")
print("done")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    msg_types_count = df.iloc[:,5:].sum().sort_values(ascending=False)
    msg_types = msg_types_count.index
    msg_group_count = df_group.sum()
    msg_group = msg_group_count.index
    msg_group_total_count = df_group_total.sum()
    msg_group_total = msg_group_total_count.index
    
    # create visuals

    color_sequence =['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']
    ncat = len(group_size)
    colors=[]
    group_colors = []
    #assign group colors
    for i in range (0, ncat):
        for j in range (0, group_size[i]):
            colors.append(color_sequence[i])          
        group_colors.append(color_sequence[i])
    print(group_colors)
  
    graphs = [
        {
            'data': [ #first graph
                Bar(
                    x=msg_types,
                    y=msg_types_count,
                   
                )
            ],
            'layout': {
                'title': 'Distribution of Message Types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': " ",'tickangle': -32, 
                },
                                             
            }             
            
        },
        
        {
            'data': [ #second graph
                Bar(
                    x=msg_group,
                    y=msg_group_count,
                    marker=dict(color=colors)
                )
            ],
            'layout': {
                'title': 'Messages grouped by need types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': " ", "tickangle":-32
                }
            }            
             
        },
         {
            'data': [ #third graph
                Bar(
                    x=msg_group_total,
                    y=msg_group_total_count,
                    marker=dict(color=group_colors)
                )
            ],
            'layout': {
                'title': 'Message count for each need type',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': " ",'tickangle': 0
                } 
               
            }             
            
        },
        {
            'data': [ #fourth graph
                Pie(
                    labels=msg_group_total,
                    values=msg_group_total_count,
                    marker=dict(colors=group_colors)
                )
            ],
            'layout': {
                'title': 'Aid Priorities, as measured by % messages received ',  
            }             
            
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()