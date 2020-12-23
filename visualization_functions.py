import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns
from sklearn import preprocessing
import matplotlib
import pandas as pd
import numpy as np
from math import pi

"""
Author: Andrés Carrillo López.

Useful visualization functions coded and collected.
Reusable functions, adaptable to many use cases.
"""

def viz_kmeans_3DScatter(col_names, data, kmeansModel):
    '''
    Given an array of 3 different column names for a 3D data array, produces a 
    3D ScatterPlot with the kmeans resulting colored clusters.
    '''
    fig = plt.figure(figsize=(12, 10))
    t = fig.suptitle(col_names[0] + '-' + col_names[1] + '-' + col_names[2] , fontsize=14)
    ax = fig.add_subplot(111, projection='3d')

    xs = list(data[0])
    ys = list(data[1])
    zs = list(data[2])

    ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w',c=kmeansModel.labels_.astype(float))

    ax.set_xlabel(col_names[0])
    ax.set_ylabel(col_names[1])
    ax.set_zlabel(col_names[2])

    return

def get_missing_plot(df, cols, title = "Dataset variables and percent missing"):
    '''
    Plots barchart with pct. of missing values per column.
    '''
    n_missing =  df[cols].isnull().sum()
    percent_missing = n_missing * 100 / len(df)
    
    missing_value_df = pd.DataFrame({'column_name': cols,
                                    'num_missing': n_missing,
                                    'percent_missing': percent_missing})
    
    missing_value_df = missing_value_df.sort_values(by=['percent_missing'])
    
        
    plt.figure(figsize=(16,6))
    
    ax = sns.barplot(x="column_name", y="percent_missing", data=missing_value_df,
                        palette=sns.color_palette("flare", len(cols)))
    
    plt.xticks(rotation='vertical')

    plt.title(title)
    ax.grid(b=True, which='major', color='#d3d3d3', linewidth=1.0)
    ax.grid(b=True, which='minor', color='#d3d3d3', linewidth=0.5)
    #plt.show()

    return

def createMultiUserTemporalClusterEvolution(userlist, df, n_clusters,cluster_col , title = "Temporal User Cluster Evolution"):
    '''
    Plots temporal evolution of a list of users belonging to a dataframe
    processed on a clustering algorithm for a given number of clusters.

    "df": dataframe containing the data cols (user, date, pred_clusters) we want a temporal plot.
    "userlist": list containing the names of the different users from a dataframe we want to plot.
    "n_clusters": number of distinct clusters.
    "cluster_col": name of the column storing the predicted clusters.

    Note: max n_clusters: 9
    Note: date column name is default to be named as "date_time".
    '''
    fig, ax= plt.subplots(figsize=(6,3))
    y_ticks_arr = list()
    clusters_list = list()
    for n_cluster in range(n_clusters):
        clusters_list.append("Cluster "+ str(n_cluster))
        
    color_list = ["green", "orange", "red", "cyan", "blue","yellow", "magenta", "black", "pink"]
    for index, user_name in enumerate(userlist):
        user_dataframe = df.where(df["user"]==user_name).dropna()
        user_series = pd.Series(user_dataframe[cluster_col].values, index=user_dataframe["date_time"])
        
        for n_cluster in range(n_clusters):
            s_cluster = user_series[user_series == n_cluster]
            inxval = matplotlib.dates.date2num(s_cluster.index.to_pydatetime())
            times= zip(inxval, np.ones(len(s_cluster)))
            plt.broken_barh(list(times), (index,1), color=color_list[n_cluster])
            
        
        y_ticks_arr.append(index)
        index += 2
    
    ax.margins(0)    
    ax.set_yticks(y_ticks_arr)
    #ax.set_yticks([])
    ax.set_yticklabels(userlist)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.legend(clusters_list, loc="upper left")
    ax.set_title(title)
    monthFmt = matplotlib.dates.DateFormatter("%b")
    ax.xaxis.set_major_formatter(monthFmt)
    plt.tight_layout()
    plt.show()

 
def plot_cluster_centers_radarplot(df, title):
    """
    Given a normalized dataframe with the following structure:
    (important cluster centre label in the last col)
    < Col1, ..., ColN, ClusterCntr>
    Plots radar plot of the N columns, categorized by a group.

    Note: Suitable for plotting e.g. cluster centers, etc. for a 
    reduced number of centers (from 2 to 8 to enhance clarification)
    """
    # Background of the spider plot
    categories = list(df)[1:]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:-1]
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([-2,0,2], ["-2","0","2"], color="black", size=10)
    plt.ylim(-2.5,2.5)
    
    c_list = ['r', 'g', 'b', 'c', 'm', 'y', 'tab:pink','lime']

    # Data of the spider plot
    group_colname = df.columns.values[-1]
    groups = df[group_colname].values

    for i, group in enumerate(groups): 
        values = df.loc[i].drop(group_colname).values.flatten().tolist()
        values += values[:-1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=group)
        ax.fill(angles, values, c_list[i], alpha=0.2)

    plt.title(title)
    plt.legend(bbox_to_anchor=(0.1, 0.1), title=group_colname)
