def viz_kmeans_3DScatter(col_names, 3D_data, kmeansModel):
    '''
    Given an array of 3 different column names for a 3D data array, produces a 
    3D ScatterPlot with the kmeans resulting colored clusters.
    '''
    fig = plt.figure(figsize=(12, 10))
    t = fig.suptitle(col_names[0] + '-' + col_names[1] + '-' + col_names[2] , fontsize=14)
    ax = fig.add_subplot(111, projection='3d')

    xs = list(3D_data[:,0])
    ys = list(3D_data[:,1])
    zs = list(3D_data[:,3])

    ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w',c=kmeansModel.labels_.astype(float))

    ax.set_xlabel(col_names[0])
    ax.set_ylabel(col_names[1])
    ax.set_zlabel(col_names[2])

    return
