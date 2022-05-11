# Script to read + visualize data
from cmath import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # to convert text into numeric vectors
import matplotlib.animation as animation
from scipy import sparse
import function_implementation as imp # OUR METHODS!

def rotate(angle):
    ax.view_init(azim=angle)


df_full_comments = pd.read_csv('data/FullAndClean.csv')

# Read the data, these data frames include also the origin column be careful


countvectorizer = CountVectorizer()
tfidfvectorizer = TfidfVectorizer()

# Convert the documents (text) to a matrix
count_wm = countvectorizer.fit_transform(df_full_comments['Processed Comments'])
tfidf_wm = tfidfvectorizer.fit_transform(df_full_comments['Processed Comments'])

count_tokens = countvectorizer.get_feature_names()
tfidf_tokens = tfidfvectorizer.get_feature_names()

df_countvect = pd.DataFrame(data = count_wm.toarray(), columns = count_tokens)
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(), columns = tfidf_tokens)


df_countvect['Origin'] = df_full_comments['Origin']
df_tfidfvect['Origin'] = df_full_comments['Origin']


print('no')
print(df_countvect.columns)
print('\n')
print(df_tfidfvect.columns)


# Rearrange
colormap = plt.cm.get_cmap('plasma')
data_plot_count = df_countvect.iloc[:, df_countvect.columns != 'Origin'].to_numpy()
data_plot_tfidfvect = df_tfidfvect.iloc[:, df_tfidfvect.columns != 'Origin'].to_numpy()

# print( np.sum( data_plot_count, axis = 1 ) )
# print( np.sum( data_plot_tfidfvect, axis = 1 ) )

# sm = plt.cm.ScalarMappable(cmap=colormap)

# # Look! The matrices seem very very sparse
# plt.figure(1)
# plt.imshow(data_plot_count, cmap=colormap)
# colorsCount = colormap(data_plot_count)
# plt.colorbar(sm)
# plt.show(block=False)
# plt.title("Counter")

# plt.figure(2)
# plt.imshow(data_plot_tfidfvect, cmap=colormap)
# colors_tfidf = colormap(data_plot_tfidfvect)
# plt.colorbar(sm)
# plt.show(block=False)
# plt.title("TFIDF")


# # But it we take a closer look, it is not a zero matrix

# data_plot_count2 = df_countvect.iloc[250:500, 250:500].to_numpy()
# data_plot_tfidfvect2 = df_tfidfvect.iloc[250:500, 250:500].to_numpy()

# plt.figure(3)
# plt.imshow(data_plot_count2, cmap=colormap)
# colorsCount = colormap(data_plot_count2)
# plt.colorbar(sm)
# plt.show(block=False)
# plt.title("Counter")

# plt.figure(4)
# plt.imshow(data_plot_tfidfvect2, cmap=colormap)
# colors_tfidf = colormap(data_plot_tfidfvect2)
# plt.colorbar(sm)
# plt.show(block=False)
# plt.title("TFIDF")

#### PCA to see how this looks like (PCA exact, from Python)

# # For the counter
# fig5 = plt.figure(5)
# ax = plt.axes(projection='3d')
# pca = PCA(n_components=3).fit(df_countvect.iloc[:, df_countvect.columns != 'Origin'])
# data3D1 = pd.DataFrame( pca.transform(df_countvect.iloc[:, df_countvect.columns != 'Origin']), columns=['PC 1', 'PC 2', 'PC 3'])
# data3D1['Origin'] = df_countvect['Origin']
# groups1 = data3D1.groupby('Origin')
# k = 0
# for name, group in groups1:
#     ax.scatter(  group['PC 1'], group['PC 2'], group['PC 3'], label = name, c = colormap(50*k), alpha = 0.5 )
#     k += 1
# plt.legend()
# rot_animation = animation.FuncAnimation(fig5, rotate, frames=np.arange(0, 362, 2), interval=100)
# rot_animation.save('Figures/ExactPCACount.gif', dpi=80, writer='imagemagick')
# plt.show(block=False)


# plt.figure(6)
# k = 0
# for name, group in groups1:
#     plt.scatter(group['PC 1'], group['PC 2'], label = name, c = colormap(50*k), alpha = 0.5 )
#     k += 1
# plt.legend()
# plt.show(block=False)

# # For the TFIDF
# fig7 = plt.figure(7)
# ax = plt.axes(projection='3d')
# pca = PCA(n_components=3).fit(df_tfidfvect.iloc[:, df_tfidfvect.columns != 'Origin'])
# data3D2 = pd.DataFrame( pca.transform(df_tfidfvect.iloc[:, df_tfidfvect.columns != 'Origin']), columns=['PC 1', 'PC 2', 'PC 3'] )
# data3D2['Origin'] = df_tfidfvect['Origin']
# k = 0
# groups2 = data3D2.groupby('Origin')
# for name, group in groups2:
#     ax.scatter( group['PC 1'], group['PC 2'], group['PC 3'], label = name, c = colormap(50*k), alpha = 0.5  )
#     k += 1
# plt.legend()
# rot_animation = animation.FuncAnimation(fig7, rotate, frames=np.arange(0, 362, 2), interval=100)
# rot_animation.save('Figures/ExactPCATFIDF.gif', dpi=80, writer='imagemagick')
# plt.show(block=False)

# plt.figure(8)
# k = 0
# for name, group in groups2:
#     plt.scatter( group['PC 1'], group['PC 2'], label = name, c = colormap(50*k), alpha = 0.5 )
#     k += 1
# plt.legend()
# plt.show(block=False)


# Test for building the PCA with numpy's SVD
# fig9 = plt.figure(9)
# ax = plt.axes(projection = '3d')
data_matrix = df_tfidfvect.iloc[:, df_tfidfvect.columns != 'Origin'].to_numpy()
print(data_matrix.shape[0])
print(data_matrix.shape[1])
# u, S, vh = np.linalg.svd( data_matrix, full_matrices=False )
# V3 = np.transpose( vh[0:3, :] ) # because we want the transpose
# Rep3 = data_matrix@V3 
# df_Rep3 = pd.DataFrame(Rep3, columns=['PC 1', 'PC 2', 'PC 3'])
# df_Rep3['Origin'] = df_tfidfvect['Origin']
# groupsRep3 = df_Rep3.groupby('Origin')
# k = 0
# for name, group in groupsRep3:
#     ax.scatter( group['PC 1'], group['PC 2'], group['PC 3'], label = name, c = colormap(50*k), alpha = 0.5  )
#     k += 1
# plt.legend()
# rot_animation = animation.FuncAnimation(fig9, rotate, frames=np.arange(0, 362, 2), interval=100)
# rot_animation.save('Figures/ExactSVDTFIDF.gif', dpi=80, writer='imagemagick')
# plt.show(block=False)

# plt.figure(10)
# k = 0
# for name, group in groupsRep3:
#     plt.scatter( group['PC 1'], group['PC 2'], label = name, c = colormap(50*k), alpha = 0.5 )
#     k += 1
# plt.legend()
# plt.show(block=False)

# Plot the decay on the singular values
# plt.figure()
# plt.plot( list(range(1, len(S)+1)), S, c = colormap(25)  )
# plt.ylabel('Singular values')
# plt.xlabel('k')
# plt.title('Decay on singular value')
# plt.show(block = False)


# WITH OUR APPROXIMATIONS
# time_svd, errors_svd, errors_table_lT, errors_table_B = imp.errorsTable(data_matrix)
# print(time_svd)
# errors_svd.to_csv('Conclusions/errors_svd.csv')
# errors_table_lT.to_csv('Conclusions/errors_table_lT.csv')
# errors_table_B.to_csv('Conclusions/errors_table_B.csv')

# # Plots

# imp.LowDimVisualization(data_matrix, df_tfidfvect['Origin'])

# imp.tic()
# imp.plotRightSingularVectors(data_matrix)

# print(imp.toc())

# plt.show()