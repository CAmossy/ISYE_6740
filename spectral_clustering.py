import numpy as np
from scipy import sparse
import scipy
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#1) Load nodes
#2) Load edges
#3) Build adjacency matrix which connects nodes
#4) Build degree degree matrix
#5) Derive Leplacian Matrix
#6) Compute the eigenvalues, find k smallest
#7) Run kmeans on k smallest eigenvectors by treating each row as a data point


#Load nodes
def load_nodes ():
    file_path = 'data/nodes.txt'
    nodes = []
    with open(file_path) as blogs:
        lines = blogs.readlines()
        for line in lines:
            nodes.append(line.split()[0])
    return(nodes)

def load_politics():
    file_path = 'data/nodes.txt'
    politics = []
    with open(file_path) as blogs:
        lines = blogs.readlines()
        for line in lines:
            politics.append(line.split()[2])
    return (np.array(politics).astype(int))

def load_edges():
    edge_file = "data/edges.txt"
    points = []
    with open(edge_file) as edges:
        lines = edges.readlines()
        for line in lines:
            points.append(line.split())
    return np.array(points).astype(int)

def drop_unconnected(A,n,deg, politics):
    #Find indicies where degrees != 0
    idx_nonzero = []
    politics_condensed = []
    for entry in range(n):
        if deg[entry] != 0:
            idx_nonzero.append(entry)
    #Drop indicies where the degree is 0 from A by dropping the row and column
    A = A[:,idx_nonzero]
    A = A[idx_nonzero,:]
    politics_condensed = politics[idx_nonzero]
    return(A, politics_condensed)



nodes = load_nodes()
edges = load_edges()
politics = load_politics()
n = len(nodes)
k_clusters = [2, 5, 10, 25]

i = edges[:, 0]-1
j = edges[:, 1]-1
v = np.ones((edges.shape[0], 1)).flatten()

A = sparse.coo_matrix((v, (i, j)), shape=(n, n))
A = (A + np.transpose(A))
A = sparse.csc_matrix.todense(A) # ## convert to dense matrix

#Find degrees for each node
deg = np.sum(A, axis=1).A1
#Drop nodes with no connections
A_condensed, pols_condensed = drop_unconnected(A,n,deg, politics)
degrees = np.sum(A_condensed, axis=1).A1
D = np.diag(1/np.sqrt(degrees))

#Construct Laplacian matrix
L = D @ A_condensed @ D

overall_mismatch = []
for k_val in range(len(k_clusters)):
    #find the eigenvalues and eigenvectors for our matrix L
    v, x = np.linalg.eig(L)
    k = k_clusters[k_val]
    idx_sorted = np.argsort(v)
    x = x[:, idx_sorted[-k:]]

    #perform k_means clustering on the eigenvectors
    kmeans = KMeans(n_clusters = k).fit(x.real)
    c_idx = np.asarray(kmeans.labels_)
    missmatch_rates = []
    items_assigned_to_cluster = []
    for i in range(k):
        items_in_cluster = len(c_idx[c_idx == i])
        items_assigned_to_cluster.append(items_in_cluster)
        cluster_veiws = np.sum(pols_condensed[c_idx == i])
        average_cluster_veiw = cluster_veiws/items_in_cluster
        #print(average_cluster_veiw)
        if average_cluster_veiw > 0.5:
            missmatch_rates.append(1-average_cluster_veiw)
        else:
            missmatch_rates.append(average_cluster_veiw)
    print("The missmatch Rate for " + str(k) + ' is '+ str(missmatch_rates))
    print("The number of items in each cluster is " + str(items_assigned_to_cluster))
    mismatched_items = []
    for i in range(len(missmatch_rates)):
        mismatched_items.append(missmatch_rates[i] * items_assigned_to_cluster[i])
    overall_mismatch.append(sum(mismatched_items) / sum(items_assigned_to_cluster))

print("The overall mismatch rate is " + str(overall_mismatch))
plt.plot(k_clusters, overall_mismatch)
plt.xlabel('k')
plt.ylabel('% Mismatch')
plt.show()

#This section is used to find the best value of k by plotting the missmatch rate for each k
overall_mismatch = []
k_val = list(range(0,98))
#Loop over k = 2 to 100
for item in k_val:
    print("Working on k = " + str(item+2))
    #find the eigenvalues and eigenvectors for our matrix L
    v, x = np.linalg.eig(L)
    k = item + 2
    idx_sorted = np.argsort(v)
    x = x[:, idx_sorted[-k:]]
    #perform k_means clustering on the eigenvectors
    kmeans = KMeans(n_clusters = k).fit(x.real)
    c_idx = np.asarray(kmeans.labels_)
    missmatch_rates = []
    items_assigned_to_cluster = []
    #Find the items in each cluster and the missmatch rates, as well as the group assignment
    for i in range(k):
        items_in_cluster = len(c_idx[c_idx == i])
        items_assigned_to_cluster.append(items_in_cluster)
        cluster_veiws = np.sum(pols_condensed[c_idx == i])
        average_cluster_veiw = cluster_veiws/items_in_cluster
        #print(average_cluster_veiw)
        if average_cluster_veiw > 0.5:
            missmatch_rates.append(1-average_cluster_veiw)
        else:
            missmatch_rates.append(average_cluster_veiw)
    #print("The missmatch Rate for " + str(k) + ' is '+ str(missmatch_rates))
    #print("The number of items in each cluster is " + str(items_assigned_to_cluster))
    mismatched_items = []
    for i in range(len(missmatch_rates)):
        mismatched_items.append(missmatch_rates[i] * items_assigned_to_cluster[i])
    overall_mismatch.append(sum(mismatched_items) / sum(items_assigned_to_cluster))
k_clusters = [k + 2 for k in k_val]
plt.plot(k_clusters, overall_mismatch)
plt.xlabel('k')
plt.ylabel('% Mismatch')
plt.show()
