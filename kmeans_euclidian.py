import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csc_matrix, find
from matplotlib import pyplot as plt
import time
import random

#Perform kmeans clustering on vectorized image given a value of k and initial centroid values
def kmeans(imageVec, k, centroids):
    currentCentroids = centroids
    #Choose the distance measure to be used
    dist = 'euclidean'
    #Initialize dummy var to track whether algorithm has converged
    converged = False
    previousIter = -1
    #Initialize the update matrix
    updated = np.full((k,np.shape(imageVec)[1]), np.nan)
    iter = 0
    #Perform k means clustering while the algorithm has not converged
    while converged == False :
        #Calculate distances to centroids, and find closest to each data point
        distances = cdist(imageVec, currentCentroids, dist)
        currentIter = 0
        #record which centroid is closest to each data point
        closest = np.argmin(distances, axis=1)
        #Update locations of centroid, to drop is used to find any empty clusters
        toDrop = []
        for centroid in range(k):
            #Record the indicies of the pixels which are closes to the current centroid
            indiciesForC = np.where(closest==centroid)
            #Pull the data specifically for the centroid in range k
            dataCentroid = imageVec[indiciesForC]
            #Check to see that it contains data
            nopoints = indiciesForC[0].shape
            #print(nopoints)
            #If the centroid does not contain any points, it is appended to be dropped
            if nopoints[0] == 0:
                toDrop.append(centroid)
                updated[centroid] = [0,0,0]
            #If the centroid contains data, take the mean value of the data and find the new centroid location
            else:
                updated[centroid] = np.mean(dataCentroid, axis=0)
                currentIter += np.sum(dataCentroid.dot(updated[centroid]))
        #If there are values in toDrop, those centroids are deleted from the list of current centroids and k is decremented to match
        if len(toDrop) > 0:
            #print(len(toDrop))
            updated = np.delete(updated, toDrop, 0)
            currentCentroids = np.delete(currentCentroids, toDrop, 0)
            k=k-len(toDrop)

        #Check to see if centroids have moved
        if currentIter == previousIter:
            print('For k = ' +str(k)+' converged in '+str(iter) +' iterations')
            #Ends the algorithm if the current and previous iterations are the same
            converged = True
        previousIter = currentIter
        currentCentroids = updated
        iter += 1

    # assign the new pixel value with new centroid
    finalDistances = cdist(imageVec, currentCentroids, dist)
    clusterAssignments = np.argmin(finalDistances, axis=1)

    #Return the final clusters and centroid locations 
    return(clusterAssignments, updated)


#Choose which image we want to compress here by changing the current entry to one of [ruby, football, hestain]
path = 'data/ruby'
original = plt.imread(path+'.bmp')
#Choose k values on which to perform clustering
k_vals = [2,4,8]
iter_time = []
#Get the number of pixels in each dimension
picSize = original.shape
#Vectorize the image in order to perform clustering
vectorized = np.array(original.reshape((picSize[0]*picSize[1],3)))

#Loop over each value of k
for k_val in range(len(k_vals)):
    #record the start time of each iteration
    iter_start = time.time()

    #initialize the centroids as pixels with random RGB values
    centroids = np.random.randint(255, size=(k_vals[k_val], 3))

    #Get the clustered Image and the centroid locations from kmeans function
    clusters, centroids = kmeans(vectorized, k_vals[k_val], centroids)
    #Transform vectorized image back to original picture
    newPic = np.full(np.shape(vectorized), fill_value = np.nan)
    for cluster in np.unique(clusters):
        newPic[np.where(clusters == cluster)] = centroids[cluster]/255
    finalPicture = np.reshape(newPic,(np.shape(original)[0],np.shape(original)[1],np.shape(original)[2]))
    #Record end time of each iteration
    iter_end = time.time()
    #Plot output and write picture each picture ot a jpeg file
    plt.imshow(finalPicture)
    plt.title('Output for K-means with k = ' + str(k_vals[k_val]))
    plt.savefig(path+'/Euclidean/Kmeans_result'+str(k_vals[k_val])+'.jpeg')
    iter_time.append(iter_end - iter_start)
    #Output the final k centroid values, where k may be less than intitial k
    print('The final centroid RGB values are ' +str(centroids))

#Output the time it took for the algorithm to converge at each k value
print('Time for convergence: ' + str(iter_time))
