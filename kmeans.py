import csv
import math
import random
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance


def calculate_distance(point1, point2, type):
    """

    Here, I am calculating the distance between point 1 and point 2. The distance metric is based on the "type" parameter.
    There are 2 types of distances - Manhattan, Euclidean and Minkowski

    """
    sum = 0.0
    if type == "Euclidean":
        for i in range(len(point1)):
            sum+=(point1[i] - point2[i])**2
        return math.sqrt(sum)
    if type == "Manhattan":
        for i in range(len(point1)):
            sum+= abs(point1[i] - point2[i])
        return sum
    if type == "Minkowski":
        sum = distance.minkowski(point1, point2, p=3)
        return sum

def get_clusters(training_data, k, dist):

    """

    Step 1:
    I initialize the centroids to k random points. The number of centroids depends on the k value defined

    Step 2:
    For each point, check the distance from the current centroids
    map each point to the nearest centroid
    STOPPING CONDITION: If the new centroid != current centroid, update the centroids to the mean of the current points mapped to that centroid
    If the centroid for 2 consecutive iterations remains the same, the k-means algorithm has converged

    """
    centroids = []
    for i in range(k):
        x = random.choice(training_data)
        centroids.append(x[:-1])

    for t in range(30):
        if t == 0:
            for point in range(len(training_data)):
                min_dist = float("inf")
                temp_centroid = []
                for c in range(len(centroids)):
                    distance = calculate_distance(training_data[point][:-1], centroids[c], dist)
                    if distance<min_dist:
                        min_dist=distance
                        temp_centroid = centroids[c]
                training_data[point].append(temp_centroid)
        else:
            temp_centroids = []
            for c in range(len(centroids)):
                sum_1 = 0
                sum_2 = 0
                sum_3 = 0
                sum_4 = 0
                sum_5 = 0
                count = 0
                for point in range(len(training_data)):
                    if training_data[point][-1] == centroids[c]:
                        count+=1
                        sum_1 += training_data[point][0]
                        sum_2 += training_data[point][1]
                        sum_3 += training_data[point][2]
                        sum_4 += training_data[point][3]
                        sum_5 += training_data[point][4]

                temp = [sum_1/count, sum_2/count, sum_3/count, sum_4/count, sum_5/count]
                temp_centroids.append(temp)

            if centroids != temp_centroids:
                centroids = temp_centroids
            else:
                break
            for point in range(len(training_data)):
                min_dist = float("inf")
                temp_centroid = []
                for c in range(len(centroids)):
                    distance = calculate_distance(training_data[point][:-2], centroids[c], dist)
                    if distance<min_dist:
                        min_dist=distance
                        temp_centroid = centroids[c]
                training_data[point][-1] = temp_centroid
    return training_data, centroids

def create_data(path):
    """

    Reading the csv file at the path inputted by the user.

    """

    with open(path, newline='') as f:
        reader = csv.reader(f)
        final_data = list(reader)
        final_data.pop(0)
        random.shuffle(final_data)

        for i in range(len(final_data)):
            for j in range(len(final_data[i])):
                final_data[i][j] = float(final_data[i][j])

    return final_data

def rms(trained_data, dist):
    """

    Calculating error to obtain the optimal k-value for the dataset

    """
    dist = "Euclidean"
    sum = 0
    for i in trained_data:
        point = i[:-2]
        centroid = i[-1]
        distance = (calculate_distance(point,centroid, dist)**2)
        sum +=distance
    return math.sqrt(sum)

def plot_error(k_vals, error):

    """

    Plotting the line chart to identify the elbow for k-means

    """

    plt.plot(k_vals,error)
    plt.xlabel('k-value')
    plt.ylabel('Cost')
    plt.show()

def test_clusters(trained_data, centroids):

    """

    Testing the clusters (i.e. Checking the percentage of 0's and 1's in each cluster)

    """

    for c in range(len(centroids)):
        count_1 = 0
        count_0 = 0
        for p in range(len(trained_data)):
            if trained_data[p][-2] == 0 and trained_data[p][-1] == centroids[c]:
                count_0 += 1
            if trained_data[p][-2] == 1 and trained_data[p][-1] == centroids[c]:
                count_1 += 1
        print ("Centroid ", c+1, ":", centroids[c])
        print("Number of 1's: ", count_1)
        print("Number of 0's: ", count_0)
        print("Percent 1's: ", round((count_1/(count_1 + count_0))*100,2))
        print("Percent 0's: ", round((count_0 / (count_1 + count_0)) * 100,2))
        print("****************")


def main():
    """

    The program can be run in command line. 3 parameters can be initialised.
    - MANDATORY: Using "--path", you can set the path to the dataset.
    - OPTIONAL: Using "--k", you can set the k value. Default value = 2
    - OPTIONAL: Using "[--distance Manhattan]", you can run k-means with Manhattan distance. Default is Euclidean Distance
    """

    dist = "Euclidean"
    path = ""
    k_v = 2
    error = []
    k_vals = []

    for i in range(len(sys.argv)):
        if sys.argv[i] == "--path":
            path = sys.argv[i+1]
        if sys.argv[i] == "--k":
            k_v = int(sys.argv[i+1])
        if sys.argv[i] == "[--distance Manhattan]":
            dist = "Manhattan"
        if sys.argv[i] == "[--distance Minkowski]":
            dist = "Minkowski"


    training_data = create_data(path)

    for k in range(2,10):
        k_vals.append(k)
        if k>2:
            for i in range(len(training_data)):
                training_data[i].remove(training_data[i][-1])
        trained_data, centroids = get_clusters(training_data, k, dist)
        error.append(rms(trained_data, dist))
    plot_error(k_vals, error)

    for i in range(len(training_data)):
        training_data[i].remove(training_data[i][-1])

    trained_data, centroids = get_clusters(training_data, k_v, dist)

    test_clusters(trained_data, centroids)



if __name__ == "__main__":
    main()