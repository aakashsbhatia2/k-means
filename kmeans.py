import csv
import math
import random
import numpy as np
import sys
from statistics import mean
import matplotlib.pyplot as plt


def calculate_euclidean_distance(point1, point2):
    sum = 0.0
    for i in range(len(point1)):
        sum+=(point1[i] - point2[i])**2
    return math.sqrt(sum)

def get_clusters(training_data, k):

    #initialise centroids
    centroids = []
    for i in range(k):
        x = random.choice(training_data)
        centroids.append(x[:-1])

    #calculate distance of each point from centroids:
    for t in range(30):
        # For first iteration, calculate distance between point and centroids and append centroid with lowest value
        if t == 0:
            for point in range(len(training_data)):
                min_dist = float("inf")
                temp_centroid = []
                for c in range(len(centroids)):
                    distance = calculate_euclidean_distance(training_data[point][:-1], centroids[c])
                    if distance<min_dist:
                        min_dist=distance
                        temp_centroid = centroids[c]
                training_data[point].append(temp_centroid)
        else:
            #calculate new centroids
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

            centroids = temp_centroids

            for point in range(len(training_data)):
                min_dist = float("inf")
                temp_centroid = []
                for c in range(len(centroids)):
                    distance = calculate_euclidean_distance(training_data[point][:-2], centroids[c])
                    if distance<min_dist:
                        min_dist=distance
                        temp_centroid = centroids[c]
                training_data[point][-1] = temp_centroid
    return training_data, centroids

def create_data():
    with open("Breast_cancer_data.csv", newline='') as f:
        reader = csv.reader(f)
        final_data = list(reader)
        final_data.pop(0)
        random.shuffle(final_data)

        for i in range(len(final_data)):
            for j in range(len(final_data[i])):
                final_data[i][j] = float(final_data[i][j])

        training_data = final_data[:int(0.80 * len(final_data))]
        testing_data = final_data[int(0.80 * len(final_data)):]
    return training_data, testing_data

def rms(trained_data):
    distance = 0
    sum = 0
    for i in trained_data:
        point = i[:-2]
        centroid = i[-1]
        distance = calculate_euclidean_distance(point,centroid)**2
        sum +=distance
    return sum

def plot_error(k_vals, error):
    plt.plot(k_vals,error)
    plt.show()

def test_kmeans(trained_data, testing_data, centroids):
    result = []
    for c in centroids:
        count_1 = 0
        count_0 = 0
        for i in trained_data:
            if i[-2] == 1.0 and i[-1]==c:
                count_1 += 1
            if i[-2] == 0.0 and i[-1] == c:
                count_0 += 1
        if count_0>count_1:
            result.append([c, 0.0])
        else:
            result.append([c, 1.0])
    print(result)
    distance = 0
    predicted_centroid = []
    prediction = 0
    correct = 0
    wrong = 0
    for i in testing_data:
        point = i[:-2]
        label = i[-1]
        min_dist = float("inf")
        for c in range(len(result)):
            distance = calculate_euclidean_distance(point, result[c][0])
            if distance<min_dist:
                min_dist=distance
                prediction = result[c][1]
        if prediction == label:
            correct+=1
        else:
            wrong+=1
    print(correct,wrong, (correct/(correct+wrong))*100)



def main():
    error = []
    k_vals = []
    for k in range(2,10):
        k_vals.append(k)
        training_data, testing_data = create_data()
        trained_data, centroids = get_clusters(training_data, k)
        error.append(rms(trained_data))
    plot_error(k_vals, error)
    training_data, testing_data = create_data()
    trained_data, centroids = get_clusters(training_data, 4)
    test_kmeans(trained_data, testing_data, centroids)



if __name__ == "__main__":
    main()