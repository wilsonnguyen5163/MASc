import pandas
import numpy
import sklearn.datasets
from decision_tree import decision_tree
from collections import Counter
import sklearn
import json
from itertools import combinations, chain
from abc import ABC, abstractmethod
import helpers
from matplotlib import pyplot as plt, patches
from copy import deepcopy
from collections import deque
from mpl_toolkits.mplot3d import Axes3D


class Cluster:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_centroid(self):
        pass

    @abstractmethod
    def add(self, object):
        pass


class Clustering:

    @abstractmethod
    def __init__(self, num_clusters, iterations):
        self.model = None
        self.num_clusters = num_clusters
        self.num_iterations = iterations

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def quality(self):
        pass

    @abstractmethod
    def get_seeds(self) -> Cluster:
        pass


class Cluster_density(Cluster):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius
        self.objects = []
        self.cores = []

    def add(self, point):
        self.objects.append(point)

    def get_cores(self):
        return self.cores

    def add_core(self, core):
        self.cores.append(core)


class Data_Point():
    def __init__(self, point):
        self.value = point
        self.visited = False
        self.owner = None
        self.is_noise = False

    def ownership(self):
        return self.owner

    def reassign_owner(self, cluster: Cluster_density):
        self.owner = cluster

    def mark_noise(self):
        self.is_noise = True


class DBSCAN(Clustering):
    def __init__(self, minpts, radius):
        self.minpts = minpts
        self.radius = radius
        self.clusters = []

    def fit(self, data):
        self.data = data
        self.num_data = len(data)
        data_points = [Data_Point(obj) for obj in self.data]
        queue = deque()
        randomizer = numpy.random.choice(
            range(self.num_data), self.num_data, replace=False)
        for index in randomizer:
            queue.append(data_points[index])

        # Using queue for fast pop
        while queue:
            Q: Data_Point = queue.popleft()
            if Q.visited:
                continue
            Q.visited = True

            if self.is_core(Q):
                cluster = Cluster_density(self.radius)
                cluster.add_core(Q)
                self.clusters.append(cluster)

                # Obtain r-neighbourhood of point P to check for density connectedness to this cluster
                candidate_set = set([point for point in data_points if self.L2_distance(
                    Q.value, point.value) <= self.radius])
                # Check for density reachable points around Q (check if P's neighbour is density reachable from Q)
                while candidate_set:
                    P = candidate_set.pop()
                    if not P.visited:
                        P.visited = True
                        if self.is_core(P):
                            candidate_set.update(
                                [x for x in data_points if self.L2_distance(x.value, P.value) <= self.radius and x != P])
                            cluster.add_core(P)
                    if P.ownership() == None:
                        P.reassign_owner(cluster)
                        cluster.add(P)
            else:
                Q.mark_noise()

    def is_core(self, p: Data_Point):
        neighbour = [obj for obj in self.data if self.L2_distance(
            obj, p.value) <= self.radius]
        if len(neighbour) >= self.minpts:
            return True
        return False

    def quality(self):
        return super().quality()

    def L2_distance(self, p1, p2):
        # Using Euclidean Distance
        if len(p1) != len(p2):
            return 0
        dist = 0
        for i in range(len(p1)):
            dist += numpy.pow(p1[i]-p2[i], 2)
        return numpy.sqrt(dist)

    def display_clusters(self):
        cores = numpy.array([cs.value
                             for cluster in self.clusters for cs in cluster.get_cores()])

        if self.data.shape[1] == 2:
            # Use 2D scatterplot
            fig, ax = plt.subplots(figsize=(15, 10))
            for core in cores:
                circle = patches.Circle(
                    core, radius=self.radius, edgecolor='blue', facecolor='lightblue', linewidth=1, alpha=0.2)
                ax.add_patch(circle)
            ax.set_aspect('equal')

            ax.scatter(self.data[:, 0], self.data[:, 1],
                       color='blue', alpha=0.3)
            if len(cores) > 1:
                ax.scatter(cores[:, 0], cores[:, 1],
                           s=100, c='red', marker='X', alpha=0.6)

            # Mark objects belonging to cluster
            for cluster in self.clusters:
                if len(cluster.objects) < 1:
                    continue
                objs = numpy.array([obj.value for obj in cluster.objects])
                ax.scatter(
                    objs[:, 0], objs[:, 1], color='green', alpha=1)

            plt.legend()
            plt.title("2D clustering result")
            plt.show()

        elif self.data.shape[1] == 3:
            # Use 3D scatterplot
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(projection='3d')
            ax.set_aspect('equal')

            u, v = numpy.mgrid[0:2*numpy.pi:20j, 0:numpy.pi:10j]
            x = self.radius*numpy.cos(u)*numpy.sin(v)
            y = self.radius*numpy.sin(u)*numpy.sin(v)
            z = self.radius*numpy.cos(v)

            for core in cores:
                ax.plot_surface(x+core[0], y+core[1],
                                z+core[2], color="b", alpha=0.2)

            ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2],
                       color='blue', alpha=0.3)
            if len(cores) > 1:
                ax.scatter(cores[:, 0], cores[:, 1], cores[:, 2],
                           s=100, c='red', marker='X', alpha=0.6)

            # Mark objects belonging to cluster
            for cluster in self.clusters:
                if len(cluster.objects) < 1:
                    continue
                objs = numpy.array([obj.value for obj in cluster.objects])
                ax.scatter(
                    objs[:, 0], objs[:, 1], objs[:, 2], color='green', alpha=1)

            ax.set_title("3D clustering")
            plt.legend()
            plt.show()


class Cluster_mean(Cluster):
    def __init__(self):
        self.centroid = None
        self.objects = []

    def update_centroid_mean(self):
        """
        Update centroid mean based on its own internal objects
        """
        if (len(self.objects) == 0):
            return
        self.centroid = numpy.mean(self.objects, axis=0)

    def get_centroid(self):
        """
        Return centroid of the cluster
        """
        return self.centroid

    def add(self, object):
        """
        Assign object to the cluster
        """
        self.objects.append(object)

    def clear(self):
        self.old_centroid = self.centroid[:]
        self.objects = []

    def has_changed(self):
        return not numpy.array_equal(self.old_centroid, self.centroid)


class K_means(Clustering):
    def __init__(self, num_clusters, iterations=1000):
        super().__init__(num_clusters, iterations=1000)
        self.clusters = []
        self.data = None

    def get_seeds(self):
        return self.clusters

    def fit(self, new_data):
        # More clusters than available training data
        self.data = new_data

        if self.num_clusters > len(self.data):
            self.num_clusters = len(self.data)
            self.clusters = [Cluster_mean() for _ in self.data]
            for i, X in enumerate(self.data):
                self.clusters[i].add(X)
                self.clusters[i].update_centroid_mean()
            return
        # Initial centroids
        if self.clusters == []:
            self.clusters = [Cluster_mean() for _ in range(self.num_clusters)]
            chosen_indices = numpy.random.choice(
                range(len(self.data)), self.num_clusters, replace=False)
            for i in range(self.num_clusters):
                self.clusters[i].add(self.data[chosen_indices[i]])
                self.clusters[i].update_centroid_mean()
        has_changed = True
        num_iteration = 0
        while has_changed or num_iteration < self.num_iterations:
            has_changed = False
            for cluster in self.clusters:
                cluster.clear()
            for object in self.data:
                # Assign object to the closest cluster
                distances = [helpers.euclidean_distance(
                    object, cluster.centroid) for cluster in self.clusters]
                closest = numpy.argmin(distances)
                self.clusters[closest].add(object)
            num_iteration += 1
            for cluster in self.clusters:
                cluster.update_centroid_mean()
                if cluster.has_changed():
                    has_changed = True

    def add_data(self, new_data: numpy.array):
        '''
        Add new data objects to clustering space
        '''
        try:
            self.data = numpy.append(self.data, new_data, axis=0)
            self.fit()
        except:
            return

    def quality(self):
      # Return the sum of squared error of the cluster
        error = 0
        for cluster in self.clusters:
            for object in cluster.objects:
                error += numpy.power(helpers.euclidean_distance(object,
                                                                cluster.centroid), 2)
        return error

    def display_clusters(self):
        centroids = numpy.array([cluster.get_centroid()
                                 for cluster in self.clusters])
        if self.data.shape[1] == 2:
            # Use 2D scatterplot
            plt.figure(figsize=(15, 10))
            plt.scatter(self.data[:, 0], self.data[:, 1],
                        cmap="viridis", alpha=0.8)
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        s=200, c='red', marker='X', label="Centroids", alpha=0.3)

            for center in self.clusters:
                for object in center.objects:
                    plt.plot([object[0], center.get_centroid()[0]], [
                        object[1], center.get_centroid()[1]], 'k-', alpha=0.5)

            plt.legend()
            plt.title("2D clustering result")
            plt.show()

        elif self.data.shape[1] == 3:
            # Use 3D scatterplot
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2],
                       cmap='viridis', alpha=0.6)
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       c='red', marker='X', label="Centroids", s=200)

            for center in self.clusters:
                for object in center.objects:
                    ax.plot([object[0], center.get_centroid()[0]], [
                        object[1], center.get_centroid()[1]], [object[2], center.get_centroid()[2]], 'k-', alpha=0.1)

            ax.set_title("3D clustering")
            plt.legend()
            plt.show()


class Cluster_medoid(Cluster):
    def __init__(self):
        super().__init__()
        self.seed = None
        self.objects = []

    def get_centroid(self):
        return self.seed

    def add(self, object):
        self.objects.append(object)

    def clear(self):
        self.objects = []

    def update_seed(self, new_centroid):
        self.seed = new_centroid


class K_medoid(Clustering):
    def __init__(self, num_clusters, iterations):
        super().__init__(num_clusters, iterations)
        self.num_clusters = num_clusters
        self.num_iterations = iterations
        self.seeds = []
        self.data = []

    def fit(self, data):
        self.data = data

        # If data is smaller than number of clusters desired
        if self.num_clusters > len(data):
            self.num_clusters = len(data)
            self.seeds = [Cluster_medoid() for _ in self.data]
            for i in range(len(self.data)):
                self.seeds[i].add(self.data[i])
                self.seeds[i].update_seed(self.data[i])
            return

        # Initial centroids:
        if self.seeds == []:
            # Choose randomly the initial points for medoids representation
            # indices = numpy.random.choice(
            #    range(len(self.data)), self.num_clusters, replace=False)
            indices = [40, 20, 14]
            self.seeds = [Cluster_medoid() for _ in range(self.num_clusters)]
            for ind, medoid in enumerate(self.seeds):
                medoid.add(self.data[indices[ind]])
                medoid.update_seed(self.data[indices[ind]])

        has_changed = True
        num_iter = 0

        # Associate objects to their closest initial medoid
        for object in self.data:
            closest = [self.distance(object, medoid.get_centroid())
                       for medoid in self.seeds]
            self.seeds[numpy.argmin(closest)].add(object)

        # Obtain current medoids partition quality
        curr_medoid_quality = self.quality(self.seeds)

        while has_changed and num_iter < self.num_iterations:
            has_changed = False
            curr_best_swap = [curr_medoid_quality, None]

            # Testing swap candidate
            for ind in range(self.num_clusters):
                test_partitions = [Cluster_medoid()
                                   for _ in range(self.num_clusters)]
                for i, medoid in enumerate(self.seeds):
                    test_partitions[i].update_seed(medoid.get_centroid())

                for choice in self.data:
                    test_partitions[ind].update_seed(choice)
                    for medoid in test_partitions:
                        medoid.clear()
                    # Reassign points to their new closest medoids
                    for object in self.data:
                        closest = [self.distance(object, medoid.get_centroid())
                                   for medoid in test_partitions]
                        test_partitions[numpy.argmin(closest)].add(object)

                    choice_cost = self.quality(test_partitions)
                    # Keeps track of best partition setup
                    if choice_cost < curr_best_swap[0]:
                        potential = deepcopy(test_partitions)
                        curr_best_swap = [choice_cost, potential]
                        has_changed = True

                # Best medoid found for ind-th cluster, update such
                if has_changed:
                    self.seeds = curr_best_swap[1]
                    curr_medoid_quality = curr_best_swap[0]
            num_iter += 1

    def distance(self, p1, p2):
        dist = 0
        for ind, attr in enumerate(p1):
            match attr:
                # Nominal attribute, do Hamming Distance
                case str():
                    pass
                case float():
                    dist += abs(attr-p2[ind])
        return dist

    def curr_quality(self):
        quality = 0
        for medoid in self.seeds:
            for object in medoid.objects:
                quality += self.distance(object, medoid.get_centroid())
        return quality

    def quality(self, medoids):
        quality = 0
        for medoid in medoids:
            for object in medoid.objects:
                quality += self.distance(object, medoid.get_centroid())
        return quality

    def display_clusters(self):
        centroids = numpy.array([cluster.get_centroid()
                                 for cluster in self.seeds])
        if self.data.shape[1] == 2:
            # Use 2D scatterplot
            plt.figure(figsize=(15, 10))
            plt.scatter(self.data[:, 0], self.data[:, 1],
                        cmap="viridis", alpha=0.8)
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        s=200, c='red', marker='X', label="Centroids", alpha=0.3)

            for medoid in self.seeds:
                for object in medoid.objects:
                    plt.plot([object[0], medoid.get_centroid()[0]], [
                        object[1], medoid.get_centroid()[1]], 'k-', alpha=0.1)

            plt.legend()
            plt.title("2D clustering result")
            plt.show()

        elif self.data.shape[1] == 3:
            # Use 3D scatterplot
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2],
                       cmap='viridis', alpha=0.6)
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       c='red', marker='X', label="Centroids", s=200)

            for medoid in self.seeds:
                for object in medoid.objects:
                    ax.plot([object[0], medoid.get_centroid()[0]], [
                        object[1], medoid.get_centroid()[1]], [object[2], medoid.get_centroid()[2]], 'k-', alpha=0.5)

            ax.set_title("3D clustering")
            plt.legend()
            plt.show()


def data_preprocessing(n=5):
    (data, labels) = sklearn.datasets.load_iris(
        return_X_y=True, as_frame=True)
    features = data.columns.tolist()
    classifier_names = list(set(labels))

    # Use pandas to sample without replacement the data to be used for testing
    X_test = data.sample(frac=1/n, replace=False, random_state=1)

    # Partition dataset into test and training sets, using holdout method
    Y_test = labels.loc[X_test.index]
    X_train = data.drop(X_test.index)
    Y_train = labels.drop(X_test.index)

    return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy(), features, classifier_names


def main():
    (X_train, Y_train, X_test, Y_test, features,
     classifier_names) = data_preprocessing(n=10)

    models = [None for _ in range(10)]
    qualities = [0 for _ in range(10)]
    for i in range(len(models)):
        models[i] = K_means(3)
        models[i].fit(X_train)
        qualities[i] = models[i].quality()

    model = models[numpy.argmin(qualities)]

    for cluster in model.clusters:
        print("Centroid: {a}".format(a=cluster.centroid))
        for object in cluster.objects:
            ind = numpy.where((X_train == object).all(axis=1))[0][0]
            print(
                "\t {a} ---- Training Label: {b}".format(a=object, b=Y_train[ind]))
        print("\n")


def test_medoid():
    centers = 3
    X, y = sklearn.datasets.make_blobs(
        300, n_features=3, centers=centers, cluster_std=1, random_state=16)

    model = K_medoid(centers, 1000)
    model.fit(X)

    centroids = numpy.array([cluster.get_centroid()
                             for cluster in model.seeds])
    print(centroids)
    model.display_clusters()


def test_density():
    X, y = sklearn.datasets.make_blobs(
        100, n_features=3, centers=2, cluster_std=1, random_state=16)

    model = DBSCAN(5, 1)
    model.fit(X)
    # centroids = [len(cluster.objects) for cluster in model.clusters]
    # print(centroids)
    model.display_clusters()


def test_kmean():
    centers = 3
    X, y = sklearn.datasets.make_blobs(
        300, n_features=2, centers=centers, cluster_std=1, random_state=16)

    model = K_means(centers, 1000)
    model.fit(X)
    model.display_clusters()


if __name__ == "__main__":
    test_kmean()
