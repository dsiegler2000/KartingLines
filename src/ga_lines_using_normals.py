import os
import pickle
import time
from datetime import timedelta
import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt


# Discretize track into points
# Throw out normal lines from the inner by looking at 3 point groups
# Revise those tangent lines to be be normal to the outside
# Put points on those normal lines
# Connect those points to make a line
# right now the genetic algorithm method is just a mess


def closest_point(point, points):
    if points.shape[0] == 0:
        return -1
    dist_2 = np.sum((points - point) ** 2, axis=1)
    return np.argmin(dist_2)


def segment_line_intersect(segment_start, segment_end, line_p, line_v):
    segment_p = segment_start
    segment_v = segment_end - segment_start
    q = line_p
    s = line_v
    p = segment_p
    r = segment_v
    rcs = np.linalg.norm(np.cross(r, s))
    if rcs is 0:
        return False
    t = np.linalg.norm(np.cross(q - p, s)) / rcs
    if t < 0 or t > 1:
        return False

    intersection_point = p + t * r

    if any(np.isnan(intersection_point).tolist()):
        return False
    return intersection_point


class Track:
    def __init__(self, inner, outer, scale_factor=0.5, refinement_factor=100, normal_segments=10):
        """
        Creates a track with the given inner and outer loops, lists of (x, y) points
        :param inner: inner loop
        :param outer: outer loop
        :param scale_factor: how much to scale up/down the track
        :param refinement_factor: how small of chunks to break the track up into
        :param normal_segments: how many chunks to split the normal lines into
        """
        if len(inner) == 0 or len(outer) == 0:
            raise ValueError("`inner` and `outer` must be populated")
        self.inner = (scale_factor * np.array(inner)).astype(np.int)
        self.outer = (scale_factor * np.array(outer)).astype(np.int)
        self.refinement_factor = refinement_factor
        self.starting_point = (0.5 * (self.inner[0] + self.outer[0])).astype(np.int)

        # Get the valid points
        self.valid_points = self.generate_valid_points()

        # Approximate length
        self.approximate_length, self.inner_dist, self.outer_dist = self.calculate_approximate_length()

        # Discretize the track
        self.discretize_track()

        # Calculate the midline
        self.midline = []
        for i in range(self.inner.shape[0]):
            inn = self.inner[i]
            out = self.outer[i % self.outer.shape[0]]
            center = (inn + out) / 2
            self.midline.append(center)
        self.midline = np.array(self.midline)
        self.normal_lines = []
        for i in range(self.midline.shape[0]):
            p1 = self.midline[i - 2]
            p2 = self.midline[i - 1]
            p3 = self.midline[i]
            tangent = p3 - p1
            normal_dir = np.array([-tangent[1], tangent[0]])
            normal_dir = normal_dir / np.linalg.norm(normal_dir)
            # See where this normal intersects with the inner and the outer
            inner_intersections = []
            outer_intersections = []
            for j in range(self.inner.shape[0]):
                intersects = segment_line_intersect(self.inner[j - 1], self.inner[j], p2, normal_dir)
                if intersects is not False:
                    inner_intersections.append(intersects)
            for j in range(self.outer.shape[0]):
                intersects = segment_line_intersect(self.outer[j - 1], self.outer[j], p2, normal_dir)
                if intersects is not False:
                    outer_intersections.append(intersects)
            closest_idx1 = closest_point(self.midline[i], np.array(inner_intersections))
            closest_idx2 = closest_point(self.midline[i], np.array(outer_intersections))

            if closest_idx1 < 0 or closest_idx2 < 0:
                continue

            a = inner_intersections[closest_idx1]
            b = outer_intersections[closest_idx2]
            self.normal_lines.append((a, b))
        self.normal_lines = np.array(self.normal_lines)
        self.normal_line_points = []
        for i in range(self.normal_lines.shape[0]):
            normal_line = self.normal_lines[i]
            p = normal_line[0]
            v = normal_line[1] - normal_line[0]
            segments = []
            for t in np.arange(0, 1.001, 1 / (normal_segments - 1)):
                segments.append(p + t * v)
            self.normal_line_points.append(segments)
        self.normal_line_points = np.array(self.normal_line_points)
        self.normal_segments = normal_segments

    @staticmethod
    def load_track(filepath):
        """
        Creates a track from a text file with x y of the inner and outer loops
        :param filepath: The file path
        :return: A track object
        """
        f = open(filepath, "r")
        lines = f.read().split("\n")
        f.close()

        inner = []
        outer = []
        i = 0
        line = lines[i]
        while line != "outer":
            split = line.split(" ")
            inner.append((int(split[0]), int(split[1])))
            i += 1
            line = lines[i]
        i += 1
        line = lines[i]
        while i < len(lines) - 1:
            split = line.split(" ")
            outer.append((int(split[0]), int(split[1])))
            i += 1
            line = lines[i]

        return Track(inner, outer)

    def generate_valid_points(self):
        """
        Returns a list of points that are inside the track using a flood fill technique
        :return: List of all points that are inside the track
        """
        im = np.zeros((max(a[0] for a in self.outer) + 50, max(a[1] for a in self.outer) + 50, 3), dtype=np.uint8)
        for i in range(len(self.inner)):
            cv2.line(im, tuple(self.inner[i - 1]), tuple(self.inner[i]), color=(255, 255, 255))
        for i in range(len(self.outer)):
            cv2.line(im, tuple(self.outer[i - 1]), tuple(self.outer[i]), color=(255, 255, 255))
        h, w = im.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im, mask, tuple(self.starting_point), (255, 255, 255))
        valid_pts = []
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if im[i, j, 0] > 200 and im[i, j, 1] > 200 and im[i, j, 2] > 200:
                    valid_pts.append((j, i))
        return valid_pts

    def discretize_track(self):
        detailed_inner = []
        detailed_outer = []
        for i in range(len(self.inner)):
            dist_pct = np.linalg.norm(self.inner[i - 1] - self.inner[i]) / self.inner_dist
            num_segments = np.round(self.refinement_factor * dist_pct)
            for t in np.arange(0, 1, 1 / num_segments):
                # Lerp
                r = np.round(self.inner[i - 1] + t * (self.inner[i] - self.inner[i - 1]))
                detailed_inner.append(r)
        for i in range(len(self.outer)):
            dist_pct = np.linalg.norm(self.outer[i - 1] - self.outer[i]) / self.outer_dist
            num_segments = np.round(self.refinement_factor * dist_pct)
            for t in np.arange(0, 1, 1 / num_segments):
                # Lerp
                r = np.round(self.outer[i - 1] + t * (self.outer[i] - self.outer[i - 1]))
                detailed_outer.append(r)
        self.outer = np.array(detailed_outer, dtype=np.int)
        self.inner = np.array(detailed_inner, dtype=np.int)

    def draw(self):
        """
        Draws the track
        :return: Image of the track
        """
        im = np.zeros((max(a[0] for a in self.outer) + 50, max(a[1] for a in self.outer) + 50, 3), dtype=np.uint8)
        for i in range(len(self.inner)):
            cv2.line(im, tuple(self.inner[i - 1]), tuple(self.inner[i]), color=(255, 255, 255))
        for i in range(len(self.outer)):
            cv2.line(im, tuple(self.outer[i - 1]), tuple(self.outer[i]), color=(255, 255, 255))
        for i in range(self.normal_line_points.shape[0]):
            for j in range(self.normal_line_points.shape[1]):
                cv2.circle(im, tuple(self.normal_line_points[i, j].astype(np.int)), 1, color=(255, 0, 255))
        return im

    def calculate_approximate_length(self):
        """
        Returns an approximate length of the track by averaging the inner and outer loop lengths
        :return: An approximate track length
        """
        inner_dist = 0
        for i in range(self.inner.shape[0]):
            inner_dist += np.linalg.norm(self.inner[i - 1] - self.inner[i])
        outer_dist = 0
        for i in range(self.outer.shape[0]):
            outer_dist += np.linalg.norm(self.outer[i - 1] - self.outer[i])

        return (inner_dist + outer_dist) / 2, inner_dist, outer_dist


class RacingLine:
    DISTANCE_WEIGHT = 300  # Distances usually in the range of 0.05-0.01
    CURVATURE_WEIGHT = 1
    OFF_TRACK_WEIGHT = 100  # Per occurance

    def __init__(self, track, use_midline=False, points=None):
        """
        Creates a racing line object
        :param track: Track this line goes to
        :param use_midline: If true, the line is generated by jittering from the midline
        :param points: Points of the racing line. If none, a random line will be generated
        """
        self._fitness_contributions = []
        self.track = track

        if points is None:
            if use_midline:
                self.points = self.generate_random_midline()
            else:
                self.points = self.generate_random_lines()
        else:
            self.points = points

        self.points = np.array(self.points)
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        """
        Calculates the fitness of a track by punishing being off track, distance of track, and curvature of track
        :return: The fitness
        """
        fitness = 0
        for i in range(self.points.shape[0]):
            # Punish distance (normalized by the length of the track)
            p0 = self.track.normal_line_points[i - 2, self.points[i - 2]]
            p1 = self.track.normal_line_points[i - 1, self.points[i - 1]]
            p2 = self.track.normal_line_points[i, self.points[i]]
            dist = RacingLine.DISTANCE_WEIGHT * (np.linalg.norm(p1 - p2) / self.track.approximate_length)

            # Punish curvature
            u = p0 - p1
            v = p1 - p2
            mult = np.linalg.norm(u) * np.linalg.norm(v)
            if -0.001 < mult < 0.001:
                mult = 0.001 if mult > 0 else -0.001
            curve = RacingLine.CURVATURE_WEIGHT * (1 - (0.5 * np.dot(u, v) / mult + 0.5))

            # Punish out of track
            off_track = 0
            # if not tuple(p2) in self.track.valid_points:
            #     off_track = RacingLine.OFF_TRACK_WEIGHT

            fitness -= dist + curve + off_track
            self._fitness_contributions.append(dist + curve + off_track)

        return fitness

    def generate_random_midline(self):
        if self.track.normal_segments % 2 == 0:
            half = int(self.track.normal_segments / 2)
            points = [half if random.random() < 0.5 else half - 1
                      for _ in range(self.track.normal_line_points.shape[0])]
        else:
            points = [int(self.track.normal_segments / 2) for _ in range(self.track.normal_line_points.shape[0])]
        return points

    def generate_random_lines(self):
        """
        Generates a random racing line
        :return: Random set of points
        """
        points = [random.randrange(0, self.track.normal_segments)
                  for _ in range(self.track.normal_line_points.shape[0])]

        return points

    def draw(self):
        """
        Draws the racing line
        :return: Image of the line
        """
        im = self.track.draw()
        for i in range(self.points.shape[0]):
            p0 = self.track.normal_line_points[i - 1, self.points[i - 1]]
            p1 = self.track.normal_line_points[i, self.points[i]]
            cv2.line(im, tuple(p0.astype(np.int)), tuple(p1.astype(np.int)),
                     (255, int(self._fitness_contributions[i] * 255 / (max(self._fitness_contributions) + 10)), 0))

        cv2.putText(im, self.__str__(), (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        return im

    def __str__(self):
        """
        String representation
        :return: Fitness: `fitness`
        """
        return "Fitness: {:10.4f}".format(self.fitness)


class Population:
    def __init__(self, use_midline=False, npy_file=None, track=None, pop_size=100, mutation_rate=0.05,
                 mutation_min_movement=5e-5, mutation_max_movement=5e-4):
        """
        Creates a population
        :param use_midline: If set to true, the population won't be randomly initialized, rather racing lines will be
        created by starting with the midline then randomly jittering the points
        :param npy_file: Numpy file to load population from. If numpy file is passed, track must also be passed
        :param track: The track to use
        :param pop_size: Population size
        :param mutation_rate: Mutation rate
        :param mutation_min_movement: Minimum amount of movement when mutating (scaled by track length)
        :param mutation_max_movement: Maximum amount of movement when mutating (scaled by track length)
        """
        if npy_file is None:
            self.track = track
            self.population = [RacingLine(track, use_midline=use_midline) for _ in range(pop_size)]
        elif npy_file is not None and track is None:
            raise ValueError("If you are loading from a numpy file you must provide a track!")
        else:
            self.population = []
            arr = np.load(npy_file)
            for i in range(arr.shape[0]):
                self.population.append(RacingLine(track, points=arr[i]))
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_min_movement = mutation_min_movement
        self.mutation_max_movement = mutation_max_movement
        self.use_midline = use_midline

    def save(self, filepath):
        """
        Saves the population to a file
        :param filepath: Where to save it to (.npy)
        :return: None
        """
        np.save(filepath, np.array([a.points for a in self.population]))

    def save_params(self, filepath):
        """
        Saves the parameters to a text file
        :param filepath: Where to save it to
        :return: None
        """
        with open(filepath, "w") as f:
            f.write(f"{__file__}")
            f.write(f"population size:                {self.population_size}\n")
            f.write(f"mutation rate:                  {self.mutation_rate}\n")
            f.write(f"midline/race line segments:     {self.population[0].points.shape[0]}\n")
            f.write(f"normal line segments:           {self.population[0].track.normal_segments}\n")
            f.write(f"line distance fitness weight:   {RacingLine.DISTANCE_WEIGHT}\n")
            f.write(f"line curvature fitness weight:  {RacingLine.CURVATURE_WEIGHT}\n")

    def jitter(self, child_point):
        child_point += 1 if random.random() < 0.5 else -1
        if child_point < 0:
            child_point = 0
        if child_point >= self.track.normal_segments:
            child_point = self.track.normal_segments - 1
        return child_point

    def crossover(self, l1, l2):
        """
        Performs crossover between two racing lines
        :param l1: Parent line 1
        :param l2: Parent line 2
        :return: The child racing line
        """
        points = []

        for i in range(l1.points.shape[0]):
            child_point = l1.points[i] if random.random() < 0.5 else l2.points[i]
            if random.random() < self.mutation_rate:
                child_point = self.jitter(child_point)
            points.append(child_point)

        return RacingLine(l1.track, points=points)

    def update(self):
        """
        Updates population by probabilistically breeding based on fitness
        :return: None
        """
        probabilities = []
        fitnesses = [a.fitness for a in self.population]
        mi = min(fitnesses)
        ma = max(fitnesses)
        if np.abs(mi - ma) < 0.01:
            mi = -0.01
        for m in self.population:
            r = -100 * (m.fitness - mi) / (mi - ma)
            probabilities.append(r)
        probabilities = np.array(probabilities) / sum(probabilities)
        new_pop = []
        for _ in range(self.population_size):
            new_pop.append(self.crossover(np.random.choice(self.population, 1, p=probabilities)[0],
                                          np.random.choice(self.population, 1, p=probabilities)[0]))
        self.population = new_pop

    def __getitem__(self, item):
        """
        Quick item getter for convenience
        :param item: Index
        :return: `self.population[item]`
        """
        return self.population[item]


def show(im, upscale=3):
    """
    Shows an image with Numpy
    :param im: Image to show
    :param upscale: Scaling factor
    :return: None (hangs program until q is pressed)
    """
    upscaled = im.copy()
    upscaled = cv2.resize(upscaled, (upscale * upscaled.shape[0], upscale * upscaled.shape[1]))
    cv2.imshow("Show", upscaled)
    while cv2.waitKey(1) != ord("q"):
        pass


# Breeding params
NUM_GENERATIONS = 550


def racing_line(track):
    """
    Generates
    :param track:
    :return:
    """
    # Using a genetic algorithm
    # Generate initial population
    population = Population(track=track, use_midline=True)
    print("Created initial population")
    histories = {"avg": [], "mi": [], "ma": []}
    t = int(time.time())
    start_time = time.time()
    logdir = os.path.join("logs", str(t))
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    population.save_params(os.path.join(logdir, "params.txt"))

    for generation in range(NUM_GENERATIONS):
        print(f"Starting generation {generation + 1} / {NUM_GENERATIONS}")

        if generation > 0 and generation % 549 == 0:
            best_idx = int(np.argmax([m.fitness for m in population.population]))
            best = population.population[best_idx]
            show(best.draw())

        # Generate next generation using crossover
        population.update()

        # Log
        fitnesses = [a.fitness for a in population.population]
        mi = min(fitnesses)
        ma = max(fitnesses)
        avg = sum(fitnesses) / len(fitnesses)
        histories["avg"].append(avg)
        histories["mi"].append(mi)
        histories["ma"].append(ma)

        # Save and log some stuff
        population.save(os.path.join(logdir, "curr_pop.npy"))
        with open(os.path.join(logdir, "training_history.obj"), "wb") as handle:
            pickle.dump(histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
        time_elapsed = time.time() - start_time
        rate = time_elapsed / (generation + 1)
        total_time = rate * NUM_GENERATIONS
        time_left = total_time - time_elapsed
        rate = time_elapsed / ((generation + 1) * population.population_size)
        print(f"Avg: {avg}")
        print(f"Min: {mi}")
        print(f"Max: {ma}")
        print(f"{timedelta(seconds=time_elapsed)} time elapsed, {timedelta(seconds=time_left)} estimated remaining, "
              f"{rate} sec / individual")
        print()

    plt.plot(histories["avg"], "r")
    plt.plot(histories["mi"], "b")
    plt.plot(histories["ma"], "g")
    plt.show()


if __name__ == "__main__":
    track = Track.load_track("track.txt")
    print("Done loading track")
    racing_line(track)
    # population = Population(npy_file="logs/1568337868/curr_pop.npy", track=track)
    # best_idx = int(np.argmax([m.fitness for m in population.population]))
    # best = population.population[best_idx]
    # show(best.draw())
    # histories = pickle.load(open("", "rb"))
    # plt.plot(histories["avg"], "r")
    # plt.plot(histories["mi"], "b")
    # plt.plot(histories["ma"], "g")
    # plt.show()
