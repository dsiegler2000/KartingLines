import cv2
import numpy as np

im_on = 0

# TODO: make a center line by averaging inner and outer and make normal lines using that


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
    def __init__(self, inner, outer, scale_factor=1, refinement_factor=150):
        """
        Creates a track with the given inner and outer loops, lists of (x, y) points
        :param inner: inner loop
        :param outer: outer loop
        :param scale_factor: how much to scale up/down the track
        """
        if len(inner) == 0 or len(outer) == 0:
            raise ValueError("`inner` and `outer` must be populated")
        self.inner = (scale_factor * np.array(inner)).astype(np.int)
        self.outer = (scale_factor * np.array(outer)).astype(np.int)
        self.starting_point = (0.5 * (self.inner[0] + self.outer[0])).astype(np.int)
        self.approximate_length, self.inner_dist, self.outer_dist = self.approximate_length()
        self.refinement_factor = refinement_factor
        self.valid_points = self.generate_valid_points()

        # Break the track up into more discrete points
        self.discretize_track()
        im = self.draw()

        self.centerline = []
        for i in range(self.inner.shape[0]):
            inn = self.inner[i]
            out = self.outer[i % self.outer.shape[0]]
            center = (inn + out) / 2
            self.centerline.append(center)
            center = center.astype(np.int)
            cv2.circle(im, (center[0], center[1]), 1, (255, 0, 0))
        self.centerline = np.array(self.centerline)
        for i in range(self.centerline.shape[0]):
            p1 = self.centerline[i - 2]
            p2 = self.centerline[i - 1]
            p3 = self.centerline[i]
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
            closest_idx = closest_point(self.centerline[i], np.array(inner_intersections))
            a = self.centerline[i] if closest_idx < 0 else inner_intersections[closest_idx]
            closest_idx = closest_point(self.centerline[i], np.array(outer_intersections))
            b = self.centerline[i] if closest_idx < 0 else outer_intersections[closest_idx]

            # normal_dir *= 10
            # a = p2 + normal_dir
            # b = p2 - normal_dir

            cv2.line(im, (int(b[0]), int(b[1])), (int(a[0]), int(a[1])), (255, 0, 255))
        show(im)
        # print(self.inner.shape)
        # print(self.outer.shape)

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

    def generate_valid_points(self):
        """
        Returns a list of points that are inside the track using a flood fill technique
        :return: List of all points that are inside the track
        """
        im = self.draw()
        h, w = im.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im, mask, tuple(self.starting_point), (255, 255, 255))
        valid_pts = []
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if im[i, j, 0] > 200 and im[i, j, 1] > 200 and im[i, j, 2] > 200:
                    valid_pts.append((j, i))
        return valid_pts

    def draw(self):
        """
        Draws the track
        :return: Image of the track
        """
        im = np.zeros((max(a[0] for a in self.outer) + 50, max(a[1] for a in self.outer) + 50, 3), dtype=np.uint8)
        for i in range(len(self.inner)):
            cv2.circle(im, (int(self.inner[i][0]), int(self.inner[i][1])), 1, (255, 255, 255))
        for i in range(len(self.outer)):
            cv2.circle(im, (int(self.outer[i][0]), int(self.outer[i][1])), 1, (255, 255, 255))

        cv2.line(im, (self.inner[0][0], self.inner[0][1]), (self.outer[0][0], self.outer[0][1]), (255, 0, 255))

        return im

    def approximate_length(self):
        """
        Returns an approximate length of the track by averaging the inner and outer loop lengths
        :return: An approximate track length and the inner and outer distances
        """
        inner_dist = 0
        for i in range(self.inner.shape[0]):
            inner_dist += np.linalg.norm(self.inner[i - 1] - self.inner[i])
        outer_dist = 0
        for i in range(self.outer.shape[0]):
            outer_dist += np.linalg.norm(self.outer[i - 1] - self.outer[i])

        return (inner_dist + outer_dist) / 2, inner_dist, outer_dist


def show(im, upscale=3):
    """
    Shows an image with Numpy
    :param im: Image to show
    :param upscale: Scaling factor
    :return: None (hangs program until q is pressed)
    """
    upscaled = im.copy()
    upscaled = cv2.resize(upscaled, (upscale * upscaled.shape[0], upscale * upscaled.shape[1]))
    cv2.imshow(str(im_on), upscaled)
    while cv2.waitKey(1) != ord("q"):
        pass


if __name__ == "__main__":
    track = Track.load_track("track.txt")

