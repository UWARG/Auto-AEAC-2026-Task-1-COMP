import numpy as np
import sklearn.mixture
from sklearn.preprocessing import StandardScaler
from typing import Optional, Sequence
from collections import Counter
from scipy.spatial.distance import cdist
from util import Target, Coordinate

# Radius in meters for determining which raw targets contribute to cluster color
CLUSTER_COLOUR_RADIUS = 0.5


def cluster_estimation(raw_targets: list[Target]) -> Optional[list[Target]]:
    extracted_points = [[t.location.x, t.location.y, t.location.z] for t in raw_targets]

    success, clusterInstance = ClusterEstimation.create(
        min_activation_threshold=5,  # Requires at least 5 points to start
        min_new_points_to_run=1,
        max_num_components=10,
        random_state=42,
        min_points_per_cluster=2,
    )

    if not success or clusterInstance is None:
        return None

    did_run, clusters = clusterInstance.run(extracted_points)

    if not did_run or not clusters:
        return None

    # Map each cluster to its constituent colors based on proximity
    raw_target_coords = np.array(extracted_points)
    raw_target_colors = [t.colour for t in raw_targets]

    clustered_targets: list[Target] = []
    for mean, _, _ in clusters:
        # Find distances from cluster mean to all raw targets
        distances = cdist([mean], raw_target_coords)[0]

        nearby_indices = np.where(distances <= CLUSTER_COLOUR_RADIUS)[0]

        # If no targets within radius, use the closest one
        if len(nearby_indices) == 0:
            nearby_indices = np.array([np.argmin(distances)])

        # Find the most common color among nearby targets
        colors_in_cluster = [raw_target_colors[i] for i in nearby_indices]
        cluster_color = Counter(colors_in_cluster).most_common(1)[0][0]

        clustered_targets.append(
            Target(
                colour=cluster_color,
                location=Coordinate(float(mean[0]), float(mean[1]), float(mean[2])),
            )
        )

    return clustered_targets


class ClusterEstimation:
    _WEIGHT_DROP_THRESHOLD = (
        0.5  # Set low to keep sparse clusters. Points per cluster are very different
    )
    _MAX_COVARIANCE_THRESHOLD = 100  # Cluster Size can be in a large range of sizes

    @classmethod
    def create(
        cls,
        min_activation_threshold: int,
        min_new_points_to_run: int,
        max_num_components: int,
        random_state: int,
        min_points_per_cluster: int,
    ) -> tuple[bool, Optional["ClusterEstimation"]]:

        if min_activation_threshold > max_num_components or max_num_components < 1:
            return False, None

        return True, cls(
            min_activation_threshold,
            min_new_points_to_run,
            max_num_components,
            random_state,
            min_points_per_cluster,
        )

    def __init__(
        self,
        min_activation_threshold: int,
        min_new_points_to_run: int,
        max_num_components: int,
        random_state: int,
        min_points_per_cluster: int,
    ) -> None:
        self._vgmm: sklearn.mixture.BayesianGaussianMixture = (
            sklearn.mixture.BayesianGaussianMixture(
                covariance_type="spherical",
                n_components=max_num_components,
                init_params="k-means++",
                weight_concentration_prior=0.001,  # Lower --> Accepts clusters with fewer points
                mean_precision_prior=1e-3,
                max_iter=3000,
                random_state=random_state,
            )
        )
        self._scaler: StandardScaler = StandardScaler()
        self._all_points: list[list[float]] = []
        self._current_bucket: list[list[float]] = []
        self._min_activation_threshold: int = min_activation_threshold
        self._min_new_points_to_run: int = min_new_points_to_run
        self._min_points_per_cluster: int = min_points_per_cluster
        self._has_ran_once: bool = False

    def run(
        self, detections: list[list[float]], run_override: bool = False
    ) -> tuple[bool, list[tuple[np.ndarray, float, float]]]:
        self._current_bucket = detections
        if not self._decide_to_run(run_override):
            return False, []

        raw_data = np.array(self._all_points)

        scaled_data = self._scaler.fit_transform(raw_data)

        self._vgmm.fit(scaled_data)

        if not self._vgmm.converged_:
            return False, []

        real_means = self._scaler.inverse_transform(self._vgmm.means_)

        model_output: list[tuple[np.ndarray, float, float]] = list(
            zip(real_means, self._vgmm.weights_, self._vgmm.covariances_)  # type: ignore[arg-type]
        )

        model_output = self._filter_by_points_ownership(model_output, scaled_data)

        model_output = self._sort_by_weights(model_output)

        if not model_output:
            return True, []

        viable_clusters = [model_output[0]]
        for i in range(1, len(model_output)):
            ratio = model_output[i][1] / (model_output[i - 1][1] + 1e-9)
            if ratio < self._WEIGHT_DROP_THRESHOLD:
                break
            viable_clusters.append(model_output[i])

        return True, self._filter_by_covariances(viable_clusters)

    def _decide_to_run(self, run_override: bool) -> bool:
        count_all = len(self._all_points)
        count_current = len(self._current_bucket)

        if not run_override:
            if count_all + count_current < self._min_activation_threshold:
                return False
            if self._has_ran_once and count_current < self._min_new_points_to_run:
                return False

        if count_all + count_current == 0:
            return False

        self._all_points.extend(self._current_bucket)
        self._current_bucket = []
        self._has_ran_once = True
        return True

    def _filter_by_points_ownership(
        self,
        model_output: Sequence[tuple[np.ndarray, float, float]],
        scaled_data: np.ndarray,
    ) -> list[tuple[np.ndarray, float, float]]:
        cluster_assignment = self._vgmm.predict(scaled_data)
        unique, counts = np.unique(cluster_assignment, return_counts=True)
        cluster_counts = dict(zip(unique, counts))

        filtered_output = []
        for i, cluster_data in enumerate(model_output):
            if cluster_counts.get(i, 0) >= self._min_points_per_cluster:
                filtered_output.append(cluster_data)
        return filtered_output

    def _filter_by_covariances(
        self, model_output: Sequence[tuple[np.ndarray, float, float]]
    ) -> list[tuple[np.ndarray, float, float]]:
        if not model_output:
            return []
        min_cov = min(item[2] for item in model_output)
        threshold = min_cov * self._MAX_COVARIANCE_THRESHOLD
        return [c for c in model_output if c[2] <= threshold]

    def _sort_by_weights(
        self, model_output: Sequence[tuple[np.ndarray, float, float]]
    ) -> list[tuple[np.ndarray, float, float]]:
        return sorted(model_output, key=lambda x: x[1], reverse=True)
