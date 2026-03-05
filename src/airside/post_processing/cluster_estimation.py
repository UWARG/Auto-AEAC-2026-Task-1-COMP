import numpy as np
import sklearn.mixture
from sklearn.preprocessing import StandardScaler


def cluster_estimation(extracted_points: list):
    success, clusterInstance = ClusterEstimation.create(
        min_activation_threshold=5,  # Requires at least 5 points to start
        min_new_points_to_run=1,
        max_num_components=10,
        random_state=42,
        min_points_per_cluster=2,
    )

    if success and clusterInstance:
        did_run, clusters = clusterInstance.run(extracted_points)

        return clusters

    else:
        return None


class ClusterEstimation:
    _WEIGHT_DROP_THRESHOLD = (
        0.5  # Set low to keep sparse clusters. Points per cluster are very different
    )
    _MAX_COVARIANCE_THRESHOLD = 100  # Cluster Size can be in a large range of sizes

    @classmethod
    def create(
        cls,
        min_activation_threshold,
        min_new_points_to_run,
        max_num_components,
        random_state,
        min_points_per_cluster,
    ):

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
        min_activation_threshold,
        min_new_points_to_run,
        max_num_components,
        random_state,
        min_points_per_cluster,
    ):
        self._vgmm = sklearn.mixture.BayesianGaussianMixture(
            covariance_type="spherical",
            n_components=max_num_components,
            init_params="k-means++",
            weight_concentration_prior=0.001,  # Lower --> Accepts clusters with fewer points
            mean_precision_prior=1e-3,
            max_iter=3000,
            random_state=random_state,
        )
        self._scaler = StandardScaler()
        self._all_points = []
        self._current_bucket = []
        self._min_activation_threshold = min_activation_threshold
        self._min_new_points_to_run = min_new_points_to_run
        self._min_points_per_cluster = min_points_per_cluster
        self._has_ran_once = False

    def run(self, detections, run_override=False):
        self._current_bucket = detections
        if not self._decide_to_run(run_override):
            return False, []

        raw_data = np.array(self._all_points)

        scaled_data = self._scaler.fit_transform(raw_data)

        self._vgmm.fit(scaled_data)

        if not self._vgmm.converged_:
            return False, []

        real_means = self._scaler.inverse_transform(self._vgmm.means_)

        model_output = list(
            zip(real_means, self._vgmm.weights_, self._vgmm.covariances_)
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

    def _decide_to_run(self, run_override):
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

    def _filter_by_points_ownership(self, model_output, scaled_data):
        cluster_assignment = self._vgmm.predict(scaled_data)
        unique, counts = np.unique(cluster_assignment, return_counts=True)
        cluster_counts = dict(zip(unique, counts))

        filtered_output = []
        for i, cluster_data in enumerate(model_output):
            if cluster_counts.get(i, 0) >= self._min_points_per_cluster:
                filtered_output.append(cluster_data)
        return filtered_output

    def _filter_by_covariances(self, model_output):
        if not model_output:
            return []
        min_cov = min(item[2] for item in model_output)
        threshold = min_cov * self._MAX_COVARIANCE_THRESHOLD
        return [c for c in model_output if c[2] <= threshold]

    def _sort_by_weights(self, model_output):
        return sorted(model_output, key=lambda x: x[1], reverse=True)
