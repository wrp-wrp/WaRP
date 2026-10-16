"""
Dataset management for WaRP Experiments

Provides a unified interface for loading various vector search datasets.
Based on the Big-ANN-Benchmarks dataset infrastructure.
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASEDIR = "./data/"
PROCESSED_DIR = Path(BASEDIR) / "processed"


def _processed_path(dataset_key: str, name: str) -> Path:
    return PROCESSED_DIR / dataset_key / name


def _has_processed_dataset(dataset_key: str) -> bool:
    return (_processed_path(dataset_key, "meta.json").exists()
            and _processed_path(dataset_key, "base.npy").exists()
            and _processed_path(dataset_key, "queries.npy").exists())


def _load_processed_vectors(dataset_key: str, name: str) -> np.ndarray:
    # mmap_mode keeps memory bounded for large datasets.
    return np.load(_processed_path(dataset_key, name), mmap_mode="r")


def load_fvecs(filename: str) -> np.ndarray:
    """Load vectors from .fvecs format (SIFT-style)."""
    with open(filename, "rb") as f:
        data = f.read()

    arr = np.frombuffer(data, dtype=np.int32)
    d = arr[0]
    rec_bytes = 4 + 4 * d
    n = len(data) // rec_bytes

    X = np.frombuffer(data, dtype=np.float32, offset=4, count=n * d).reshape(n, d)
    return X.copy()


def load_bvecs(filename: str) -> np.ndarray:
    """Load vectors from .bvecs format (uint8)."""
    with open(filename, "rb") as f:
        data = f.read()

    if len(data) < 4:
        raise ValueError(f"Invalid bvecs file: {filename}")

    d = int(np.frombuffer(data, dtype=np.int32, count=1)[0])
    rec_bytes = 4 + d
    n = len(data) // rec_bytes
    if len(data) % rec_bytes != 0:
        raise ValueError(f"Invalid bvecs file size: {filename}")

    dims = np.ndarray(shape=(n,), dtype=np.int32, buffer=data, offset=0, strides=(rec_bytes,))
    if not np.all(dims == d):
        raise ValueError(f"Inconsistent bvecs record header in {filename}")

    vectors = np.ndarray(shape=(n, d), dtype=np.uint8, buffer=data, offset=4, strides=(rec_bytes, 1))
    return vectors.astype(np.float32)


def load_ivecs(filename: str) -> np.ndarray:
    """Load integer vectors from .ivecs format (groundtruth)."""
    with open(filename, "rb") as f:
        data = f.read()

    arr = np.frombuffer(data, dtype=np.int32)
    d = arr[0]
    rec_bytes = 4 + 4 * d
    n = len(data) // rec_bytes

    I = np.frombuffer(data, dtype=np.int32, offset=4, count=n * d).reshape(n, d)
    return I.copy()


def load_bin_format(filename: str) -> Tuple[int, int, np.ndarray]:
    """Load vectors from competition .bin format (header + data)."""
    with open(filename, "rb") as f:
        nb_d = np.fromfile(f, dtype=np.int32, count=2)
        nb, d = nb_d[0], nb_d[1]
        vectors = np.fromfile(f, dtype=np.float32).reshape((nb, d))
    return nb, d, vectors


class Dataset:
    """Base dataset class."""

    def __init__(self):
        self.nb = 0  # Number of base vectors
        self.nq = 0  # Number of queries
        self.d = 0   # Dimension
        self.name = ""
        self.basedir = ""

    def prepare(self):
        """Download and prepare the dataset."""
        raise NotImplementedError

    def get_base_vectors(self) -> np.ndarray:
        """Return base vectors as (nb, d) array."""
        raise NotImplementedError

    def get_queries(self) -> np.ndarray:
        """Return query vectors as (nq, d) array."""
        raise NotImplementedError

    def get_groundtruth(self, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Return groundtruth indices and distances."""
        raise NotImplementedError

    def distance_metric(self) -> str:
        """Return 'euclidean' or 'ip' (inner product)."""
        return "euclidean"

    def __str__(self):
        return f"{self.name} (nb={self.nb:,}, nq={self.nq:,}, d={self.d})"


class SIFT1M(Dataset):
    """SIFT1M dataset (1M vectors, 128D)."""

    def __init__(self):
        super().__init__()
        self.name = "SIFT1M"
        self.nb = 1000000
        self.nq = 10000
        self.d = 128
        self.basedir = Path(BASEDIR) / "sift1m"

    def prepare(self):
        """Download SIFT1M if not present."""
        if _has_processed_dataset("sift1m"):
            logger.info(f"✓ SIFT1M processed dataset ready at {_processed_path('sift1m', '')}")
            return

        self.basedir.mkdir(parents=True, exist_ok=True)

        required_files = [
            "sift_base.fvecs",
            "sift_query.fvecs",
            "sift_groundtruth.ivecs"
        ]

        all_exist = all((self.basedir / f).exists() for f in required_files)

        if not all_exist:
            logger.info(f"SIFT1M not found in {self.basedir}")
            logger.info("Download SIFT1M dataset files to data/sift1m/")
            raise FileNotFoundError("SIFT1M dataset not found")

        logger.info(f"✓ SIFT1M dataset ready at {self.basedir}")

    def get_base_vectors(self) -> np.ndarray:
        """Load base vectors."""
        if _has_processed_dataset("sift1m"):
            return _load_processed_vectors("sift1m", "base.npy")
        path = self.basedir / "sift_base.fvecs"
        logger.info(f"Loading base vectors from {path}")
        return load_fvecs(str(path))

    def get_queries(self) -> np.ndarray:
        """Load query vectors."""
        if _has_processed_dataset("sift1m"):
            return _load_processed_vectors("sift1m", "queries.npy")
        path = self.basedir / "sift_query.fvecs"
        logger.info(f"Loading queries from {path}")
        return load_fvecs(str(path))

    def get_groundtruth(self, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Load groundtruth."""
        if _has_processed_dataset("sift1m"):
            I = _load_processed_vectors("sift1m", "gt_I.npy")
            if k is not None and k < I.shape[1]:
                I = I[:, :k]
            return I, None
        path = self.basedir / "sift_groundtruth.ivecs"
        logger.info(f"Loading groundtruth from {path}")
        I = load_ivecs(str(path))

        if k is not None and k < I.shape[1]:
            I = I[:, :k]

        # SIFT groundtruth doesn't include distances, return None
        return I, None


class SIFT100M(Dataset):
    """SIFT100M dataset (100M vectors, 128D)."""

    def __init__(self):
        super().__init__()
        self.name = "SIFT100M"
        self.nb = 100000000
        self.nq = 10000
        self.d = 128
        self.basedir = Path(BASEDIR) / "sift100m"

    def prepare(self):
        """Check if SIFT100M is available."""
        self.basedir.mkdir(parents=True, exist_ok=True)

        base_file = self.basedir / "bigann_base.bvecs"
        if not base_file.exists():
            logger.warning(f"SIFT100M not found in {self.basedir}")
            logger.info("Download from: http://corpus-texmex.irisa.fr/")
            raise FileNotFoundError("SIFT100M dataset not found")

        logger.info(f"✓ SIFT100M dataset ready at {self.basedir}")

    def get_base_vectors(self) -> np.ndarray:
        """Load base vectors (warning: 100M vectors!)."""
        path = self.basedir / "bigann_base.bvecs"
        logger.warning("Loading 100M vectors - this will use ~50GB RAM!")
        return load_bvecs(str(path))

    def get_queries(self) -> np.ndarray:
        """Load query vectors."""
        path = self.basedir / "bigann_query.bvecs"
        return load_bvecs(str(path))

    def get_groundtruth(self, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Load groundtruth."""
        path = self.basedir / "bigann_groundtruth.ivecs"
        I = load_ivecs(str(path))

        if k is not None and k < I.shape[1]:
            I = I[:, :k]

        return I, None


class SIFT10M(Dataset):
    """SIFT10M dataset (10M-vector prefix of BigANN, 128D)."""

    def __init__(self):
        super().__init__()
        self.name = "SIFT10M"
        self.nb = 10000000
        self.nq = 10000
        self.d = 128
        self.basedir = Path(BASEDIR) / "sift100m"

    def prepare(self):
        """Require the processed SIFT10M subset prepared from BigANN raw files."""
        if _has_processed_dataset("sift10m"):
            logger.info(f"✓ SIFT10M processed dataset ready at {_processed_path('sift10m', '')}")
            return

        base_file = self.basedir / "bigann_base.bvecs"
        query_file = self.basedir / "bigann_query.bvecs"
        if base_file.exists() and query_file.exists():
            raise FileNotFoundError(
                "SIFT10M raw BigANN files detected, but the processed subset is missing. "
                "Run `python3 -m artifact prepare-dataset --dataset sift10m` first."
            )

        raise FileNotFoundError(
            "SIFT10M dataset not found. Expected processed files under data/processed/sift10m/ "
            "or run `python3 -m artifact prepare-dataset --dataset sift10m --download`."
        )

    def get_base_vectors(self) -> np.ndarray:
        """Load processed SIFT10M base vectors via mmap."""
        if not _has_processed_dataset("sift10m"):
            raise FileNotFoundError(
                "Processed SIFT10M dataset missing. Run python3 -m artifact prepare-dataset --dataset sift10m."
            )
        return _load_processed_vectors("sift10m", "base.npy")

    def get_queries(self) -> np.ndarray:
        """Load processed SIFT10M query vectors via mmap."""
        if not _has_processed_dataset("sift10m"):
            raise FileNotFoundError(
                "Processed SIFT10M dataset missing. Run python3 -m artifact prepare-dataset --dataset sift10m."
            )
        return _load_processed_vectors("sift10m", "queries.npy")

    def get_groundtruth(self, k: int = 100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load groundtruth if present; SIFT10M processing does not generate it by default."""
        gt_path = _processed_path("sift10m", "gt_I.npy")
        if not gt_path.exists():
            return None, None

        I = _load_processed_vectors("sift10m", "gt_I.npy")
        if k is not None and k < I.shape[1]:
            I = I[:, :k]
        return I, None


class GIST1M(Dataset):
    """GIST1M dataset (1M vectors, 960D)."""

    def __init__(self):
        super().__init__()
        self.name = "GIST1M"
        self.nb = 1_000_000
        self.nq = 1_000
        self.d = 960
        self.basedir = Path(BASEDIR) / "gist1m"

    def prepare(self):
        if _has_processed_dataset("gist1m"):
            logger.info(f"✓ GIST1M processed dataset ready at {_processed_path('gist1m', '')}")
            return
        raise FileNotFoundError(
            "GIST1M processed dataset missing. Run python3 -m artifact prepare-dataset --dataset gist1m."
        )

    def get_base_vectors(self) -> np.ndarray:
        if _has_processed_dataset("gist1m"):
            return _load_processed_vectors("gist1m", "base.npy")
        raise FileNotFoundError("GIST1M processed dataset missing; run python3 -m artifact prepare-dataset --dataset gist1m")

    def get_queries(self) -> np.ndarray:
        if _has_processed_dataset("gist1m"):
            return _load_processed_vectors("gist1m", "queries.npy")
        raise FileNotFoundError("GIST1M processed dataset missing; run python3 -m artifact prepare-dataset --dataset gist1m")

    def get_groundtruth(self, k: int = 100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gt_path = _processed_path("gist1m", "gt_I.npy")
        if not gt_path.exists():
            return None, None
        I = _load_processed_vectors("gist1m", "gt_I.npy")
        if k is not None and k < I.shape[1]:
            I = I[:, :k]
        return I, None


class RandomDataset(Dataset):
    """Random synthetic dataset for testing."""

    def __init__(self, nb: int = 100000, nq: int = 1000, d: int = 128, seed: int = 42):
        super().__init__()
        self.name = f"Random-{nb//1000}K"
        self.nb = nb
        self.nq = nq
        self.d = d
        self.seed = seed
        self.basedir = Path(BASEDIR) / f"random_{nb}_{d}"

        # Generate data
        np.random.seed(seed)
        self._base_vectors = np.random.randn(nb, d).astype(np.float32)
        self._queries = np.random.randn(nq, d).astype(np.float32)

    def prepare(self):
        """No preparation needed for random data."""
        logger.info(f"✓ Random dataset ready (nb={self.nb:,}, nq={self.nq:,}, d={self.d})")

    def get_base_vectors(self) -> np.ndarray:
        """Return random base vectors."""
        return self._base_vectors

    def get_queries(self) -> np.ndarray:
        """Return random queries."""
        return self._queries

    def get_groundtruth(self, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Compute groundtruth using brute force."""
        logger.info("Computing groundtruth with brute force...")

        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='brute')
        nbrs.fit(self._base_vectors)
        D, I = nbrs.kneighbors(self._queries)

        return I, D


class GLOVE(Dataset):
    """GloVe word embeddings dataset."""

    def __init__(self):
        super().__init__()
        self.name = "GLOVE-1M"
        self.nb = 1_000_000
        self.nq = 10000
        self.d = 100
        self.basedir = Path(BASEDIR) / "GLOVE"

    def prepare(self):
        """Check if GLOVE is available."""
        if _has_processed_dataset("glove"):
            logger.info(f"✓ GLOVE processed dataset ready at {_processed_path('glove', '')}")
            return

        # Prefer the unified processed dataset. Raw sources are handled by python3 -m artifact prepare-dataset.
        h5_path = self.basedir / "glove-100-angular.hdf5"
        logger.warning(f"GLOVE processed dataset not found at {_processed_path('glove', '')}")
        if not h5_path.exists():
            raise FileNotFoundError(
                "GLOVE raw file not found. Expected data/GLOVE/glove-100-angular.hdf5. "
                "Run: python3 -m artifact prepare-dataset --dataset glove --download"
            )
        raise FileNotFoundError(
            "GLOVE processed dataset not found. Raw HDF5 exists, but this repo expects the processed .npy format. "
            "Run: python3 -m artifact prepare-dataset --dataset glove"
        )

    def get_base_vectors(self) -> np.ndarray:
        """Load base vectors."""
        if _has_processed_dataset("glove"):
            return _load_processed_vectors("glove", "base.npy")
        raise FileNotFoundError("GLOVE processed dataset missing; run python3 -m artifact prepare-dataset --dataset glove")

    def get_queries(self) -> np.ndarray:
        """Load query vectors."""
        if _has_processed_dataset("glove"):
            return _load_processed_vectors("glove", "queries.npy")
        raise FileNotFoundError("GLOVE processed dataset missing; run python3 -m artifact prepare-dataset --dataset glove")

    def get_groundtruth(self, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Load groundtruth if available."""
        gt_path = self.basedir / f"gt_{self.nb}_{self.nq}_{self.d}"
        if not gt_path.exists():
            logger.warning("Groundtruth not found, computing...")
            return None, None

        with open(gt_path, "rb") as f:
            nq, k_gt = np.fromfile(f, dtype=np.uint32, count=2)
            I = np.fromfile(f, dtype=np.uint32, count=nq * k_gt).reshape(nq, k_gt)
            D = np.fromfile(f, dtype=np.float32, count=nq * k_gt).reshape(nq, k_gt)

        if k is not None and k < k_gt:
            I = I[:, :k]
            D = D[:, :k]

        return I, D


class DEEP1B(Dataset):
    """DEEP1B dataset prepared as a reproducible 1M working slice."""

    def __init__(self):
        super().__init__()
        self.name = "DEEP1B-1M"
        self.nb = 1_000_000
        self.nq = 10_000
        self.d = 96
        self.basedir = Path(BASEDIR) / "deep1b"

    def prepare(self):
        if _has_processed_dataset("deep1b"):
            logger.info(f"✓ DEEP1B processed dataset ready at {_processed_path('deep1b', '')}")
            return
        raise FileNotFoundError(
            "DEEP1B processed dataset missing. Run python3 -m artifact prepare-dataset --dataset deep1b."
        )

    def get_base_vectors(self) -> np.ndarray:
        if _has_processed_dataset("deep1b"):
            return _load_processed_vectors("deep1b", "base.npy")
        raise FileNotFoundError("DEEP1B processed dataset missing; run python3 -m artifact prepare-dataset --dataset deep1b")

    def get_queries(self) -> np.ndarray:
        if _has_processed_dataset("deep1b"):
            return _load_processed_vectors("deep1b", "queries.npy")
        raise FileNotFoundError("DEEP1B processed dataset missing; run python3 -m artifact prepare-dataset --dataset deep1b")

    def get_groundtruth(self, k: int = 100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gt_path = _processed_path("deep1b", "gt_I.npy")
        if not gt_path.exists():
            return None, None
        I = _load_processed_vectors("deep1b", "gt_I.npy")
        D = _processed_path("deep1b", "gt_D.npy")
        dist = _load_processed_vectors("deep1b", "gt_D.npy") if D.exists() else None
        if k is not None and k < I.shape[1]:
            I = I[:, :k]
            if dist is not None:
                dist = dist[:, :k]
        return I, dist


class DEEP10M(Dataset):
    """DEEP10M dataset prepared as a reproducible 1M working slice."""

    def __init__(self):
        super().__init__()
        self.name = "DEEP10M-1M"
        self.nb = 1_000_000
        self.nq = 10_000
        self.d = 96
        self.basedir = Path(BASEDIR) / "deep10m"

    def prepare(self):
        if _has_processed_dataset("deep10m"):
            logger.info(f"✓ DEEP10M processed dataset ready at {_processed_path('deep10m', '')}")
            return
        raise FileNotFoundError(
            "DEEP10M processed dataset missing. Run python3 -m artifact prepare-dataset --dataset deep10m."
        )

    def get_base_vectors(self) -> np.ndarray:
        if _has_processed_dataset("deep10m"):
            return _load_processed_vectors("deep10m", "base.npy")
        raise FileNotFoundError("DEEP10M processed dataset missing; run python3 -m artifact prepare-dataset --dataset deep10m")

    def get_queries(self) -> np.ndarray:
        if _has_processed_dataset("deep10m"):
            return _load_processed_vectors("deep10m", "queries.npy")
        raise FileNotFoundError("DEEP10M processed dataset missing; run python3 -m artifact prepare-dataset --dataset deep10m")

    def get_groundtruth(self, k: int = 100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gt_path = _processed_path("deep10m", "gt_I.npy")
        if not gt_path.exists():
            return None, None
        I = _load_processed_vectors("deep10m", "gt_I.npy")
        D = _processed_path("deep10m", "gt_D.npy")
        dist = _load_processed_vectors("deep10m", "gt_D.npy") if D.exists() else None
        if k is not None and k < I.shape[1]:
            I = I[:, :k]
            if dist is not None:
                dist = dist[:, :k]
        return I, dist


class DEEP1M(Dataset):
    """DEEP image features prepared as a reproducible 1M working slice."""

    def __init__(self):
        super().__init__()
        self.name = "DEEP1M"
        self.nb = 1_000_000
        self.nq = 10_000
        self.d = 96
        self.basedir = Path(BASEDIR) / "deep1m"

    def prepare(self):
        if _has_processed_dataset("deep1m"):
            logger.info(f"✓ DEEP1M processed dataset ready at {_processed_path('deep1m', '')}")
            return
        raise FileNotFoundError(
            "DEEP1M processed dataset missing. Run python3 -m artifact prepare-dataset --dataset deep1m --download."
        )

    def get_base_vectors(self) -> np.ndarray:
        if _has_processed_dataset("deep1m"):
            return _load_processed_vectors("deep1m", "base.npy")
        raise FileNotFoundError("DEEP1M processed dataset missing; run python3 -m artifact prepare-dataset --dataset deep1m")

    def get_queries(self) -> np.ndarray:
        if _has_processed_dataset("deep1m"):
            return _load_processed_vectors("deep1m", "queries.npy")
        raise FileNotFoundError("DEEP1M processed dataset missing; run python3 -m artifact prepare-dataset --dataset deep1m")

    def get_groundtruth(self, k: int = 100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gt_path = _processed_path("deep1m", "gt_I.npy")
        if not gt_path.exists():
            return None, None
        I = _load_processed_vectors("deep1m", "gt_I.npy")
        D = _processed_path("deep1m", "gt_D.npy")
        dist = _load_processed_vectors("deep1m", "gt_D.npy") if D.exists() else None
        if k is not None and k < I.shape[1]:
            I = I[:, :k]
            if dist is not None:
                dist = dist[:, :k]
        return I, dist


class MSong(Dataset):
    """Million-song-style angular features prepared as a 1M working slice."""

    def __init__(self):
        super().__init__()
        self.name = "MSong-1M"
        self.nb = 1_000_000
        self.nq = 10_000
        self.d = 420
        self.basedir = Path(BASEDIR) / "msong"

    def prepare(self):
        if _has_processed_dataset("msong"):
            logger.info(f"✓ MSONG processed dataset ready at {_processed_path('msong', '')}")
            return
        raise FileNotFoundError(
            "MSONG processed dataset missing. Run python3 -m artifact prepare-dataset --dataset msong "
            "(and set VECIO_MSONG_URL if needed)."
        )

    def get_base_vectors(self) -> np.ndarray:
        if _has_processed_dataset("msong"):
            return _load_processed_vectors("msong", "base.npy")
        raise FileNotFoundError("MSONG processed dataset missing; run python3 -m artifact prepare-dataset --dataset msong")

    def get_queries(self) -> np.ndarray:
        if _has_processed_dataset("msong"):
            return _load_processed_vectors("msong", "queries.npy")
        raise FileNotFoundError("MSONG processed dataset missing; run python3 -m artifact prepare-dataset --dataset msong")

    def get_groundtruth(self, k: int = 100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gt_path = _processed_path("msong", "gt_I.npy")
        if not gt_path.exists():
            return None, None
        I = _load_processed_vectors("msong", "gt_I.npy")
        D = _processed_path("msong", "gt_D.npy")
        dist = _load_processed_vectors("msong", "gt_D.npy") if D.exists() else None
        if k is not None and k < I.shape[1]:
            I = I[:, :k]
            if dist is not None:
                dist = dist[:, :k]
        return I, dist


class Random1M128(Dataset):
    """Deterministic 1M random Gaussian vectors materialized in processed format."""

    def __init__(self):
        super().__init__()
        self.name = "Random-1M-128D"
        self.nb = 1_000_000
        self.nq = 10_000
        self.d = 128
        self.basedir = Path(BASEDIR) / "random1m128"

    def prepare(self):
        if _has_processed_dataset("random1m128"):
            logger.info(f"✓ Random1M128 processed dataset ready at {_processed_path('random1m128', '')}")
            return
        raise FileNotFoundError(
            "Random1M128 processed dataset missing. Run python3 -m artifact prepare-dataset --dataset random1m128."
        )

    def get_base_vectors(self) -> np.ndarray:
        if _has_processed_dataset("random1m128"):
            return _load_processed_vectors("random1m128", "base.npy")
        raise FileNotFoundError("Random1M128 processed dataset missing; run python3 -m artifact prepare-dataset --dataset random1m128")

    def get_queries(self) -> np.ndarray:
        if _has_processed_dataset("random1m128"):
            return _load_processed_vectors("random1m128", "queries.npy")
        raise FileNotFoundError("Random1M128 processed dataset missing; run python3 -m artifact prepare-dataset --dataset random1m128")

    def get_groundtruth(self, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        gt_path = _processed_path("random1m128", "gt_I.npy")
        if gt_path.exists():
            I = _load_processed_vectors("random1m128", "gt_I.npy")
            D = _processed_path("random1m128", "gt_D.npy")
            dist = _load_processed_vectors("random1m128", "gt_D.npy") if D.exists() else None
            if k is not None and k < I.shape[1]:
                I = I[:, :k]
                if dist is not None:
                    dist = dist[:, :k]
            return I, dist

        logger.warning("Random1M128 groundtruth not found, computing...")
        from sklearn.neighbors import NearestNeighbors
        base = self.get_base_vectors()
        queries = self.get_queries()
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
        nbrs.fit(base)
        D, I = nbrs.kneighbors(queries)
        return I, D


class COCO(Dataset):
    """COCO image features dataset (100K vectors, 768D)."""

    def __init__(self):
        super().__init__()
        self.name = "COCO"
        self.nb = 100000
        self.nq = 500
        self.d = 768
        self.basedir = Path(BASEDIR) / "COCO"

    def prepare(self):
        """Check if COCO is available."""
        if _has_processed_dataset("coco"):
            logger.info(f"✓ COCO processed dataset ready at {_processed_path('coco', '')}")
            return

        self.basedir.mkdir(parents=True, exist_ok=True)

        data_file = self.basedir / f"data_{self.nb}_{self.d}"
        queries_file = self.basedir / f"queries_{self.nq}_{self.d}"

        # Check for alternative file names
        alt_data_files = [
            self.basedir / f"data_{self.nb}_{self.d}",
            self.basedir / f"data_{self.nb}_{self.d}.fvecs",
            self.basedir / "learn_COCO.fvecs",
            self.basedir / "data_100000_768.fvecs",
        ]

        alt_query_files = [
            self.basedir / f"queries_{self.nq}_{self.d}",
            self.basedir / f"queries_{self.nq}_{self.d}.fvecs",
            self.basedir / "query_COCO.fvecs",
            self.basedir / "queries_500_768.fvecs",
        ]

        data_exists = any(f.exists() for f in alt_data_files)
        queries_exist = any(f.exists() for f in alt_query_files)

        if not data_exists or not queries_exist:
            logger.warning(f"COCO not found in {self.basedir}")
            logger.info("Download from Google Drive:")
            logger.info("  https://drive.google.com/drive/folders/1Hp6SI8YOFPdWbmC1a4_-1dZWxZH3CHMS")
            raise FileNotFoundError("COCO dataset not found")

        logger.info(f"✓ COCO dataset ready at {self.basedir}")

    def get_base_vectors(self) -> np.ndarray:
        """Load base vectors."""
        if _has_processed_dataset("coco"):
            return _load_processed_vectors("coco", "base.npy")
        # Try different file names
        candidates = [
            self.basedir / f"data_{self.nb}_{self.d}",
            self.basedir / f"data_{self.nb}_{self.d}.fvecs",
            self.basedir / "learn_COCO.fvecs",
            self.basedir / "data_100000_768.fvecs",
        ]

        for path in candidates:
            if path.exists():
                logger.info(f"Loading base vectors from {path}")
                if path.suffix == '.fvecs':
                    return load_fvecs(str(path))
                else:
                    _, _, vectors = load_bin_format(str(path))
                    return vectors

        raise FileNotFoundError("COCO base vectors not found")

    def get_queries(self) -> np.ndarray:
        """Load query vectors."""
        if _has_processed_dataset("coco"):
            return _load_processed_vectors("coco", "queries.npy")
        # Try different file names
        candidates = [
            self.basedir / f"queries_{self.nq}_{self.d}",
            self.basedir / f"queries_{self.nq}_{self.d}.fvecs",
            self.basedir / "query_COCO.fvecs",
            self.basedir / "queries_500_768.fvecs",
        ]

        for path in candidates:
            if path.exists():
                logger.info(f"Loading queries from {path}")
                if path.suffix == '.fvecs':
                    return load_fvecs(str(path))
                else:
                    _, _, vectors = load_bin_format(str(path))
                    return vectors

        raise FileNotFoundError("COCO query vectors not found")

    def get_groundtruth(self, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Load groundtruth if available."""
        gt_candidates = [
            self.basedir / f"gt_{self.nb}_{self.nq}_{self.d}",
            self.basedir / "coco_groundtruth.ivecs",
        ]

        for gt_path in gt_candidates:
            if gt_path.exists():
                logger.info(f"Loading groundtruth from {gt_path}")
                if gt_path.suffix == '.ivecs':
                    I = load_ivecs(str(gt_path))
                    if k is not None and k < I.shape[1]:
                        I = I[:, :k]
                    return I, None
                else:
                    with open(gt_path, "rb") as f:
                        nq, k_gt = np.fromfile(f, dtype=np.uint32, count=2)
                        I = np.fromfile(f, dtype=np.uint32, count=nq * k_gt).reshape(nq, k_gt)
                        D = np.fromfile(f, dtype=np.float32, count=nq * k_gt).reshape(nq, k_gt)

                    if k is not None and k < k_gt:
                        I = I[:, :k]
                        D = D[:, :k]
                    return I, D

        logger.warning("COCO groundtruth not found")
        return None, None


class OpenImage(Dataset):
    """OpenImage-Streaming dataset (1M vectors, 512D)."""

    def __init__(self):
        super().__init__()
        self.name = "OpenImage"
        self.nb = 1000000
        self.nq = 10000
        self.d = 512
        self.basedir = Path(BASEDIR) / "openimage-streaming"

    def prepare(self):
        """Check if OpenImage is available."""
        if _has_processed_dataset("openimage"):
            logger.info(f"✓ OpenImage processed dataset ready at {_processed_path('openimage', '')}")
            return

        self.basedir.mkdir(parents=True, exist_ok=True)

        data_file = self.basedir / f"data_{self.nb}_{self.d}"
        queries_file = self.basedir / f"queries_{self.nq}_{self.d}"

        # Check for alternative file names
        alt_data_files = [
            self.basedir / f"data_{self.nb}_{self.d}",
            self.basedir / f"data_{self.nb}_{self.d}.fvecs",
            self.basedir / "learn_image.fvecs",
            self.basedir / "data_1000000_512.fvecs",
        ]

        alt_query_files = [
            self.basedir / f"queries_{self.nq}_{self.d}",
            self.basedir / f"queries_{self.nq}_{self.d}.fvecs",
            self.basedir / "query_image.fvecs",
            self.basedir / "queries_10000_512.fvecs",
        ]

        data_exists = any(f.exists() for f in alt_data_files)
        queries_exist = any(f.exists() for f in alt_query_files)

        if not data_exists or not queries_exist:
            logger.warning(f"OpenImage not found in {self.basedir}")
            logger.info("Download from Google Drive:")
            logger.info("  https://drive.google.com/drive/folders/1ZkWOrja-0A6C9yh3ysFoCP6w5u7oWjQx")
            raise FileNotFoundError("OpenImage dataset not found")

        logger.info(f"✓ OpenImage dataset ready at {self.basedir}")

    def get_base_vectors(self) -> np.ndarray:
        """Load base vectors."""
        if _has_processed_dataset("openimage"):
            return _load_processed_vectors("openimage", "base.npy")
        # Try different file names
        candidates = [
            self.basedir / f"data_{self.nb}_{self.d}",
            self.basedir / f"data_{self.nb}_{self.d}.fvecs",
            self.basedir / "learn_image.fvecs",
            self.basedir / "data_1000000_512.fvecs",
        ]

        for path in candidates:
            if path.exists():
                logger.info(f"Loading base vectors from {path}")
                if path.suffix == '.fvecs':
                    return load_fvecs(str(path))
                else:
                    _, _, vectors = load_bin_format(str(path))
                    return vectors

        raise FileNotFoundError("OpenImage base vectors not found")

    def get_queries(self) -> np.ndarray:
        """Load query vectors."""
        if _has_processed_dataset("openimage"):
            return _load_processed_vectors("openimage", "queries.npy")
        # Try different file names
        candidates = [
            self.basedir / f"queries_{self.nq}_{self.d}",
            self.basedir / f"queries_{self.nq}_{self.d}.fvecs",
            self.basedir / "query_image.fvecs",
            self.basedir / "queries_10000_512.fvecs",
        ]

        for path in candidates:
            if path.exists():
                logger.info(f"Loading queries from {path}")
                if path.suffix == '.fvecs':
                    return load_fvecs(str(path))
                else:
                    _, _, vectors = load_bin_format(str(path))
                    return vectors

        raise FileNotFoundError("OpenImage query vectors not found")

    def get_groundtruth(self, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Load groundtruth if available."""
        gt_candidates = [
            self.basedir / f"gt_{self.nb}_{self.nq}_{self.d}",
            self.basedir / "openimage_groundtruth.ivecs",
        ]

        for gt_path in gt_candidates:
            if gt_path.exists():
                logger.info(f"Loading groundtruth from {gt_path}")
                if gt_path.suffix == '.ivecs':
                    I = load_ivecs(str(gt_path))
                    if k is not None and k < I.shape[1]:
                        I = I[:, :k]
                    return I, None
                else:
                    with open(gt_path, "rb") as f:
                        nq, k_gt = np.fromfile(f, dtype=np.uint32, count=2)
                        I = np.fromfile(f, dtype=np.uint32, count=nq * k_gt).reshape(nq, k_gt)
                        D = np.fromfile(f, dtype=np.float32, count=nq * k_gt).reshape(nq, k_gt)

                    if k is not None and k < k_gt:
                        I = I[:, :k]
                        D = D[:, :k]
                    return I, D

        logger.warning("OpenImage groundtruth not found")
        return None, None


# Dataset registry
DATASETS = {
    'sift1m': SIFT1M,
    'sift10m': SIFT10M,
    'sift100m': SIFT100M,
    'gist1m': GIST1M,
    'glove': GLOVE,
    'deep1m': DEEP1M,
    'deep1b': DEEP1B,
    'deep10m': DEEP10M,
    'msong': MSong,
    'coco': COCO,
    'openimage': OpenImage,
    'random1m128': Random1M128,
    'random-100k': lambda: RandomDataset(100000, 1000, 128),
    'random-1m': lambda: RandomDataset(1000000, 10000, 128),
    'random-small': lambda: RandomDataset(10000, 100, 128),
}


def get_dataset(name: str, **kwargs) -> Dataset:
    """
    Get a dataset by name.

    Args:
        name: Dataset name (e.g., 'sift1m', 'glove', 'random-100k')
        **kwargs: Additional arguments for dataset constructor

    Returns:
        Dataset instance

    Example:
        >>> ds = get_dataset('sift1m')
        >>> ds.prepare()
        >>> vectors = ds.get_base_vectors()
    """
    if name not in DATASETS:
        available = ', '.join(DATASETS.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    dataset_class = DATASETS[name]

    # Handle lambda constructors
    if callable(dataset_class) and not isinstance(dataset_class, type):
        dataset = dataset_class()
    else:
        dataset = dataset_class(**kwargs)

    return dataset


def list_datasets():
    """List all available datasets."""
    print("\nAvailable Datasets:")
    print("=" * 80)

    for name in sorted(DATASETS.keys()):
        try:
            ds = get_dataset(name)
            print(f"  {name:20s} - {ds}")
        except Exception as e:
            print(f"  {name:20s} - Error: {e}")

    print("=" * 80)


if __name__ == "__main__":
    # Test dataset loading
    list_datasets()

    # Test SIFT1M
    print("\nTesting SIFT1M:")
    try:
        ds = get_dataset('sift1m')
        ds.prepare()
        print(f"  Base vectors shape: {ds.get_base_vectors().shape}")
        print(f"  Query vectors shape: {ds.get_queries().shape}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test Random
    print("\nTesting Random dataset:")
    ds = get_dataset('random-small')
    ds.prepare()
    print(f"  Base vectors shape: {ds.get_base_vectors().shape}")
    print(f"  Query vectors shape: {ds.get_queries().shape}")
