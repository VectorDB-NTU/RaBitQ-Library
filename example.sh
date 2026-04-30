#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/gist}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/gist_example}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

BASE="${DATA_DIR}/gist_base.fvecs"
QUERY="${DATA_DIR}/gist_query.fvecs"
GT="${DATA_DIR}/gist_groundtruth.ivecs"

mkdir -p "${ROOT_DIR}/bin" "${DATA_DIR}" "${LOG_DIR}"
cd "${ROOT_DIR}"

run_and_log() {
    local name="$1"
    shift
    local log_file="${LOG_DIR}/${name}.log"

    echo
    echo "===== ${name} ====="
    "$@" 2>&1 | tee "${log_file}"
}

print_recall_summary() {
    local name="$1"
    local log_file="${LOG_DIR}/${name}.log"

    echo
    echo "----- ${name} summary -----"
    awk '
        BEGIN { IGNORECASE = 1 }
        /^[[:space:]]*(EF|nprobe)[[:space:]]/ {
            in_table = 1
            next
        }
        in_table && NF >= 3 && $1 ~ /^[0-9]+$/ {
            last = $0
            if (best_line == "" || ($3 + 0) > best_recall) {
                best_recall = $3 + 0
                best_line = $0
            }
        }
        END {
            if (best_line != "") {
                print "best_recall_line: " best_line
                print "last_line:        " last
            } else {
                print "No recall table found. Please inspect " FILENAME
            }
        }
    ' "${log_file}"
}

echo "===== build ====="
cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build"
cmake --build "${ROOT_DIR}/build" -j "${BUILD_JOBS}"

if [[ ! -f "${BASE}" || ! -f "${QUERY}" || ! -f "${GT}" ]]; then
    echo
    echo "===== download gist ====="
    if [[ ! -f "${DATA_DIR}/gist.tar.gz" ]]; then
        wget -P "${DATA_DIR}" ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
    fi
    tar -xzvf "${DATA_DIR}/gist.tar.gz" -C "${DATA_DIR}"
fi

echo
echo "Logs will be written to ${LOG_DIR}"

run_and_log symqg_indexing \
    "${ROOT_DIR}/bin/symqg_indexing" "${BASE}" 32 400 "${DATA_DIR}/symqg_32.index"
run_and_log symqg_querying \
    "${ROOT_DIR}/bin/symqg_querying" "${DATA_DIR}/symqg_32.index" "${QUERY}" "${GT}"

run_and_log ivf_clustering_4096 \
    python "${ROOT_DIR}/python/ivf.py" "${BASE}" 4096 \
        "${DATA_DIR}/gist_centroids_4096.fvecs" "${DATA_DIR}/gist_clusterids_4096.ivecs"
run_and_log ivf_rabitq_indexing \
    "${ROOT_DIR}/bin/ivf_rabitq_indexing" "${BASE}" \
        "${DATA_DIR}/gist_centroids_4096.fvecs" "${DATA_DIR}/gist_clusterids_4096.ivecs" \
        3 "${DATA_DIR}/ivf_4096_3.index"
run_and_log ivf_rabitq_querying \
    "${ROOT_DIR}/bin/ivf_rabitq_querying" "${DATA_DIR}/ivf_4096_3.index" "${QUERY}" "${GT}"

run_and_log hnsw_clustering_16 \
    python "${ROOT_DIR}/python/ivf.py" "${BASE}" 16 \
        "${DATA_DIR}/gist_centroids_16.fvecs" "${DATA_DIR}/gist_clusterids_16.ivecs"
run_and_log hnsw_rabitq_indexing \
    "${ROOT_DIR}/bin/hnsw_rabitq_indexing" "${BASE}" \
        "${DATA_DIR}/gist_centroids_16.fvecs" "${DATA_DIR}/gist_clusterids_16.ivecs" \
        16 200 5 "${DATA_DIR}/hnsw_5.index"
run_and_log hnsw_rabitq_querying \
    "${ROOT_DIR}/bin/hnsw_rabitq_querying" "${DATA_DIR}/hnsw_5.index" "${QUERY}" "${GT}"

echo
echo "===== recall summaries ====="
print_recall_summary symqg_querying
print_recall_summary ivf_rabitq_querying
print_recall_summary hnsw_rabitq_querying
