# compiling
pip install .

# Download the dataset
mkdir -p ./data/gist
if [ ! -f ./data/gist/gist.tar.gz ]; then
    wget -P ./data/gist ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
fi
tar -xzvf ./data/gist/gist.tar.gz -C ./data

# indexing and querying for symqg
python  ./sample/python/symqg_indexing.py --max-degree 32 --ef-construction 400 ./data/gist/gist_base.fvecs ./data/gist/symqg_32.index 

python  ./sample/python/symqg_querying.py ./data/gist/symqg_32.index ./data/gist/gist_query.fvecs ./data/gist/gist_groundtruth.ivecs

# indexing and querying for RabitQ+ with ivf
python ./sample/python/ivf_rabitq_indexing.py --total-bits 5 --num-clusters 4096 ./data/gist/gist_base.fvecs ./data/gist/ivf_4096_5.index

python ./sample/python/ivf_rabitq_querying.py ./data/gist/ivf_4096_5.index ./data/gist/gist_query.fvecs ./data/gist/gist_groundtruth.ivecs

# indexing and querying for RabitQ+ with hnsw
python ./sample/python/hnsw_rabitq_indexing.py --total-bits 5 --num-clusters 16 --degree 16 --ef-construction 200 ./data/gist/gist_base.fvecs ./data/gist/hnsw_5.index

python ./sample/python/hnsw_rabitq_querying.py ./data/gist/hnsw_5.index ./data/gist/gist_query.fvecs ./data/gist/gist_groundtruth.ivecs
