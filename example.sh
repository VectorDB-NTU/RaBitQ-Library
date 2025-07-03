# compiling 
mkdir build bin 
cd build 
cmake ..
make -j 10
cd ..

if [ ! -d ./data/gist ]; then
    mkdir -p ./data/gist
fi

if [ ! -f ./data/gist/gist_base.fvecs ]; then
    # download the dataset
    wget -P ./data/gist ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
    tar -xzvf ./data/gist/gist.tar.gz -C ./data/gist
fi

# indexing and querying for symqg
./bin/symqg_indexing ./data/gist/gist_base.fvecs 32 400 ./data/gist/symqg_32.index

./bin/symqg_querying ./data/gist/symqg_32.index ./data/gist/gist_query.fvecs ./data/gist/gist_groundtruth.ivecs

# indexing and querying for RabitQ+ with ivf, please refer to python/ivf.py for more information about clustering
python ./python/ivf.py ./data/gist/gist_base.fvecs 4096 ./data/gist/gist_centroids_4096.fvecs ./data/gist/gist_clusterids_4096.ivecs

./bin/ivf_rabitq_indexing ./data/gist/gist_base.fvecs ./data/gist/gist_centroids_4096.fvecs ./data/gist/gist_clusterids_4096.ivecs 3 ./data/gist/ivf_4096_3.index

./bin/ivf_rabitq_querying ./data/gist/ivf_4096_3.index ./data/gist/gist_query.fvecs ./data/gist/gist_groundtruth.ivecs

# indexing and querying for RabitQ+ with hnsw, do clustering first
python ./python/ivf.py ./data/gist/gist_base.fvecs 16 ./data/gist/gist_centroids_16.fvecs ./data/gist/gist_clusterids_16.ivecs

./bin/hnsw_rabitq_indexing ./data/gist/gist_base.fvecs ./data/gist/gist_centroids_16.fvecs ./data/gist/gist_clusterids_16.ivecs 16 200 5 ./data/gist/hnsw_5.index

./bin/hnsw_rabitq_querying ./data/gist/hnsw_5.index ./data/gist/gist_query.fvecs ./data/gist/gist_groundtruth.ivecs
