# compiling 
mkdir build bin 
cd build 
cmake ..
make 

# Download the dataset
wget -P ./data/msong ftp://ftp.irisa.fr/local/texmex/corpus/msong.tar.gz
tar -xzvf ./data/msong/msong.tar.gz -C ./data/msong

# indexing and querying for symqg
./bin/symqg_indexing ./data/msong/msong_base.fvecs 32 400 ./data/msong/symqg_32.index

./bin/symqg_querying ./data/msong/symqg_32.index ./data/msong/msong_query.fvecs ./data/msong/msong_groundtruth.ivecs

# indexing and querying for RabitQ+ with ivf, please refer to python/ivf.py for more information about clustering
python ./python/ivf.py ./data/msong/msong_base.fvecs 4096 ./data/msong/msong_centroids_4096.fvecs ./data/msong/msong_clusterids_4096.ivecs

./bin/ivf_rabitq_indexing ./data/msong/msong_base.fvecs ./data/msong/msong_centroids_4096.fvecs ./data/msong/msong_clusterids_4096.ivecs 3 ./data/msong/ivf_4096_3.index

./bin/ivf_rabitq_querying ./data/msong/ivf_4096_3.index ./data/msong/msong_query.fvecs ./data/msong/msong_groundtruth.ivecs

# indexing and querying for RabitQ+ with hnsw, do clustering first
python ./python/ivf.py ./data/msong/msong_base.fvecs 16 ./data/msong/msong_centroids_16.fvecs ./data/msong/msong_clusterids_16.ivecs

./bin/hnsw_rabitq_indexing ./data/msong/msong_base.fvecs ./data/msong/msong_centroids_16.fvecs ./data/msong/msong_clusterids_16.ivecs 16 200 5 ./data/msong/hnsw_5.index

./bin/hnsw_rabitq_querying ./data/msong/hnsw_5.index ./data/msong/msong_query.fvecs ./data/msong/msong_groundtruth.ivecs
