int hnsw_index_header_smoke();
int ivf_index_header_smoke();

int main() { return hnsw_index_header_smoke() + ivf_index_header_smoke(); }
