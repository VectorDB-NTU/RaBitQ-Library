int hnsw_header_smoke();
int ivf_header_smoke();
int symqg_header_smoke();

int main() { return hnsw_header_smoke() + ivf_header_smoke() + symqg_header_smoke(); }
