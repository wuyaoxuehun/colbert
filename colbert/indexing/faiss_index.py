import time

import faiss

from colbert.indexing.faiss_index_gpu import FaissIndexGPU
from colbert.utils.utils import print_message


class FaissIndexing:
    def __init__(self, dim, partitions):
        self.dim = dim
        self.partitions = partitions

        self.gpu = FaissIndexGPU()
        self.quantizer, self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        # quantizer = faiss.IndexFlatL2(self.dim)  # faiss.IndexHNSWFlat(dim, 32)
        # quantizer = faiss.IndexFlatIP(self.dim)
        quantizer = faiss.IndexFlatL2(self.dim)
        # index = faiss.IndexIVFPQ(quantizer, self.dim, self.partitions, 16, 8)
        # index = faiss.IndexIVFPQ(quantizer, self.dim, self.partitions, 32, 8)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.partitions, 16, 8)
        # index = quantizer
        return quantizer, index
        # return quantizer

    def train(self, train_data):
        print_message(f"#> Training now (using {self.gpu.ngpu} GPUs)...")

        if self.gpu.ngpu > 0:
            self.gpu.training_initialize(self.index, self.quantizer)

        s = time.time()
        print('getting train')
        self.index.train(train_data)
        print(time.time() - s)
        print('training last')

        if self.gpu.ngpu > 0:
            self.gpu.training_finalize()

    def add(self, data):
        print_message(f"Add data with shape {data.shape} (offset = {self.offset})..")

        if self.gpu.ngpu > 0 and self.offset == 0:
            self.gpu.adding_initialize(self.index)

        if self.gpu.ngpu > 0:
            print('using gpu add')
            self.gpu.add(self.index, data, self.offset)
            print('finished gpu add')
        else:
            self.index.add(n=len(data), x=data)

        self.offset += data.shape[0]

    def save(self, output_path):
        print_message(f"Writing index to {output_path} ...")

        self.index.nprobe = 10  # just a default
        faiss.write_index(self.index, output_path)
