import struct
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from tqdm import tqdm
import os
from os import makedirs
from os.path import join
from os.path import exists
from itertools import groupby
import requests
from tqdm import tqdm
import shutil
import gzip

class one_sample_object:
    def __init__(self):
        self.image_lt  = None
        self.image_rt  = None
        self.category  = None
        self.instance  = None
        self.elevation = None
        self.azimuth   = None
        self.lighting  = None

class smallNORB:
    n_samples = 24300
    def __init__(self, dataset_dir, set):
        self.set = set
        self.dataset_dir = dataset_dir
        self.dataset_files = {
            'train': {
                'cat':  join(dataset_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat'),
                'info': join(dataset_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat'),
                'dat':  join(dataset_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
            },
            'test':  {
                'cat':  join(dataset_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat'),
                'info': join(dataset_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat'),
                'dat':  join(dataset_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')
            }
        }
        
        if (self.set is 'train'):
            self.process_downloaded_data('train', 'dat', 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz')
            self.process_downloaded_data('train', 'cat', 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz')
            self.process_downloaded_data('train', 'info', 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz')
        if (self.set is 'test'):
            self.process_downloaded_data('test', 'dat', 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz')
            self.process_downloaded_data('test', 'cat', 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz')
            self.process_downloaded_data('test', 'info', 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz')
        
        self.data = {
            self.set : [one_sample_object() for _ in range(smallNORB.n_samples)],
        }
        
        self.fill_data_in_objects(self.set)
            
    def process_downloaded_data(self, set, type, url):
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        if not os.path.exists(self.dataset_files[set][type]):
            if not os.path.exists(self.dataset_files[set][type]+'.gz'):
                print("Download "+set+" dataset - file: "+type)
                self.download(url, self.dataset_dir)
            if not os.path.exists(self.dataset_files[set][type]):
                with gzip.open(self.dataset_files[set][type]+'.gz', 'rb') as f_in:
                    print("Unpack "+set+" dataset - file: "+type)
                    with open(self.dataset_files[set][type], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    
    def download(self, url, dir):
        chunk_size = 1024
        r = requests.get(url, stream = True)
        total_size = int(r.headers['content-length'])
        filename = url.split('/')[-1]

        with open(join(dir, filename), 'wb') as f:
            for data in tqdm(iterable = r.iter_content(chunk_size = chunk_size), total = total_size/chunk_size, unit = 'KB'):
                f.write(data)    
    def load_data(self):
        if self.set is 'train':
            train_dat_lt = []
            train_dat_rt = []
            train_cat = []
            train_info = []
            for i, norb_example in enumerate(self.data['train']):
                train_dat_lt.append(norb_example.image_lt)
                train_dat_rt.append(norb_example.image_rt)
                train_cat.append(norb_example.category)
                train_info.append([norb_example.instance, norb_example.elevation, norb_example.azimuth, norb_example.lighting])
            return (np.array(train_dat_lt).astype(np.float32), np.array(train_dat_rt).astype(np.float32), np.array(train_cat).astype(np.int32), np.array(train_info).astype(np.int32))
        if self.set is 'test':
            test_dat_lt = []
            test_dat_rt = []
            test_cat = []
            test_info = []
            for i, norb_example in enumerate(self.data['test']):
                test_dat_lt.append(norb_example.image_lt)
                test_dat_rt.append(norb_example.image_rt)
                test_cat.append(norb_example.category)
                test_info.append([norb_example.instance, norb_example.elevation, norb_example.azimuth, norb_example.lighting])
            return (np.array(test_dat_lt).astype(np.float32), np.array(test_dat_rt).astype(np.float32), np.array(test_cat).astype(np.int32), np.array(test_info).astype(np.int32))
        
        
        

    def fill_data_in_objects(self, dataset_split):
        print("Parse binary files .mat to numpy arrays")
        dat_data  = self.parse_NORB_dat_file(self.dataset_files[dataset_split]['dat'])
        cat_data  = self.parse_NORB_cat_file(self.dataset_files[dataset_split]['cat'])
        info_data = self.parse_NORB_info_file(self.dataset_files[dataset_split]['info'])
        for i, small_norb_example in enumerate(self.data[dataset_split]):
            small_norb_example.image_lt   = dat_data[2 * i]
            small_norb_example.image_rt   = dat_data[2 * i + 1]
            small_norb_example.category  = cat_data[i]
            small_norb_example.instance  = info_data[i][0]
            small_norb_example.elevation = info_data[i][1]
            small_norb_example.azimuth   = info_data[i][2]
            small_norb_example.lighting  = info_data[i][3]
    
    @staticmethod
    def matrix_type_from_magic(magic_number):
        convention = {'1E3D4C51': 'single precision matrix',
                      '1E3D4C52': 'packed matrix',
                      '1E3D4C53': 'double precision matrix',
                      '1E3D4C54': 'integer matrix',
                      '1E3D4C55': 'byte matrix',
                      '1E3D4C56': 'short matrix'}
        magic_str = bytearray(reversed(magic_number)).hex().upper()
        return convention[magic_str]
        
    @staticmethod
    def parse_small_NORB_header(file_pointer):
        # Read magic number
        magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        file_header_data = {'magic_number': magic,
                            'matrix_type': smallNORB.matrix_type_from_magic(magic),
                            'dimensions': dimensions}
        return file_header_data

    @staticmethod
    def parse_NORB_cat_file(file_path):
    
        with open(file_path, mode='rb') as f:
            header = smallNORB.parse_small_NORB_header(f)

            num_examples, = header['dimensions']

            struct.unpack('<BBBB', f.read(4))  # ignore this integer
            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            examples = np.zeros(shape=num_examples, dtype=np.int32)
            for i in tqdm(range(num_examples), desc='Loading categories...'):
                category, = struct.unpack('<i', f.read(4))
                examples[i] = category

            return examples

    @staticmethod
    def parse_NORB_dat_file(file_path):
        with open(file_path, mode='rb') as f:

            header = smallNORB.parse_small_NORB_header(f)

            num_examples, channels, height, width = header['dimensions']

            examples = np.zeros(shape=(num_examples * channels, height, width), dtype=np.uint8)

            for i in tqdm(range(num_examples * channels), desc='Loading images...'):

                # Read raw image data and restore shape as appropriate
                image = struct.unpack('<' + height * width * 'B', f.read(height * width))
                image = np.uint8(np.reshape(image, newshape=(height, width)))

                examples[i] = image

        return examples

    @staticmethod
    def parse_NORB_info_file(file_path):
        
        with open(file_path, mode='rb') as f:

            header = smallNORB.parse_small_NORB_header(f)

            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            num_examples, num_info = header['dimensions']

            examples = np.zeros(shape=(num_examples, num_info), dtype=np.int32)

            for r in tqdm(range(num_examples), desc='Loading info...'):
                for c in range(num_info):
                    info, = struct.unpack('<i', f.read(4))
                    examples[r, c] = info

        return examples