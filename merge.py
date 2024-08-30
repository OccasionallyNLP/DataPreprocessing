import os
import json
from tqdm import tqdm
import random
import argparse
import glob
import multiprocessing
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--output_data',type=str)
    args = parser.parse_args()
    return args

def star_map(in_path, out_path):
    fout = open(out_path,'a',encoding='utf-8')
    fin = open(in_path, 'r', encoding='utf-8')
    for i in tqdm(fin):
        t = json.loads(i)
        fout.write(json.dumps(t, ensure_ascii=False)+'\n')
    return

def main(args):
    #fout = open('/gpfs/bigmodel/midm_proxy/Core-r0.5.0/data/raw/midm_v2/en/fineweb_4k.jsonl','a',encoding='utf-8')
    now = datetime.now()
    input_paths = glob.glob(args.input_data+'/*')
    print(len(input_paths))
    input_datas = input_paths[:int(len(input_paths)*0.35)]
    inputs = [(i, args.output_data) for i in input_datas]

    # heuristic
    pool = multiprocessing.Pool(processes=int(os.cpu_count()*0.9))
    pool.starmap_async(star_map, inputs)
    pool.close()
    pool.join()
    done = datetime.now()
    print(f'time elapsed : {done-now}')

if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
