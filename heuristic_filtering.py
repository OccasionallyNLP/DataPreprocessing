from tqdm import tqdm
import multiprocessing
import json
import os
import copy
import datatrove
from utils import *
import argparse
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.data import Document
from typing import List
import time
import logging

def get_args():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--logging_path', type=str)
    parser.add_argument('--input_path', type=str)
    args  = parser.parse_args()
    return args

def apply_filter(filter_for_fineweb, dataset_i:datatrove.data.Document):
    result = filter_for_fineweb.filter(dataset_i)
    if result==True:
        return True
    else:
        dataset_i.metadata['reason'] = result[1]
        return dataset_i

def write_json(address, name, data_i):
    with open(os.path.join(address,name+'.jsonl'),'a',encoding = 'utf-8') as f:
        f.write(json.dumps(data_i,ensure_ascii=False)+'\n')
    
    
def map_filter(data_i, name, output_path):
    dataset_i = Document(text=data_i['text'], id=data_i['adlr_id'], metadata=data_i)
    # url
    url_filter=URLFilter()
    is_true = apply_filter(url_filter, dataset_i)
    if is_true != True:
        write_json(os.path.join(output_path,'removed/url'), name, is_true.metadata)
        return
    else:
        gopher_repetition_filter = GopherRepetitionFilter(language='ko')
        is_true = apply_filter(gopher_repetition_filter, dataset_i)
        if is_true != True:
            write_json(os.path.join(output_path,'removed/gopher_rep'), name, is_true.metadata)
            return
        else:
            gopher_quality_filter = GopherQualityFilter(min_stop_words=None,min_avg_word_length=None,language='ko')
            is_true = apply_filter(gopher_quality_filter, dataset_i)
            if is_true != True:
                write_json(os.path.join(output_path,'removed/gopher_quality'), name, is_true.metadata)
                return
            else:
                c4_filter = C4QualityFilter(filter_no_terminal_punct=False,language='ko')
                is_true = apply_filter(c4_filter, dataset_i)
                if is_true != True:
                    write_json(os.path.join(output_path,'removed/c4'), name, is_true.metadata)
                    return
                else:
                    write_json(output_path, name, dataset_i.metadata)
                    return

if __name__=='__main__':
    args = get_args()
    import time
    now = time.time()
    logging.basicConfig(filename=args.logging_path, level=logging.INFO)
    cpu_count = int(os.cpu_count()*0.8) # heuristic
    os.makedirs(args.output_dir, exist_ok=True)
    for filter_name in ['url','gopher_rep','gopher_quality','c4']:
        os.makedirs(os.path.join(args.output_dir,f'removed/{filter_name}'), exist_ok=True)    
    data = load_jsonl(args.input_path, True)
    name = os.path.splitext(os.path.split(args.input_path)[1])[0]
    inputs = [(a, name, args.output_dir) for a in data]
    pool = multiprocessing.Pool(processes=cpu_count)
    pool.starmap_async(map_filter, inputs)
    pool.close()
    pool.join()
    tmp = (ime.time()-now)//60
    print(f'time-eplased: {tmp} min')
    print(f'{args.input_path} is done')
