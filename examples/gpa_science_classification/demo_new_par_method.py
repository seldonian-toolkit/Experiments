import numpy as np

import concurrent.futures
from tqdm import tqdm

def chunker(data_fracs,n_trials,n_workers):
    n_tot = len(data_fracs)*n_trials
    chunk_size = n_tot // n_workers
    chunk_sizes = []
    for i in range(0,n_tot,chunk_size):
        if (i+chunk_size) > n_tot:
            chunk_sizes.append(n_tot-i)
        else:
            chunk_sizes.append(chunk_size)
    print(f"chunk_sizes: {chunk_sizes}")
    assert sum(chunk_sizes) == n_tot
    # flatten data fracs and trials so we can make chunked tuples ((data_frac,trial_index),...) 
    data_fracs_vector = np.array([x for x in data_fracs for y in range(n_trials)])
    trials_vector = np.array(
        [x for y in range(len(data_fracs)) for x in range(n_trials)]
    )
    chunked_list = []
    start_ix = 0
    for chunk_size in chunk_sizes:
        chunked_list.append([[data_fracs_vector[ii],trials_vector[ii]] for ii in range(start_ix,start_ix+chunk_size)])
        start_ix += chunk_size
    return chunked_list


def func_to_parallelize(tup_list):
    # print(f"In a parallel process with tup_list: {tup_list}")
    for tup in tup_list:
        data_frac,trial_i = tup
        print(f"Processing data_frac,trial_i: {data_frac},{trial_i}")
    return

if __name__ == "__main__":
    data_fracs = 0.1*np.arange(1,11)
    n_trials = 10
    n_workers = 4
    chunked_list = chunker(data_fracs,n_trials,n_workers)
    print(f"chunked_list: {chunked_list}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit the square_elements function to the executor for each chunk
        # futures = [executor.submit(func_to_parallelize, num) for num in chunk_sizes]

        results = tqdm(executor.map(func_to_parallelize, chunked_list),total=len(chunked_list))
        for res in results:
            print(res)    