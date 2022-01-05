import json, collections, logging, os
import requests
import numpy as np
import pandas as pd
from operator import itemgetter as at
from tqdm import tqdm
from encoders import parse_schema
from pathlib import Path
from joblib import delayed, Parallel


def main(args):
	data_dir = Path(args.data_dir)
	with open(data_dir / "schema.json",'r') as f:
		schema = json.load(f)
	partition_dir=Path(args.partition_dir)
	partition_dir.mkdir(exist_ok=True)
	# copy schema
	logging.debug("Copy schema file")
	with (partition_dir/"schema.json").open('w') as f:
		json.dump(schema, f)
	#

	schema = parse_schema(schema)
	encode = schema["encode_fn"]

	partitions = collections.defaultdict(pd.DataFrame)

	logging.debug("Repartition")
	p_files = [f for f in data_dir.glob("*.parquet")]
	for partition in tqdm(schema["partitions"]):
		kv=list(zip(schema["filters"],partition))
		subindex = 0
		for f in p_files:
			df = pd.read_parquet(f)
			for k,v in kv:
				df=df[df[k]==v]
			n = len(df)//args.repartition_limit
			for sp in range(n+1):
				increment_df = df.iloc[:args.repartition_limit]
				if len(partitions[partition+(subindex,)])+len(increment_df)>args.repartition_limit:
					subindex+=1
				partitions[partition+(subindex,)]=pd.concat([partitions[partition+(subindex,)],increment_df])
				df=df.iloc[args.repartition_limit:]
				if len(increment_df)>0:
					subindex+=1

	logging.debug("Save to local json files")
	# save jsons
	j_files = []
	for partition,df in partitions.items():
		p = args.partition_sep.join(map(str,partition))
		j_file = partition_dir/(p)
		df.to_json(j_file, orient='records')
		j_files.append(j_file)

	logging.debug("Encoding each partition")
	def fstack(f,lst):
		"""Combines np.vstack and parallelism"""
		if len(lst)==0:
			return np.array([])
		elif len(lst)<1024:
			return np.vstack(Parallel(-1)([f(d) for d in lst]))
		return np.vstack([fstack(f,lst[:len(lst)//2]),fstack(f,lst[len(lst)//2:])])

	start_num_idx=0
	for jf in tqdm(j_files):
		partition = jf.name.split(args.partition_sep)
		subindex = int(partition[-1])
		partition = partition[:-1]
		with open(jf,'r') as f:
			data = json.load(f)
		arr = fstack(delayed(encode), data)
		np.save(jf.with_suffix('.npy'), arr)
		with jf.with_suffix('.meta').open('w') as f:
			index_num = schema["index_num"](dict(zip(schema["filters"],partition)))
			meta = {
				"size":len(arr),
				"start_num_idx":start_num_idx,
				"ids": [d['id'] for d in data],
				"partition": partition,
				"subindex": subindex,
				"index_num":index_num,
				"metric":schema["metric"],
				"dim":schema["dim"],
				}
			json.dump(meta,f)
		start_num_idx+=len(arr)
		#jf.unlink(missing_ok=True) #TODO: why not working ?
		# os.remove(str(jf))



if __name__=="__main__":
    import sys
    from argparse import ArgumentParser
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    argparse = ArgumentParser()
    argparse.add_argument('--data_dir',     default=str(Path(__file__).absolute().parent.parent / "data/ny")           ,type=str, help='data dir')
    argparse.add_argument('--partition_dir', default=str(Path(__file__).absolute().parent.parent / "data/ny/partitioned")   ,type=str, help='partition dir')
    argparse.add_argument('--partition_sep', default='~'   ,type=str, help='partition separator')
    argparse.add_argument('--repartition_limit', default=10000   ,type=int, help='repartition size limit')
    sys.exit(main(argparse.parse_args()))

