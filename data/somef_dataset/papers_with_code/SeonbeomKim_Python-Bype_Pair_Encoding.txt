# Python-Byte_Pair_Encoding
Byte Pair Encoding (BPE)


## Env
   * Python 3
   * Numpy 1.15
   * tqdm
   * multiprocessing
   
## Paper
   * Byte-Pair Encoding (BPE): https://arxiv.org/abs/1508.07909  
      
## Command
   * learn BPE from document
```
python bpe_learn.py 
	-train_path 1_document 2_document ... K_document
	-voca_out_path voca_path/voca_file_name
	-bpe_out_path 1_BPE_document 2_BPE_document ... K_BPE_document
	-train_voca_threshold 1 
	-num_merges 30000 
	-multi_proc=-1 (-1:use all process, 1:not use)
	-final_voca_size 30000 or -final_voca_threshold 50
```

   * apply BPE to document
```
python bpe_apply.py
	-data_path 1_document 2_document ... K_document
	-voca_path voca_path/voca_file_name
	-bpe_out_path 1_BPE_document 2_BPE_document ... K_BPE_document
```
   
## Reference
   * https://lovit.github.io/nlp/2018/04/02/wpm/
