# Project-7
Receptive Field based on the network in Page 6 of  https://arxiv.org/pdf/1409.4842.pdf 

**Layers**|**k**|**s**|**jin**|**jout**|**rin**|**rout**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
Conv|7|2|1|2|1|7
Maxpooling|3|2|2|4|7|11
Conv|3|1|4|4|11|19
Maxpooling|3|2|4|8|19|27
Conv|5|1|8|8|27|59
Conv|5|1|8|8|59|91
Maxpooling|3|2|8|16|91|107
Conv|5|1|16|16|107|171
Conv|5|1|16|16|171|235
Conv|5|1|16|16|235|299
Conv|5|1|16|16|299|363
Conv|5|1|16|16|363|427
Maxpooling|3|2|16|32|427|459
Conv|5|1|32|32|459|587
Conv|5|1|32|32|587|715
Avg Pool|7|1|32|32|715|907
