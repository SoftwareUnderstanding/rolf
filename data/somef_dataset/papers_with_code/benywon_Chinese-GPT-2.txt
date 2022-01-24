An extra-easy Chinese GPT-2 model based on Pytorch 
---------
###Information:

The model is based on shared weight Transformer (a.k.a albert https://arxiv.org/abs/1909.11942)

the model is trained on 4.5 billion chinese internet web pages

The model has 2048 hidden size and 12 layers, which is about 97 million parameters.

------
###Usage

First, install extra packages by: pip3 install -r requirements.txt

Then, download the pre-trained model from baidu netdisk:
url: https://pan.baidu.com/s/1uW7sBytbtS5LEmSOlWlr-A 
code: nafi 

python3 interface --input=input.txt  --output=output.txt --cuda=True --input_max_length=32 --output_max_length=128

the result will be generated to the output file

----
###Examples

If your input is '如何做一个好人，首先我们应该主要通过学习来'

The output would be

'维护自己,从自身的做起,从小做起,学会做一个好人,首先培养自己人格。做人生最不应该要的孩子。作为一名学生,我们应该培养自己的人格,做一个好孩子,并不断地总结自己对学校和老师的态度和结果。在学习上要多学习,学会学习、运用、实践。学习一定要先学习,才能形成良好的学习精神。学习要有自己独到的见解,能运用好你的知识,才能形成良好的学习心理。要学会独立做事、求实干。当别人知道你在用人的时候,应该冷静的思考,不要把自己看得的看的太重。要学会独立做事,要学会独立做事。当别人知道自己在你用人的时候,应该冷静的思考,不要把自己看得的看的太重。要学会独立做事。学会独立做事的人,自己应该做到。不要对他人说别人的好,他们也会拒绝别人。学会自作主张。自己不应该追求自我。自己是一个充满正能量的人,而这种正能量,才是未来生活不可缺少的希望。对他人来说,别人可以给予你的最大机会,但是你不能。而且很多时候,我要学会“自作主张”。凡事都要自作主张,只为自己的行为去奋斗。不要把自己看成是好人、好一个人,人生就是一场活的'








 
