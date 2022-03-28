## これはノートです。

# AWS のシークレットキーの作成
AWSコンソールから

# AWS CLIのインストール
PS aws --version  
バージョン確認

# AWS Access Key等の設定
```bash
PS aws configure
```
下記の設定を行う  
AWS Access Key ID  
AWS Secret Access Key  
Default region name # ap-northeast-1 (東京リージョン)を指定するのがよい  
Default output format   # jsonを指定 # WEBページだと大文字だが小文字で指定しろ
```bash
PS cat ~/.aws/credentials
[default]
aws_access_key_id = XXXXXXXXXXXXXXXXXX
aws_secret_access_key = YYYYYYYYYYYYYYYYYYY

PS cat ~/.aws/config
[profile default]
region = ap-northeast-1
output = json
```

# S3バケット作成・アップロード・確認・削除
```bash
# S3 bucketの作成
PS $bucketName="mybucket-twobooks20200712"
PS echo $bucketName
mybucket-twobooks20200712
PS echo "s3://${bucketName}"
s3://mybucket-twobooks20200712
PS aws s3 mb "s3://${bucketName}"
make_bucket: mybucket-twobooks20200712
PS aws s3 ls
2020-07-12 11:44:55 mybucket-twobooks20200712

# S3 bucket へのアップロード
PS echo "Hello world!" > hello_world.txt
PS aws s3 cp hello_world.txt "s3://${bucketName}/hello_world.txt"

# S3 bucket 内の確認
aws s3 ls "s3://${bucketName}" --human-readable

# S3 bucket の削除
#デフォルトでは，バケットは空でないと削除できない．空でないバケットを強制的に削除するには --force のオプションを付ける．
aws s3 rb "s3://${bucketName}" --force
```

# CloudFormation
AWSでのリソースを管理するための仕組み.  
Infrastructure as Code (IaC)．  
CloudFormation を記述するには，基本的に JSON (JavaScript Object Notation) と呼ばれるフォーマットを使う．

# AWS CDK
CloudFormation 記述が複雑.  
基本的に変数やクラスといった概念が使えない　(厳密には，変数に相当するような機能は存在する)．  
また，記述の多くの部分は繰り返しが多く，自動化できる部分も多い．  
解決してくれるのが， AWS Cloud Development Kit (CDK) ．  
CDKは Python などのプログラミング言語を使って CloudFormation を自動的に生成してくれるツール．  
CDKを使うことで，CloudFormation に相当するクラウドリソースの記述を，より親しみのあるプログラミング言語を使って行うことができる．かつ，典型的なリソース操作に関してはパラメータの多くの部分を自動で決定してくれるので，記述しなければならない量もかなり削減される．

# SSH コマンドの基本的な使い方
```bash
$ ssh <user name>@<host name>
```
user name は接続する先のユーザー名．  
host name はアクセスする先のサーバーのIPアドレスやDNSによるホストネームが入る． 
SSH コマンドでは，ログインのために使用する秘密鍵ファイルを -i もしくは --identity_file のオプションで指定することができる．
```bash
# 基本書式
$ ssh -i Ec2SecretKey.pem <user name>@<host name>
# .sshフォルダからのSSH
C:\Users\2books\.ssh>ssh -i "HirakeGoma.pem" ec2-user@<ip address>
# Linuxと同じ形でも使える
$ ssh -i ~/.ssh/HirakeGoma.pem ec2-user@<IP address>
# ポートフォワーディングで接続
.ssh$ ssh -i HirakeGoma.pem -L localhost:8931:localhost:8888 ec2-user@<IP address>
# PSなら以下でOK
PS ssh -i $HOME/.ssh/HirakeGoma.pem -L localhost:8931:localhost:8888 ec2-user@13.115.248.26
# SSH接続の終了
$ exit
```

# chmod 400 をWindowsで
https://qiita.com/sumomomomo/items/28d54e35bfa5bc524cf5  
PowerShellで下記
```bash
$path = "$HOME\.ssh\HirakeGoma.pem"
icacls.exe $path /reset
icacls.exe $path /GRANT:R "$($env:USERNAME):(R)"
icacls.exe $path /inheritance:r
```

# Amazon EC2 キーペアの作成、表示、削除
https://docs.aws.amazon.com/ja_jp/cli/latest/userguide/cli-services-ec2-keypairs.html  
あまりPowershell使いたくないからコンソールで作ったほうが早いかも（pemファイルのダウンロードまでされる）
```bash
PS C:\>aws ec2 create-key-pair --key-name HirakeGoma --query 'KeyMaterial' --output text | out-file -encoding ascii -filepath HirakeGoma.pem

# キーの削除
$ aws ec2 delete-key-pair --key-name "HirakeGoma"
# ローカルのキーも削除しとく
```

# VPC(Virtual Private Cloud)
VPCはAWS上にプライベートな仮想ネットワーク環境を構築するツール．  
複数のサーバーを連動させて計算を行う場合に互いのアドレスなどを管理する必要があり，そのような場合にVPCは有用．  
EC2インスタンスは必ずVPCの中に配置されなければならない，という制約がある.ハンズオンでもミニマルなVPCを構成している．
```python
from aws_cdk import (core, aws_ec2 as ec2,)
import os

class MyFirstEc2(core.Stack):

    def __init__(self, scope: core.App, name: str, key_name: str, **kwargs) -> None:
        super().__init__(scope, name, **kwargs)

        # <1>
        vpc = ec2.Vpc(
            self, "MyFirstEc2-Vpc",
            max_azs=1, #avaialibility zoneの設定．障害などを気にしないなら1でOK
            cidr="10.10.0.0/23", # VPC内のIPv4のレンジを指定．
            # 10.10.0.0/23 は10.10.0.0 ~ 10.10.1.255の512個のアドレス範囲．
            subnet_configuration=[ #サブネットの設定．
            # priavte subnet と public subnet の二種類ある.
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                )
            ],
            nat_gateways=0,
            # これを0にしておかないと，NAT Gateway の利用料金が発生してしまうので，注意！
        )
```

# Security Group
EC2インスタンスに付与することのできる仮想ファイアーウォール  
特定のIPアドレスから来た接続を許したり　(インバウンド・トラフィックの制限) ，逆に特定のIPアドレスへのアクセスを禁止したり (アウトバウンド・トラフィックの制限) することができる．  
```python
from aws_cdk import (core, aws_ec2 as ec2,)
import os

class MyFirstEc2(core.Stack):

    def __init__(self, scope: core.App, name: str, key_name: str, **kwargs) -> None:
        super().__init__(scope, name, **kwargs)

        # <2>
        sg = ec2.SecurityGroup(
            # EC2にログインしたのち，ネットからプログラムなどをダウンロードできるよう，
            # allow_all_outbound=True のパラメータを設定している．
            self, "MyFirstEc2Vpc-Sg",
            vpc=vpc,
            allow_all_outbound=True,
        )
        sg.add_ingress_rule(
            # SSHによる外部からの接続を許容
            # すべてのIPv4アドレスからのポート22番へのアクセスを許容している．
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(22),
        )
```

# EC2
t2.micro であれば月に750時間までは無料で利用することができる．  
t2.micro の $0.0116 / hour という金額は，on-demandインスタンスというタイプを選択した場合の価格である．  
Spot instance と呼ばれるインスタンスも存在する．Spot instance は，AWSのデータセンターの負荷が増えた場合，AWSの判断により強制シャットダウンされる可能性があるが，その分大幅に安い料金設定になっている．  
科学計算で，コストを削減する目的で，このSpot Instanceを使う事例も報告されている
```python
from aws_cdk import (core, aws_ec2 as ec2,)
import os

class MyFirstEc2(core.Stack):

    def __init__(self, scope: core.App, name: str, key_name: str, **kwargs) -> None:
        super().__init__(scope, name, **kwargs)

                # <3>
        host = ec2.Instance(
            self, "MyFirstEc2Instance",
            instance_type=ec2.InstanceType("t2.micro"), # t2.micro を設定
            machine_image=ec2.MachineImage.latest_amazon_linux(),# OSはAmazon Linux 
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            security_group=sg,
            key_name=key_name
        )
```

# CDKのdeployまで
空フォルダを初期化してから
```bash
$ md cdkdir
$ cd cdkdir
$ cdk init app --language=python
$ python -m venv .env
### Powershellの場合 ###
PS Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# 又は
PS Set-ExecutionPolicy RemoteSigned -Scope Process
# 以上を実行してからその後
PS .env\Scripts\activate 
########################
$ .env\Scripts\activate.bat
# ここから仮想環境
(.env)$ pip install -r requirements.txt
(.env)$ cdk deploy -c key_name="HirakeGoma"  # deploy完了　
# stackの削除
$ cdk destroy
```

# AMI (Amazon Machine Image) 
AMI は，AWSや他のサードパーティーから提供されており，EC2のコンソール画面でインスタンスを起動するときに検索することができる．  
あるいは， AWS CLI を使って，次のコマンドでリストを取得することができる (参考)．  
awsコマンドの使い方  
https://qiita.com/takech9203/items/4b9394ddd3f190835ca5
```bash
$ aws ec2 describe-images --owners amazon
```
ディープラーニングで頻繁に使われるプログラムが予めインストールしてあるAMIが，DLAMI (Deep Learning AMI) である．DLAMIには TensorFlow, PyTorch などの人気の高いディープラーニングのフレームワーク・ライブラリが既にインストールされているため，EC2インスタンスを起動してすぐさまディープラーニングの計算を実行できる．
例えば、Amazon Linux 2 をベースにした DLAMI はAMI ID = ami-09c0c16fc46a29ed9．  
AWS CLI を使って，このAMIの詳細情報を取得．
```cmd
$ aws ec2 describe-images --owners amazon --image-ids "ami-09c0c16fc46a29ed9"
```

# オンデマンドインスタンスの EC2 vCPU 制限引き上げリクエストを計算する方法
g4dn.xlarge インスタンスをDeploy する際に下記のエラーが出た。
You have requested more vCPU capacity than your current vCPU limit of 0 allows for the instance bucket that the specified instance type belongs to.
https://aws.amazon.com/jp/premiumsupport/knowledge-center/ec2-on-demand-instance-vcpu-increase/
AWS Service Quotas を使用して、更新された vCPU クォータを表示および管理できる。

# Elastic Container Service (ECS)
Elastic Container Service (ECS) とは， Docker を使った計算機クラスターを AWS 上に作成するためのツール.  
ECS は，タスク (Task) と呼ばれる単位で管理された計算ジョブを受け付ける．  
システムにタスクが投下されると，ECS はまず最初にタスクで指定された Docker イメージを外部レジストリからダウンロードしてくる．  
外部レジストリとしては， DockerHub や ECR (AWS 独自の Docker レジストリ) を指定することができる．  
次に， ECS はクラスターのスケーリング(自動拡張・自動シャットダウン)を行う．  
クラスターの中に配置できる仮想インスタンスは， EC2 に加えて， Fargate と呼ばれる ECS での利用に特化した仮想インスタンスを選択することができる．  
最後に， ECS はタスクの配置を行う． クラスター内で，計算負荷が小さい仮想インスタンスを選び出し，そこに Docker イメージを配置することで指定された計算タスクが開始される．  
これら一連のタスクの管理・クラスターのスケーリングを， ECS はほとんど自動でやってくれる． ユーザーは，クラスターのスケーリングやタスクの配置に関してのパラメータを指定するだけでよい． パラメータには，目標とする計算負荷 (例えば80%の負荷を維持する，など) などが含まれる．  

# Fargate
EC2 と同様に，仮想サーバー上で計算を走らせるためのサービスであるが，特に ECS での利用に特化されたもの．  
EC2 に比べるといろいろな制約があるが，ECS に特化した結果，利用者の側で設定しなければならないパラメータが少なく便利．  
Fargate では，EC2と同様にCPUの数・RAM のサイズを必要な分だけ指定できる． 執筆時点 (2020/06) では，CPU は 0.25 - 4 コア， RAM は 0.5 - 30 GB の間で選択することができる   
(https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)  
Fargate による仮想インスタンスは，ECS によって動的に管理することができる．  
すなわち，ECSは, タスクに応じて動的に Fargate インスタンスを立ち上げ，タスクの完了を検知して動的にインスタンスをシャットダウンすることができる．(EC2 を使っても可能だが，設定が多く，ハードルが高い．)  
また，インスタンスの起動時間でも，EC2 が2-3分程度の時間を要するのに対し，Fargate は典型的には30-60秒程度の時間で済む．  
したがって，Fargate を用いることでより俊敏にクラスターのスケーリングをすることが可能になる．

# ECS と Fargate
```python
# 空の ECS クラスターを定義
cluster = ecs.Cluster(
    self, "EcsClusterQaBot-Cluster",
    vpc=vpc,
)

# Fargate インスタンスを使ったタスクを定義
taskdef = ecs.FargateTaskDefinition(
    self, "EcsClusterQaBot-TaskDef",
    cpu=1024, # 1 CPU
    memory_limit_mib=4096, # 4GB RAM
)

# タスクの実行でで使用する Docker image を定義
container = taskdef.add_container(
    "EcsClusterQaBot-Container",
    image=ecs.ContainerImage.from_registry(
        "registry.gitlab.com/tomomano/intro-aws/handson03:latest"
    ),
)
```

# Transformers
pytorch-transformers と pytorch-pretrained-bert で知られるやつ  
State-of-the-art Natural Language Processing for PyTorch and TensorFlow 2.0  
https://github.com/huggingface/transformers  

学習済みのモデルを用いる場合，私達が行うのは，与えられた入力をモデルに投げて予測を行う (推論) のみである．  
推論の演算は，CPU だけでも十分高速に行うことができるので，コストの削減と，よりシンプルにシステムを設計をする目的で，GPU は利用しない選択も合理的．  
一般的に，ニューラルネットは学習のほうが圧倒的に計算コストが大きく，そのような場合に GPU はより威力を発揮する．  
ハンズオンで使用する Question & Answering システムには， DistilBERT という Transformer を基にした言語モデルが用いられている．(原著論文 https://arxiv.org/abs/1910.01108 )
また， huggingface/transformers の DistilBert についてのドキュメンテーションは こちら https://huggingface.co/transformers/index.html

# openの際のエンコーディング指定
encoding="utf-8_sig"を使う。  
run_task.py のjsonを開く際にEncodingエラーが発生する。
（UnicodeDecodeError: 'cp932' illegal multibyte sequence）
下記の通り修正する。
```python
    with open("problems.json", "r",encoding="utf-8_sig") as f:
            problems = json.load(f)
```

# REST API
REST API は，Method と URI (Universal Resource Identifier) の組からなる．  
```bash
# Method   URL(1.1までがEndPoint？)
GET https://api.twitter.com/1.1/statuses/home_timeline
```
メソッドとは,"どのような操作を行いたいか"を抽象的に表す ("動詞"と捉えてもよい)．REST API では典型的には次に示したメソッドが用いられる．
|  メソッド  |  動作  |
| ---- | ---- |
|  GET  |  要素を取得する  |
|  POST  |  新しい要素を作成する  |
|PUT|既存の要素を新しい要素と置き換える|
|PATCH|既存の要素の一部を更新する|
|DELETE|要素を削除する|

# Serverless architecture
## Lambda
Lambda では，まずユーザーは実行したいプログラムを予め登録しておく．  
プログラムは，Python, Node.js, ruby などの主要な言語がサポートされている．  
そして，プログラムを実行したいときに，そのプログラムを実行 (invoke する)コマンドを Lambda に送信する．  
Lambda では，invoke のリクエストを受け取ると直ちにプログラムの実行を開始し，実行結果をクライアントやその他の計算機に返す．
同時に複数のリクエストが来た場合でも，AWS はそれらを実行するための計算リソースを割り当て，並列的に処理を行ってくれる．  
原理上は，数千から数万のリクエストが同時に来たとしても，Lambda はそれらを同時に実行することができる．  
このような，占有された仮想サーバーの存在なしに，動的に関数を実行するサービスを FaaS (Function as a Service) と呼ぶ．

## サーバーレスストレージ: S3
従来的なストレージ (ファイルシステム) では，必ずホストとなるマシンと OS が存在しなければならない．  
従って，それほどパワーは必要ないまでも，ある程度の CPU リソースを割かなければならない．  
また，従来的なファイルシステムでは，データ領域のサイズは最初に作成するときに決めなければならず，後から容量を増加させることはしばしば困難．  
よって，従来的なクラウドでは，ストレージを借りるときには予めディスクのサイズを指定せねばならず，ディスクの容量が空であろうと満杯であろうと，同じ利用料金が発生することになる．  
Simple Storage Service (S3) は，サーバーレスなストレージシステムを提供する．  
S3 では，予めデータ保存領域の上限は定められていない．  
データを入れれば入れた分だけ，保存領域は拡大していく (仕様上はペタバイトスケールのデータを保存することが可能である)．  
ストレージにかかる料金も，保存してあるデータの総容量で決定される．  
その他，データの冗長化やバックアップなども，API を通じて行うことができる． これらの観点から，S3 も サーバーレスクラウドの一部として取り扱われることが一般的である．

## サーバーレスデータベース: DynamoDB
従来的に有名なデータベースとしては MySQL, PostgreSQL, MongoDB などが挙げられる．  
データベースと普通のストレージの違いは，データの検索機能にある．  
検索機能を実現するには，当然 CPU の存在が必須である．  
従って，従来的なデータベースを構築する際は，ストレージ領域に加えて，たくさんのCPUを搭載したマシンが用いられることが多い．  
また，格納するデータが巨大な場合は複数マシンにまたがった分散型のシステムが設計される.分散型システムの場合は，データベースへのアクセス負荷に応じて適切なスケーリングがなされる必要がある．  

DynamoDB は，サーバーレスなデータベースである．  
DynamoDB は分散型のデータベースであるが，データベースのスケーリングは AWS によって行われる． ユーザーとしては，特になにも考えずに，送りたいだけのリクエストをデータベースに送信すればよい． データベースへの負荷が増減したときのスケーリングは， DynamoDB が自動で行ってくれる．

## その他のサーバーレスクラウドの構成要素
その他，サーバーレスクラウドを構成するための構成要素  

API Gateway: API を構築する際のルーティングを担う．
Fargate: Fargate も，サーバーレスクラウドの要素の一部． Lambda では実行できないような，メモリーや複数CPUを要するような計算などを行うために用いる．
Simple Notification Service (SNS): サーバーレスのサービス間 (Lambda と DynamoDB など) でイベントをやり取りするためのサービス．
Step Functions: サーバーレスのサービス間のオーケストレーションを担う．

## サーバーレスアーキテクチャの欠点
サーバーレスアーキテクチャは万能ではない．  
まだまだ新しい技術なだけに，欠点，あるいはサーバーフルなシステムに劣る点は，数多くある．  
ひとつ大きな欠点をあげるとすれば，サーバーレスのシステムは各クラウドプラットフォームに固有なものなので，特定のプラットフォームでしか運用できないシステムになってしまう点．  
AWSで作成したサーバーレスのシステムを，Google のクラウドに移植するには，かなり大掛かりなプログラムの書き換えが必要になる．  
一方，serverful なシステムであれば，プラットフォーム間のマイグレーションは比較的簡単に行うことができる．  
その他，サーバーレスコンピューティングの欠点や今後の課題などは，次の論文で詳しく議論されている． 興味のある読者は読んでみると良い．

Hellerstein et al., "Serverless Computing: One Step Forward, Two Steps Back" arXiv (2018)
https://arxiv.org/abs/1812.03651

# LambdaのCDK,deploy
メモリーオーバーした場合は？？？
```python
# lambdaで実行する関数を定義してる
FUNC = """
import time
from random import choice, randint
def handler(event, context):
    time.sleep(randint(2,5))
    pokemon = ["Charmander", "Bulbasaur", "Squirtle"]
    message = "Congratulations! You are given " + choice(pokemon)
    print(message)
    return message
"""

class SimpleLambda(core.Stack):

    def __init__(self, scope: core.App, name: str, **kwargs) -> None:
        super().__init__(scope, name, **kwargs)

        
        handler = _lambda.Function(
            self, 'LambdaHandler',
            # runtimeを指定。Python3.7 の他に，
            # Node.js, Java, Ruby, Go などを指定可能
            runtime=_lambda.Runtime.PYTHON_3_7,
            # 実行する関数のコードを指定する．
            # ここでは， FUNC=…​ で定義した文字列を渡している
            # ファイルのパスを渡すことも可能．
            code=_lambda.Code.from_inline(FUNC),
            # コードにサブ関数があるときに，メインとサブを区別する． 
            # handlerという関数をメイン関数として実行させる
            handler="index.handler",
            # メモリーは128MBを最大で使用することを指定． 
            # メモリーオーバーした場合は
            memory_size=128,
            # タイムアウト時間を10秒に設定． 
            # 10秒以内に終了しなかった場合，エラーが返される．
            timeout=core.Duration.seconds(10),
            # アドバンストな設定なので説明は省略する．
            dead_letter_queue_enabled=True,
        )
```

# DynamoDB のCDK,deploy
```python
class SimpleDynamoDb(core.Stack):
    def __init__(self, scope: core.App, name: str, **kwargs) -> None:
        super().__init__(scope, name, **kwargs)

        table = ddb.Table(
            self, "SimpleTable",
            # DynamoDBテーブルにはPartitionKey定義必須
            # Partitionkeyとは，レコードごとの固有のID
            # 同一のPartitionkeyの要素はテーブル内に一つのみ
            # Partitionkeyがない要素はテーブル内に存在できない． 
            # Partitionkeyにitem_idという名前をつけている
            partition_key=ddb.Attribute(
                name="item_id",
                type=ddb.AttributeType.STRING
            ),
            # ddb.BillingMode.PAY_PER_REQUESTで基本的に良い
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            # 省略
            removal_policy=core.RemovalPolicy.DESTROY
        )
```

## DynamoDB の操作
DynamoDBの削除はしなくていいのか？
```python
import boto3
ddb = boto3.resource('dynamodb')
table = ddb.Table("XXXX") # XXXX はTableのName

# TableにItem登録
table.put_item(
   Item={
       'item_id': 'bec7c265-46e2-4065-91d8-80b2e8dcc9c2',
       'first_name': 'John',
       'last_name': 'Doe',
       'age': 25,
    }
)

# TableからItemを取得
table.get_item(
   Key={"item_id": 'bec7c265-46e2-4065-91d8-80b2e8dcc9c2'}
).get("Item")

# Tableから全Itemを取得
table.scan().get("Items")
```

