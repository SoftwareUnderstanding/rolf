# Effective TensorFlow

Esta artigo foi orignalmente escrito por [Vahid
Kazemi](https://github.com/vahidk) e traduzido ao português por [Gabriel
Mouzella Silva](https://www.linkedin.com/in/gabrielmouzella/). Para acessar o
artigo original [clique aqui](https://github.com/vahidk/EffectiveTensorflow).

# Indicações <a name="indicacoes"></a>

**Dicas  execução**
* Carregue todos os pacotes que serão necessários no início
do arquivo.
* Para executar comandos da shell dentro do notebook, utilize o
prefixo `!` antes do comando de shell. (e.g. para listar os arquivos da pasta
atual faça `! ls` para UNIX ou `! dir` para Windows.

```{.python .input}
import tensorflow as tf
import numpy as np
```



# Parte I: Fundamentos de TensorFlow

## TensorFlow Básico

A maior diferença entre TensorFlow e outras bibliotecas computacionais tal como
NumPy é que **operações em TensorFlow são simbólicas**.
* Isso é um conceito
poderoso que permite ao TensorFlow fazer todo tipo de coisa (por exemplo
**diferenciação automática**) que não são possíveis em bibliotecas imperativas
como o NumPy.
* Porém aumenta a complexidade, o que o torna de mais difícil
compreenção.

Nosso objetivo é de desmistificar TensorFlow e prover algumas
diretrizes e boas práticas para um melhor uso do TensorFlow.

Vamos iniciar com
um simples exemplo, queremos multiplicar duas matrizes randômicas. Primeiramente
vejamos a implementação feita em NumPy:

```{.python .input}
x = np.random.normal(size=[10, 10])
y = np.random.normal(size=[10, 10])
z = np.dot(x, y)

print(z)
```

Agora vamos executar o mesmo cálculo porém dessa vez em TensorFlow:

```{.python .input}
tf.reset_default_graph()

#	criando os tensores (arestas)
x = tf.random_normal([10, 10])
y = tf.random_normal([10, 10])

#	criando as operações (nos)
z = tf.matmul(x, y)

#criando o grafo e executando-lo em uma sesão
sess = tf.Session()
z_val = sess.run(z)

print(z_val)
```

Observação:
* `NumPy`: imediatamente executa o cálculo e gera o resultado,
*
`TensorFlow`:  Nos dá somente um identificador (do tipo Tensor) para um nó no
gráfico que representa o resultado.

Se tentarmos escrever o valor de z
diretamente, teremos algo do tipo:

```{.python .input}
tf.reset_default_graph()

#	criando os tensores
x = tf.random_normal([10, 10])
y = tf.random_normal([10, 10])

#	criando as operações
z = tf.matmul(x, y)

print(z)
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Tensor(\"MatMul:0\", shape=(10, 10), dtype=float32)\n"
 }
]
```

Uma vez que ambas as entradas têm um formato definido, TensorFlow é capaz de
inferir o formato do tensor de saída, assim como seu tipo.

A fim de computar o
valor do tensor, faz-se necessário criar uma sessão e avaliá-la usando o método
`Session.run()` .

---
**Dica**: Caso esteja utilizando `Jupyter notebook` assegure-se de chamar
`tf.reset_default_graph()` no começo para limpar o gráfico simbólico antes de
definir novos nós.

---

Para entendermos quão poderoso computação simbólica pode ser vajemos um outro
exemplo.

1. Assuma que tenhamos amostras de uma curva (digamos $f(x) = 5x^2 +
3$) e queremos estimar $f(x)$ baseado nessas amostras.
1. Definimos a função
paramétricas $g(x, w) = w_0 x^2 + w_1 x + w_2$, que esta em função de $x$ e
parâmetros $w$, nosso objetivo é encontrar os parâmetros tal que $g(x, w) ≈
f(x)$.
1. Isso pode ser feito minimizando a seguinte função de perda $L(w) =
\sum(f(x) - g(x, w))^2$. 

Apesar de que haja uma solução fechada para este
simples problema, optamos por utilizar uma aproximação mais generalizada, que
pode ser aplicada a qualquer função diferencial arbitrária utilizando-se do
gradiente descendente estocástico.
Simplesmente calcula-se o gradiente médio de
$L(w)$ com relação a $w$ em um conjunto de amostras e move-se na direção oposta.
O código em `TensorFlow` ficaria assim:

```{.python .input}
tf.reset_default_graph()

# Placeholders are used to feed values from python to TensorFlow ops. We define
# two placeholders, one for input feature x, and one for output y.
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Assuming we know that the desired function is a polynomial of 2nd degree, we
# allocate a vector of size 3 to hold the coefficients. The variable will be
# automatically initialized with random noise.
w = tf.get_variable("w", shape=[3, 1])

# We define yhat to be our estimate of y.
f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
yhat = tf.squeeze(tf.matmul(f, w), 1)

# The loss is defined to be the l2 distance between our estimate of y and its
# true value. We also added a shrinkage term, to ensure the resulting weights
# would be small.
loss = tf.nn.l2_loss(yhat - y) + 0.1 * tf.nn.l2_loss(w)

# We use the Adam optimizer with learning rate set to 0.1 to minimize the loss.
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

def generate_data():
    x_val = np.random.uniform(-10.0, 10.0, size=100)
    y_val = 5 * np.square(x_val) + 3
    return x_val, y_val

sess = tf.Session()
# Since we are using variables we first need to initialize them.
sess.run(tf.global_variables_initializer())
for _ in range(1000):
    x_val, y_val = generate_data()
    _, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})
    #print(loss_val)

w_val = sess.run([w])
print('w = ',w_val)
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "('w = ', [array([[ 4.9922128e+00],\n       [-3.5868271e-04],\n       [ 3.4673905e+00]], dtype=float32)])\n"
 }
]
```

O resultado é uma aproximação relativamente próxima dos nossos parâmetros.
Essa é somente a ponta do iceberg que TensorFlow pode fazer.
Muitos problemas
tais como *otimizar uma grande rede neural* com milhões de parâmetros pode ser
implementada de maneira eficiente em Tensorflow em poucas linhas de código.
TensorFlow encarrega-se do dimensionamento de vários dispositivos, threads e
suporta uma variedade de plataformas.

## Entendendo formatos estáticos e dinâmicos

Tensores em TensorFlow apresenta **atributo de forma estática** que é
**determinada durante a contrução do gráfico**.
A forma estática poderá ser
subespecificada. Por exemplo. pode-se definir um forma do tensor como `[None,
128]`:

```{.python .input}
a = tf.placeholder(tf.float32, [None, 128])
```

Isso significa que
* a primeira dimensão pode ser de qualquer tamanho e será
determinada dinamicamente durante a `Session.run()`.

Pode-se consultar o
formato estático do Tensor da seguinte maneira:

```{.python .input}
static_shape = a.shape.as_list()  # returns [None, 128]
print('static_shape = ',static_shape)
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "('static_shape = ', [32, 128])\n"
 }
]
```

Para obter o formato dinâmico do tensor pode-se chamar `tf.shape`, que retorna
um tensor representando o formato do tensor dado:

```{.python .input}
dynamic_shape = tf.shape(a)
print('dynamic_shape = ',dynamic_shape)
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "('dynamic_shape = ', <tf.Tensor 'Shape:0' shape=(2,) dtype=int32>)\n"
 }
]
```

O formato estático de um tensor pode ser definido com o método
`Tensor.set_shape()`:

```{.python .input}
a.set_shape([32, 128])  # static shape of a is [32, 128]
print('static_shape1 = ',a.shape.as_list())
a.set_shape([None, 128])  # first dimension of a is determined dynamically
print('static_shape2 = ',a.shape.as_list())
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "('static_shape1 = ', [32, 128])\n('static_shape2 = ', [32, 128])\n"
 }
]
```

Pode-se modificar um dado tensor dinamicamente usando `tf.reshape` function:

```{.python .input}
print('a_shape = ',a.shape.as_list())
a =  tf.reshape(a, [128, 32])
print('a_shape = ',a.shape.as_list())
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "('a_shape = ', [32, 128])\n('a_shape = ', [128, 32])\n"
 }
]
```

Pode ser conveniente ter uma função que retorna o formato estático quando
disponível, caso contrário retorne o formato dinâmico. A função utilidade abaixo
faz exatamente isso:

```{.python .input}
def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0] for s in zip(static_shape, dynamic_shape)]
  return dims
```

Imaginemos que se queira converter um Tensor de dimensão 3 para um um de
dimensão 2 colapsando a segunda e terceira dimensão em uma. Pode-se utilizar  a
função `get_shape()` para fazê-lo:

```{.python .input}
b = tf.placeholder(tf.float32, [None, 10, 32])
shape = get_shape(b)
print('shape_before = ',get_shape(b))
b = tf.reshape(b, [shape[0], shape[1] * shape[2]])
print('shape_after = ',get_shape(b))
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "('shape_before = ', [<tf.Tensor 'unstack_2:0' shape=() dtype=int32>, 10, 32])\n('shape_after = ', [<tf.Tensor 'unstack_3:0' shape=() dtype=int32>, 320])\n"
 }
]
```

Note que isso funciona independente se o formato é estaticamente especificado ou
não.

Pode-se escrever uma função de redimensionamento de propósito geral para
colapsar qualquer lista de qualquer dimensão:

```{.python .input}
def reshape(tensor, dims_list):
  shape = get_shape(tensor)
  dims_prod = []
  for dims in dims_list:
    if isinstance(dims, int):
      dims_prod.append(shape[dims])
    elif all([isinstance(shape[d], int) for d in dims]):
      dims_prod.append(np.prod([shape[d] for d in dims]))
    else:
      dims_prod.append(tf.prod([shape[d] for d in dims]))
  tensor = tf.reshape(tensor, dims_prod)
  return tensor
```

Agora colapsar a segunda dimensão torna-se muito fácil:

```{.python .input}
b = tf.placeholder(tf.float32, [None, 10, 32])
print('shape_before = ',get_shape(b))
b = reshape(b, [0, [1, 2]])
print('shape_after = ',get_shape(b))
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "('shape_before = ', [<tf.Tensor 'unstack_4:0' shape=() dtype=int32>, 10, 32])\n('shape_after = ', [<tf.Tensor 'unstack_6:0' shape=() dtype=int32>, 320])\n"
 }
]
```

## Escopos e quando utilizá-los

Variáveis e tensores em TensorFlow tem o atributo nome que é usado para
identificá-los no gráfico simbólico.

Caso não seja especificado o nome quando
se cria uma variável ou tensor, TensorFlow automaticamente designa um nome:

```{.python .input}
a = tf.constant(1)
print(a.name)  
b = tf.Variable(1)
print(b.name)  
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Const_2:0\nVariable_2:0\n"
 }
]
```

Pode-se sobrescrever o nome *defaulf* especificando explicitamente o nome:

```{.python .input}
a = tf.constant(1, name="a")
print(a.name)  
b = tf.Variable(1, name="b")
print(b.name)  
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "a:0\nb:0\n"
 }
]
```

TensorFlow tem duas maneiras de modificar o nome dos tensores e variáveis. O
primeiro é `tf.name_scope`:

```{.python .input}
with tf.name_scope("scope"):
  a = tf.constant(1, name="a")
  print(a.name)  
  
  b = tf.Variable(1, name="b")
  print(b.name)  

  c = tf.get_variable(name="c", shape=[])
  print(c.name)  
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "scope/a:0\nscope/b:0\nc:0\n"
 }
]
```

Note que há duas maneiras de definir uma nova variável em TensorFlow criando-se
um objeto `tf.Variable` ou a partir de `tf.get_variable`.
* Utilizando-se de
`tf.get_variable` com um novo nome resulta em criar uma uma nova variável, porém
se a variável com o mesmo nome existir resultará em uma exceção *ValueError*,
dizendo que redeclarar uma variável não é permitido.
* `tf.name_scope` afeta o
nome de tensores e variáveis criadas com `tf.Variable`, porém não impacta as
variáveis criadas com `tf.get_variable`.

Diferentemente de `tf.name_scope`,
`tf.variable_scope` modifica o nome da variável criada também com
`tf.get_variable`:

```{.python .input}
with tf.variable_scope("scope"):
  a = tf.constant(1, name="a")
  print(a.name)  # prints "scope/a:0"

  b = tf.Variable(1, name="b")
  print(b.name)  # prints "scope/b:0"

  c = tf.get_variable(name="c", shape=[])
  print(c.name)  # prints "scope/c:0"
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "scope/a:0\nscope/b:0\nscope/c:0\n"
 }
]
```

```{.python .input}
with tf.variable_scope("scope"):
  a1 = tf.get_variable(name="a", shape=[])
  a2 = tf.get_variable(name="a", shape=[])  # Disallowed
```

Mas e se quisessemos reutilizar uma variável previamente declarada?
`tf.variable_scope` também apresenta uma funcionalidade para fazê-lo:

```{.python .input}
with tf.variable_scope("scope"):
  a1 = tf.get_variable(name="a", shape=[])
with tf.variable_scope("scope", reuse=True):
  a2 = tf.get_variable(name="a", shape=[])  # OK
```

Isso se faz útil por exemplo quando se utiliza camadas de redes neurais
integradas:

```{.python .input}
with tf.variable_scope('my_scope'):
  features1 = tf.layers.conv2d(image1, filters=32, kernel_size=3)
# Use the same convolution weights to process the second image:
with tf.variable_scope('my_scope', reuse=True):
  features2 = tf.layers.conv2d(image2, filters=32, kernel_size=3)
```

Alternativamente pode-se usar reuse para tf.AUTO_REUSE que diz ao TensorFlow
para criarr uma nova variável se uma variável co mesmo nome não existir, caso
contrário reutilizá-la:

```{.python .input}
with tf.variable_scope("scope", reuse=tf.AUTO_REUSE):
  features1 = tf.layers.conv2d(image1, filters=32, kernel_size=3)
  features2 = tf.layers.conv2d(image2, filters=32, kernel_size=3)
```

Caso queira-se fazer muitos compartilhamento de variáveis mantendo-se o controle
de quando definir novas variáveis e quando reutiliza-las então pode ser
complicado e sujeito a erros.
`tf.AUTO_REUSE` simplifica a tarefa porém adiciona
o risco de compartilhar variáveis que supostamente não deveriam ser
compartilhadas. O *template* do TensorFlow é outra maneira de resolver o mesmo
problema sem tal risco:

```{.python .input}
conv3x32 = tf.make_template("conv3x32", lambda x: tf.layers.conv2d(x, 32, 3))
features1 = conv3x32(image1)
features2 = conv3x32(image2)  # Will reuse the convolution weights.
```

Pode-se tornar qualquer função em um *template* do TensorFlow. Na primeira
chamada para um *template*, as variáveis definidas dentro da função seriam
declaradas e nas chamadas subsequentes seriam automaticamente reutilizadas.

## Difusão: Pontos fortes e fracos

TensorFlow suporta a difusão de operações
elemento a elemento.
Normalmente quando se deseja fazer operações como adição ou
multiplicação, é necessário certificar-se que as dimensões dos operandos estajam
de acordo, por exemplo, não se pode adicionar um tensor de dimensão `[3,2]` com
um tensor `[3,4]`.
Porém há uma excessão, caso se tem somente uma dimensão.
TensorFlow implicitamente organiza os tensores em uma dimensão para que o
formato seja igual ao do outro operando.
Portanto é aceitável adicionar um
tensor de dimensão `[3, 2]` com um tensor de dimensão `[3, 1]`.

```{.python .input}
a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[1.], [2.]])
# c = a + tf.tile(b, [1, 2])
c = a + b
with tf.Session() as session:
  result  = session.run(c)  #result = v.eval() #forma equivalente
  print(result)
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[2. 3.]\n [5. 6.]]\n"
 }
]
```

Difusão nos permite agrupar implicitamente, o que torna o código menor, e mais
eficiente em relação à memória, uma vez que não é necessário guardar o resultado
da operação de agrupamento.
Um lugar onde pode ser facilmente implementado é ao
combinar *features* de tamanhos distintos.
A fim de concatenar *features* de
tamanhos diferentes normalmente agrupa-se os tensores de entrada, concatena o
resultado e aplica-se alguma não-linearidade.
**Esse é um procedimento padrão
entre várias arquiteturas de redes neurais**.

```{.python .input}
a = tf.random_uniform([5, 3, 5])
b = tf.random_uniform([5, 1, 6])

# concat a and b and apply nonlinearity
tiled_b = tf.tile(b, [1, 3, 1])
c = tf.concat([a, tiled_b], 2)
d = tf.layers.dense(c, 10, activation=tf.nn.relu)

with tf.Session() as session:
  #print(session.run(tiled_b))
  print(session.run(c))
  #print(session.run(d)) 
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[[0.6937026  0.15428507 0.2800038  0.08377886 0.12188792 0.5092124\n   0.695078   0.4601586  0.82269514 0.7481663  0.7347907 ]\n  [0.16978633 0.36053598 0.16227639 0.7982862  0.6463525  0.5092124\n   0.695078   0.4601586  0.82269514 0.7481663  0.7347907 ]\n  [0.6232879  0.43789756 0.80180585 0.6007143  0.54549825 0.5092124\n   0.695078   0.4601586  0.82269514 0.7481663  0.7347907 ]]\n\n [[0.22090471 0.31535435 0.5629263  0.3793478  0.60918427 0.68534255\n   0.6267303  0.71711516 0.5630053  0.6170057  0.5886011 ]\n  [0.5502802  0.55038464 0.07872856 0.8033217  0.83834624 0.68534255\n   0.6267303  0.71711516 0.5630053  0.6170057  0.5886011 ]\n  [0.4749012  0.43129456 0.61200035 0.7015667  0.90547967 0.68534255\n   0.6267303  0.71711516 0.5630053  0.6170057  0.5886011 ]]\n\n [[0.51492727 0.49063408 0.6472235  0.42694485 0.9332788  0.94186914\n   0.09389567 0.76210356 0.00971448 0.85871506 0.03518164]\n  [0.0576483  0.06008446 0.15991223 0.6457871  0.30569184 0.94186914\n   0.09389567 0.76210356 0.00971448 0.85871506 0.03518164]\n  [0.7202405  0.23647058 0.6942618  0.62667763 0.4135698  0.94186914\n   0.09389567 0.76210356 0.00971448 0.85871506 0.03518164]]\n\n [[0.17352045 0.14470255 0.1059376  0.46654844 0.48657966 0.4362011\n   0.90611553 0.76159    0.47261274 0.36685407 0.7907051 ]\n  [0.5611588  0.10189164 0.53168106 0.3823582  0.95938146 0.4362011\n   0.90611553 0.76159    0.47261274 0.36685407 0.7907051 ]\n  [0.5209532  0.5431979  0.327083   0.95845103 0.13369536 0.4362011\n   0.90611553 0.76159    0.47261274 0.36685407 0.7907051 ]]\n\n [[0.27793443 0.02233255 0.59707713 0.54039264 0.4431789  0.418213\n   0.234949   0.6185926  0.31132996 0.88933694 0.9839337 ]\n  [0.8138777  0.42909408 0.63658106 0.52854526 0.68817234 0.418213\n   0.234949   0.6185926  0.31132996 0.88933694 0.9839337 ]\n  [0.2202568  0.55266035 0.5153638  0.01934719 0.3002057  0.418213\n   0.234949   0.6185926  0.31132996 0.88933694 0.9839337 ]]]\n"
 }
]
```

Porém isso pode ser feito de maneira mais eficiente com uso de difusão. Usa-se o
fato de que `f(m(x+y))` é igual a `f(mx+my)`.
Por fim pode-se fazer operações
lineares separadamente utilizando a difusão para fazer concatenação implícita:

```{.python .input}
pa = tf.layers.dense(a, 10, activation=None)
pb = tf.layers.dense(b, 10, activation=None)
d = tf.nn.relu(pa + pb)
```

Na verdade essa parte do código é bastante generalista e pode ser aplicada a
tensores de dimensões arbitrárias contanto que seja possível fazer a difusão
entre tensores:

```{.python .input}
def merge(a, b, units, activation=tf.nn.relu):
    pa = tf.layers.dense(a, units, activation=None)
    pb = tf.layers.dense(b, units, activation=None)
    c = pa + pb
    if activation is not None:
        c = activation(c)
    return c
```

Uma forma mais generalista dessa função pode ser encontrada neste
[livro](https://github.com/vahidk/EffectiveTensorflow#merge) (livro em inglês).
Até o momento discutiu-se o lado bom da difusão. Porém quais problemas existem?
Suposições implícitas quase sempre torna difícil debugar. Considere o exemplo
abaixo:

```{.python .input}
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b)

with tf.Session() as session:
  print(session.run(c))
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "12.0\n"
 }
]
```

Qual você pensaria que seria o valor de `c`?
Se você disse `6` você está errado.
Será `12`.
Isso porque quando o *rank* de dois tensores não combinam,
**TensorFlow automaticamente expande a primeira dimensão do tensor de menor rank
antes de fazer a operação elemento a elemento**, portanto o resultado da adição
seria `[[2,3],[3,4]]`, e a redução de todos os parâmetros daria 12.

**A maneira
de evitar esse problema é ser tão explícito quanto se poda**.
Case houvessemos
especificado qual a dimensão nós gostaríamos de reduzir, encontrar esse bug
teria sido muito mais fácil:

```{.python .input}
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b, 0)

with tf.Session() as session:
  print(session.run(c))
```

```{.json .output n=0}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[5. 7.]\n"
 }
]
```

Aqui o valor de `c` seria `[5,7]`, e nós teríamos adivinhado baseado no formato
do resultado que há alguma coisa errada.
**Uma regra geral é sempre especificar
as dimensões em operação de redução e quando se utiliza `tf.squeeze`.**

## Importando dados ao TensorFlow

O TensorFlow foi desenvolvido para trabalhar
de maneira eficiente com grandes quantidades de dados. Portanto é importante não
alimentar seu modelo TensorFlow a fim de maximizar sua performance. Existem
várias maneiras de alimentar dados ao TensorFlow.

TensorFlow is designed to
work efficiently with large amount of data. So it's important not to starve your
TensorFlow model in order to maximize its performance. There are various ways
that you can feed your data to TensorFlow.

**Constantes**

A maneira mais simples é declarar os dados como constantes:

```{.python .input}
import tensorflow as tf
import numpy as np

actual_data = np.random.normal(size=[100])

data = tf.constant(actual_data)
```

Essa forma pode ser muito eficiente, porém não muito flexível. Um problema é que
em ao usar seu modelo com outro dataset deve-se reescrever o grafo. Além de que
têm-se que carregar todos os dados na memória de uma vez a mantê-los na memória,
o que só funcionaria para pequenos datasets.

**Espaços reservados (placeholders)**

Usar espaços reservadoe resolver ambos os
problemas acima citados:

```{.python .input}
import tensorflow as tf
import numpy as np

data = tf.placeholder(tf.float32)

prediction = tf.square(data) + 1

actual_data = np.random.normal(size=[100])

tf.Session().run(prediction, feed_dict={data: actual_data})
```

O operador de espaços reservados retora um tensor cujo vajor é capturado a
partir do argumento `feed_dict` na função `Session.run`. Note que executar
`Session.run` sem alimentar os valores dos dados, neste caso, resultará em erro.

**Python ops**

Outra maneira de alimentar dados ao TensorFlow é utilizando
*Python ops*:

```{.python .input}
def py_input_fn():
    actual_data = np.random.normal(size=[100])
    return actual_data

data = tf.py_func(py_input_fn, [], (tf.float32))
```

*Python ops* permite converter uma função Python normal em uma operação em
TensorFlow.

**Dataset API**

A forma recomendada de ler dados em TensorFlow é utilizando
**dataset API**:

```{.python .input}
actual_data = np.random.normal(size=[100])
dataset = tf.contrib.data.Dataset.from_tensor_slices(actual_data)
data = dataset.make_one_shot_iterator().get_next()
```

Caso você tenha que ler seus dados a partir de um arquivo pode ser mais
eficiente escrever no formato `TFrecord` e utilizar `TFRecordDataset` para ler:

```{.python .input}
dataset = tf.contrib.data.TFRecordDataset(path_to_data)
```

Veja os documentos oficiais para um exemplo de como escrever os dados em formato
`TFrecord`.

Dataset API permite que você faça processamento eficiente de dados
usando pipelines de maneira fácil. Por exemplo, assim processamos nossos dados
no código abaixo:
[trainer.py](https://github.com/vahidk/TensorflowFramework/blob/master/trainer.py):

```{.python .input}
dataset = ...
dataset = dataset.cache()
if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(batch_size * 5)
dataset = dataset.map(parse, num_threads=8)
dataset = dataset.batch(batch_size)
```

Após ler os dados, utilizamos o método `Dataset.cache` para armazenar-lo em
memória a fim de aumentar a eficiência. Durante o modo de treino, repetimos o
dataset indefinidamente. Isso permite processar todo o dataset muitas vezes. Nós
embaralhamos o dataset para pegar *batches* com diferentes distribuições de
dados. Após utiliza-se a função `Dataset.map` para fazer o pré-processamento em
registros brutos e converter os dados a um formato utilizável pelo modelo. Por
fim cria-se os *batches* de amostras chamando `Dataset.batch`.

## Tire vantagens da sobrecarga de operadores

Assim como NumPy, TensorFlow
sobrecarrega um número de operadores Python para facilitar a construção de
grafos e tornar o código mais fácil de ler.

A operação de repartição é um dos operadores que pode facilitar a indexação de
tensores:

```{.python .input}
z = x[begin:end]  # z = tf.slice(x, [begin], [end-begin])
```

Porém há de ser cuidadoso ao utilizar a operação de repartição. A operação de
repartição é bastante ineficiente e melhor se evitada, especialmente quando o
número de repartições é alto. Para entender quão ineficiente tal operação pode
ser  vejamos um exemplo. Queremos manualmente fazer uma redução através as
linhas da matriz:

```{.python .input}
import tensorflow as tf
import time

x = tf.random_uniform([500, 10])

z = tf.zeros([10])
for i in range(500):
    z += x[i]

sess = tf.Session()
start = time.time()
sess.run(z)
print("Took %f seconds." % (time.time() - start))
```

Em um MacBook Pro, essa operação demorou 2.67 segundos para rodar. a razão disso
é que se está chamando a operação 500 vezes, o que tornará o código bastante
lento para rodar. Uma melhor alternatica seria usar a opreração `tf.unstack`
para separar a matriz em uma lista de vetores de uma só vez:

```{.python .input}
z = tf.zeros([10])
for x_i in tf.unstack(x):
    z += x_i
```

Essa operação demorou 0.18 segundos. Com certeza, a maeira correta de fazer essa
simples redução é usando a operação `tf.reduce_sum`:

```{.python .input}
z = tf.reduce_sum(x, axis=0)
```

Dessa forma demorou 0.008 segundos, o que é 300x mais rápido que a implementação
original.

TensorFlow também sobrecarrega uma gama de operações aritiméticas e
lógicas de maneira mais eficiente:

```{.python .input}
z = -x  # z = tf.negative(x)
z = x + y  # z = tf.add(x, y)
z = x - y  # z = tf.subtract(x, y)
z = x * y  # z = tf.mul(x, y)
z = x / y  # z = tf.div(x, y)
z = x // y  # z = tf.floordiv(x, y)
z = x % y  # z = tf.mod(x, y)
z = x ** y  # z = tf.pow(x, y)
z = x @ y  # z = tf.matmul(x, y)
z = x > y  # z = tf.greater(x, y)
z = x >= y  # z = tf.greater_equal(x, y)
z = x < y  # z = tf.less(x, y)
z = x <= y  # z = tf.less_equal(x, y)
z = abs(x)  # z = tf.abs(x)
z = x & y  # z = tf.logical_and(x, y)
z = x | y  # z = tf.logical_or(x, y)
z = x ^ y  # z = tf.logical_xor(x, y)
z = ~x  # z = tf.logical_not(x)
```

Pode-se também usar as versões aumentadas dessas operações. Por exemplo `x += y`
e `x **= 2` também são válidos.

Note que Python não permite a sobrecarga das
palavras-chave `"and"`, `"or"` e `"not"`. 

TensorFlow não permite utilizar
tensores como booleanos, tal ação pode resultar em erro:

```{.python .input}
x = tf.constant(1.)
if x:  # This will raise a TypeError error
    ...
```

Pode-se usar `tf.cond(x,...)` caso queira checar o valor do tensor, ou usar "`if
x is None`" para checar o valor da variável.

Outras operações que não são
suportadas é a de igual(==) e de diferente(!=), operadores que são permitidos em
NumPy porém não em TensorFlow. Utilize a versão do TensorFlow que são `tf.equal`
e `tf.not_equal`.

## Entendendo a ordem de execução e controle de dependências

Como discutido no primeiro item, TensorFlow não roda imediatamente operações que
são definidas, mas cria nós correspondentes em um grafo que pode ser avaliado
com o método `Session.run()`. Isso permite que o TensorFlow faça otimizações no
tempo de execução para determinar a ordem de execução ótima e possível corte de
nós não utilizados. Caso tenha-se somente `tf.Tensord` no grafo não há
necessidade de preocupar-se com dependencias, porém provavelmente existe no
código também  `tf.Variables`, o que torna a situação mais complicada. Meu
conselho é utilizar Variáveis somente se Tensores não forem suficientes para a
tarefa. Talvez isso não faça muito sentido, portanto vamos começar com um
exemplo.

```{.python .input}
import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)
a = a + b

tf.Session().run(a)
```

Avaliar "`a`" retornará o valor 3 como esperado. Note que criou-se 3 tensores,
dois tensores constantes e um tensor para guardar o resultado da adição. Note
que não se pode sobrescrever o valor de um tensor. Caso queira-se modificar o
valor do tensor tem-se que criar um novo tensor. Como foi feito aqui.



---
DICA: Caso você não defina um novo grafo, TensorFlow automaticamente cria um
grafo por *default* . Pode-se usar `tf.get_default_graph()` para acessar o
grafo. Você pode então inspecionar o grafo, como por exemplo imprimindo todos os
tensores:


---

```{.python .input}
print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
```

```{.json .output n=0}
[
 {
  "ename": "NameError",
  "evalue": "ignored",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m\u001b[0m",
   "\u001b[0;31mNameError\u001b[0mTraceback (most recent call last)",
   "\u001b[0;32m<ipython-input-1-1f23664bdb1f>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcontrib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgraph_editor\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_tensors\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_default_graph\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m: name 'tf' is not defined"
  ]
 }
]
```

Diferente de tensores, variáveis podem ser atualizadas. Portanto vejamos como
utilizariamos variáveis para fazer a mesma tarefa:

```{.python .input}
a = tf.Variable(1)
b = tf.constant(2)
assign = tf.assign(a, a + b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(assign))
```

Novamente obtem-se 3 como esperado. Note que `tf.assign` retorna um tensor
representando o valor atribuido. Até o momento tudo parece bom, porém vejamos um
exemplo um pouco mais complicado:

```{.python .input}
a = tf.Variable(1)
b = tf.constant(2)
c = a + b

assign = tf.assign(a, 5)

sess = tf.Session()
for i in range(10):
    sess.run(tf.global_variables_initializer())
    print(sess.run([assign, c]))
```

Note que o tensor c não tem um valor determinístico. Esse valor pode ser 3 ou 7
dependendo de qual adição ou atribuição é executada primeiro.

Note que a ordem
em que se define operações em seu código não importa para a execução do
TensorFlow. A única coisa que importa é o controle de dependências. Controle de
dependências para tensores é bem direta. Cada vez que utiliza-se um tensor em
uma operação, essa operação define uma dependência implícita para aquele tensor.
Porém as coisas podem ficar complicadas com variáveis porque elas podem ter
vários valores.

Quando se esta lidando com variáveis, se faz necessário definir
explicitamente as dependências usando `tf.control_dependencies()` como a seguir:

```{.python .input}
a = tf.Variable(1)
b = tf.constant(2)
c = a + b

with tf.control_dependencies([c]):
    assign = tf.assign(a, 5)

sess = tf.Session()
for i in range(10):
    sess.run(tf.global_variables_initializer())
    print(sess.run([assign, c]))
```

Assim estará assegurado que a operação `assign` será chamada depois da adição.

## Operações de controle de fluxo: condicionais e loops

Ao construir modelos
complexos como redes neurais recorrentes, as vezes se faz necessário controlar o
fluxo de operações a partir de condicionais e loops. Nesta seção introduzimos as
operações mais comumente utilizada para controle de fluxo.

Digamos que você queira decidir se deve multiplicar ou adicionar dois tensores
dados baseados em um predicado. Isso pode ser implementado simplesmente com
`tf.cond` que atua como o condicional if `if`:

```{.python .input}
a = tf.constant(1)
b = tf.constant(2)

p = tf.constant(True)

x = tf.cond(p, lambda: a + b, lambda: a * b)

print(tf.Session().run(x))
```

Como o predicado é Verdadeiro nesse caso, a saída seria o resultado da adição,
que é 3.

Na maioria das vezes, quando se usa TensorFlow, utiliza-se grandes tensores e se
deseja fazer operações em bateladas. Uma operação condicional relacionada é
`tf.where`, que como `tf.cond` recebe um predicado, porém seleciona a saída
baseada em uma condição em batelada.

```{.python .input}
a = tf.constant([1, 1])
b = tf.constant([2, 2])

p = tf.constant([True, False])

x = tf.where(p, a + b, a * b)

print(tf.Session().run(x))
```

Isso irá retornar `[3, 2]`.

Outro método de controle de fluxo bastante utilizado é `tf.while_loop`. Que
permite construir um loop dinâmico em TensorFlow, que opera em sequencias de
tamanho variável. Vejamos como gerar a sequencia de Fibonacci com
`tf.while_loop`:

```{.python .input}
n = tf.constant(5)

def cond(i, a, b):
    return i < n

def body(i, a, b):
    return i + 1, b, a + b

i, a, b = tf.while_loop(cond, body, (2, 1, 1))

print(tf.Session().run(b))
```

Iso irá imprimir 5. `tf.while_loop` recebe uma função condição, e uma função de
corpo, em adicão aos valores iniciais para variáveis de *loop*. Essas vairáveis
de *loop* são então atualizadas por múltiplas chamadas na função de corpo até
que a condição retorne falso.

Agora imagine que queiramos manter toda a
sequência da série de Fibonacci. Teremos que atualizar o corpo da função para
manter o histórico dos valores correntes:

```{.python .input}
n = tf.constant(5)

def cond(i, a, b, c):
    return i < n

def body(i, a, b, c):
    return i + 1, b, a + b, tf.concat([c, [a + b]], 0)

i, a, b, c = tf.while_loop(cond, body, (2, 1, 1, tf.constant([1, 1])))

print(tf.Session().run(c))
```

Agora se tentarmos rodar esse código o TensorFlow irá reclamar que o formato da
quarta variável  loop está mudando. Portanto deve-se explicitar que a mudança é
intencional:

```{.python .input}
i, a, b, c = tf.while_loop(
    cond, body, (2, 1, 1, tf.constant([1, 1])),
    shape_invariants=(tf.TensorShape([]),
                      tf.TensorShape([]),
                      tf.TensorShape([]),
                      tf.TensorShape([None])))
```

Isso não está somente ficando feio, mas também ineficiente. Note que estamos
construindo um monte de tensores intermediários que não utilizamos. TensorFlow
tem uma  melhor forma de solucionar esse tipo de arrays crescentes. Conheça
`tf.TensorArray`. Façamos a mesma coisa, porém dessa vez com vetores de tensor:

```{.python .input}
n = tf.constant(5)

c = tf.TensorArray(tf.int32, n)
c = c.write(0, 1)
c = c.write(1, 1)

def cond(i, a, b, c):
    return i < n

def body(i, a, b, c):
    c = c.write(i, a + b)
    return i + 1, b, a + b, c

i, a, b, c = tf.while_loop(cond, body, (2, 1, 1, c))

c = c.stack()

print(tf.Session().run(c))
```

*While loops* e vetores de tensor do TensorFlow são ferramentas essenciais para
construir Redes neurais recorrentes complexas. Como exercício tente implementar
[*beam search*](https://en.wikipedia.org/wiki/Beam_search) usando
`tf.while_loop`. Você pode fazê-lo mais eficientemente com vetores de tensor?

## Prototipando kernels e visualizações avançadas com operações Python

Operações
de kernel em TensorFlow são escritos inteiramente em C++ pela sua eficiência.
Porém escrever um TensorFlow kernel em C++ pode ser bastante doloroso. Portanto,
antes de passar horas implementando seu kernel, você pode querer prototipar algo
rapidamente, porém de maneira ineficiente. Com `tf.py_func()` você pode
transformar qualquer parte de código python em uma operação de TensorFlow.

Por exemplo, assim pode-se implementar uma simples kernel de não linearidade
ReLU em TensorFlowem python:

```{.python .input}
import numpy as np
import tensorflow as tf
import uuid

def relu(inputs):
    # Define the op in python
    def _relu(x):
        return np.maximum(x, 0.)

    # Define the op's gradient in python
    def _relu_grad(x):
        return np.float32(x > 0)

    # An adapter that defines a gradient op compatible with TensorFlow
    def _relu_grad_op(op, grad):
        x = op.inputs[0]
        x_grad = grad * tf.py_func(_relu_grad, [x], tf.float32)
        return x_grad

    # Register the gradient with a unique id
    grad_name = "MyReluGrad_" + str(uuid.uuid4())
    tf.RegisterGradient(grad_name)(_relu_grad_op)

    # Override the gradient of the custom op
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        output = tf.py_func(_relu, [inputs], tf.float32)
    return output

To verify that the gradients are correct you can use TensorFlow's gradient checker:

x = tf.random_normal([10])
y = relu(x * x)

with tf.Session():
    diff = tf.test.compute_gradient_error(x, [10], y, [10])
    print(diff)
```

`compute_gradient_error()` calcula o gradiente numericamente e retorna a
diferença entre o gradiente provido. O que se busca é uma diferença muito
pequena.

Note que essa é uma implementação bastante ineficiente, e é utilizável
somente para prototipagem, uma vez que código Python não é paralelizável e não
irá rodar na GPU. Uma vez verificada a ideia, você definitivamente irá querer
escrevê-la como um kernel em C++.

Na prática utilizamos operações em python
para vizualização no Tensorboard. Considere o caso em que você esteja
construindo um modelo de classificador de imagem e queira vizualizar as
predições do modelo durante o treinamento. TensorFlow permite visualizar imagens
com a função `tf.summary.image()`:

```{.python .input}
image = tf.placeholder(tf.float32)
tf.summary.image("image", image)
```

Porém visualiza-se comente a imagem de entrada. Para que se possa visualizar a
predição, tem-se que encontrar uma maneira de adicionar anotações às imagens, o
que pode ser quase impossível com as operações existentes. Uma maneira mais
fácil de fazê-lo é fazendo o desenho em python, e envolve-lo com uma operação
python:

```{.python .input}
import io
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

def visualize_labeled_images(images, labels, max_outputs=3, name="image"):
    def _visualize_image(image, label):
        # Do the actual drawing in python
        fig = plt.figure(figsize=(3, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.imshow(image[::-1,...])
        ax.text(0, 0, str(label),
          horizontalalignment="left",
          verticalalignment="top")
        fig.canvas.draw()

        # Write the plot as a memory file.
        buf = io.BytesIO()
        data = fig.savefig(buf, format="png")
        buf.seek(0)

        # Read the image and convert to numpy array
        img = PIL.Image.open(buf)
        return np.array(img.getdata()).reshape(img.size[0], img.size[1], -1)

    def _visualize_images(images, labels):
        # Only display the given number of examples in the batch
        outputs = []
        for i in range(max_outputs):
            output = _visualize_image(images[i], labels[i])
            outputs.append(output)
        return np.array(outputs, dtype=np.uint8)

    # Run the python op.
    figs = tf.py_func(_visualize_images, [images, labels], tf.uint8)
    return tf.summary.image(name, figs)
```

Nopte que uma vez que o sumário são somente avaliados de vez em quando (não a
cada passo), essa implementação pode ser usada em pretica sem se preocupar com a
eficiência.

## Processamento com Multi-GPU e paralelismo de dados

Caso você esteja
escrevendo um _software_ em uma linguagem como C++ para um computador com um só
processador, fazê-lo rodar em multiplas GPUs em paralelo requereria reescrever o
software do zero. Porém esse não é o caso com TensorFlow. Por conta de sua
natureza simbólica, TensorFlow pode esconder toda essa complexidade, tornando
fácil escalar seu programa entre multiplos CPUs e GPUs.

Comecemos por um exemplo simples de adição de dois vetores em uma CPU:

```{.python .input}
import tensorflow as tf

with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
   a = tf.random_uniform([1000, 100])
   b = tf.random_uniform([1000, 100])
   c = a + b

tf.Session().run(c)
```

A mesma operação pode ser feita de maneira simples em uma GPU:

```{.python .input}
with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
    a = tf.random_uniform([1000, 100])
    b = tf.random_uniform([1000, 100])
    c = a + b
```

Mas e se você possui duas GPUs e queira utilizar ambas? Para fazer isso, podemos
separar os dados e usar uma GPU separada para precessar cada metade:

```{.python .input}
split_a = tf.split(a, 2)
split_b = tf.split(b, 2)

split_c = []
for i in range(2):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
        split_c.append(split_a[i] + split_b[i])

c = tf.concat(split_c, axis=0)
```

Vamos escrever isso de uma maneira mais generalizada para que possamos
substituir a operação de adição por qualquer outra operação:

```{.python .input}
def make_parallel(fn, num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                out_split.append(fn(**{k : v[i] for k, v in in_splits.items()}))

    return tf.concat(out_split, axis=0)


def model(a, b):
    return a + b

c = make_parallel(model, 2, a=a, b=b)
```

Você pode substituir o modelo por qualquer função que tenha como entrada uma
série de tensores e retorne um tensor como resultado com a condição que ambps,
entrada e saída estejam em _batchs_. Note que nós também adicionamos um alcance
variável e setamos a reutilização como verdadeiro. Isso garante que utilizaremos
as mesmas variáveis para processar as duas metades. Isso será útil no nosso
próximo exemplo.

Vejamos um exemplo um pouco mais prático. Queremos treinar a
rede neural em multiplos GPUs. Durante o treinamento nós não somente
necessitamos calcular o passo à frente como também precisa calcular o passo
atrás (os gradientes). porém como podemos paralelizar o cálculo do gradiente?
Isso acaba por ser bastante simples.

Lembre-se do primeiro item que nós
queríamos treinar um polinômio de segunda ordem para algumas amostras.
Organizamos um pouco o código para ter uma pilha das operações na função modelo:

```{.python .input}
import numpy as np
import tensorflow as tf

def model(x, y):
    w = tf.get_variable("w", shape=[3, 1])

    f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
    yhat = tf.squeeze(tf.matmul(f, w), 1)

    loss = tf.square(yhat - y)
    return loss

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

loss = model(x, y)

train_op = tf.train.AdamOptimizer(0.1).minimize(
    tf.reduce_mean(loss))

def generate_data():
    x_val = np.random.uniform(-10.0, 10.0, size=100)
    y_val = 5 * np.square(x_val) + 3
    return x_val, y_val

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for _ in range(1000):
    x_val, y_val = generate_data()
    _, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})

_, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})
print(sess.run(tf.contrib.framework.get_variables_by_name("w")))
```

Agora vamos usar `make_parallel` que escrevemos para paralelizar. Precisamos
modificar somente duas linhas de código do código acima:

```{.python .input}
loss = make_parallel(model, 2, x=x, y=y)

train_op = tf.train.AdamOptimizer(0.1).minimize(
    tf.reduce_mean(loss),
    colocate_gradients_with_ops=True)
```

A unica coisa que precisamos para mudar para paralelizar o _backpropagation_ de
gradientes é colocar a flag `colocate_gradients_with_ops` como verdadeiro. Para
assegurar que a operação de gradiente rode na mesma GPU que a operação original.

## Debugando modelos TensorFlow

A natureza simbólica do TensorFlow o torna
relativamente mais difícil de debugar em comparação com código python regular.
Aqui introduzimos algumas ferramentas incluídas no TensorFlow para tornar a
tarefa de debugar mais fácil.

Provavelmente o erro mais comum que se pode fazer ao utilizar TensorFlow é
passar tensores de tamanhos errados às operações. Muitas operações do TensorFlow
podem operar com tensores de ranks e tamanhos diferentes. Isso pode ser
conveniente quando se utiliza uma API, porém pode causar dor de cabeça quando as
coisas dão errado.

Por exemplo, considere a operação `tf.matmul`, que
multiplica duas matrizes:

```{.python .input}
a = tf.random_uniform([2, 3])
b = tf.random_uniform([3, 4])
c = tf.matmul(a, b)  # c is a tensor of shape [2, 4]
```

Porém a mesma função também faz multiplicação matricial em _batch_:

```{.python .input}
a = tf.random_uniform([10, 2, 3])
b = tf.random_uniform([10, 3, 4])
tf.matmul(a, b)  # c is a tensor of shape [10, 2, 4]
```

Outro exemplo que discutimos antes na seção de difusão é a operação de adição
que suporta difusão:

```{.python .input}
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = a + b  # c is a tensor of shape [2, 2]
```

### Validando seus tensores com operações `tf.assert*`

Uma maneira de reduzir as chances de comportamento indesejado é verificar
explicitamente o rank ou dimensão de tensores intermediários com operações
`tf.assert*`.

```{.python .input}
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
check_a = tf.assert_rank(a, 1)  # This will raise an InvalidArgumentError exception
check_b = tf.assert_rank(b, 1)
with tf.control_dependencies([check_a, check_b]):
    c = a + b  # c is a tensor of shape [2, 2]
```

Lembre-se que nós de afirmação, assim como outras operações são parte do grafo e
se não from avaliados são podado durante `Session.run()`. Portanto assegure-se
de criar dependências explícitas para operações de afirmação, para forçar o
TensorFlow a executá-los.

Você pode também  afirmações para validar o valore de
tensores no _runtime_:

```{.python .input}
check_pos = tf.assert_positive(a)
```

Veja os documentos oficiais para [lista completa de operações de
afirmação](https://github.com/tensorflow/docs/tree/master/site/en/api_guides/python).

### Registrando valores de tensores com `tf.Print`

Outra função inerente útil para debugar é `tf.Print` que registra os tensores
dados para o erro padrão:

```{.python .input}
input_copy = tf.Print(input, tensors_to_print_list)
```

Note que `tf.Print` retorna a cópia de seu primeiro argumento como uma saída.
Uma maneira de forçar `tf.Print` a rodar é passar sua saída para outra operação
que seja executada. Por exemplo, se você quer escrever o valor dos tensores a e
b antes de adicionar então poderiamos fazer algo assim:

```{.python .input}
a = ...
b = ...
a = tf.Print(a, [a, b])
c = a + b
```

Alternativamente podemos manualmente definir o controle de dependência.

### Checando o gradiente com `tf.compute_gradient_error`

Nem todas as operações
de TensorFlow vêm com gradientes, e é facil construir graphs (não
intencionalemente) para o qual TensorFlow não consegue calcular os gradientes.

Vejamos um exemplo:

```{.python .input}
import tensorflow as tf

def non_differentiable_softmax_entropy(logits):
    probs = tf.nn.softmax(logits)
    return tf.nn.softmax_cross_entropy_with_logits(labels=probs, logits=logits)

w = tf.get_variable("w", shape=[5])
y = -non_differentiable_softmax_entropy(w)

opt = tf.train.AdamOptimizer()
train_op = opt.minimize(y)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    sess.run(train_op)

print(sess.run(tf.nn.softmax(w)))
```

Estamos utilizando `tf.softmax_cross_entropy_with_logits` para definier a
entropia de uma distribuição categórica. E por fim utilizamos o otimizador Adam
para encontrar os pesos com máxima entropia. Se você fez um curso de teoria da
informação, você saberia que distribuição uniforme contem máxima entropia.
Portanto esperaria-se que o resultado fosse [0.2, 0.2, 0.2, 0.2, 0.2]. Porém
caso você rode isso, o resultado gerado pode ser algo inesperado como:

```{.python .input}
[ 0.34081486  0.24287023  0.23465775  0.08935683  0.09230034]
```

Acontece que `tf.nn.softmax_cross_entropy_with_logits` tem gradiente indefinido
no que dis relação aos rótulos! Porém como podemos identificar esse erro se não
soubessemos disso?

Felizmente para nós, TensorFlow vem com um diferenciador
numérico que pode ser utilizado para encontrar erro simbólicos em gradientes.
Vejamos como isso funciona:

```{.python .input}
with tf.Session():
    diff = tf.test.compute_gradient_error(w, [5], y, [])
    print(diff)
```

Se você rodar isso, você verá que a diferença entre os gradientes numérico e
simbólico é consideravelmente altas (0.06 - 0.1 nas minhas tentativas).

Agora
vamos corrigir nossa função com uma versão diferenciável da entropia e chegar
outra vez:

```{.python .input}
import tensorflow as tf
import numpy as np

def softmax_entropy(logits, dim=-1):
    plogp = tf.nn.softmax(logits, dim) * tf.nn.log_softmax(logits, dim)
    return -tf.reduce_sum(plogp, dim)

w = tf.get_variable("w", shape=[5])
y = -softmax_entropy(w)

print(w.get_shape())
print(y.get_shape())

with tf.Session() as sess:
    diff = tf.test.compute_gradient_error(w, [5], y, [])
    print(diff)
```

A diferença deve ser ~0.0001 o que é muito melhor.

Agora se rodarmos o
otimizador outra vez com a versão correta podemos que os pesos finais são:

```{.python .input}
[ 0.2  0.2  0.2  0.2  0.2]
```

O que é exatamente o que esperávamos.

[Sumário
TensorFlow](https://github.com/tensorflow/docs/tree/master/site/en/api_guides/python),
e [tfdbg (TensorFlow
Debugger)](https://github.com/tensorflow/docs/tree/master/site/en/api_guides/python)
são outras ferramentas que podem ser utilizadas para debugar. Por favor vá aos
documentos oficiais para aprender mais.

## Estabilidade numérica em TensorFlow

Ao utilizar qualquer módulo de computação
numérica como NumPy ou TensorFlow, é importante atentar-se que escrever o código
matematicamente correto, não necessariamente leva a resultados corretos. Também
se faz necessário assegurar-se que os cálculos são estáveis.

Vamos começar com um exemplo simples. Desde o ensino fundamental sabemos que
x*y/y é igual a x para qualquer valor de x diferente de zero. Porém vejamos se
isso é sempre verdade na prática:

```{.python .input}
import numpy as np

x = np.float32(1)

y = np.float32(1e-50)  # y would be stored as zero
z = x * y / y

print(z)  # prints nan
```

A razão para o resultado incorreto é simplesmente que y é muito pequeno para um
tipo float32. Um problema similar ocorre também quando y é muito grande:

```{.python .input}
y = np.float32(1e39)  # y would be stored as inf
z = x * y / y

print(z)  # prints 0
```

O menor número positivo que o tipo float32 pode representar é 1.4013e-45 e
qualquer valor menor é guardado como zero. Da mesma maneira, qualquer número
acima de 3.40282e+38 é guardado como infinito.

```{.python .input}
print(np.nextafter(np.float32(0), np.float32(1)))  # prints 1.4013e-45
print(np.finfo(np.float32).max)  # print 3.40282e+38
```

Para assegurar-se que seus cálculos são estáveis, é preciso evitar valores muito
pequeno ou muito grandes. Isso pode soar um pouco óbvio, porém esse tipo de
problema pode ser extremamente difícil de debugar, especialmente quando se está
usando gradiente descendente em TensorFlow. Isso porque você não somente tem que
se assegurar que todos os valores no caminho direto estão em um intervalo
válido, assim como tem que se assegurar que no caminho inverso (durante o
cálculo de gradiente) também estejam em um intervalo válido.

Vejamos um exemplo
real. Queremos calcular o softmax de um vetor de
[logits](https://pt.wikipedia.org/wiki/Logit). Uma implementação 
ingênua seria
algo mais ou menos assim:

```{.python .input}
import tensorflow as tf

def unstable_softmax(logits):
    exp = tf.exp(logits)
    return exp / tf.reduce_sum(exp)

tf.Session().run(unstable_softmax([1000., 0.]))  # prints [ nan, 0.]
```

Note que calcular a exonencial de logits para valores relativamente pequenos
resulta em resultados gigantes que estão fora do intervalo do float32. O maior
valor logit para nossa implementação ingênua do softmax é ln(3.40282e+38) =
88.7, qualquer coisa acima disso retornaria NaN.

Mas como podemos fazê-la mais
estável? A solução é bastante simples. É fácil ver que exp(x - c) / ∑ exp(x - c)
= exp(x) / ∑ exp(x). Portanto podemos subtrair qualquer constante do resultado
logit e o resultado permanece o mesmo. Escolhemos essa constante para ser o
máximo de logits. Dessa forma o domínio da função exponencial seria limitado a
[-inf,0], e consequentemente seu intervalo seria [0.0,1.0], o que é desejável:

```{.python .input}
import tensorflow as tf

def softmax(logits):
    exp = tf.exp(logits - tf.reduce_max(logits))
    return exp / tf.reduce_sum(exp)

tf.Session().run(softmax([1000., 0.]))  # prints [ 1., 0.]
```

Vejamos um caso mais complicado. Considere que temos um problema de
classificação. Usamos a função softmax para produzir as probabilidades de nossos
logits. Definimos a função de perda ara ser a entropia cruzada entre a predição
e os rótulos. Lembre-se que a entropia cruzada para uma distribuição categórica
pode ser definida como xe(p, q) = -∑ p_i log(q_i). Portanto uma implementação
ingênua da entropia cruzada seria algo como:

```{.python .input}
def unstable_softmax_cross_entropy(labels, logits):
    logits = tf.log(softmax(logits))
    return -tf.reduce_sum(labels * logits)

labels = tf.constant([0.5, 0.5])
logits = tf.constant([1000., 0.])

xe = unstable_softmax_cross_entropy(labels, logits)

print(tf.Session().run(xe))  # prints inf
```

```{.json .output n=0}
[
 {
  "ename": "NameError",
  "evalue": "ignored",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m\u001b[0m",
   "\u001b[0;31mNameError\u001b[0mTraceback (most recent call last)",
   "\u001b[0;32m<ipython-input-3-b8f0b41f6b2d>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      3\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0;34m-\u001b[0m\u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreduce_sum\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlabels\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mlogits\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m \u001b[0mlabels\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconstant\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0.5\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m0.5\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      6\u001b[0m \u001b[0mlogits\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconstant\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1000.\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m0.\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mNameError\u001b[0m: name 'tf' is not defined"
  ]
 }
]
```

Note que nessa implementação que à medida que a saída do softmax se aproxima do
zero, o regristo de saída se aproxima de infinito o que causa instabilidade na
computação. Podemos reescrever essa função expandindo o softmax e fazendo
algumas simplificções:

```{.python .input}
def softmax_cross_entropy(labels, logits):
    scaled_logits = logits - tf.reduce_max(logits)
    normalized_logits = scaled_logits - tf.reduce_logsumexp(scaled_logits)
    return -tf.reduce_sum(labels * normalized_logits)

labels = tf.constant([0.5, 0.5])
logits = tf.constant([1000., 0.])

xe = softmax_cross_entropy(labels, logits)

print(tf.Session().run(xe))  # prints 500.0

We can also verify that the gradients are also computed correctly:

g = tf.gradients(xe, logits)
print(tf.Session().run(g))  # prints [0.5, -0.5]
```

O que é correto.

Deixe-me lebrar-lhes outra vez que cuidado extra tem de ser
tomado quando se usa gradiente descendente para se assegurar que o intervalo de
valores que a função está utilizando, assim como seus gradientes para cada
camada estão dentro do intervalo de estabilidade. Funções exponencial e
logarítimo quando utilizadas ingenuamente são especialmente problemáticas porque
podem variar de valores muito pequenos a valores muito grandes rapidamente.

## Construindo um framework de treinamento de rede neural com API _learn_

Por
simplicidade, na maioria dos exemplos aqui executados nós criamos sessões
manualmente e não nos preocupamos em salvar e carregar _checkpoints_ porém
normalmente não fazemos assim na prática. Você provavelmente quererá usar o API
_learn_ para cuidar do gerenciamento e registro de sessões. Nós fornecemos um
framework simples porém prático para treinamento de redes neurais utilizando
TensorFlow. Nesse item explicaremos como esse framework funciona.

Ao se
trabalhar com modelos de redes neurais normalmente tem-se uma separação do
conjunto de treino e de teste. Treina-se o modelo com o conjunto de treion, e de
vez em quando avalia-se junto ao conjunto de teste e calcula-se as métricas.
Também é necessário salvar os parâmetros do modelo como um _checkpoint_, e
idealmente espera-se ser capaz de parar e retomar o treinamento de qualquer
ponto. O API _learn_ do TensorFlow é feito para tornar essa tarefa mais simples,
deixando-nos livres para focar no desenvolvimento do modelo em si.

A forma mais básica de usar o API `tf.learn` é usando o objeto `tf.Estimator`
diretamente. É necessário definir um model que defina a função de perda, a
operação de treino, um ou um conjunto de predições, e um conjunto de métricas de
avaliação (opcional):

```{.python .input}
import tensorflow as tf

def model_fn(features, labels, mode, params):
    predictions = ...
    loss = ...
    train_op = ...
    metric_ops = ...
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metric_ops)

params = ...
run_config = tf.contrib.learn.RunConfig(model_dir=FLAGS.output_dir)
estimator = tf.estimator.Estimator(
    model_fn=model_fn, config=run_config, params=params)
```

Para treinar o modelo basta simplesmente chamar a função `Estimator.train()`
inserindo uma função de entrada para leitura dos dados:

```{.python .input}
def input_fn():
    features = ...
    labels = ...
    return features, labels

estimator.train(input_fn=input_fn, max_steps=...)
```

E para avaliar o modelo basta chamar `Estimator.evaluate()`:

```{.python .input}
estimator.evaluate(input_fn=input_fn)
```

```{.json .output n=0}
[
 {
  "ename": "NameError",
  "evalue": "ignored",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m\u001b[0m",
   "\u001b[0;31mNameError\u001b[0mTraceback (most recent call last)",
   "\u001b[0;32m<ipython-input-1-27863659c21c>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mestimator\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mevaluate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minput_fn\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0minput_fn\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m: name 'estimator' is not defined"
  ]
 }
]
```

O objeto `Estimator` pode ser bom o suficiente para casos simples, portém
TensorFlow fornece um objeto de maior hierarquia chamado `Experiment` que
fornece alguma funcionalidades adicionais. Criando um objeto `Experiment` é
muito fácil:

```{.python .input}
experiment = tf.contrib.learn.Experiment(
    estimator=estimator,
    train_input_fn=train_input_fn,
    eval_input_fn=eval_input_fn)
```

Agora podemos chamar a função `train_and_evaluate` para calcular as métricas
enquanto treina:

```{.python .input}
experiment.train_and_evaluate()
```

Uma maneira ainda mais alto nível de rodar experimento é usando a função
`learn_runner.run()`. Assim fica a função principal:

```{.python .input}
import tensorflow as tf

tf.flags.DEFINE_string("output_dir", "", "Optional output dir.")
tf.flags.DEFINE_string("schedule", "train_and_evaluate", "Schedule.")
tf.flags.DEFINE_string("hparams", "", "Hyper parameters.")

FLAGS = tf.flags.FLAGS

def experiment_fn(run_config, hparams):
  estimator = tf.estimator.Estimator(
    model_fn=make_model_fn(),
    config=run_config,
    params=hparams)
  return tf.contrib.learn.Experiment(
    estimator=estimator,
    train_input_fn=make_input_fn(tf.estimator.ModeKeys.TRAIN, hparams),
    eval_input_fn=make_input_fn(tf.estimator.ModeKeys.EVAL, hparams))

def main(unused_argv):
  run_config = tf.contrib.learn.RunConfig(model_dir=FLAGS.output_dir)
  hparams = tf.contrib.training.HParams()
  hparams.parse(FLAGS.hparams)

  estimator = tf.contrib.learn.learn_runner.run(
    experiment_fn=experiment_fn,
    run_config=run_config,
    schedule=FLAGS.schedule,
    hparams=hparams)

if __name__ == "__main__":
  tf.app.run()
```

A flag `schedule` decide qual função membro do objeto `Experiment` é chamado.
Portanto, se você por exemplo setar `schedule` para "train_and_evaluate",
`experiment.train_and_evaluate()` seria chamado.

A função de entrada retorna
dois tensores (ou dicionário de tensores) fornecendo os recursos e rótulos a
serem passados para o modelo:

```{.python .input}
def input_fn():
    features = ...
    labels = ...
    return features, labels
```

Veja
[mnist.py](https://github.com/vahidk/TensorflowFramework/blob/master/dataset/mnist.py)
para exemplo de como ler os dados com o API dataset. Para aprender diferentes
maneiras de ler os dados em TensorFlow leia [esse
item](https://github.com/vahidk/EffectiveTensorflow#data).  

O framework também
vem com uma simples rede de classificação convulocional em
[alexnet.py](https://github.com/vahidk/TensorflowFramework/blob/master/model/alexnet.py)
que inclui um exemplo.

Isso é tudo! Isso é tudo o que se necessita para começar
com o API _learn_ de TensorFlow. Recomendo analisar o [código
fonte](https://github.com/vahidk/TensorFlowFramework) do framework e visitar o
API python oficial para aprender mais sobre o API _learn_.

# Parte II: Cookbook

Essa seção inclui a implementação de várias operações
comuns em TensorFlow

## Verificar Dimensão

```{.python .input}
def get_shape(tensor):
  """Returns static shape if available and dynamic shape otherwise."""
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims
```

## Obter _Batch_

```{.python .input}
def batch_gather(tensor, indices):
  """Gather in batch from a tensor of arbitrary size.

  In pseudocode this module will produce the following:
  output[i] = tf.gather(tensor[i], indices[i])

  Args:
    tensor: Tensor of arbitrary size.
    indices: Vector of indices.
  Returns:
    output: A tensor of gathered values.
  """
  shape = get_shape(tensor)
  flat_first = tf.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
  indices = tf.convert_to_tensor(indices)
  offset_shape = [shape[0]] + [1] * (indices.shape.ndims - 1)
  offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
  output = tf.gather(flat_first, indices + offset)
  return output
```

## Beam Search

```{.python .input}
import tensorflow as tf

def rnn_beam_search(update_fn, initial_state, sequence_length, beam_width,
                    begin_token_id, end_token_id, name="rnn"):
  """Beam-search decoder for recurrent models.

  Args:
    update_fn: Function to compute the next state and logits given the current
               state and ids.
    initial_state: Recurrent model states.
    sequence_length: Length of the generated sequence.
    beam_width: Beam width.
    begin_token_id: Begin token id.
    end_token_id: End token id.
    name: Scope of the variables.
  Returns:
    ids: Output indices.
    logprobs: Output log probabilities probabilities.
  """
  batch_size = initial_state.shape.as_list()[0]

  state = tf.tile(tf.expand_dims(initial_state, axis=1), [1, beam_width, 1])

  sel_sum_logprobs = tf.log([[1.] + [0.] * (beam_width - 1)])

  ids = tf.tile([[begin_token_id]], [batch_size, beam_width])
  sel_ids = tf.zeros([batch_size, beam_width, 0], dtype=ids.dtype)

  mask = tf.ones([batch_size, beam_width], dtype=tf.float32)

  for i in range(sequence_length):
    with tf.variable_scope(name, reuse=True if i > 0 else None):

      state, logits = update_fn(state, ids)
      logits = tf.nn.log_softmax(logits)

      sum_logprobs = (
          tf.expand_dims(sel_sum_logprobs, axis=2) +
          (logits * tf.expand_dims(mask, axis=2)))

      num_classes = logits.shape.as_list()[-1]

      sel_sum_logprobs, indices = tf.nn.top_k(
          tf.reshape(sum_logprobs, [batch_size, num_classes * beam_width]),
          k=beam_width)

      ids = indices % num_classes

      beam_ids = indices // num_classes

      state = batch_gather(state, beam_ids)

      sel_ids = tf.concat([batch_gather(sel_ids, beam_ids),
                           tf.expand_dims(ids, axis=2)], axis=2)

      mask = (batch_gather(mask, beam_ids) *
              tf.to_float(tf.not_equal(ids, end_token_id)))

  return sel_ids, sel_sum_logprobs
```

## Combinar - Merge

```{.python .input}
import tensorflow as tf

def merge(tensors, units, activation=tf.nn.relu, name=None, **kwargs):
  """Merge features with broadcasting support.

  This operation concatenates multiple features of varying length and applies
  non-linear transformation to the outcome.

  Example:
    a = tf.zeros([m, 1, d1])
    b = tf.zeros([1, n, d2])
    c = merge([a, b], d3)  # shape of c would be [m, n, d3].

  Args:
    tensors: A list of tensor with the same rank.
    units: Number of units in the projection function.
  """
  with tf.variable_scope(name, default_name="merge"):
    # Apply linear projection to input tensors.
    projs = []
    for i, tensor in enumerate(tensors):
      proj = tf.layers.dense(
          tensor, units, activation=None,
          name="proj_%d" % i,
          **kwargs)
      projs.append(proj)

    # Compute sum of tensors.
    result = projs.pop()
    for proj in projs:
      result = result + proj

    # Apply nonlinearity.
    if activation:
      result = activation(result)
  return result
```

## Entropia

```{.python .input}
import tensorflow as tf

def softmax_entropy(logits, dim=-1):
  """Compute entropy over specified dimensions."""
  plogp = tf.nn.softmax(logits, dim) * tf.nn.log_softmax(logits, dim)
  return -tf.reduce_sum(plogp, dim)
```

## Divergência-KL

```{.python .input}
def gaussian_kl(q, p=(0., 0.)):
  """Computes KL divergence between two isotropic Gaussian distributions.

  To ensure numerical stability, this op uses mu, log(sigma^2) to represent
  the distribution. If q is not provided, it's assumed to be unit Gaussian.

  Args:
    q: A tuple (mu, log(sigma^2)) representing a multi-variatie Gaussian.
    p: A tuple (mu, log(sigma^2)) representing a multi-variatie Gaussian.
  Returns:
    A tensor representing KL(q, p).
  """
  mu1, log_sigma1_sq = q
  mu2, log_sigma2_sq = p
  return tf.reduce_sum(
    0.5 * (log_sigma2_sq - log_sigma1_sq +
           tf.exp(log_sigma1_sq - log_sigma2_sq) +
           tf.square(mu1 - mu2) / tf.exp(log_sigma2_sq) -
           1), axis=-1)
```

## Paralelizar

```{.python .input}
def make_parallel(fn, num_gpus, **kwargs):
  """Parallelize given model on multiple gpu devices.

  Args:
    fn: Arbitrary function that takes a set of input tensors and outputs a
        single tensor. First dimension of inputs and output tensor are assumed
        to be batch dimension.
    num_gpus: Number of GPU devices.
    **kwargs: Keyword arguments to be passed to the model.
  Returns:
    A tensor corresponding to the model output.
  """
  in_splits = {}
  for k, v in kwargs.items():
    in_splits[k] = tf.split(v, num_gpus)

  out_split = []
  for i in range(num_gpus):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        out_split.append(fn(**{k : v[i] for k, v in in_splits.items()}))

  return tf.concat(out_split, axis=0)
```

## ReLU simples

```{.python .input}
def leaky_relu(tensor, alpha=0.1):
    """Computes the leaky rectified linear activation."""
    return tf.maximum(tensor, alpha * tensor)
```

## Normalização de _Batch_

```{.python .input}
def batch_normalization(tensor, training=False, epsilon=0.001, momentum=0.9, 
                        fused_batch_norm=False, name=None):
  """Performs batch normalization on given 4-D tensor.
  
  The features are assumed to be in NHWC format. Noe that you need to 
  run UPDATE_OPS in order for this function to perform correctly, e.g.:

  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = optimizer.minimize(loss)

  Based on: https://arxiv.org/abs/1502.03167
  """
  with tf.variable_scope(name, default_name="batch_normalization"):
    channels = tensor.shape.as_list()[-1]
    axes = list(range(tensor.shape.ndims - 1))

    beta = tf.get_variable(
      'beta', channels, initializer=tf.zeros_initializer())
    gamma = tf.get_variable(
      'gamma', channels, initializer=tf.ones_initializer())

    avg_mean = tf.get_variable(
      "avg_mean", channels, initializer=tf.zeros_initializer(),
      trainable=False)
    avg_variance = tf.get_variable(
      "avg_variance", channels, initializer=tf.ones_initializer(),
      trainable=False)

    if training:
      if fused_batch_norm:
        mean, variance = None, None
      else:
        mean, variance = tf.nn.moments(tensor, axes=axes)
    else:
      mean, variance = avg_mean, avg_variance
   
    if fused_batch_norm:
      tensor, mean, variance = tf.nn.fused_batch_norm(
        tensor, scale=gamma, offset=beta, mean=mean, variance=variance, 
        epsilon=epsilon, is_training=training)
    else:
      tensor = tf.nn.batch_normalization(
        tensor, mean, variance, beta, gamma, epsilon)

    if training:
      update_mean = tf.assign(
        avg_mean, avg_mean * momentum + mean * (1.0 - momentum))
      update_variance = tf.assign(
        avg_variance, avg_variance * momentum + variance * (1.0 - momentum))

      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance)

  return tensor
```
