# DL_cv-mdt_NVIDIA_Cert_Course_StudyPI
NVIDIA 국제 인증과정 - 컴퓨터 비전 딥러닝 &amp; 다중 데이터 유형 딥러닝 기초


* caffe -> Torch에 흡수
* keras -> TensorFlow에 흡수

![](https://github.com/lkeonwoo94/DL_cv-mdt_NVIDIA_Cert_Course_StudyPI/blob/master/ML.png)


참고자료 : [가장 빠르게 딥러닝 이해하기](https://www.slideshare.net/yongho/ss-79607172/49)
[Optimizer정리-수식](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html)


> 참고자료에서 내가 몰랐던 부분

![SGD](https://github.com/lkeonwoo94/DL_cv-mdt_NVIDIA_Cert_Course_StudyPI/blob/master/SGD.png)   
![opt](https://github.com/lkeonwoo94/DL_cv-mdt_NVIDIA_Cert_Course_StudyPI/blob/master/opt.png)
![optmizer발달](https://github.com/lkeonwoo94/DL_cv-mdt_NVIDIA_Cert_Course_StudyPI/blob/master/optmizer%EB%B0%9C%EB%8B%AC.png)

---

안녕하세요,
CV, MDT강의를 통틀어 도움이 될만한 질문과 답변을 정리해봤습니다.
심심하실 때 한번씩 복습삼아 읽어봐주시면 도움이 되리라 생각합니다^^
질문 주신분들 감사드리고, 참가자 여러분들 모두 고생 많으셨습니다.
---------------------------------------------------------------------------------------------------
1. 컨볼루션 갯수나 필터 갯수는 자기가 직접 정하는 건가요? , ' 컨볼루션 - 풀링 - 컨볼루션 - 풀링 ' 이 아닌 '컨볼루션 - 컨볼루션 - 컨볼루션 - 풀링 - 풀링 - ...' 순으로 사용자가 직접 바꾸어도 상관 없나요 ?! 또한 왜 이 때 Fully Connected Layer를 사용했는지 등등 층을 쌓는 순서에 대해 궁금해요.




-> 갯수들을 직접 정할 수 있습니다.
-> 그 순서는 풀링을 컨벌루선 중앙에 배치하는게 일반적입니다. 예를들어, 컨벌루션을 연속으로 쓰고 싶은 경우에는 컨-컨-컨-풀-컨-컨-컨-풀과 같이 사용하기도 하고, 얼마든지 직접 바꾸어 사용하기도 합니다. 아래링크의 VGG와 같은 유명한 네트워크도 컨컨-풀-컨컨-풀- 이렇게 구성합니다.
https://www.google.com/search?q=vggnet&sxsrf=ACYBGNTG3d5ZhZa_LxW-eOAVA5SldW9pCw:1581309366464&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjVv8W4lMbnAhWLMN4KHSouBzcQ_AUoAXoECAwQAw&biw=1805&bih=928#imgrc=AON1iVApOf8dvM




-> 풀링은 컨벌루션을 통해 depth의 숫자가 깊어지기 직전에 보통 이루어 집니다. depth만 깊어지기만하다가는 memory가 너무 많이 필요로되기 때문에 깊어지기 직전에 메모리를 줄일 목적과 적당히 컨벌루션이 이루어진 시점에서 dominant한 feature들을 뽑아내는 목적으로 쓰입니다.




-> Fully connected layer는 flatten을 시킴으로써 공간적보를 없애버립니다. 그렇기 때문에 초반에는 입력으로부터 공간정보를 활용하는 feature들을 convolution으로 충분히 뽑아낸 뒤, 그 feature들을 flatten을 시켜 개인지 고양인지 mapping시켜보기 위한 fully connected layer를 배치시키는 것 입니다.




2. 이미지를 input으로 받고 컨볼루션과 풀링을 거쳐서 depth가 늘어나고 width, height가 줄어들면서 일렬로 펴지게 되는 과정을 거치게 되는데 꼭 이 과정을 커쳐야만 하는지요? Input으로 받고 컨볼루션 필터 사이즈나, 풀링 사이즈를 많이 키워서 input size를 확확 줄이면은 안되는지?




-> 유용하게 쓰일 수 있는 고차원 정보를 추출하기 위해 CNN구성의 앞단에는 저차원, 점점 중에서 고차원으로 추출을 할 수 있습니다. 이와같이 유용한 고차원을 뽑기 위해서는 한번에 영상사이즈를 줄여버리면 잘 뽑히지 못하는 문제가 발생합니다. 아래 링크를 통해 보시면 아시겠지만, 결국 사람얼굴이라는 feature를 뽑기 위해 단계별로 feature들을 뽑는과정이 있는것을 알 수 있습니다.
https://www.google.com/search?q=deep+learning+face+&tbm=isch&ved=2ahUKEwjxjazAlMbnAhWRG6YKHcgjB_sQ2-cCegQIABAA&oq=deep+learning+face+&gs_l=img.3..0j0i30l3j0i24l2j0i8i30l4.274192.276551..276821...2.0..1.189.1684.18j1......0....1..gws-wiz-img.....10..35i39j0i131j35i362i39.akvy1Pfq4qU&ei=xt1AXvGSMpG3mAXIx5zYDw&bih=928&biw=1805#imgrc=4Q9C-i666c9lgM




3. 유명한 네트워크 모델들 (AlexNet ...) 내부를 보면 여러개로 쌓여진 층을 확인할 수 있는데, 층이 이렇게 쌓여진 이유같은게 존재할까요? ( 그냥 단지 많은 노가다를 했는데 이렇게 쌓는게 제일 성능이 좋았다더라. 라던지 ?)




-> 층을 깊게 쌓은 이유는 저차원 정보에서 고차원 정보를 추출하기 위함이고, 이를 위해 층을 많이 쌓고, 점점 깊어질수록 feature를 더 뽑게 합니다. 하지만 예전에는 깊게만 쌓는다고 좋을까?라고해서 깊게만 쌓았더니 성과가 좋지 못했으나, ResNet이라는 알고리즘이 깊게 쌓을수록 residual을 레이어간에 잘 전달만 해주면 performance가 높아진다라는 논문을 쓰기도 했습니다.
아래링크 참고해주세요!
논문원본: https://arxiv.org/abs/1512.03385
한글설명: https://blog.naver.com/sohyunst/221666285678




4. 나중에 영상 이미지를 가지고 deep learing과 AI에 활용하려면 구체적으로 어떻게 해야 하는지, deep learning을 활용하는데 어느 정도를 알아야 할까요??




-> 우선 data, algorithm, GPU가 필요하다는건 알고 계시죠?! data는 labeling된 (정답이 있는) 데이터를 말합니다!
우선 이 세개가 필수요소라 보시면 되고, github에 classification, detection, segmentation, 등등이 running되게하는 소스코드가 있습니다. 소스코드들을 훑어봤을 때 우리의 데이터에 맞게 돌아가게만 소스코드들이 요구하는 데이터포맷만 포맷팅해줄 수 있을정도의 python 문법을 사용할 줄 아시면 됩니다.




예를 들어, csv type의 파일형태로 detection algorithm을 running시키게 하는 소스코드를 사용하시겠다 하면 선생님의 data들을 csv type에 맞춰 python 코드들로 만들기만 하면 그 detection code가 돌아갑니다.
처음부터 공부하고 접근하기보다, 예제형태의 소스코드들을 구글링해서 필요한 스킬들을 얻어가시면 될 듯 합니다.




5. 아산 병원에서는 의료 영상에 대한 deep learning 연구를 어떻게 하고 있는지요?




-> 저희는 철저히 의료진의 unmet needs에 의해 연구를 시작합니다. 의료진의 그 needs를 공학자들과 discussion들을 통해 서로 이해하고나서 IRB를 통해 데이터사용에 대한 심의를 받게 되고, 통과하게 되면 연구목적으로 특정시기, 목적에 의해 의료 데이터를 활용할 수 있게 됩니다.
의료진들도 격주 열리는 회의에 항상 참여하여 적극적으로 인공지능 성능들에 대한 discussion을 하면서 서로의 feedback을 활발히 주고받고 있습니다.
결국에는 의료진들도 workflow에 개선시키고자 unmet needs를 통해 연구를 하면서 논문실적을 끝으로 연구를 마무리하는 조금의 괴리감(?) 을 갖게하는 몇몇 주제들도 있긴 하지만, 의대생들의 교육목적이나 사업화, 혹은 기술이전등으로 결과들을 내고 있다고 보시면 됩니다.




6. Dropout을 넣으면 해당 히든 레이어 중 일부 노드가 이용되지 않으니 연산속도가 올라갈꺼라 생각했는데, 오히려 늘어나는 것 같은데 왜 그럴까요?




-> 개념상 노드를 off시키기만 하기 때문에 쉬워보이고 학습하기 위한 weight가 증가하는 것은 아닙니다만,
randomness에 의해 off되는 노드들도 매번 바뀌는 등의 소스코드상 구현단의 추가적인 computation cost들이 필요할 것으로 보입니다.
그래서 dropout이 실제 동작하기 위한 소스코드를 찾아봤는데 소스드가 100줄정도 되더군요!
https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/nn_ops.py#L4232-L4317




7. CNN에서 각 픽셀값들의 fully connected network 로 구성하지 않고 앞단에 convolution 과정을 거치는 근본적인 이유가 무엇인가요? convolution 과정을 통해 얻는 이점이 무엇인가요?




-> Convolutional neural network를 통해 공간정보로부터 다양한 feature를 추출할 수 있기 때문입니다. 공간정보를 없애면서 처리하는 fully connected layer이 맨 앞에 배치된 채로 feature를 추출하게 되버리면 영상에서 공간상에서 나타날 수 있는 형태 (눈, 코, 입, 등등)를 나타내는 정보들을 추출할 수 없는 것이죠.

```
arXiv.orgarXiv.org
Deep Residual Learning for Image Recognition
Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We...
tensorflow/python/ops/nn_ops.py:4232-4317
```
```python
@tf_export("nn.dropout", v1=[])
def dropout_v2(x, rate, noise_shape=None, seed=None, name=None):
  """Computes dropout.

  With probability `rate`, drops elements of `x`. Input that are kept are
  scaled up by `1 / (1 - rate)`, otherwise outputs `0`.  The scaling is so that
  the expected sum is unchanged.

  **Note:** The behavior of dropout has changed between TensorFlow 1.x and 2.x.
  When converting 1.x code, please use named arguments to ensure behavior stays
  consistent.

  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.

  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    name: A name for this operation (optional).

  Returns:
    A Tensor of the same shape of `x`.

  Raises:
    ValueError: If `rate` is not in `(0, 1]` or if `x` is not a floating point
      tensor.
  """
  with ops.name_scope(name, "dropout", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if not x.dtype.is_floating:
      raise ValueError("x has to be a floating point tensor since it's going to"
                       " be scaled. Got a %s tensor instead." % x.dtype)
    if isinstance(rate, numbers.Real):
      if not (rate >= 0 and rate < 1):
        raise ValueError("rate must be a scalar tensor or a float in the "
                         "range [0, 1), got %g" % rate)
      if rate > 0.5:
        logging.log_first_n(
            logging.WARN, "Large dropout rate: %g (>0.5). In TensorFlow "
            "2.x, dropout() uses dropout rate instead of keep_prob. "
            "Please ensure that this is intended.", 5, rate)

    # Early return if nothing needs to be dropped.
    if isinstance(rate, numbers.Real) and rate == 0:
      return x
    if context.executing_eagerly():
      if isinstance(rate, ops.EagerTensor):
        if rate.numpy() == 0:
          return x
    else:
      rate = ops.convert_to_tensor(
          rate, dtype=x.dtype, name="rate")
      rate.get_shape().assert_has_rank(0)

      # Do nothing if we know rate == 0
      if tensor_util.constant_value(rate) == 0:
        return x

    noise_shape = _get_noise_shape(x, noise_shape)
    # Sample a uniform distribution on [0.0, 1.0) and select values larger than
    # rate.
    #
    # NOTE: Random uniform actually can only generate 2^23 floats on [1.0, 2.0)
    # and subtract 1.0.
    random_tensor = random_ops.random_uniform(
        noise_shape, seed=seed, dtype=x.dtype)
    keep_prob = 1 - rate
    scale = 1 / keep_prob
    # NOTE: if (1.0 + rate) - 1 is equal to rate, then we want to consider that
    # float to be selected, hence we use a >= comparison.
    keep_mask = random_tensor >= rate
    ret = x * scale * math_ops.cast(keep_mask, x.dtype)
    if not context.executing_eagerly():
      ret.set_shape(x.get_shape())
    return ret
```
<https://github.com/tensorflow/tensorflow|tensorflow/tensorflow>tensorflow/tensorflow | Added by GitHub
