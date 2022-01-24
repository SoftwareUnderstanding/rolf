# SSD
Chúng ta đã tìm hiểu về một số mô hình phát hiện vật thể như họ R-CNN hay YOLO. Trong bài này chúng ta tìm hiểu thêm một mô hình phát hiện vật thể mới có tên là SSD.

Đầu vào của SSD là toạ độ bounding box của vật thể (các offsets) và nhãn của vật thể chứa trong đó (giống các mô hình phát hiện vật thể khác).

SSD (single shot multibox detector) chỉ cần 1 lần duy nhất đưa ảnh qua mạng NN để có thể phát hiện các vật thể trong ảnh trong khi đó region proposal network (RPN) dựa trên R-CNN cần đến 2 lần đưa ảnh qua mạng NN, một lần để tạo region proposals và một lần để phát hiện vật thể trong mỗi proposal. SSD nhanh hơn so với phương pháp dựa trên RPN.

Tại thời điểm dự đoán, mạng NN sẽ dự đoán confidence xuất hiện mỗi class trong mỗi **default box** và sự thay đổi của box (offsets so với default box) để khớp với hình dạng vật thể hơn. Mạng NN cũng dự đoán từ nhiều feature maps với độ phân giải khác nhau để xử lý object nhiều kích thước. SSD ban đầu được train trên Framework Caffe.

SSD300 đạt được 74.3% mAP với 59 FPS trên VOC2017, SSD500 đạt được 76.9% mAP với 22 FPS vượt trội hơn hẳn so với [Faster R-CNN (73.2% mAP với 7 FPS)](https://towardsdatascience.com/review-faster-r-cnn-object-detection-f5685cb30202) và [YOLOv1 (63.4% mAP với 45 FPS)](https://towardsdatascience.com/yolov1-you-only-look-once-object-detection-e1f3ffec8a89). Hiện này YOLO đã có đến phiên bản thứ 5 cải thiện hơn rất nhiều, chúng ta sẽ không bàn ở đây.

## 1. Multbox Detector
![](images/0.png)
*SSD: Multiple bounding boxes for localization (loc) and confidence (cof)*

SSD chỉ cần ảnh đầu vào và các ground-truth boxes cho mỗi object trong quá trình training.

* Sau khi đi qua một số lớp Convolution để trích xuất đặc trưng, chúng ta nhận được **feature layer kích thước mxn với p channels** (như vậy mỗi channel chúng ta có mxn vị trí (location)). Như ở trên chúng ta nhận được feature layer với kích thước 8x8 hoặc 4x4 (chưa tính channels). Convolutional layer 3x3 được áp dụng lên feature layers mxnxp này.
* Ứng với mỗi vị trí (location) chúng ta có **k bounding boxes**. k bounding boxes này có kích thước và tỉ lệ khác nhau (như trên hình). Ý tưởng chính là hình chữ nhậ đứng phù hợp với người, hình chữ nhật ngang thì phù hợp với ô tô.
* Ứng với mỗi bounding box chúng ta sẽ tính confidence cho tất cả classes **c class scores** $(c_{1}, c_{2},...,c_{n})$ (chú ý hơi khác với YOLO, YOLO dự đoán objectness confidence của box - có object hay không, và tiếp theo dự đoán class scores) và **4 offsets** tương đối so với default boxes.
* Trong quá trình training, đầu tiên chúng ta cần khớp default boxes với ground-truth boxes. Ví dụ ở trên chúng ta khớp 2 default boxes với mèo và một default box với chó, chúng được coi là các positive examples.
* Cuối cùng chúng ta sẽ có $ m\times n\times k\times (c+4) $ outputs
![3](images/3.png)
Đó cũng là lý do bài báo có tên là *SSD: Single Shot Multibox Detector*
Bây giờ người ta hay sử dụng **ResNet** làm base model hơn.

## 2. SSD Network architecture
![1](images/1.png)
*SSD(top) and YOLO(bottom)*

SSD được xây dựng dựa trên bas model VVG-16 có loại bỏ các lớp fully connected layers. Thay vì sử dụng các lớp FC như mạng VGG-16 ban đầu SSD sử dụng các lớp convolution phụ (bắt đầu từ lớp `Conv6`). **Việc này giúp model có thể trích xuất đặc trưng ở nhiều tỉ lệ khác nhau và giảm kích thước đầu vào ở các Conv layer tiếp theo.**
![2](images/2.png)
*Kiến trúc VGG-16*

Để có thể phát hiện chính xác hơn nhiều layers khác nhau của feature map đã được cho đi qua Conv layer `3x3` để phát hiện vật thể như hình bên trên.
* Mô hình SSD300 đầu vào có kích thước `300x300x3`
* Ví dụ như hình trên tại `Conv4_3` có kích thước `38x38x512` được áp lên 3x3 Conv layer để phát hiện vật thể trê feature map. Đầu ra từ `Conv4_3` để dự đoán là `38x38x4x(c+4)`. Giả sử chúng ta có 20 object classes cộng với background khi đó số outputs từ `Conv4_3` là `38x38x4x(21+4) = 144400`. Nếu chỉ tính số lượng bounding boxes chúng ta có `38x38x4=5776`. Chú ý ngoài áp Conv layer để nhận diện vật thể chúng ta còn áp Conv layer thích hợp để nhận được lớp tiếp theo. Điều này cũng tương tự đối với các lớp kế tiếp.
* Tương tự như vậy chúng ta có:
    * `Conv7`: `19x19x6=2166` boxes (6 boxes cho mỗi vị trí)
    * `Conv8_2`: `10x10x6=600` boxes (6 boxes cho mỗi vị trí)
    * `Conv9_2`: `5x5x6=150` boxes (6 boxes cho mỗi vị trí)
    * `Conv10_2`: `3x3x4=36` boxes (4 boxes cho mỗi vị trí)
    * `Conv11_2`: `1x1x4=4` boxes (4 boxes cho mỗi vị trí)

Chú ý số lượng bounding boxes ở mỗi vị trí cho các layers có thể khác nhau. Tổng cộng lại chúng ta có `5776 + 2166 + 600 + 150 + 36 + 4 = 8732` bounding boxes. Ở phiên bản **YOLOv1** có `7x7` vị trí với 2 bounding boxes cho mỗi vị trí, như vậy **YOLOv1** có tổng cộng 98 bounding boxes. Nên nhớ output của mỗi bounding box sẽ có dạng:
$$ y^{T} = [\underbrace{x, y, w, h}_{\text{bounding box}}, \underbrace{c_1, c_2,..., c_C}_{\text{scores of C classes}}] $$
* Conv layer dự đoán có kích thước lớn dùng để phát hiện vật thể có kích thước nhỏ và ngược lại.

## 3. Training
### 3.1. Loss function
**Matching strategy:** Trong suốt quá trinh training chúng ta cần xác định default boxes khớp với ground-truth. Đối với mỗi groud-truth box chúng ta chọn các default boxes có `jaccard overlap` hay IoU lớn hơn `0.5` (coi là positive examples). 

$ x_{ij}^{p} = \left\{1, 0 \right\}$ thể hiện matching (sự khớp) của **default box $i$ với ground-truth box $j$** của nhãn thứ $p$. $\sum_{i}x_{ij}^p \geq 1$ - trong quá trình mapping chúng ta có thể có nhiều default bounding box $i$ được map vào cùng 1 ground truth box $j$ với cùng 1 nhãn $p$.

Loss function bao gồm 2 thành phần: $ L_{loc} $ và $ L_{conf} $ (loss function của bài toán image classification chỉ có $L_{conf}$ thôi)
$$ L(x, c, l, g) = \frac{1}{N}[L_{conf}(x, c) + \alpha L_{loc}(x, l, g)] \tag{1} $$

trong đó $N$ là số lượng các default boxes matching với ground truth boxes. Nếu $N=0$ chúng ta set loss = 0.

#### 3.1.1. Localization loss
$$ L_{loc}(x, l ,g) = \sum_{i \in Pos}^{N}\sum_{m \in \{cx, cy, w, h\}} x^{k}_{ij} \space L_1^\text{smooth}(l_i^m - \hat{g}_j^m) $$

**Localization loss** là một hàm Smooth L1 đo lường sai số giữa tham số của **box dự đoán (predicted box) ($ l $) và ground truth box ($ g $).**
Các tham số này bao gồm offsets cho tâm $(cx, cy)$ của default bounding box, chiều dài ($h$) và chiều rộng ($w$). Loss này cũng tương tự với loss của Faster R-CNN.

Localization loss chỉ xét cho các positive matching example ($i \in Pos$) giữa predicted box và ground-truth box. Thành phần $\sum_{m \in {x, y, w, h}} x^{k}_{ij} \space L_1^\text{smooth}(l_i^m - \hat{g}_j^m)$ chính là tổng khoảng cách giữa **predicted box ($l$)** và  ground-truth box ($g$) trên ở 4 offsets ($cx, cy, w, h$). $ cx, cy $ ở đây chính là tọa độ tâm. **$ d $ là kí kiệu cho default bounding box**.

$$ \hat{g}_j^{cx} = \frac{g_j^{cx}-d_{i}^{cx}}{d_{i}^{w}} \triangleq t_{x} $$

$$ \hat{g}_j^{cy} = \frac{g_j^{cy}-d_{i}^{cy}}{d_{i}^{h}} \triangleq t_{y} $$

$$ \hat{g}_j^{w} =log\frac{g_j^{w}}{d_i^{w}} \triangleq t_{w} $$

$$ \hat{g}_j^{h} =log\frac{g_j^{h}}{d_i^{h}} \triangleq t_{h} $$

Kí hiệu $\triangleq$ là đặt vế phải bằng vế phải. $ t_{x}, t_{y}, t_w, t_h $ nhận giá trị trong khoảng $ (-\infty, +\infty) $ và dùng để tinh chỉnh kích thước của bounding box. $ t_{x}, t_{y} $ càng lớn thì khoảng cách giữa tâm của **ground-truth $ g $ và default box $ d $** càng lớn. $t_w, t_h $ càng lớn thì chênh lệch giữa chiều dài và chiều rộng của ground-truth box và default box càng lớn. $ (t_x, t_y, t_w, t_h) $ là bộ tham số chuẩn hóa kích thước của ground-truth box $g$ theo kích thước của default box $d$. 

Tương tự như vậy chúng ta có thể xác định được bộ tham số thể hiện mối liên hệ giữa **predicted box $l$ và default box $d$** bằng cách thay $g$ bằng $l$ trong các phương trình trên. Khi đó khoảng cách giữa predicted box và ground truth box sẽ càng gần nếu khoảng cách giữa các bộ tham số chuẩn hóa giữa chúng càng gần, tức khoảng cách giữa 2 vector $g$ và $l$ càng nhỏ.

Nhắc lại một chút về hàm smooth $L_1^{smooth}$:
$$ L_1^\text{smooth}(x) = \begin{cases}
    0.5 x^2             & \text{if } \vert x \vert < 1\\
    \vert x \vert - 0.5 & \text{otherwise}
\end{cases} $$

Trường hợp $x$ là một véc tơ thì thay $\left| x\right|$ ở vế phải bằng giá trị norm chuẩn bậc 1 của $x$ kí hiệu là $\left| x\right|$.

Trong phương trình của hàm localization loss thì các hằng số mà ta đã biết chính là $g$. Biến cần tìm giá trị tối ưu chính là $l$. Sau khi tìm ra được nghiệm tối ưu của $l$ ta sẽ tính ra predicted box nhờ phép chuyển đổi từ default box tương ứng.

**Bổ sung:** làm rõ thêm về chuyển đổi kích thước.

Nếu để nguyên các giá trị tọa độ tâm và kích thước của khung hình sẽ rất khó để xác định sai số một cách chuẩn xác. Ta hãy so sánh sai số trong trường hợp khung hình lớn và khung hình bé. Trong trường hợp khung hình lớn có predicted box và ground truth box rất khớp nhau. Tuy nhiên do khung hình quá to nên khoảng cách tâm của chúng sẽ lớn một chút, giả định là aspect ratio của chúng bằng nhau. Còn trường hợp khung hình bé, sai số của tâm giữa predicted box và ground truth box có thể bé hơn trường hợp khung hình lớn về số tuyệt đối. Nhưng điều đó không có nghĩa rằng predicted box và ground truth box của khung hình bé là rất khớp nhau. Chúng có thể cách nhau rất xa.

Do đó chúng ta cần phải chuẩn hóa kích thước width, height và tâm sao cho không có sự khác biệt trong trường hợp khung hình bé và lớn. Một phép chuẩn hóa các offset được thực hiện như sau:
![5](images/5.png)

#### 3.1.2. Confidence loss

$$ L_{conf}(x, c) = -\sum_{i \in Pos} x_{ij}^{p} \text{log}(\hat{c}_{i}^p) - \sum_{i \in Neg}\text{log}(\hat{c}_{i}^0) $$

$ L_{conf} $ chính là softmax loss trên toàn bộ confidences của các classes ($c$).  
* Đối với mỗi **positive match prediction**, chúng ta phạt loss function theo confidence score của các nhãn tương ứng. Do positive match prediction nên vùng dự đoán có vật thể chính xác là chứa vật thể. Do đó việc dự đoán nhãn cũng tương tự như bài toán image classification với softmax loss $-\sum_{i \in Pos} x_{ij}^{p} \text{log}(\hat{c}_{i}^p)$. Nhớ lại $x_{ij}^{p} = \left\{1, 0 \right\}$ thể hiện matching default box $i$ với ground-truth box $j$ cho nhãn $p$, còn $(\hat{c}_{i}^p)$ chính là xác suất xuất hiện nhãn $p$ trong default box $i$. Điều này cũng tương tự như bài toán classification với nhiều nhãn với loss là $-\sum_{i}^{}y^{(i)}\ast log(\hat{y}^{i})$
* Đối với mỗi một **negative match prediction**, chúng ta phạt loss function theo confidence score của nhãn ‘0’ là nhãn đại diện cho background không chứa vật thể. Do không chứa vật thể nên chỉ có duy nhất background `0`, xác suất xảy ra background $ x_{ij}^{0} = 1$, do đó loss là $ -\sum_{i \in Neg}\text{log}(\hat{c}_{i}^0) $.

Ở đây $$\hat{c}_{i}^p = \frac{exp({c}_{i}^p)}{\sum_{p}^{}exp({c}_{i}^p)}$$


## 3.2. Lựa chọn kích cỡ (scale) và hệ số tỉ lệ (aspect ratio) cho box mặc định
**Scale:** độ phóng đại so với ảnh gốc. Nếu ảnh gốc có kích thước $(w, h)$, sau khi scale ảnh mới sẽ có kích thước là $(ws, hs)$. $s\in \left [ 0,1 \right ]$ là hệ số
scale. 

**Aspect ratio:** hệ số tỉ lệ hay tỉ lệ cạnh $\frac{w}{h}$ xác định hình dạng tương đối của khung hình chứa vật thể, người thường có aspect ration < 1, ô tô có aspect ration > 1.

Giả sử chúng ta có $m$ feature maps để dự đoán. Scale của default boxes cho mỗi feature map được tính như sau:
$$ s_k = s_{min} + \frac{s_{max} - s_{min}}{m-1}(k-1), k \in [1,m] $$

Trong đó $k$ là số thứ tự layer dùng để dự đoán do đó nó nằm từ 1 đến $m$, $s_{min} = 0.2$, $s_{max} = 0.9$.
* $k=1$ - tương đương với layer `Conv4_3` và $s_{1} = s_{min} = 0.2$. Điều này nghĩa là sao? Tại `Conv4_3` layer sẽ phát hiện object với scale nhỏ (bản thân `Conv4_3` layer là layer đầu tiên để dự đoán, có kích thước lớn nhất, chia làm nhiều cell nhất, do đó nó có khả năng phát hiện các vật thể nhỏ).
* $k=m$ - tương đương với layer `Conv11_2` và $s_{m} = s_{max} = 0.9.2$. Điều này nghĩa là sao? Tại `Conv11_2` layer sẽ phát hiện object với scale lớn (bản thân `Conv11_2` layer là layer cuối cùng để dự đoán, có kích thước nhot nhất, chia làm ít cell nhất, do đó nó có khả năng phát hiện các vật thể lớn).
Giả sử chúng ta có $m$ feature maps để dự đoán, chúng ta sẽ tính $s_{k}$ cho $k-th$ feature map.

Đối với layer có 6 dự đoán, chúng ta đặt các tỉ lệ (aspect ratios) khác nhau cho các default boxes và biểu diễn là $ a_{r}\in \left\{1, 2, 3, \frac{1}{2}, \frac{1}{3} \right\} $. Sau đó chúng ta có thể tính được height và width cho mỗi default box theo công thức sau:
$$w_{k}^a = s_{k} * \sqrt{{a_{r}}} $$

$$h_{k}^a = \frac{s_{k}} {\sqrt{{a_{r}}}} $$

Đối với trường hợp aspect ratio $ a_{r} = 1$ ta sẽ thêm một defaul box có scale $s_k' = \sqrt{s_ks_{k+1}}$ để tạo thành 6 default boxes cho mỗi vị trí của feature map.

## 3.3. Hard negative mining
Sau quá trình matching (khớp default boxes và groud-truth box) có rất nhiều negative example (các bounding box với IOU so với ground-truth box thấp). Thay vì sử dụng tất cả negative examples, chúng ta sắp xếp chúng dựa vào **highest confidence loss** (ít có khả năng chứa vật thể nhất) cho mỗi default box và lấy những cái đầu tiên sao cho tỉ lệ giữa negatives và positives tối đa là `3:1` (tránh mất cân bằng quá)
Điều có thể thể làm tăng quá trình tối ưu và ổn định hơn khi training.

## 3.4. Data augmentation
Để model mạnh mẽ với object nhiều kích thước, hình dạng mỗi training image được chọn ngẫu nhiên từ các lựa chọn sau:
* Sử dụng ảnh gốc
* Lấy một patch với minimum jaccard IoU với các vật thể: 0.1, 0.3, 0.5, 0.7, 0.9
* Lấy ngẫu nhiên một patch

Kích thucows của mỗi patch là `[0.1, 1]` so với kích thước ảnh gốc và có aspect ratio nằm giữa 0.2 và 2.

## 6. Inferences using SSD
SSD sử dụng các default boxes với tỉ lệ, hình dạng khác nhau trên các layers. SSD loại bỏ các predictions có confidence nhỏ hớn 0.01. Sau đó cũng áp dụng NMS các overlap 0.45 và giữ 200 detections cho mỗi image.


![0](images/0.jpeg)
## 7. Kết quả
* SSD300: `300x300` ảnh đầu vào, nhanh hơn
* SSD512: `512x512` ảnh đầu vào, độ phân giải cao hơn. chính xác hơn.

# Tài liệu tham khảo
1. https://arxiv.org/abs/1512.02325
2. https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11
3. https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab
4. https://jonathan-hui.medium.com/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06
5. https://medium.com/featurepreneur/object-detection-using-single-shot-multibox-detection-ssd-and-opencvs-deep-neural-network-dnn-d983e9d52652
6. https://github.com/weiliu89/caffe/tree/ssd#models Github của tác giả
7. https://github.com/pierluigiferrari/ssd_keras
8. https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html 
9. https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html#ssd-single-shot-multibox-detector Blog này rất hay