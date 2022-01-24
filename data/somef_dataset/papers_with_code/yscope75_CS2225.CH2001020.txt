<center><h1>Image Captioning with Resnet and Seq-2-Seq model</h1></center>

**Mục lục**

[Giới thiệu ](https://github.com/yscope75/CS2225.CH2001020 "introduction")
[Kiến trúc mô hình](https://github.com/yscope75/CS2225.CH2001020 "model-arch")
[Dữ liệu](https://github.com/yscope75/CS2225.CH2001020 "dataset")
[Training](https://github.com/yscope75/CS2225.CH2001020 "training")
[Kết quả và đánh giá mô hình](https://github.com/yscope75/CS2225.CH2001020 "result")
[Tài liệu tham khảo](https://github.com/yscope75/CS2225.CH2001020 "references")
##Giới thiệu 
Link github này là đồ án môn học nhận diện thị giác và ứng dụng (CS2225).
- Notebook của đồ án là: Image_captioning_Master_courses.ipynb
- Đồ án xây dựng mô hình gán chú thích cho ảnh sử dụng pretrained model từ Image Net, cụ thể là Resnet152.
- Được xây dựng trên framwork Pytorch. 

##Kiến trúc mô hình 
Encoder: sử dụng Resnet152 để trích xuất thông tin hình ảnh, chiều của đặc trưng đầu ra là 10x10x2048.
Attention: Mô hình sử dụng cơ chế Attention theo thiết kế của Bahdanau.
- Decoder: Sử dụng mô hình Seq-2-Seq  ([Sutskever at el.] [1])
Tại mỗi bước thời gian trong quá trình tạo text, Decoder xem sét hidden layer của bước thời gian trước đó kết hợp với input tại bước đang xét và vector trả về từ Attention trên Tensor đặt trưng của Encoder.
##Dữ liệu
Tập dữ liệu được lấy từ task Image Captioning của bộ dữ liệu COCO từ Microsoft.
Dữ liệu dùng để huấn luyện mô hình của đồ án chỉ sử dụng tập train của dữ liệu gốc, sau đó được phân chia thành 3 tập tương ứng là train, validation, test với tỉ lệ tương ứng là 60, 20, 20.
##Training
Do giới hạn thời gian và phần cứng nên mô hình của đồ án được huấn luyện với 5 lần lập sử dụng GPU Tesla V100 chạy trên Google Colab.
#Kết quả và đánh giá mô hình
Mô hình được đánh giá bới độ đo Perplexity dùng trong Language Modeling.
Kết quả training mô hình và giá trị loss như sau:

Tập  | Loss | PPL
------------- | ------------- | ------------- 
Train | 3.380 | 28.382 
Val | 4.966 | 143.466
Test | 4.491 | 139.846

##Tài liệu tham khảo 
[1]: https://arxiv.org/abs/1409.3215
