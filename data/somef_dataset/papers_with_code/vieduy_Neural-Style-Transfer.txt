
<img src='result image/perceptual loss result/2_UIT_stars.jpg'> 

##### Môn: CS331.L21

# ĐỒ ÁN MÔN HỌC: THỊ GIÁC MÁY TÍNH NÂNG CAO

### Thành viên
1. Hoàng Viễn Duy
2. Trần Hoàng Việt
3. Nguyễn Anh Khoa

**Trong thư mục drive này sẽ chứa những file như sau**:

 1. Slide báo cáo
 2. Mã nguồn thực thi các mô hình nhóm đã trình bày trong slide
 3. Web Demo để người dùng có thể tương tác trực tiếp
 4. Ảnh kết quả khi chạy những mô hình trên
 5. Bộ ảnh content và style mà nhóm đã sử dụng

Nhóm sẽ hướng dẫn chi tiết cách chạy từng mô hình.
## Baseline
**Mã nguồn của mô hình này được viết trên TensorFlow**.

 1. **Traning:** `python neural_style.py --content <content file> --styles <style file> --output <output file>` 
 2. **Chạy** `python neural_style.py --help` để có thể xem hết phần tham số.
 
Phương pháp baseline này chỉ training một ảnh content với một ảnh style tương ứng.
### **Một số kết quả thực nghiệm**
<img src='result image/baseline result/TF-HUB/chicago.starrynew.png' width="200" height="200"> <img src='result image/baseline result/TF-HUB/phongcanh.stars.png' width="200" height="200"> <img src='result image/baseline result/TF-HUB/logouit.ocean.png' width="200" height="200"> <img src='result image/baseline result/TF-HUB/brad_tit.stars.png' width="200" height="200"> 


## Fast Neural Style 
**Trong thư mục perceptual_loss bao gồm những file sau**:
 - **Content_image** : folder chứa ảnh content
 - **Models : folder chứa models**
 - **Notebook**: folder chứa các notebook để train
 - **Result**: folder chứa ảnh kết quả
 - **Stylize.py**: file test
 - **Train.py**: file training
 
 **Biến đổi ảnh hoặc folder chứa các ảnh:** `python stylize.py` 
 **Training:** 
 - Open file jupyter notebook trong folder notebook với colab
 - Thay các đường dẫn tới dataset, style image và save image vào các biến DATASET_PATH, STYLE_IMAGE_PATH, SAVE_IMAGE_PATH
 - Run các cell trong notebook
 - Quá trình training xấp xỉ 2 tiếng/style với GPU colab và 4 tiếng /style với CPU colab
 - Một số model đã train lưu trong folder models
 
 ### **Một số hình ảnh sau khi chạy thực nghiệm**
 <img src='result image/perceptual loss result/2_phongcanh_stars.jpg' width="200" height="200"> <img src='result image/perceptual loss result/3_logo_uit.jpg' width="200" height="200"> <img src='result image/perceptual loss result/1_phongcanh.jpg' width="200" height="200"> <img src='result image/perceptual loss result/chicago.jpg' width="200" height="200"> 

## WebDemo
**WebDemo được xây dựng dựa trên framework Flask.** 
**Hướng dẫn sử dụng:** 

 1. `pip install -r requirements.txt.` để cài đặt các packages cần thiết
 2. Truy cập `https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/tutorials/lap_style.md` để tải 4 file pretrained của mô hình và lưu trong thư mục `WebDemo/PaddleGAN`
 3. Chạy `python app.py` và truy cập `http://127.0.0.1:8000/` để sử dụng ứng dụng

### **Một số hình ảnh khi chạy thành công ứng dụng**
<img src='result image/perceptual loss result/web_1.png'> <img src='result image/perceptual loss result/web_2.png'> 

## So sánh kết quả giữa các mô hình
<img src='result image/perceptual loss result/Screenshot from 2021-07-07 15-58-10.png'> 

# Tài liệu tham khảo

1. A neural algorithm of artistic style (https://arxiv.org/abs/1508.06576)
2. Perceptual Losses for Real-Time Style Transfer and Super-Resolution (https://arxiv.org/abs/1603.08155)
3. Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality Artistic Style Transfer (https://arxiv.org/pdf/2104.05376.pdf)
4. https://viblo.asia/p/bien-anh-cua-ban-thanh-cac-tac-pham-hoi-hoa-voi-neural-style-transfer-WAyK8dXoKxX
5. https://towardsdatascience.com/perceptual-losses-for-real-time-style-transfer-and-super-resolution-637b5d93fa6d
6. https://github.com/rrmina/fast-neural-style-pytorch
7. https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/tutorials/lap_style.md
8. https://www.youtube.com/watch?v=VQEMptfWpLk
