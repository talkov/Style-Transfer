# 🎨 Style-Transfer: Neural Style Transfer with ResNet18

This repository implements a **modified version** of the neural style transfer method presented in the paper [*In the Light of Feature Distributions: Moment Matching for Neural Style Transfer*](https://arxiv.org/abs/XXXX), replacing the VGG network with a **ResNet18** model trained on the **WikiArt** dataset. Our approach enables the generation of stylized images guided by diverse artistic styles.

---

## 📖 Overview

### 🔍 Original Paper
The original method uses a **VGG network** to extract and manipulate content and style features for neural style transfer.

### 💡 Our Contribution
We adapt this approach by introducing the following modifications:
1. **Training ResNet18 on WikiArt**:  
   The ResNet18 model was fine-tuned on **13 distinct artistic styles** from the WikiArt dataset, creating a robust style classification framework.
2. **Replacing VGG with ResNet18**:  
   The **ResNet18 model** was utilized to extract style features, enabling the style transfer process.

These changes explore the impact of using a different backbone for style extraction and classification.

---

## 🖥️ System Architecture

The diagram below illustrates the system pipeline:

![image](https://github.com/user-attachments/assets/3e4fabe1-5c99-4fc2-8ac2-ffd5768680ae)

### Description:
1. **Input Images**:  
   - A content image (e.g., parrots) and a style image (e.g., *The Starry Night*) are provided as inputs.  
2. **Feature Extraction**:  
   - The **VGG network** (trained on ImageNet) and the **ResNet18 network** (trained on WikiArt) are used to extract features from selected convolutional layers.
3. **Style Transfer**:  
   - The **CMD Moment Matching Algorithm** combines the extracted features, enabling the transfer of style features onto the content image.
4. **Output Image**:  
   - The final result is a stylized image that combines the content of the input image with the artistic style of the style image.

This architecture highlights the use of **two neural networks** for feature extraction and the application of **moment matching** for transferring style.

---

## ✨ Features

- **Custom Style Classification**:  
  ResNet18 trained on 13 unique styles from the **WikiArt dataset**.
- **Neural Style Transfer**:  
  Applies learned style features to infuse the artistic essence of one image onto another.
- **Flexible Framework**:  
  Easily extensible for additional styles or adjustments to the training pipeline.

---

## 📊 Dataset

The **WikiArt dataset** was used to train the ResNet18 model. This dataset contains a diverse collection of artworks spanning various styles, enabling effective learning of distinct artistic characteristics.  
More details about the dataset: [WikiArt Dataset on HuggingFace](https://huggingface.co/datasets/huggan/wikiart).

---

## 🖼️ Example Output

Below are examples of content images stylized using our approach:


### Stylized Outputs
![image](https://github.com/user-attachments/assets/5fe9d6b9-3080-4af5-b084-af88ab5b7da2)

![image](https://github.com/user-attachments/assets/af7dc40c-6716-45ec-8c23-dbc04030eabd)

![image](https://github.com/user-attachments/assets/3d3ebb5c-9853-4204-86a0-8d93536a4b21)

---

## 📂 References

- **Original Paper**:  
  [In the Light of Feature Distributions: Moment Matching for Neural Style Transfer](https://arxiv.org/abs/XXXX)
- **WikiArt Dataset**:  
  [HuggingFace WikiArt Dataset](https://huggingface.co/datasets/huggan/wikiart)

---

Feel free to explore the repository and contribute! 🚀
