# Style-Transfer
Style Transfer project


This repository implements a modified version of the method presented in the paper "In the Light of Feature Distributions: Moment Matching for Neural Style Transfer", where we replace the VGG network used for style extraction with a ResNet18 model. The ResNet18 was trained to classify 13 different art styles using the WikiArt dataset, allowing us to generate stylized images guided by diverse artistic styles.

Overview
Original Paper
The method is based on the paper "In the Light of Feature Distributions: Moment Matching for Neural Style Transfer", which uses a VGG network to extract and manipulate content and style features.

Our Contribution
In this project, we adapt the original method by:

Training ResNet18 on the WikiArt Dataset:
The ResNet18 model was fine-tuned on 13 distinct artistic styles from the WikiArt dataset, providing a robust style classification framework.
Replacing VGG with ResNet18:
The ResNet18 model was used to extract style features, enabling the style transfer process.
By making these changes, we aim to explore the effects of using a different backbone for style extraction and classification.

Features
Custom Style Classification: Trained ResNet18 on 13 unique styles from WikiArt.
Style Transfer: Applies the learned style features to transfer the artistic essence of one image to another.
Flexible Framework: Easily extensible to include additional styles or adjust the training pipeline.

![image](https://github.com/user-attachments/assets/3e4fabe1-5c99-4fc2-8ac2-ffd5768680ae)

Example Output
Here is an example of a content image stylized using ResNet18:

![image](https://github.com/user-attachments/assets/e7a90556-dd10-468e-9e81-fabf5d1ace11)

![image](https://github.com/user-attachments/assets/90de0d22-d6c3-4332-bb77-267c453906b9)

![image](https://github.com/user-attachments/assets/8c0c891b-d4a1-48f3-9de9-948a4f842e3c)

Dataset
The WikiArt dataset was used for training the style classification model. This dataset contains a diverse set of artworks across various styles, enabling effective learning of distinct artistic characteristics.

References
Original Paper: "In the Light of Feature Distributions: Moment Matching for Neural Style Transfer"
WikiArt Dataset: huggingface.co/datasets/huggan/wikiart






