---
Title
Pix2Pix
---
---
개요
---
본 블로그는 Project를 수행함에 있어 그 과정을 기록하고, 자료로 남기기 위하여 작성한다. 해당 블로그는 Image to Image, GAN(Generative Adversarial Network) 알고리즘을 학습하고, 이를 구현하는 것을 목적으로 한다. 사용 환경은 Google COLAB을 사용하였고, 기본 사용언어는 Python을 사용하였다.

---
Project
---
GAN(Generative Adversarial Network)
GAN은 대표적인 비지도학습(unsupervised learning)의 한 종류로 서로 대립하는 두 시스템이 서로 경쟁하는 방식으로 학습이 진행한다. Generator와 Discriminator 간의 유사도를 비교하며 학습하는 알고리즘이다. Image 데이터를 Train Data와 Validation Data로 나누어 학습 시킨다음 노이즈가 포함되어 있는 이미지를 Generator에 보내면 실제 Image와 비슷한 Image를 생성해낸다. 그럼 Discriminator는 실제 Image와 Generator가 생성해낸 가짜 Image를 비교하여 정확도를 높여간다.

 1. 환경 Set up
    Google colab을 사용하면서 GPU를 할당해야 한다. GPU 할당은 아래와 같이 수행했다
    <script src="https://gist.github.com/Programnewb/6a2faa8675e156471e5aead9ac52c2a5.js"></script>
