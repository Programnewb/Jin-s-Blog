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
 2. 데이터 셋 동적 다운로드 수행
    데이터 셋 다운로드를 수행한다. PC 성능에 따라 차이가 있지만 꽤나 많은 시간이 소요된다.
    <script src="https://gist.github.com/Programnewb/f9fa7b1d821336f2490c2c656009a384.js"></script>
 3. 다운로드 경로 설정
    다운로드 경로를 설정 해준다.
    <script src="https://gist.github.com/Programnewb/971648a1ddb7449feb6c40ac6032119d.js"></script>
 4. Image 랜덤 시드 설정
    경로 설정이 완료되었으면 다운로드한 Image에서 추출해올 이미지를 랜덤으로 정한다. 랜덤 이미지를 설정하면서, Train Data와 Validation Data의 갯수로 설정한다. 여기서는 4:1로 설정하
    였다.
    <script src="https://gist.github.com/Programnewb/c7843007eea89c75a0d2bda46ee8efb9.js"></script>
    Random Image가 잘 불러 와졌는지 확인
    <script src="https://gist.github.com/Programnewb/eeb8d07d501b41f176c3b637129ee432.js"></script>
 5. Data Loader 생성
    <script src="https://gist.github.com/Programnewb/5ed5df9723b508a209d43eb365ee6ca4.js"></script>
    Data 전처리
    <script src="https://gist.github.com/Programnewb/7ef6c710b5afb44fe1ff48bec368fb51.js"></script>
 6. Data Loader 사용
    <script src="https://gist.github.com/Programnewb/aaf3757ae20cd44c258cc37342b9f861.js"></script>
 7. SRCNN
    <script src="https://gist.github.com/Programnewb/cb532949f9b9acc9a84f4d4a7f0c50b9.js"></script>
