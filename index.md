---
title: Home
layout: Home
---

Pix2Pix
===
---
개요
---
본 블로그는 Project를 수행함에 있어 그 과정을 기록하고, 자료로 남기기 위하여 작성한다. 해당 블로그는 Image to Image, GAN(Generative Adversarial Network) 알고리즘을 학습하고, 이를 구현하는 것을 목적으로 한다. 사용 환경은 Google COLAB을 사용하였고, 기본 사용언어는 Python을 사용하였다.

---
Project
---
GAN(Generative Adversarial Network)
GAN은 대표적인 비지도학습(unsupervised learning)의 한 종류로 서로 대립하는 두 시스템이 서로 경쟁하는 방식으로 학습이 진행한다. Generator와 Discriminator 간의 유사도를 비교하며 학습하는 알고리즘이다. Image 데이터를 Train Data와 Validation Data로 나누어 학습 시킨다음 노이즈가 포함되어 있는 이미지를 Generator에 보내면 실제 Image와 비슷한 Image를 생성해낸다. 그럼 Discriminator는 실제 Image와 Generator가 생성해낸 가짜 Image를 비교하여 정확도를 높여간다.

Pooling
 : Convolution을 거쳐서 나온 activation maps이 있을때 이를 이루는 convolution lyater을 resizing하여 새로운 layer를 얻는 것
 
  ![image](https://user-images.githubusercontent.com/117645947/208297506-80651f59-88b0-4bf6-b121-974023084cef.png)

Max pooling
 : Pooling에서 최댓값을 뽑아내는것
 
  ![image](https://user-images.githubusercontent.com/117645947/208297647-f80e1916-ef5a-4bac-8423-1426b32fc3af.png)
Max pooling 목적
 : Overfitting을 방지하기 위함
   예제)
   *feature = elements of input data

   *parameter = elements of weight matrix

   size가 96x96인 image가 주어져 있고(즉, feature의 수는 96x96개), 이를 400개의 filter로 ***Convolution**한 size 8x8x400의
   convolution layers가 있습니다.  각 convolution layer에는 (stride를 1이라 가정하면) (96–8+1)x(96–8+1)=89x89=7921개의 
   featur가 있고, 이런 layer가 400개이니 총 feature의 수는 89x89x400=3,168,400 이렇게 feature가 많아지면 **overfitting**의 우
   려가 있음

   Pooling 사용 시
    ![image](https://user-images.githubusercontent.com/117645947/208297795-2925a05b-053c-41f6-a0bf-020f7cd3df7b.png)

 1. 환경 Set up
    Google colab을 사용하면서 GPU를 할당해야 한다. GPU 할당은 아래와 같이 수행했다.
    ```python
    # GPU 할당되었는지 확인 코드

    from psutil import virtual_memory
    import torch

    # colab-GPU 사용 확인
    gpu_info = !nvidia-smi
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
      print('GPU 연결 실패!')
    else:
      print(gpu_info)

    # RAM 사용량 체크
    ram_gb = virtual_memory().total / 1e9
    print('{:.1f} gigabytes of available RAM\n'.format(ram_gb))

    # pytorch-GPU 연결 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('학습을 진행하는 기기:',device)

    # 구글 드라이브 연결. 만약 직접 데이터셋을 사용한다면 주석 해제.
    # from google.colab import drive
    # drive.mount('/content/drive')
    ```
    
 2. 데이터 셋 동적 다운로드 수행
    데이터 셋 다운로드를 수행한다. PC 성능에 따라 차이가 있지만 꽤나 많은 시간이 소요된다.
    ```python
    ## 데이터 셋 동적으로 다운로드
    ## !는 명령어

    !pip install fastai==2.4
    ```
    
 3. 다운로드 경로 설정
    다운로드 경로를 설정 해준다.
    ```python
    from fastai.data.external import untar_data, URLs
    import glob

    coco_path = untar_data(URLs.COCO_SAMPLE)
    paths = glob.glob(str(coco_path) + "/train_sample/*.jpg")
    # train_sample 경로 상의 모든 jpg 파일을 다운 받는다
    # *.jpg -> jpg의 모든 항목 *은 모든이라는 의미
    ```
 4. Image 랜덤 시드 설정
    경로 설정이 완료되었으면 다운로드한 Image에서 추출해올 이미지를 랜덤으로 정한다. 랜덤 이미지를 설정하면서, Train Data와    
    Validation Data의 갯수로 설정한다. 여기서는 4:1로 설정하였다.
    ```python
    import numpy as np

    np.random.seed(1) # 랜덤 시드 정하기
    chosen_paths = np.random.choice(paths, 5000, replace=False)
    # 5000장의 이미지를 랜덤으로 저장하되 중복된 이미지는 안들고 온다
    index = np.random.permutation(5000)

    train_paths = chosen_paths[index[:4000]] # 앞의 4000장을 train 이미지로 사용
    val_paths = chosen_paths[index[4000:]]

    print(len(train_paths))
    print(len(val_paths))
    ```
    Random Image가 잘 불러 와졌는지 확인
    ```python
    import matplotlib
    import matplotlib.pyplot as plt

    sample = matplotlib.image.imread(train_paths[3])
    plt.imshow(sample)
    plt.axis("off")
    plt.show()
    ```
    
 5. Data Loader 생성
    ```python
    ### 기본적인 dataloader 만드는 법

    import torch
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    # 이걸 꼭해야 데이터 로더 쓸수 잇다

    ### 데이터셋 클래스 선언

    class myDataset(Dataset):

      # 생성자 만들기
      def __init__(self, x, y):
        self.x = x
        self.y = y

      # 굉장히 중요
      def __getitem__(self, index):
        return self.x[index], self.y[index]

      # 전체 길이 함수
      def __len__(self):
        return self.x.shape[0]

    x = np.random.randint(0, 100, 5) # x에 랜덤 숫자 생성
    y = np.random.randint(0, 100, 5) # y에 랜덤 숫자 생성
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    dataset = myDataset(x,y)
    dataloader = DataLoader(dataset, batch_size=3, num_workers=2, pin_memory=True)
    # DataLoader 돌려보기
    x, y = next(iter(dataloader)) # 하나씩 뽑아 보겠다 => 돌면서(다음것(dataloader))
    ```
    
    Data 전처리
    ```python
    # Data 전처리
    from torch.utils.data import Dataset, Dataloader
    from torchvision import transforms
    from PIL import Image
    from skimage.color import rgb2lab, lab2rgb
    import numpy as np

    class ColorizationDataset(Dataset):
      def __init__(self, paths, mode='train'):
        self.mode = mode
        self.paths = paths

        if mode == 'train':
          self.transforms = transforms.Compose([
              transforms.Resize((256,256), Image.BICUBIC),
              # 이미지 사이즈 조정
              # Image.BICUBIC 이라는 알고리즘 사용하여 Resize 진행
              transforms.RandomHorizontalFlip()
          ])
        elif mode == 'val':
          self.transforms = transforms.Resize((256,256), Image.BICUBIC)
        else:
          raise Exception("train or validation only!!!!")

      def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        # image를 self의 경로에 있는 것을 불러온다 (**회색 이미지가 있으니까 "RGB"로 변환!)
        img = np.array(self.transforms(img))
        # 이미지 변환 (배열형태)
        img = rgb2lab(img).astype("float32") # RGB 채널을 LAB 채널로 변환해 주는 것!
        img = transforms.ToTensor()(img) # Tensor 형태로 이미지 변환
        L = img[[0], ...] /50. -1  # -1 에서 1 사이로 정규화를 진행
        ab = img[[1,2], ...] /110. # -1 에서 1 사이로 정규화를 진행

        return {'L': L, 'ab':ab}

      def __len__(self):
        return len(self.paths)
    ```
    
 6. Data Loader 사용
    ```python
    # Data loader

    dataset_train = ColorizationDataset(train_paths, mode='train')
    dataset_val = ColorizationDataset(val_paths, mode='val')

    dataloader_train = DataLoader(dataset_train, batch_size=16, num_workers=2, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=16, num_workers=2, pin_memory=True)
    ```
 7. SRCNN
    ```python
    #SRCNN 구성
    import torch.nn as nn

    class SRCNN(nn.Module):
      def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5//2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5//2)
        self.relu = nn.ReLu(inplace=True)  

      def forward(self, x):
        x = self.conv1(x)
        x = self.relu()
        x = self.conv2(x)
        x = self.relu()
        x = self.conv3(x)

        return x
    ```
  8. Generator
     ```python
     import torch.nn as nn

     class generator(nn.Module):
       def __init__(self):
         super(generator, self).__init__()

         self.input_layer = nn.Sequential(
             nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
         )

         self.encoder_1 = nn.Sequential(
             nn.LeakyReLU(0.2, True),
             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
             nn.BatchNorm2d(128)
         )

         self.encoder_2 = nn.Sequential(
             nn.LeakyReLU(0.2, True),
             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
             nn.BatchNorm2d(256)
         )

         self.encoder_3 = nn.Sequential(
             nn.LeakyReLU(0.2, True),
             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True),
             nn.BatchNorm2d(512)
         )

         self.encoder_4 = nn.Sequential(
             nn.LeakyReLU(0.2, True),
             nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True),
             nn.BatchNorm2d(512)
         )

         self.encoder_5 = nn.Sequential(
             nn.LeakyReLU(0.2, True),
             nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True),
             nn.BatchNorm2d(512)
         )

         self.encoder_6 = nn.Sequential(
             nn.LeakyReLU(0.2, True),
             nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True),
             nn.BatchNorm2d(512)
         )

         self.middle = nn.Sequential(
             nn.LeakyReLU(0.2, True),
             nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
             nn.ReLU(True),
             nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(512)
         )

         self.decoder_6 = nn.Sequential(
             nn.ReLU(True),
             # middle + encoder6의 값이 합쳐져 1024
             nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(512),
             nn.Dropout(0.5)
         )

         self.decoder_5 = nn.Sequential(
             nn.ReLU(True),
             nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(512),
             nn.Dropout(0.5)
         )

         self.decoder_4 = nn.Sequential(
             nn.ReLU(True),
             nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(512),
             nn.Dropout(0.5)
         )

         self.decoder_3 = nn.Sequential(
             nn.ReLU(True),
             nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(256),
             nn.Dropout(0.5)
         )

         self.decoder_2 = nn.Sequential(
             nn.ReLU(True),
             nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(128),
             nn.Dropout(0.5)
         )

         self.decoder_1 = nn.Sequential(
             nn.ReLU(True),
             nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(64),
             nn.Dropout(0.5)
         )

         self.output_layer = nn.Sequential(
             nn.ReLU(True),
             nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
             nn.Tanh()
         )



       def forward(self, x):
         input_layer = self.input_layer(x)

         encoder_1 = self.encoder_1(input_layer)
         encoder_2 = self.encoder_2(encoder_1)
         encoder_3 = self.encoder_3(encoder_2)
         encoder_4 = self.encoder_4(encoder_3)
         encoder_5 = self.encoder_5(encoder_4)
         encoder_6 = self.encoder_6(encoder_5)

         middle = self.middle(encoder_6)

         cat_6 = torch.cat((middle, encoder_6), dim=1)
         decoder_6 = self.decoder_6(cat_6)
         cat_5 = torch.cat((decoder_6, encoder_5), dim=1)
         decoder_5 = self.decoder_5(cat_5)
         cat_4 = torch.cat((decoder_5, encoder_4), dim=1)
         decoder_4 = self.decoder_4(cat_4)
         cat_3 = torch.cat((decoder_4, encoder_3), dim=1)
         decoder_3 = self.decoder_3(cat_3)
         cat_2 = torch.cat((decoder_3,encoder_2), dim=1)
         decoder_2 = self.decoder_2(cat_2)
         cat_1 = torch.cat((decoder_2, encoder_1), dim=1)
         decoder_1 = self.decoder_1(cat_1)

         output = self.output_layer(decoder_1)

         return output
     ```
