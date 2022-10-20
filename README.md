# AI-Beforehairshop

> [BeforeHairshop AI 모델(Barbershop++)](https://github.com/Before-Hairshop/Barbershop-Plus-Plus)을 이용해서 AI 모델을 사용자들에게 서빙한다. AI 모델에 대한 설명은 Barbershop++ 논문 혹은 레포지토리를 통해 확인하길 바랍니다.


### Prequsite
#### 라이브러리
- ninja 설치
``` shell
!wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
!sudo unzip ninja-linux.zip -d /usr/local/bin/
!sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
```

- pretrained pSp model 다운로드
``` shell

```

#### 코드 수정
- 가상 헤어스타일링을 할 이미지는 W+, FS space 상의 Latent vector를 미리 얻어서, 시간을 단축시키도록 한다.