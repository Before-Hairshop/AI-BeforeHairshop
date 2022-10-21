# AI-Beforehairshop

> [BeforeHairshop AI 모델(Barbershop++)](https://github.com/Before-Hairshop/Barbershop-Plus-Plus)을 이용해서 AI 모델을 사용자들에게 서빙한다. AI 모델에 대한 설명은 Barbershop++ 논문 혹은 레포지토리를 통해 확인하길 바랍니다.


### Prequsite
#### 라이브러리
- ninja 설치
``` shell
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
```

- dlib 다운로드
``` shell
pip install cmake
pip install dlib
```


- pretrained pSp model 다운로드
``` python
CODE_DIR = 'pixel2style2pixel'

def get_download_model_command(file_id, file_name):
    """ Get wget download command for downloading the desired model and save to directory ../pretrained_models. """
    current_directory = os.getcwd()
    save_path = os.path.join(os.path.dirname(current_directory), CODE_DIR, "pretrained_models")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url

MODEL_PATHS = {
    "ffhq_encode": {"id": "1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0", "name": "psp_ffhq_encode.pt"},
    "ffhq_frontalize": {"id": "1_S4THAzXb-97DbpXmanjHtXRyKxqjARv", "name": "psp_ffhq_frontalization.pt"},
    "celebs_sketch_to_face": {"id": "1lB7wk7MwtdxL-LL4Z_T76DuCfk00aSXA", "name": "psp_celebs_sketch_to_face.pt"},
    "celebs_seg_to_face": {"id": "1VpEKc6E6yG3xhYuZ0cq8D2_1CbT0Dstz", "name": "psp_celebs_seg_to_face.pt"},
    "celebs_super_resolution": {"id": "1ZpmSXBpJ9pFEov6-jjQstAlfYbkebECu", "name": "psp_celebs_super_resolution.pt"},
    "toonify": {"id": "1YKoiVuFaqdvzDP5CZaqa3k5phL-VDmyz", "name": "psp_ffhq_toonify.pt"}
}

path = MODEL_PATHS[experiment_type]
download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])

# 마지막으로는 download_command 터미널창에서 실행시키기
# !{download_command}
```

- shape_predictor_68_face_landmarks.dat 라이브러리 다운로드
``` shell
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
```

#### 코드 수정
- 가상 헤어스타일링을 할 이미지는 W+, FS space 상의 Latent vector를 미리 얻어서, 시간을 단축시키도록 한다.