from genericpath import isfile
import subprocess
import boto3
import json
import os
from PIL import Image, ExifTags
from secret import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, AWS_REQUEST_SQS_NAME, AWS_RESPONSE_SQS_NAME, AWS_SQS_REGION
from secret import AWS_RESPONSE_SQS_URL, AWS_REQUEST_SQS_URL, AWS_S3_BUCKET_REGION, AWS_S3_BUCKET_NAME, RAW_USER_INPUT_IMAGE_PATH, PREPROCESSING_USER_INPUT_IMAGE_PATH
import logging
from botocore.exceptions import ClientError


def get_request_queue():
    aws_session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_SQS_REGION)
    sqs = aws_session.resource('sqs')

    queue = sqs.get_queue_by_name(QueueName=AWS_REQUEST_SQS_NAME)
    return queue


def get_response_queue():
    aws_session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_SQS_REGION)
    sqs = aws_session.resource('sqs')

    queue = sqs.get_queue_by_name(QueueName=AWS_RESPONSE_SQS_NAME)
    return queue

# Getting Response Queue from SQS
response_queue = get_response_queue()

# Getting Request Queue from SQS
request_queue = get_request_queue()

logger = logging.getLogger(__name__)

# Getting S3 Bucket
session = boto3.Session(
    aws_access_key_id = AWS_ACCESS_KEY,
    aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
    region_name = AWS_S3_BUCKET_REGION
)
s3_resource = session.resource('s3')

def download_image_from_s3(user_id):
    # S3로부터 이미지 다운로드 받는다.
    s3_resource.meta.client.download_file(Bucket=AWS_S3_BUCKET_NAME, Key='/' + str(user_id) + '/' + 'profile.jpg', Filename=RAW_USER_INPUT_IMAGE_PATH)





def main():

    ### ========= Flow ========= (while True 문으로 돈다)
    # 1. SQS로부터 메시지 수신
    # 2. AI inference 진행
    #   2.1 face_alignment 실행 
    #   2.2 pSp 인코딩
    #   2.3 Barbershop 추론
    # 3. SQS로 메시지 송신

    while True:
        # ============= [Step 0] 이전 결과들 지우기 =============
        # unprocessed 폴더, input/face 폴더, output/W+ & output/FS 폴더 내 생성된 파일 삭제 

        # ============= [Step 1] 메시지 수신 =============
        messages = response_queue.meta.client.receive_message(
            QueueUrl=AWS_REQUEST_SQS_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=2,
            MessageAttributeNames=['All']
        )

        if 'Messages' not in messages:
            continue

        for message in messages['Messages']:
            data = message['Body']
            data = json.loads(data)

            logger.info("Message from Request Queue : ", data)

            request_queue.meta.client.delete_message(
                QueueUrl=AWS_REQUEST_SQS_URL,
                ReceiptHandle=message['ReceiptHandle']
            )

            # AWS S3로부터 이미지 다운로드
            download_image_from_s3(param_user_id)

            # =================== [Step 2] Barbershop++ 추론 ===================
            # ===== [Step 2.1] face_alignment 실행 - Barbershop.align_face.py (만약, 얼굴이 인식되지 않으면, 얼굴 인식되지 않았다는 에러 메시지 SQS로 송신한다) =====

            image_path = RAW_USER_INPUT_IMAGE_PATH # 알맞게 이미지 path 수정할 것

            ## 먼저, face_alignment 시켜줄 수 있도록, 제대로된 방향으로 rotate 시켜준다.
            try:
                image = Image.open(image_path)
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation]=='Orientation':
                        break
                exif=dict(image._getexif().items())

                if exif[orientation] == 3:
                    image=image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image=image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image=image.rotate(90, expand=True)
                    
                image.save(PREPROCESSING_USER_INPUT_IMAGE_PATH)

            except (AttributeError, KeyError, IndexError):
                logger.exception("image preprocessing(rotate) fail!! (Message body : { result : %s, user_id : %s , image_id : %s})", "이미지 회전 실패", param_user_id, image_id)
                pass
            
            # 전처리(rotate) 되지 않은 이미지를 삭제한다.
            delete_raw_input_image_cmd = 'rm  ' + RAW_USER_INPUT_IMAGE_PATH
            subprocess.call(delete_raw_input_image_cmd, shell=True)

            # align face 진행
            os.chdir("/home/ubuntu/Barbershop-Plus-Plus/Barbershop")
            subprocess.call("/home/ubuntu/Barbershop-Plus-Plus/Barbershop/align_face.py", shell=True)
            
            # align 안되어있으면, 실패했다고 리턴함. (폴더에 한 개가 아닌 0개 혹은 2개 생성되면 실패) - reference image는 5개
            if len(os.listdir("/content/drive/MyDrive/Barbershop-Plus-Plus/Barbershop/input/face")) != 6:
                fail_body_json = {
                    'result' : 'fail',
                    'user_id' : param_user_id,
                    'image_id' : image_id
                }

                fail_message_body_str = json.dumps(fail_body_json)
                try:
                    # Send message to Request Queue
                    response_queue.send_message(MessageBody=fail_message_body_str, QueueUrl=AWS_RESPONSE_SQS_URL)
                    logger.info("Send fail message success! (Message body : { result : %s, user_id : %s , image_id : %s})", "fail", param_user_id, image_id)
                    continue
                except ClientError as error:
                    logger.exception("Send fail message failed! (Message body : { result : %s, user_id : %s , image_id : %s})", "fail", param_user_id, image_id)
                    continue
            


            # ===== [Step 2.2] pSp 인코딩 =====
            os.chdir("/home/ubuntu/Barbershop-Plus-Plus/pixel2style2pixel/")
            subprocess.call("/home/ubuntu/Barbershop-Plus-Plus/psp_encoding.py", shell=True)



            # ===== [Step 2.3] Barbershop 추론 =====
            os.chdir("/home/ubuntu/Barbershop-Plus-Plus/Barbershop/")





    


if __name__ == "__main__":
	main()