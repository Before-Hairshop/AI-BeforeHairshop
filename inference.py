from genericpath import isfile
import subprocess
import boto3
import json
import os
from PIL import Image, ExifTags
from secret import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, AWS_REQUEST_SQS_NAME, AWS_RESPONSE_SQS_NAME, AWS_SQS_REGION
from secret import AWS_RESPONSE_SQS_URL, AWS_REQUEST_SQS_URL, AWS_S3_BUCKET_REGION, AWS_S3_BUCKET_NAME, RAW_USER_INPUT_IMAGE_PATH, PREPROCESSING_USER_INPUT_IMAGE_PATH
import logging
import requests
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

### ===========
reference_cnt = 5
### ===========



def download_image_from_s3(member_id, virtual_member_image_id):
    # S3로부터 이미지 다운로드 받는다.
    s3_resource.meta.client.download_file(Bucket=AWS_S3_BUCKET_NAME, Key='ai_input/' + str(member_id) + '/' + str(virtual_member_image_id) + '.jpg', Filename=PREPROCESSING_USER_INPUT_IMAGE_PATH)

def create_presigned_post(object_name,
                          fields=None, conditions=None, expiration=3600):
    """Generate a presigned URL S3 POST request to upload a file

    :param bucket_name: string
    :param object_name: string
    :param fields: Dictionary of prefilled form fields
    :param conditions: List of conditions to include in the policy
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Dictionary with the following keys:
        url: URL to post to
        fields: Dictionary of form fields and values to submit with the POST
    :return: None if error.
    """

    # # Generate a presigned S3 POST URL
    # s3_client = boto3.client('s3')
    try:
        response = s3_resource.meta.client.generate_presigned_post(AWS_S3_BUCKET_NAME,
                                                     object_name,
                                                     Fields=fields,
                                                     Conditions=conditions,
                                                     ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL and required fields
    return response


def main():

    ### ========= Flow ========= (while True 문으로 돈다)
    # 1. SQS로부터 메시지 수신
    # 2. AI inference 진행
    #   2.1 face_alignment 실행 
    #   2.2 pSp 인코딩
    #   2.3 Barbershop 추론
    # 3. S3에 inference 된 결과 이미지 업로드
    # 4. SQS로 메시지 송신 

    while True:
        # ============= [Step 0] 이전 결과들 지우기 =============
        # unprocessed 폴더, input/face 폴더, output/W+ & output/FS 폴더 내 생성된 파일 삭제 

        # ============= [Step 1] 메시지 수신 =============
        messages = response_queue.meta.client.receive_message(
            QueueUrl=AWS_REQUEST_SQS_URL,
            MaxNumberOfMessages=2,
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

            param_member_id = data['memberId']
            param_virtual_member_image_id = data['virtualMemberImageId']

            # AWS S3로부터 유저 이미지 다운로드
            download_image_from_s3(param_member_id, param_virtual_member_image_id)

            # =================== [Step 2] Barbershop++ 추론 ===================
            # ===== [Step 2.1] face_alignment 실행 - Barbershop.align_face.py (만약, 얼굴이 인식되지 않으면, 얼굴 인식되지 않았다는 에러 메시지 SQS로 송신한다) =====

            image_path = PREPROCESSING_USER_INPUT_IMAGE_PATH # 알맞게 이미지 path 수정할 것
            
            # align face 진행
            os.chdir("/home/ubuntu/Barbershop-Plus-Plus/Barbershop")
            subprocess.call("/home/ubuntu/Barbershop-Plus-Plus/Barbershop/align_face.py", shell=True)

            ## 먼저, face_alignment 시켜줄 수 있도록, 제대로된 방향으로 rotate 시켜준다.
            try:
                if len(os.listdir("/home/ubuntu/Barbershop-Plus-Plus/Barbershop/input/face")) != 6:
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

                    # align face 진행
                    subprocess.call("/home/ubuntu/Barbershop-Plus-Plus/Barbershop/align_face.py", shell=True)

            except (AttributeError, KeyError, IndexError):
                logger.exception("image preprocessing(rotate) fail!! (Message body : { result : %s, memberId : %s , virtualMemberImageId : %s})", "이미지 회전 실패", param_member_id, param_virtual_member_image_id)
                pass
            
            # # 전처리(rotate) 되지 않은 이미지를 삭제한다.
            # delete_raw_input_image_cmd = 'rm  ' + RAW_USER_INPUT_IMAGE_PATH
            # subprocess.call(delete_raw_input_image_cmd, shell=True)

            # # align face 진행
            # os.chdir("/home/ubuntu/Barbershop-Plus-Plus/Barbershop")
            # subprocess.call("/home/ubuntu/Barbershop-Plus-Plus/Barbershop/align_face.py", shell=True)
            
            # align 안되어있으면, 실패했다고 리턴함. (폴더에 한 개가 아닌 0개 혹은 2개 생성되면 실패) - reference image는 5개
            if len(os.listdir("/home/ubuntu/Barbershop-Plus-Plus/Barbershop/input/face")) != 6:
                fail_body_json = {
                    'result' : 'fail',
                    'memberId' : param_member_id,
                    'virtualMemberImageId' : param_virtual_member_image_id
                }

                fail_message_body_str = json.dumps(fail_body_json)

                for file in os.listdir("/home/ubuntu/Barbershop-Plus-Plus/Barbershop/input/face"):
                    if file == "1.png" or file == "2.png" or file == "3.png" or file == "4.png" or file == "5.png":
                        continue
                    delete_cmd = "rm /home/ubuntu/Barbershop-Plus-Plus/Barbershop/input/face/" + file
                    subprocess.call(delete_cmd, shell=True)

                try:
                    # Send message to Request Queue
                    response_queue.send_message(MessageBody=fail_message_body_str, QueueUrl=AWS_RESPONSE_SQS_URL)
                    logger.info("Send fail message success! (Message body : { result : %s, memberId : %s , virtualMemberImageId : %s})", "fail", param_member_id, param_virtual_member_image_id)
                    continue
                except ClientError as error:
                    logger.exception("Send fail message failed! (Message body : { result : %s, memberId : %s , virtualMemberImageId : %s})", "fail", param_member_id, param_virtual_member_image_id)
                    continue
            


            # ===== [Step 2.2] pSp 인코딩 =====
            print('psp 호출')
            os.chdir("/home/ubuntu/Barbershop-Plus-Plus/pixel2style2pixel/")
            subprocess.call("/home/ubuntu/psp_encoding.py", shell=True)



            # ===== [Step 2.3] Barbershop 추론 =====
            print('barbershop 호출')
            os.chdir("/home/ubuntu/Barbershop-Plus-Plus/Barbershop/")
            inference_cmd = 'python3 main.py --im_path1 input_image.png --sign realistic --smooth 1'
            subprocess.call(inference_cmd, shell=True)

            # ===== [Step 3] S3에 inference된 결과 이미지 업로드(presigned url 발급 + s3에 다운로드) =====
            # Generate a presigned S3 POST URL
            for i in range(reference_cnt):
                object_name = 'ai_result/' + str(param_member_id) + '/' + str(param_virtual_member_image_id) + '_' + str(i+1) + '.jpg' 
                response = create_presigned_post(object_name)
                if response is None:
                    continue

                # Demonstrate how another Python program can use the presigned URL to upload a file
                result_image_name = '/home/ubuntu/Barbershop-Plus-Plus/Barbershop/output/input_image_' + str(i+1) + '_' + str(i+1) + '_realistic.png'
                with open(result_image_name, 'rb') as f:
                    files = {'file': (object_name, f)}
                    http_response = requests.post(response['url'], data=response['fields'], files=files)
                # If successful, returns HTTP status code 204
                logging.info(f'File upload HTTP status code: {http_response.status_code}')
            

            # ===== [Step 4] 메시지 송신 =====
            success_msg_json = {
                'result' : 'success',
                'memberId' : param_member_id,
                'virtualMemberImageId' : param_virtual_member_image_id
            }

            success_message_body_str = json.dumps(success_msg_json)

            try:
                # Send message to Request Queue
                response_queue.send_message(MessageBody=success_message_body_str, QueueUrl=AWS_RESPONSE_SQS_URL)
                logger.info("Send message success! (Message body : { result : %s, memberId : %s, virtualMemberImageId : %s })", 'success', param_member_id, param_virtual_member_image_id)
            except ClientError as error:
                logger.exception("Send message failed! (Message body : { result : %s, memberId : %s, virtualMemberImageId : %s })", 'success', param_member_id, param_virtual_member_image_id)

                raise error
            
            # ===== [Step 5] inference 된 결과 및 input 이미지 데이터 지우기 =====
            # unprocessed
            delete_preprocessing_input_cmd = 'rm ' + '/home/ubuntu/Barbershop-Plus-Plus/Barbershop/unprocessed/input_image.jpg'
            subprocess.call(delete_preprocessing_input_cmd, shell=True)


            # input/face
            delete_align_input_face_cmd = 'rm  ' + '/home/ubuntu/Barbershop-Plus-Plus/Barbershop/input/face/input_image.png'
            subprocess.call(delete_align_input_face_cmd, shell=True)

            # output/W+
            # delete_output_W_img_cmd = 'rm ' + '/home/ubuntu/Barbershop-Plus-Plus/Barbershop/output/W+/input_image.png'
            delete_output_W_npy_cmd = 'rm ' + '/home/ubuntu/Barbershop-Plus-Plus/Barbershop/output/W+/input_image.npy'
            # subprocess.call(delete_output_W_img_cmd, shell=True)
            subprocess.call(delete_output_W_npy_cmd, shell=True)

            # output/FS
            delete_output_FS_img_cmd = 'rm ' + '/home/ubuntu/Barbershop-Plus-Plus/Barbershop/output/FS/input_image.png'
            delete_output_FS_npy_cmd = 'rm ' + '/home/ubuntu/Barbershop-Plus-Plus/Barbershop/output/FS/input_image.npz'
            subprocess.call(delete_output_FS_img_cmd, shell=True)
            subprocess.call(delete_output_FS_npy_cmd, shell=True)

            # result - via_mask
            for i in range(reference_cnt):
                delete_result_mask_cmd = 'rm ' + '/home/ubuntu/Barbershop-Plus-Plus/Barbershop/output/vis_mask_input_image_' + str(i + 1) + '_realistic.png'
                subprocess.call(delete_result_mask_cmd, shell=True)

            # result - align
            delete_result_align_cmd = 'rm -rf /home/ubuntu/Barbershop-Plus-Plus/Barbershop/output/Align_realistic/'
            subprocess.call(delete_result_align_cmd, shell=True)

            # result - blend
            delete_result_blend_cmd = 'rm -rf /home/ubuntu/Barbershop-Plus-Plus/Barbershop/output/Blend_realistic/'
            subprocess.call(delete_result_blend_cmd, shell=True)

            # result - final
            for i in range(reference_cnt):
                delete_result_real_cmd = 'rm ' + '/home/ubuntu/Barbershop-Plus-Plus/Barbershop/output/input_image_' + str(i + 1) + '_' + str(i + 1) + '_realistic.png'
                subprocess.call(delete_result_real_cmd, shell=True)

if __name__ == "__main__":
	main()