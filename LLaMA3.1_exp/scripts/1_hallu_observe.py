import csv
import string
import torch
import tempfile
import os
import random

from deep_translator import GoogleTranslator
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from llm import llm

# File paths
data_file_path = '../data/Poly-FEVER_all.tsv'
record_file_path = 'observe_LLaMA3.1_70B_record.tsv'
output_file_path = 'observe_LLaMA3.1_70B_output.tsv'
header = ['en', 'zh-CN', 'hi', 'ar', 'bn', 'ja', 'ko', 'ta', 'th', 'ka', 'am']

# Number of parallel jobs
n_jobs = 4

# Percentage of data to select
select_percent = 1

constrain_translations = {
    'en': "Assess the truthfulness of the user's claim and provide a response. \nUse 'true' to indicate that the claim is true, 'false' to indicate that it is false. \nYour response should only consist of 'true' or 'false', without any additional characters or punctuation.", 
    'zh-CN': "评估用户声明的真实性并提供回复。\n使用'true'表示声明为真，使用'false'表示声明为假。\n您的回复应仅包含'true'或'false'，不包含任何其他字符或标点符号。", 
    'hi': "उपयोगकर्ता के दावे की सत्यता का आकलन करें और प्रतिक्रिया दें।\n\nयह दर्शाने के लिए कि दावा सत्य है, 'true' का प्रयोग करें, तथा यह दर्शाने के लिए कि यह असत्य है, 'false' का प्रयोग करें।\n\nआपकी प्रतिक्रिया में केवल 'true' या 'false' ही होना चाहिए, तथा उसमें कोई अतिरिक्त वर्ण या विराम चिह्न नहीं होना चाहिए।",  
    'ar': "قم بتقييم صدق ادعاء المستخدم وقدم ردًا.\nاستخدم 'true' للإشارة إلى أن الادعاء صحيح، و'false' للإشارة إلى أنه خاطئ.\nيجب أن تتكون إجابتك فقط من 'true' أو 'false'، دون أي أحرف إضافية أو علامات ترقيم.", 
    'bn': "ব্যবহারকারীর দাবির সত্যতা মূল্যায়ন করুন এবং একটি প্রতিক্রিয়া প্রদান করুন। \nদাবিটি সত্য তা নির্দেশ করতে 'true' ব্যবহার করুন, এটি মিথ্যা তা নির্দেশ করতে 'false' ব্যবহার করুন। \nআপনার উত্তরে শুধুমাত্র 'true' বা 'false' থাকা উচিত, কোনো অতিরিক্ত অক্ষর বা বিরাম চিহ্ন ছাড়াই।", 
    'ja': "ユーザーの主張の真実性を評価し、回答を提供します。\n主張が真実であることを示すには「true」を使用し、偽であることを示すには「false」を使用します。\n回答は「true」または「false」のみで構成され、追加の文字や句読点は使用しないでください。", 
    'ko': "사용자 주장의 진실성을 평가하고 응답을 제공하세요. \n주장이 사실임을 나타내려면 'true'을 사용하고 거짓임을 나타내려면 'false'을 사용하세요. \n응답은 추가 문자나 구두점 없이 'true' 또는 'false'으로만 구성되어야 합니다.", 
    'ta': "பயனரின் கூற்றின் உண்மைத்தன்மையை மதிப்பிட்டு பதிலை வழங்கவும். \nகூற்று உண்மை என்பதைக் குறிக்க 'true', அது தவறு என்பதைக் குறிக்க 'false' என்பதைப் பயன்படுத்தவும். \nஉங்கள் பதிலில் கூடுதல் எழுத்துகள் அல்லது நிறுத்தற்குறிகள் இல்லாமல் 'true' அல்லது 'false' மட்டுமே இருக்க வேண்டும்.", 
    'th': "ประเมินความจริงของการกล่าวอ้างของผู้ใช้และตอบกลับ \nใช้ 'true' เพื่อระบุว่าการกล่าวอ้างเป็นจริง 'false' เพื่อระบุว่าเป็นเท็จ \nคำตอบของคุณควรประกอบด้วย 'true' หรือ 'false' เท่านั้น โดยไม่มีอักขระหรือเครื่องหมายวรรคตอนเพิ่มเติม", 
    'ka': "შეაფასეთ მომხმარებლის პრეტენზიის სინამდვილე და უპასუხეთ. \nგამოიყენეთ 'true' რათა მიუთითოთ, რომ პრეტენზია მართალია, 'false' მიუთითოთ, რომ ის მცდარია. \nთქვენი პასუხი უნდა შედგებოდეს მხოლოდ 'true' ან 'false', ყოველგვარი დამატებითი სიმბოლოებისა და პუნქტუაციის გარეშე.", 
    'am': "የተጠቃሚውን የይገባኛል ጥያቄ እውነተኝነት ይገምግሙ እና ምላሽ ይስጡ። \nየይገባኛል ጥያቄው እውነት መሆኑን ለማመልከት 'true'ን ተጠቀም፣ ውሸት መሆኑን ለማመልከት 'false'። \nየእርስዎ ምላሽ 'true' ወይም 'false' ብቻ ነው፣ ያለ ምንም ተጨማሪ ቁምፊዎች ወይም ሥርዓተ-ነጥብ መያዝ አለበት።"
}

# Classification Prompt
classification_prompt = """Classify the input as 'true' or 'false' based solely on the indicative words or phrases within it. 
Use 'true' for it contains affirming words like 'Correct,' 'TRUE,', "really" or 'the truth.' 
Use 'false' for it contains negating or contradictory phrases like 'Fake,' 'False,' or any form of correction or contradiction within the input. 
Respond with only 'true' or 'false' for the input, without any additional text, characters, or punctuation."""

def check_gpu_memory():
    """ Returns the free and total memory of the first CUDA device """
    if torch.cuda.is_available():
        device_index = 0  # Explicitly set to the first device
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        free_memory = total_memory - torch.cuda.memory_allocated(device)
        return free_memory, total_memory
    else:
        return 0, 0  # Default if no GPU is available

def process_row(row, temp_file_path):
    idx, label, *claims = row
    label = label.lower().translate(str.maketrans('', '', string.punctuation))
    record_row = [idx, label]
    local_T_cnt = {key: 0 for key in header}
    local_total_cnt = {key: 0 for key in header}

    with torch.no_grad():
        for i, claim in enumerate(claims):
            language = header[i]
            constrain = constrain_translations[language]

            try:
                response_label = llm.fact_check(constrain, claim)
                if response_label not in ['true', 'false']:
                    trans_response = GoogleTranslator(source=language, target='en').translate(response_label)
                    response_label = llm.classify_sentence(classification_prompt, trans_response)

                record_row.append(response_label)
                if response_label == label:
                    local_T_cnt[language] += 1
                local_total_cnt[language] += 1
            except Exception as e:
                record_row.append(f"Error: {str(e)}")

            # Clear unused memory to prevent memory leaks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(temp_file_path, 'a', newline='', encoding='utf-8') as file_temp:
        writer = csv.writer(file_temp, delimiter='\t')
        writer.writerow(record_row)

    return local_T_cnt, local_total_cnt

if __name__ == "__main__":
    # Adjust parallel jobs based on system capacity
    # free_memory, total_memory = check_gpu_memory()
    # n_jobs = max(1, int(psutil.cpu_count(logical=False) / 2)) if free_memory / total_memory < 0.1 else psutil.cpu_count(logical=False)
    # print(f"Using {n_jobs} parallel jobs")

    # Initialize a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', newline='', encoding='utf-8')
    temp_file_path = temp_file.name

    # Data processing
    filtered_data = []
    with open(data_file_path, 'r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        all_rows = list(tsv_reader)  # Read all rows into a list

        # Calculate the number of rows to select based on the percentage
        num_rows_to_select = int(select_percent * len(all_rows))

        # Generate random indices to select
        selected_indices = random.sample(range(len(all_rows)), num_rows_to_select)

        for i, row in enumerate(all_rows):
            if i in selected_indices:
                selected_row = [row[0], row[1]] + row[2:13]
                filtered_data.append(selected_row)

    # Parallel processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(row, temp_file_path) for row in tqdm(filtered_data, total=len(filtered_data))
    )
    temp_file.close()

    # Read from the temporary file and aggregate results
    T_cnt = {key: 0 for key in header}
    total_cnt = {key: 0 for key in header}
    for local_T_cnt, local_total_cnt in results:
        for language in header:
            T_cnt[language] += local_T_cnt[language]
            total_cnt[language] += local_total_cnt[language]

    with open(temp_file_path, 'r', newline='', encoding='utf-8') as file_temp, \
            open(record_file_path, 'w', newline='', encoding='utf-8') as file_record, \
            open(output_file_path, 'w', newline='', encoding='utf-8') as file_output:
        temp_reader = csv.reader(file_temp, delimiter='\t')
        record_writer = csv.writer(file_record, delimiter='\t')
        output_writer = csv.writer(file_output, delimiter='\t')

        # Write headers
        record_writer.writerow(['index', 'Label'] + header)
        output_writer.writerow(['Language', 'Total Counts', 'True Counts', 'Correct Percentage'])

        # Write records
        for row in temp_reader:
            record_writer.writerow(row)

        # Write output file with percentages
        for language in header:
            correct_percentage = round((T_cnt[language] / total_cnt[language]) * 100, 2) if total_cnt[language] > 0 else 0.0
            output_writer.writerow([language, total_cnt[language], T_cnt[language], correct_percentage])

    # Clean up the temporary file
    os.unlink(temp_file_path)