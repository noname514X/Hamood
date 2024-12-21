import requests
from lxml import etree
import json
import os
import re
from urllib.parse import urljoin
from rembg import remove
from PIL import Image
import io  
import jieba
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from imageio.v2 import imread
import nltk
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

#https://www.whitehouse.gov/wp-content/uploads/2021/01/03_thomas_jefferson.jpg

#                                                      real:
#https://www.whitehouse.gov/wp-content/uploads/2021/01/41_george_hw_bush.jpg
#https://www.whitehouse.gov/wp-content/uploads/2021/01/42_bill_clinton.jpg
#https://www.whitehouse.gov/wp-content/uploads/2021/01/36_lyndon_johnson.jpg
#https://www.whitehouse.gov/wp-content/uploads/2021/01/37_richard_nixon.jpg
#https://www.whitehouse.gov/wp-content/uploads/2021/01/39_jimmy_carter.jpg

#                                                      code:
#https://www.whitehouse.gov/wp-content/uploads/2021/01/41_george_h_w_bush.jpg
#https://www.whitehouse.gov/wp-content/uploads/2021/01/42_william_j_clinton.jpg
#https://www.whitehouse.gov/wp-content/uploads/2021/01/36_lyndon_b_johnson.jpg
#https://www.whitehouse.gov/wp-content/uploads/2021/01/37_richard_m_nixon.jpg
#https://www.whitehouse.gov/wp-content/uploads/2021/01/39_james_carter.jpg

#Washington https://www.whitehouse.gov/about-the-white-house/presidents/george-washington/
#Washington XPath
#//*[@id="content"]/article/section/div/div/p[1]/text()
#//*[@id="content"]/article/section/div/div/p[11]/text()

#Adams https://www.whitehouse.gov/about-the-white-house/presidents/john-adams/
#Adams XPath
#//*[@id="content"]/article/section/div/div/p[2]/text()

myheader = {'User-Agent':'Mozilla/5.0 '}
presidentURL = 'https://www.whitehouse.gov/about-the-white-house/presidents/'
imgpath = 'president'
despath = 'president'
wordcloudpath = 'wordcloud'
xpath_expression = '//*[@id="content"]/article/section/div/div/p[{i}]/text()'
nltk.download('punkt')
nltk.download('punkt_tab') 

def download_president_image(president_number, president_name):
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)

    formatted_number = str(president_number).zfill(2)

    base_url = "https://www.whitehouse.gov/wp-content/uploads/2021/01/"
    image_url = f"{base_url}{formatted_number}_{president_name.lower()}.jpg"
    print(image_url)

    try:
        response = requests.get(image_url, headers=myheader, timeout=10)
        if response.status_code == 200:
            image_path = os.path.join(imgpath, f"{formatted_number}_{president_name}.jpg")
            whiteimagepath = os.path.join(imgpath, f"{formatted_number}w_{president_name}.jpg")
            with open(image_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {president_name}")
            #return True
            with open(image_path, 'rb') as input_file:
                 input_data = input_file.read()
                 output_data = remove(input_data)
                 output_image = Image.open(io.BytesIO(output_data))
                 white_background = Image.new("RGB", output_image.size, (255, 255, 255))
                 white_background.paste(output_image, (0, 0), output_image)
                 white_background.save(whiteimagepath, 'JPEG', quality=95)
            return True
        else:
            print(f"Failed to download {president_name}: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {president_name}: {str(e)}")
        return False










def download_president_description(president_name,president_name2,president_number):
    url = f"https://www.whitehouse.gov/about-the-white-house/presidents/{president_name}/"
    headers = {'User-Agent': 'Mozilla/5.0 '}
    response = requests.get(url, headers=headers)
    formatted_number = str(president_number).zfill(2)

    if response.status_code == 200:
        tree = etree.HTML(response.content)
        description = []
        

        tree = etree.HTML(response.content)
        

        p_elements = tree.xpath('//*[@id="content"]/article/section/div/div/p')
        

        p_count = len(p_elements)
        
        for p in range(1, p_count):
            text = tree.xpath(f'//*[@id="content"]/article/section/div/div/p[{p}]/text()')
            description.append(text)
        

        if not os.path.exists('president'):
            os.makedirs('president')
        

        file_path = os.path.join('president', f"{formatted_number}_{president_name2}.txt")
        

        with open(file_path, 'w', encoding='utf-8') as file:
            for paragraph in description:
                if isinstance(paragraph, list):
                    paragraph = paragraph[0] if paragraph else ''
                file.write(paragraph + "\n\n")
        
        print(f"Description for {president_name} has been saved to {file_path}")
        return description
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return []


        











def GetPresidentList():
    response = requests.get(url = presidentURL,headers = myheader, timeout = 5)
    if response.status_code == 200:
        with open('presidentURL.html','wb') as f:
            f.write(response.content)
            print('ok')
        response.encoding = response.apparent_encoding
        html = etree.HTML(response.text)
        presidentList = []
        presidentList2 = []
        presidentList3 = []
        for i in range(1,46):
            presidents = html.xpath(f'//*[@id="content"]/article/section/div/div/div/div/div[{i}]/div/div/a/span')
            for president in presidents:
                presidentList.append(president.text.strip()) 
            for president in presidents:
                cleaned_name = president.text.strip()    
                cleaned_name = cleaned_name.rstrip('.')
                cleaned_name = re.sub(r'[ .]+', '_', cleaned_name)
                presidentList2.append(cleaned_name)
                download_president_image(i, cleaned_name)

                cleaned_name2 = cleaned_name.replace('_', '-')
                presidentList3.append(cleaned_name2)
                download_president_description(cleaned_name2,cleaned_name,i)

                GetWordCloud(f'president/{str(i).zfill(2)}_{cleaned_name}.txt', f'president/{str(i).zfill(2)}w_{cleaned_name}.jpg')

    return presidentList, presidentList2





def GetWordCloud(textFile, imageFile):
    # 加载停用词
    with open ('stopwords.txt','r',encoding='utf-8') as f:
        stopword = {w.strip() for w in f}

    with open(textFile, 'r', encoding='utf-8') as f:
        text = f.read()

    wordsplit = word_tokenize(text)
    
    wordText = ' '.join([w for w in wordsplit if w.lower() not in stopword and len(w) > 1])

    try:
        bg_image = np.array(Image.open(imageFile))
    except FileNotFoundError:
        print(f"警告: 图片文件 {imageFile} 未找到，跳过此文件。")
        return  # 跳过当前的词云图生成，直接返回
    
    bg_image = imread(imageFile)
    wc = WordCloud(background_color='white',
                   width=1000,
                   height=800,
                   mask=bg_image,
                   prefer_horizontal=0.5,  # 0-1之间的float数，设置字体显示方向的概率，默认为0.9，表示90%为水平显示
                   stopwords=stopword,
                   contour_width=3,  # 轮廓宽度
                   contour_color='red'  # 轮廓颜色
                   ).generate(wordText)
    # 保存图片

    textFileName = os.path.split(textFile)[1]
    imgname, ext = os.path.splitext(textFileName)
    imgname = imgname + '_Wordcloud.png'

    output_folder = 'Wordcloud'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    save_path = os.path.join(output_folder, imgname)

    wc.to_file(save_path)

    #plt.imshow(wc)  # 用plt显示图片
    #plt.axis('off')  # 不显示坐标轴
    #plt.show()
    #plt.close()

    print(f"词云图已保存为: {save_path}")




if __name__ == "__main__":
    presidents, cleaned_names = GetPresidentList()