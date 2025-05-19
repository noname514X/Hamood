import re

def find_three_letter_words(text):

  words = re.findall(r'\b\w{3}\b', text.lower())
  return words

if __name__ == "__main__":
  english_text = input("请输入一段英文文本：")
  three_letter_words = find_three_letter_words(english_text)
  if three_letter_words:
    print("这段英文中长度为3的单词有：")
    for word in three_letter_words:
      print(word)
  else:
    print("这段英文中没有长度为3的单词。")