# 텍스트를 딥러닝 모델에 넣어 분류하기 이전, 전처리를 위한 함수들
# from konlpy.tag import Mecab
import re

# mecab = Mecab()


# def mecab_tokenize(text):
#     """
#     Mecab을 사용하여 tokenized text 반환
#     """
#     return " ".join(mecab.morphs(text))


def clean_etc_reg_ex(title):
    """
    정규식을 통해 기타 공백과 기호, 숫자등을 제거
    """
    title = re.sub(
        r"[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]", "", title
    )  # remove punctuation
    title = re.sub(r"[∼%①②⑤⑪…→·]", "", title)
    title = re.sub(r"\d+", "", title)  # remove number
    title = re.sub(r"\s+", " ", title)  # remove extra space
    title = re.sub(r"<[^>]+>", "", title)  # remove Html tags
    title = re.sub(r"\s+", " ", title)  # remove spaces
    title = re.sub(r"^\s+", "", title)  # remove space from start
    title = re.sub(r"\s+$", "", title)  # remove space from the end
    title = re.sub("[一-龥]", "", title)
    return title


def slice_from_behind(text, num_of_chars):
    return text[-num_of_chars:]

def preprocess(text):
    text = clean_etc_reg_ex(text)
    # text = mecab_tokenize(text)
    return slice_from_behind(text, num_of_chars=500)