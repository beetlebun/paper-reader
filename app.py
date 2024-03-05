from PyPDF2 import PdfReader
from rank_bm25 import BM25Okapi
import unicodedata
import json
import nltk
import os
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
lemmatizer = nltk.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words("english"))

corpora: list[dict] = []


def get_file_paths() -> list[str]:
    files = os.listdir("../artigos/")
    pdf_files = []
    
    for file in files:
        if '.pdf' in file:
            pdf_files.append(f"../artigos/{file}")

    return pdf_files


def get_file_title(file_path: str) -> str:
    return PdfReader(open(file_path, "rb")).metadata.title


def get_document_from_file(file_path: str) -> str:
    document = ""

    for page in PdfReader(file_path).pages:
        document += page.extract_text()

    return document


def lemmatize(text: str) -> str:
    tokenized_words = nltk.word_tokenize(text)

    tagged_words = nltk.pos_tag(tokenized_words)
    lemmatized_words = []

    tag_to_arg = {
        "V": "v", # verbo
        "N": "n", # substantivo
        "J": "a", # adjetivo
        "R": "r", # advérbio
    }

    for word, tag in tagged_words:
        if tag[0] in tag_to_arg:
            lemmatized_words.append(lemmatizer.lemmatize(word, tag_to_arg[tag[0]]))
        else:
            lemmatized_words.append(word)

    return " ".join([word.lower() for word in lemmatized_words])


def replace_circumflex(text: str) -> str:
    new_text = text

    if " ´ı" in text:
        new_text = new_text.replace(" ´ı", "í")

    if " ´e" in text:
        new_text = new_text.replace(" ´e", "é")

    if " ´u" in text:
        new_text = new_text.replace(" ´u", "ú")

    if " ´a" in text:
        new_text = new_text.replace(" ´a", "á")

    if " ´o" in text:
        new_text = new_text.replace(" ´o", "ó")

    return new_text


def replace_line_breaks(text: str) -> str:
    new_text = text

    if "\n" in text:
        new_text = new_text.replace("\n", " ")

    return new_text


def remove_stopwords(text: str) -> str:
    tokenized_words = nltk.word_tokenize(text)
    result_words = [word for word in tokenized_words if word.casefold() not in stop_words]

    return " ".join([word.lower() for word in result_words])


def remove_punctuation(text: str) -> str:
    tokenized_words = nltk.word_tokenize(text)
    result_words = [unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore').decode('utf-8') for word in tokenized_words]
    result_words = [re.sub(r'[-%,.!?;*:|()\[\]“”`´]', '', word) for word in result_words]

    return " ".join([word.lower() for word in result_words])


def remove_nonwords(text: str) -> str:
    tokenized_words = nltk.word_tokenize(text)
    new_wordset = []

    for word in tokenized_words:
        if (word != "'s") and (not(word.isnumeric())) and (len(word) > 1):
            new_wordset.append(word)

    return " ".join([word.lower() for word in new_wordset])


def extract_references(text: str) -> tuple[str]:
    references = "NOT FOUND"
    new_text = text
    pos_references = text.lower().find("references [")

    if pos_references == -1:
        pos_references = text.lower().find("r eferences [")

    if pos_references == -1:
        pos_references = text.lower().find("r eferences [")

    if pos_references == -1:
        pos_references = text.lower().find("r\neferences [")

    if pos_references != -1:
        new_text = text[:pos_references]
        references = text[pos_references:]
        references = references.replace("references [", "", 1).strip()
        references = references.replace("r eferences [", "", 1).strip()
        references = references.replace("r\neferences [", "", 1).strip()
        references = references.replace("r eferences [", "", 1).strip()
        return (new_text, references)

    return (new_text, references)


def most_common_terms():
    if not(corpora_is_loaded()):
        print("Experimente processar os artigos antes!")
        return

    i = 1
    for obj in corpora:
        bag_of_words = obj['bag_of_words']
        ranking_most_common = bag_of_words.most_common(10)

        print(f".....'{obj['name']}'")

        for word in ranking_most_common:
            print(f".....{word}")

        print()

        i += 1


def search_term():
    if not(corpora_is_loaded()):
        print("Experimente processar os artigos antes!")
        return

    query = input("Insira query (termo de busca): ")
    tokenized_query = query.lower().split(" ")

    ranking = []
    corpus = []

    for obj in corpora:
        corpus.append(obj["processed_text"])

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    doc_scores = bm25.get_scores(tokenized_query)

    x = 0
    for score in doc_scores:
        if score > 0.0:
            ranking.append((corpora[x]["name"], score))
        x += 1

    if len(ranking) != 0:
        print(f'\nArtigos que contêm termo de busca ("{query}"):')
    else:
        print(f'\nNenhum artigo contém o termo de busca ("{query}")')

    sorted_ranking = sorted(ranking, key=lambda e: e[1], reverse=True)

    i = 1
    for entry in sorted_ranking:
        msg = f"#{i}...'{entry[0]}'...({entry[1]})"
        print(msg)
        i += 1


def extract_info(paragraphs: list[str], first_order_terms: list[str], second_order_terms: list[str], third_order_terms: list[str], filter_terms: list[str]) -> list[str]:
    data = []
    filtered_data = []

    for paragraph in paragraphs:
        if any(word in paragraph.lower() for word in first_order_terms) and any(word in paragraph.lower() for word in second_order_terms):
            data.append(paragraph)

    for info in data:
        if any(term in info.lower() for term in third_order_terms) and not any(re.search(r'\b' + term + r'\b', info.lower()) for term in filter_terms):
            filtered_data.append(info)

    return filtered_data


def load_corpora(file_paths: list[str]):
    try:
        with open("log.txt", "w", encoding="utf-8") as file:
            for file_path in file_paths:
                print(f"Processando {file_path}...")

                name = get_file_title(file_path)
                original_text = get_document_from_file(file_path)
                lemmatized_text = lemmatize(extract_references(replace_line_breaks(replace_circumflex(original_text)))[0])

                lemmatized_paragraphs = nltk.sent_tokenize(lemmatized_text)
                processed_text = remove_stopwords(remove_punctuation(lemmatized_text))
                bag_of_words = nltk.FreqDist(word.lower() for word in nltk.word_tokenize(remove_nonwords(processed_text)))

                objective = extract_info(lemmatized_paragraphs, 
                    ["objective", "aim", "aims", "propose", "proposes", "purpose", "intention", "present"], 
                    ["study", "paper", "approach", "method"], 
                    ["this paper", "this study", "this work", "we present", "the propose", "be propose", "be proposed"], 
                    ["issue", "contribution", "contributions"])

                references = extract_references(
                    extract_references(
                        replace_line_breaks(
                            replace_circumflex(original_text)))[1])

                obj = {
                    'name': name,
                    'paragraphs': lemmatized_paragraphs,
                    'processed_text': processed_text,
                    'bag_of_words': bag_of_words,
                    'objective': objective,
                    'references': references
                    }

                # file.write(f"nome: {obj['name']}\n\ncorpus: {obj['corpus']}\n\n---------------------------------------------------\n\n")
                json.dump(obj, file, ensure_ascii=False, indent=4)

                corpora.append(obj)

        print("\nArtigos processados com sucesso!\nInformações salvas no arquivo [dados_extraidos.txt]")

    except Exception as err:
        print(f"\nErro ao processar artigos:\n{err}")


def corpora_is_loaded() -> bool:
    return len(corpora) > 0


def start_app():
    while True:
        input_msg = "\n0. Sair\n1. Processar artigos\n2. Exibir 10 termos mais comuns\n3. Buscar termo nos artigos\n\nEscolha uma operação: "

        if corpora_is_loaded():
            input_msg = "\n0. Sair\n1. Reprocessar artigos\n2. Exibir 10 termos mais comuns\n3. Buscar termo nos artigos\n\nEscolha uma operação: "

        opr = input(input_msg)

        if not(opr.isnumeric()):
            print("Tente digitar um número para selecionar uma operação")
            continue
        
        opr_num = int(opr)

        if not(opr_num == 1 or opr_num == 2 or opr_num == 3 or opr_num == 0):
            print("Operação inválida, tente novamente")

        if opr_num == 0:
            print("Adeus! :]")
            break

        if opr_num == 1:
            load_corpora(get_file_paths())

        if opr_num == 2:
            most_common_terms()

        if opr_num == 3:
            search_term()


start_app()