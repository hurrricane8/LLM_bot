import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def extract_text_from_tex(tex_path):
    with open(tex_path, "r", encoding="utf-8") as file:
        return file.read()


def preprocess_tex(text):
    # Remove comments and unnecessary whitespace
    text = re.sub(r"(?<!\\)%.*", "", text)  # Remove comments
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with one
    return text.strip()


def chunk_text(text, chunk_size=2000, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.create_documents([text])


def main():
    tex_folder = "papers"
    output_folder = "models/faiss_index"
    ensure_directory_exists(tex_folder)
    ensure_directory_exists(output_folder)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    all_chunks = []

    tex_files = [f for f in os.listdir(tex_folder) if f.endswith(".tex")]
    if not tex_files:
        print("Нет файлов .tex для обработки.")
        return

    for tex_file in tex_files:
        tex_path = os.path.join(tex_folder, tex_file)
        raw_text = extract_text_from_tex(tex_path)
        clean_text = preprocess_tex(raw_text)
        chunks = chunk_text(clean_text)
        all_chunks.extend(chunks)

    if all_chunks:
        # Создаем FAISS-базу
        vector_store = FAISS.from_documents(all_chunks, embeddings)
        vector_store.save_local(output_folder)
        print(f"База данных FAISS успешно сохранена в {output_folder}.")
    else:
        print("Не удалось обработать файлы .tex, данные отсутствуют.")


if __name__ == '__main__':
    main()
