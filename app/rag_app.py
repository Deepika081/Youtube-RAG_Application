from youtube_transcript_api import YouTubeTranscriptApi, YouTubeTranscriptApiException
from urllib.parse import urlparse, parse_qs
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from uuid import uuid4
from langchain_community.document_loaders.telegram import text_to_docs
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


class Demo:

    def __init__(self):

        self.api_key = os.getenv('GROQ_API_KEY')

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.client = chromadb.Client()
        self.llm = ChatGroq(
            api_key=self.api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=512,
        )


    def _extract_video_id(self, url:str):
        parsed = urlparse(url)
        if parsed.hostname == 'youtu.be':
            return parsed.path[1:]
        if parsed.hostname in ('www.youtube.com', 'youtube.com'):
            return parse_qs(parsed.query).get('v', [None])[0]
        return None
    
    def _fetch_transcript(self, video_id:str):
        try:
            yt_api = YouTubeTranscriptApi()
            transcript = yt_api.fetch(video_id,languages=['en'])
            text = " ".join([i.text for i in transcript])
            return text
        except YouTubeTranscriptApiException:
            raise RuntimeError('Transcription error')
        
    def _chunk_text(self, text:str):
        splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",
            tokens_per_chunk=384,
            chunk_overlap=50
        )
        chunks = splitter.split_text(text)
        return chunks
    
    def _create_vector_store(self, video_id:str):
        collection_name = f"video_{video_id}"
        vector_store = Chroma(
        client=self.client,
        collection_name=collection_name,
        embedding_function=self.embedding_model
        )
        count = vector_store._collection.count()
        is_new = count == 0
        return vector_store, is_new
    
    def _embed_and_store(self, chunks:list[str],vector_store):
        docs = text_to_docs(chunks)    
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=uuids)

    def _retrieve(self, query:str,vector_store):
        retrieved_docs = vector_store.similarity_search(query, k=10)
        return retrieved_docs
    
    def _format_chat_history(self, chat_history):
            formatted = ""
            for msg in chat_history:
                formatted += f"{msg['role'].upper()}: {msg['content']}\n"
            return formatted
    
    def _generate_answer(self, docs, query: str, chat_history: list):

        context = " ".join([doc.page_content for doc in docs])

        prompt = PromptTemplate(
            template="""
            You are a helpful assistant answering questions about a YouTube video.

            Conversation so far:
            {chat_history}

            Video context:
            {context}

            Current question:
            {question}

            Rules:
            - Use the video context as the primary source
            - Use conversation history only to resolve references (e.g., "that", "this part")
            - If the answer is not in the context, say "Context insufficient."

            Answer:
            """,
            input_variables=["chat_history","context", "question"],
        )

        chain = prompt | self.llm
        history_text = self._format_chat_history(chat_history)

        response = chain.invoke({
            "chat_history": history_text,
            "context": context,
            "question": query
        })

        return response.content


    def fetch_response(self, video_url: str, query: str, chat_history: list):
        video_id = self._extract_video_id(video_url)

        vector_store, is_new = self._create_vector_store(video_id)
        if is_new:
            transcript = self._fetch_transcript(video_id)
            chunks = self._chunk_text(transcript)
            self._embed_and_store(chunks, vector_store)

        docs = self._retrieve(query, vector_store)
        answer = self._generate_answer(docs, query, chat_history)
        return answer