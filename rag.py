import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import re
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
load_dotenv()

class ArxivPaperCrawler:
    def __init__(self, delay=1):
        self.delay = delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def crawl_paper(self, url):
        """논문의 모든 섹션을 구조화해서 크롤링"""
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            paper_data = {
                'url': url,
                'crawled_at': datetime.now().isoformat(),
                'title': self._extract_title(soup),
                'authors': self._extract_authors(soup),
                'abstract': self._extract_abstract(soup),
                'sections': self._extract_sections(soup),
                'references': self._extract_references(soup),
                'equations': self._extract_equations(soup),
                'figures': self._extract_figures(soup)
            }
            
            return paper_data
            
        except Exception as e:
            print(f"크롤링 에러: {e}")
            return None
    
    def _extract_title(self, soup):
        """제목 추출"""
        title_selectors = [
            'h1.ltx_title',
            'h1',
            '.ltx_title',
            'title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        return "제목을 찾을 수 없음"
    
    def _extract_authors(self, soup):
        """저자 정보 추출"""
        authors = []
        
        # 여러 방법으로 저자 찾기
        author_containers = soup.find_all('div', class_='ltx_authors') or []
        
        for container in author_containers:
            author_elems = container.find_all('span', class_='ltx_personname')
            for author in author_elems:
                authors.append(author.get_text().strip())
        
        # 대안 방법
        if not authors:
            author_links = soup.find_all('a', href=re.compile(r'author:'))
            authors = [link.get_text().strip() for link in author_links]
        
        return authors
    
    def _extract_abstract(self, soup):
        """초록 추출"""
        abstract_selectors = [
            'div.ltx_abstract',
            '.abstract',
            'blockquote'
        ]
        
        for selector in abstract_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        return "초록을 찾을 수 없음"
    
    def _extract_sections(self, soup):
        """섹션별 내용 추출"""
        sections = []
        
        # 섹션 헤더 찾기
        section_headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], 
                                      class_=re.compile(r'ltx_title'))
        
        for i, header in enumerate(section_headers):
            section = {
                'level': int(header.name[1]),
                'title': header.get_text().strip(),
                'content': ''
            }
            
            # 다음 섹션까지의 내용 수집
            current = header.find_next_sibling()
            content_parts = []
            
            while current and (i+1 >= len(section_headers) or current != section_headers[i+1]):
                if current and current.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    content_parts.append(current.get_text().strip())
                current = current.find_next_sibling() if current else None
                
                if not current:
                    break
            
            section['content'] = '\n'.join(content_parts)
            sections.append(section)
        
        return sections
    
    def _extract_references(self, soup):
        """참고문헌 추출"""
        references = []
        
        # 참고문헌 섹션 찾기
        ref_section = soup.find('section', class_='ltx_bibliography') or \
                     soup.find('div', class_='references')
        
        if ref_section:
            ref_items = ref_section.find_all('li') or ref_section.find_all('div', class_='ltx_bibitem')
            for item in ref_items:
                references.append(item.get_text().strip())
        
        return references
    
    def _extract_equations(self, soup):
        """수식 추출"""
        equations = []
        
        # LaTeX 수식 찾기
        eq_elements = soup.find_all(['div', 'span'], class_=re.compile(r'ltx_equation|math'))
        
        for eq in eq_elements:
            equations.append({
                'type': eq.get('class', []),
                'content': eq.get_text().strip()
            })
        
        return equations
    
    def _extract_figures(self, soup):
        """그림/표 정보 추출"""
        figures = []
        
        # 그림 캡션 찾기
        fig_elements = soup.find_all('figure') + soup.find_all('div', class_='ltx_figure')
        
        for fig in fig_elements:
            caption = fig.find('figcaption') or fig.find('div', class_='ltx_caption')
            img = fig.find('img')
            
            figure_data = {
                'caption': caption.get_text().strip() if caption else '',
                'src': img.get('src') if img else ''
            }
            figures.append(figure_data)
        
        return figures

class PaperRAGProcessor:
    def __init__(self, api_key: str = None, model_name: str = "text-embedding-3-small"):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API 키가 필요합니다. 환경 변수 'OPENAI_API_KEY'를 설정하거나 인자로 전달해주세요.")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.embedding_dim = self._get_embedding_dimension()

    def _get_embedding_dimension(self):
        # 임베딩 차원을 동적으로 확인
        try:
            response = self.client.embeddings.create(input=["test"], model=self.model_name)
            return len(response.data[0].embedding)
        except Exception as e:
            print(f"임베딩 차원 확인 중 오류 발생: {e}. 기본값 1536을 사용합니다.")
            return 1536

    def _get_openai_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """OpenAI 임베딩 모델을 사용하여 텍스트 리스트의 임베딩을 가져옵니다."""
        # OpenAI API는 빈 문자열을 처리하지 못하므로 대체합니다.
        texts = [text.replace("\n", " ").strip() or " " for text in texts]
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            embeddings = [np.array(embedding.embedding) for embedding in response.data]
            return embeddings
        except Exception as e:
            print(f"OpenAI 임베딩 생성 중 오류 발생: {e}")
            return [np.zeros(self.embedding_dim) for _ in texts]

    def process_paper(self, paper_data):
        """논문 데이터를 RAG용으로 처리"""
        chunks = []
        
        # 1. 초록은 별도 처리
        if paper_data.get('abstract') and paper_data['abstract'] != "초록을 찾을 수 없음":
            chunks.append({
                'text': paper_data['abstract'],
                'type': 'abstract',
                'metadata': {'section': 'abstract', 'importance': 'high'}
            })
        
        # 2. 섹션별 구조적 청킹
        for section in paper_data.get('sections', []):
            if section['content'].strip():  # 빈 내용이 아닌 경우만
                section_chunks = self._chunk_section(section)
                chunks.extend(section_chunks)
        
        # 3. 임베딩 생성 (배치 처리)
        texts_to_embed = [chunk['text'] for chunk in chunks]
        if texts_to_embed:
            embeddings = self._get_openai_embeddings(texts_to_embed)
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk['embedding'] = embeddings[i]
                else:
                    # 임베딩 생성에 실패한 경우
                    chunk['embedding'] = np.zeros(self.embedding_dim)
        
        return [chunk for chunk in chunks if 'embedding' in chunk]
    
    def _chunk_section(self, section):
        """섹션을 의미론적으로 청킹"""
        full_text = f"{section['title']}\n{section['content']}"
        
        # 문단별로 1차 분할
        paragraphs = full_text.split('\n\n')
        chunks = []
        current_chunk = ''
        
        for para in paragraphs:
            para = para.strip()
            if not para:  # 빈 문단 건너뛰기
                continue
                
            test_chunk = current_chunk + '\n\n' + para if current_chunk else para
            
            # 200 단어 기준
            if len(test_chunk.split()) <= 200:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'type': 'section',
                        'metadata': {'section': section['title']}
                    })
                current_chunk = para
        
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'type': 'section',
                'metadata': {'section': section['title']}
            })
        
        return chunks

def crawl_and_embed_paper(url: str, openai_api_key: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    논문 URL을 받아 내용을 크롤링하고 청크로 나눈 뒤 임베딩을 생성합니다.

    Args:
        url (str): 크롤링할 논문의 ArXiv URL.
        openai_api_key (str, optional): OpenAI API 키. Defaults to None.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]: 
            - 임베딩이 포함된 청크 리스트
            - 논문 정보 딕셔너리
    """
    print("1. 논문 크롤링 및 전처리 시작...")
    
    # 1. 크롤러 및 프로세서 초기화
    crawler = ArxivPaperCrawler()
    processor = PaperRAGProcessor(api_key=openai_api_key)

    # 2. 논문 크롤링
    print(f"URL에서 논문 크롤링 중: {url}")
    paper_data = crawler.crawl_paper(url)
    
    if not paper_data:
        print("논문 크롤링에 실패했습니다.")
        return None, None

    paper_info = {
        'title': paper_data['title'],
        'authors': paper_data['authors'],
        'url': url
    }
    print(f"논문 제목: {paper_info['title']}")

    # 3. 청킹 및 임베딩
    print("텍스트 청킹 및 임베딩 생성 중...")
    chunks_with_embedding = processor.process_paper(paper_data)
    
    print(f"총 {len(chunks_with_embedding)}개의 청크가 생성 및 임베딩되었습니다.")
    print("-" * 50)
    
    return chunks_with_embedding, paper_info


def answer_question(
    question: str, 
    chunks: List[Dict[str, Any]], 
    paper_info: Dict[str, Any],
    openai_api_key: str = None, 
    top_k: int = 3
) -> str:
    """
    사용자 질문에 대해 가장 관련성 높은 논문 청크를 찾아 GPT-4o 모델로 답변을 생성합니다.

    Args:
        question (str): 사용자 질문.
        chunks (List[Dict[str, Any]]): 임베딩된 논문 청크 리스트.
        paper_info (Dict[str, Any]): 논문 정보.
        openai_api_key (str, optional): OpenAI API 키. Defaults to None.
        top_k (int, optional): 답변 생성에 사용할 청크 수. Defaults to 3.

    Returns:
        str: GPT-4o가 생성한 답변.
    """
    print("2. 질문에 대한 답변 생성 시작...")

    # 1. OpenAI 클라이언트 및 프로세서 초기화
    client = openai.OpenAI(api_key=openai_api_key)
    processor = PaperRAGProcessor(api_key=openai_api_key)
    
    # 2. 질문 임베딩
    print("질문 임베딩 중...")
    question_embedding = processor._get_openai_embeddings([question])[0]
    
    if question_embedding.sum() == 0:
        return "질문 임베딩 생성에 실패했습니다."

    # 3. 유사도 높은 청크 검색
    print("관련성 높은 논문 내용 검색 중...")
    similarities = []
    for chunk in chunks:
        similarity = cosine_similarity(
            question_embedding.reshape(1, -1),
            chunk['embedding'].reshape(1, -1)
        )[0][0]
        similarities.append((chunk, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    top_chunks = [item[0] for item in similarities[:top_k]]
    
    # 4. GPT-4o 프롬프트 구성
    print("GPT-4o 모델로 답변 생성 요청 중...")
    context = "\n\n---\n\n".join([chunk['text'] for chunk in top_chunks])
    
    prompt = f"""당신은 주어진 논문 내용을 바탕으로 질문에 답변하는 전문 AI 연구원입니다.
다음은 '{paper_info['title']}' 논문의 일부 내용입니다.

--- 논문 내용 시작 ---
{context}
--- 논문 내용 끝 ---

이 내용을 바탕으로 다음 질문에 대해 한국어로 명확하고 상세하게 답변해주세요.

질문: {question}
"""

    # 5. OpenAI API 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 전문적인 AI 연구원입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        print("답변 생성 완료.")
        print("-" * 50)
        return response.choices[0].message.content
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

def main():
    # OpenAI API 키는 환경변수 `OPENAI_API_KEY`에서 자동으로 로드됩니다.
    try:
        # 키 존재 여부만 확인
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError()
    except ValueError:
        print("OPENAI_API_KEY 환경 변수를 설정하고 다시 시도해주세요.")
        return
    
    # 1. 크롤링 및 임베딩 함수 호출
    url = "https://ar5iv.labs.arxiv.org/html/2201.12091" # 예제 논문
    chunks, paper_info = crawl_and_embed_paper(url)
    
    if not chunks:
        return

    # 2. QA 함수 호출
    questions = [
        "이 논문의 주요 기여는 무엇인가요?",
        "실험 방법론에 대해 설명해주세요",
        "결론에서 제시한 한계점은 무엇인가요?",
        "이 논문에서 사용한 모델의 이름은 무엇인가?"
    ]
    
    for q in questions:
        print(f"\n[질문] {q}")
        answer = answer_question(q, chunks, paper_info)
        print(f"\n[답변]\n{answer}")
        print("\n" + "="*80)


if __name__ == "__main__":
    main()
