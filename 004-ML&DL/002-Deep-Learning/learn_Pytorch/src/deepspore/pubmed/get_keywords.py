from Bio import Entrez, Medline
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time


# 可视化：条形图
def plot_bar_chart(freq_dict, journal, top_n=20):
    top_items = freq_dict.most_common(top_n)
    keywords, freqs = zip(*top_items)
    plt.figure(figsize=(12, 6))
    plt.bar(keywords, freqs)
    plt.title(f"Top {top_n} Keywords in {journal}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# 可视化：词云
def plot_wordcloud(freq_dict, journal):
    wc = WordCloud(width=800, height=400, background_color="white")
    wc.generate_from_frequencies(freq_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Keyword WordCloud for {journal}")
    plt.show()


# 获取关键词
def fetch_keywords_for_journal(journal_name, max_count=200):
    query = f'{journal_name}[Journal]'
    search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_count)
    search_results = Entrez.read(search_handle)
    id_list = search_results["IdList"]

    fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="medline", retmode="text")
    records = Medline.parse(fetch_handle)

    keywords = []
    for record in records:
        if "OT" in record:
            keywords.extend(record["OT"])
    return [kw.lower().strip() for kw in keywords if kw.strip()]



def keywords_plot(journals: list = ['Cell', 'Nature', 'Science'], max_articles: int = 200, email: str | None = None) -> list:
    '''Plot the keywords plot according to the journals.'''
    # 设置你的邮箱地址
    if email:
        Entrez.email = email
    else:
        Entrez.email = "zhao_sy@126.com"

    # 要检索的期刊列表
    # journals = ["Nature", "Cell", "Science"]
    # max_articles = 200  # 每个期刊最多下载的文章数量

    all_keywords = []
    # 主逻辑
    for journal in journals:
        print(f"📚 检索期刊: {journal}")
        keywords = fetch_keywords_for_journal(journal, max_articles)
        freq = Counter(keywords)
        all_keywords.extend(keywords)
        
        if not freq:
            print(f"⚠️ 未找到 {journal} 的关键词")
            continue

        plot_bar_chart(freq, journal, top_n=20)
        plot_wordcloud(freq, journal)
        time.sleep(1/3)

    return all_keywords
