from elasticsearch import Elasticsearch
from openai import OpenAI

# 初始化
es = Elasticsearch("http://localhost:9200")
client = OpenAI(
    base_url='https://api.siliconflow.cn/v1',
    api_key='sk-zhxeidatuglqramhktwnkguzmcmtxlkdjhcuqjtrdbfyngrk'
)


def fetch_doc(title):
    res = es.search(
        index="markdown_docs",
        query={
            "match": {
                "title": title,
            }
        }
    )
    print(res)
    hits = res["hits"]["hits"]
    if not hits:
        return None
    return hits[0]


def call_openai(prompt, model="deepseek-ai/DeepSeek-V2.5"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False  # 这里不需要流式输出，简单演示
    )
    # 取第一个完整回答文本
    return response.choices[0].message.content


def update_doc(index, doc_id, fields):
    es.update(index=index, id=doc_id, body={"doc": fields})


def main(title):
    hit = fetch_doc(title)
    if not hit:
        print("文档不存在")
        return
    doc_id = hit["_id"]
    source = hit["_source"]
    content = source.get("content", "")

    # 关键词提取
    prompt_kw = f"""请从以下文档内容中提取5~10个最重要的主题关键词，返回格式为JSON数组：

内容：
\"\"\"
{content}
\"\"\"
"""
    keywords_json = call_openai(prompt_kw)

    # 内容摘要
    prompt_summary = f"""请为以下文档内容生成一段简洁的摘要，控制在100字以内：

内容：
\"\"\"
{content}
\"\"\"
"""
    summary = call_openai(prompt_summary)

    # 文档归类（学术论文、电子书、其他）
    prompt_category = f"""请根据以下文档内容，判断该文档属于下面哪个类别，直接返回类别名称：
- 学术论文
- 电子书
- 其他

内容：
\"\"\"
{content}
\"\"\"
"""
    category = call_openai(prompt_category)

    # 更新 ES 文档
    update_fields = {
        "keywords": keywords_json,
        "summary": summary,
        "category": category
    }
    update_doc("markdown_docs", doc_id, update_fields)

    print(f"更新成功，关键词：{keywords_json}\n摘要：{summary}\n分类：{category}")


if __name__ == "__main__":
    main("1page")
