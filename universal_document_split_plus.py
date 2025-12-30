import re
from copy import deepcopy


class pred_frame:
    def __init__(self,params):
        pass  

    def remove_html(self, text):
        # 匹配<hx>（x为数字）并替换为\n
        text = re.sub(r"<h\d+>", "\n", text)
        # 匹配<p>并替换为\n
        text = re.sub(r"<p>", "\n", text)
        # 移除其他HTML标签
        text = re.sub(r"<[\s\S]*?>", "", text)
        
        return text.strip()


    def remove_num(self, text):
        text = re.sub(r"<[\s\S]*?>", "", text)
        text = re.sub(r"<[\s\S]*?>", "", text)

        new_text = re.sub(r"^[\d\.、一二三四五六七八九十]+(?![\u4e00-\u9fff])", "", text)
        new_text = re.sub(r":", "", new_text)
        new_text = re.sub(r"：", "", new_text)
        new_text = new_text.strip()

        res_text = new_text if new_text else text
        return res_text
    
    def doc_split(self, document, n=1,status = []):
            chunks = {}
            chunks_columns = {}

            if n > 10:
                return chunks,chunks_columns

            chunk_titles = re.findall(f"<h{n}>[\s\S]+?</h{n}>", document)
            for chunk_id, chunk_t in enumerate(chunk_titles):
                if chunk_id == len(chunk_titles) - 1:
                    tail = ")"
                else:
                    tail = f"?)<h{n}>"

                try:
                    chunk_content = re.findall(f"{chunk_t}([\s\S]+{tail}", document)[0]
                    # try:
                    #     preface = re.findall(f"(.*?)(?=<h{n+1}>)", chunk_content)[0]
                    # except Exception as e:
                    preface = ""

                    cleaned_title = self.remove_num(chunk_t) if self.remove_num(chunk_t) else chunk_t
                    

                    cleaned_content = self.remove_html(chunk_content)

                    # 递归处理下一级标题，构建嵌套字典
                    chunks[cleaned_title] = {}

                    chunks[cleaned_title]["chunk_title"] = chunk_t
                    chunks[cleaned_title]["chunk_content"] = chunk_content
                    chunks[cleaned_title]["chunk_cleaned_title"] = cleaned_title
                    chunks[cleaned_title]["chunk_cleaned_content"] = cleaned_content

                    chunks[cleaned_title]["chunk_level"] = re.findall(r"<h([\d])+?>", chunk_t)[0]
                    chunks[cleaned_title]["chunk_preface"] = preface
                    chunks[cleaned_title]["chunk_nodes"] = status + [cleaned_title] 
                    chunks[cleaned_title]["chunk_sop_nodes"] = '->'.join(chunks[cleaned_title]["chunk_nodes"])

                    # chunks_columns
                    chunks[cleaned_title]["chunk_subsections"],chunks_columns[f"{chunks[cleaned_title]['chunk_title']}"] = self.doc_split(chunk_content, n + 1,chunks[cleaned_title]["chunk_nodes"])

                except Exception as e:
                    print(e)

            return chunks,chunks_columns

            
    def predict(self, params):
        ## 解析入参
        status = params.get("status", "")
        reason = params.get("reason", "")
        data = params.get("data", {})

        result = {"status": status, "reason": reason, "data": []}
        if status == "fail": return result
        try:
            doc_chunks_all = {key: data.get(key, "" if isinstance(data.get(key, ""), str) else {}) for key in data.keys()}
            ##相关字段处理
            doc_id = doc_chunks_all["doc_id"]
            doc_name = doc_chunks_all["doc_name"]
            doc_content = doc_chunks_all["doc_content"]
            if not (doc_id and doc_name and doc_content):
                status = "fail"
                result = {"status": status, "reason": f"字段doc_id or doc_name or doc_content为空,其中 doc_id:{doc_id},doc_name:{doc_name},doc_content:{doc_content}", "data": []}
                
            doc_name = doc_name.replace("|", '或')
            # 获取文档标题级别
            doc_title_n = [int(n) for n in re.findall(r"<h([\d])+?>", doc_content)]+[1]
            header = """<h{0}>{1}</h{0}>""".format(min(doc_title_n)-1, doc_name)
            doc_content = header+doc_content
            # 预处理替换
            doc_content = doc_content.replace("|", '或')
            min_doc_title_n = min(doc_title_n)-1
            max_doc_title_n = max(doc_title_n)
            # 去除文档内颜色（颜色会使正则失效）
            doc_content = re.sub("rgba?\([\s\S]+?\)", "", doc_content)
            # 切片
            doc_chunks,doc_columns = self.doc_split(doc_content, min_doc_title_n,[])
            doc_chunks_all["doc_max_title"]= f"<h{max_doc_title_n}>"
            doc_chunks_all["doc_min_title"]= f"<h{min_doc_title_n}>"
            doc_chunks_all["doc_section"]= doc_chunks
            doc_chunks_all["doc_columns"]= doc_columns

            result["status"] = "success"
            result["data"] = doc_chunks_all
        except Exception as e:
            result["status"] = "fail"
            result["reason"] = f"model: universal_document_split error, {str(e)}"
            result["data"] = []

        return result

def init(params):
    model = pred_frame(params)
    return model



if __name__ == '__main__':
    model = init(None)
    import json
    with open("input.json", "r") as file:
        a = json.load(file)
    x = model.predict(a)
    # print(x)
    with open('output.json', 'w', encoding='utf-8') as file:
        json.dump(x, file, ensure_ascii=False, indent=4)
