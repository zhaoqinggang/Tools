import re
import json
import os
from copy import deepcopy


class pred_frame:
    def __init__(self):
        cur_path = os.path.dirname(__file__)      
        stop_words_path = cur_path + '/stop_words.txt'
        with open(stop_words_path, 'r', encoding='utf-8') as file:
            self.lines = [line.strip() for line in file]
    
    def split_text(self,text, unit_size=500, overlap_ratio=0.2):

        links = [(m.group(0), m.start(0), m.end(0)) for m in re.finditer(r"\$\[\[.*?\]\]", text)]

        result = []
        length = len(text)
        overlap = int(unit_size * overlap_ratio)

        start = 0
        end = unit_size

        while end < length:
            for link,x,y in links:
                if end > x and end < y:
                    end = min(end,x)

            result.append(text[start:end])

            start = end - overlap
            for link,x,y in links:
                if start > x and start < y:
                    start = min(start,x)
            end = start + unit_size


        # 添加最后一个部分
        result.append(text[start:])

        return result 
    

    def convert_dict_values_to_string(self,d):
        """
        此函数接收一个字典，遍历其所有的键-值对，
        并使用 json.dumps方法将每个值转换为字符串。
        """
        for key, value in d.items():
            if not isinstance(value, str):
                d[key] = json.dumps(value,ensure_ascii=False)
        return d
    
    

    def traverse_chunk_subsections(self, public_data, chunk_subsections):
        result = []  # 结果列表
        for key in chunk_subsections:
            value = chunk_subsections[key]
            private_data = {
                "chunk_preface": value["chunk_preface"],
                "chunk_sop_nodes": value["chunk_sop_nodes"],
                "chunk_cleaned_title": value["chunk_cleaned_title"],
                "chunk_cleaned_content": value["chunk_cleaned_content"],
                "chunk_nodes": value["chunk_nodes"],
                "chunk_level": value["chunk_level"],
                "chunk_title": value["chunk_title"],
                "chunk_content": value["chunk_content"],
            }
            private_data.update(public_data)
            result.append(private_data)

            # 如果有嵌套，继续递归遍历
            if value["chunk_subsections"]:
                result.extend(self.traverse_chunk_subsections(public_data, value["chunk_subsections"]))
        return result 

    def build_index(self,chunks):
        result = {}
        chunks = [self.convert_dict_values_to_string(chunk) for chunk in chunks]

        ##选择一个就好，每个字段都创建索引，会增加耗时
        # keys_list = list(chunks[0].keys()) if list(chunks[0].keys()) else []
        retrieval_keys_list = ['chunk_cleaned_title']
        recall_keys_list = ['chunk_cleaned_title','chunk_cleaned_content','retrieval_key','doc_name','doc_id','doc_type','doc_summary','doc_category']

        for key in retrieval_keys_list:
            key_list = []
            for chunk in chunks:
                chunk["doc_section"] = ""
                chunk["retrieval_key"] = key
                value = chunk[key]

                ##跳过构建索引
                skip_build_index_flag = False
                for line in self.lines:
                    if value.startswith(line):
                        skip_build_index_flag = True
                        break
                if chunk["chunk_level"] not in ["0","1"]:
                    skip_build_index_flag = True
                elif chunk["chunk_level"] != "0" and chunk["doc_category"] == "action":
                    skip_build_index_flag = True
                elif chunk["chunk_level"] != "0" and chunk["doc_category"] == "sop_question":
                    skip_build_index_flag = True

                if skip_build_index_flag:
                    continue


                chunk["retrieval_value"] = value
                chunk["retrieval_value_length"] = str(len(value))

                assert type(value) == str
                if not value:
                    continue
                texts = self.split_text(value)
                if len(texts)<=1:
                    copy_chunk1 = deepcopy(chunk)
                    copy_chunk1["retrieval_key_is_trunction"]='0'
                    if copy_chunk1[key] not in [x[key] for x in key_list]:
                        copy_chunk1["retrieval_value_length"] = str(len(copy_chunk1[key]))
                        key_list.append({k: copy_chunk1[k] for k in recall_keys_list if k in copy_chunk1})
                else:
                    for text in texts:
                        copy_chunk2 = deepcopy(chunk)
                        copy_chunk2["retrieval_key_is_trunction"]='1'
                        copy_chunk2[key] = text
                        if copy_chunk2[key] not in [x[key] for x in key_list]:
                            copy_chunk2["retrieval_value_length"] = str(len(copy_chunk2[key]))
                            key_list.append({k: copy_chunk2[k] for k in recall_keys_list if k in copy_chunk2})
            result[key] = key_list
        return result      

        
    def build_chunk(self, data):
        # 收集公有数据
        public_data = {key: data.get(key, "" if isinstance(data.get(key, ""), str) else {}) for key in data.keys()}
        chunk = self.traverse_chunk_subsections(public_data, public_data["doc_section"])
        return chunk


    def predict(self, params):
        status = params.get("status", "")
        reason = params.get("reason", "")
        data = params.get("data", {})
        
        
        result = {"status": status, "reason": reason, "structured_data":{}}
        if status == "fail": return result
        try:
            status,reason = "success",""
            structured_data = self.build_index(self.build_chunk(data))
        except Exception as e:
            status,reason,structured_data= "fail",f"model: build_index error, {str(e)}",[]
            
        result["status"] = status
        result["reason"] = reason
        result["structured_data"] = structured_data
        return result
        

def init(params):
    model = pred_frame()
    return model

if __name__ == '__main__':
    
    model = init(None)
    import json
    with open("input.json", "r") as file:
        a = json.load(file)
    x = model.predict(a)
    #print(x)
    with open('output.json', 'w', encoding='utf-8') as file:
        json.dump(x, file, ensure_ascii=False, indent=4)
    # for key,value in enumerate(x["data"]):
    #     print(key,value)
