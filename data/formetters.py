import random
import json

def _get_documtent_text(doc):
    return f'text: {doc["text"]}'

def get_baseline_no_rag_formatter(train=False):
    def formatter(data):
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            text = f"""Your task is to generate a comprehensive and factual response to the following query:
            query: {user_prompt}
            response:"""
            if train:
                text = text + f" {data['output'][i]}<eos>"
            texts.append(text)
        return texts
    return formatter

def get_baseline_no_rag_cot_formatter():
    def formatter(data):
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            text = f"""Your task is to generate a comprehensive and factual response to the following query. You should first think step by step about the information that is needed to be present in the answer to the query and then generate a response that is both comprehensive and factually accurate. You should start your thinking by "thought:" and your final response to the query by "response:".
            query: {user_prompt}
            thought:"""
            texts.append(text)
        return texts
    return formatter

def get_baseline_rag_formatter(num_contexts, train=False):
    def formatter(data):
        assert "context" in data, 'context should be in the dataset'
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            context = data['context'][i][:num_contexts]
            combined_context = "\n\n".join([_get_documtent_text(doc) for doc in context])
            text = f"""Your task is to generate a comprehensive and factual response to the given query. You can use the information provided in the context to generate a more comprehensive and factual response.
            query: 
            {user_prompt}
            context: 
            {combined_context}
            response:"""
            if train:
                text = text + f" {data['output'][i]}<eos>"
            texts.append(text)
        return texts
    return formatter

def get_baseline_rag_formatter_test_hamed(num_contexts, train=False):
    def formatter(data):
        assert "context" in data, 'context should be in the dataset'
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            context = data['context'][i][:num_contexts]
            combined_context = "\n\n".join([_get_documtent_text(doc) for doc in context])
            text = f"""Your task is to read the given information carefully and generate only the word "done" in your output and nothing else.
            query: 
            {user_prompt}
            context: 
            {combined_context}
            response:"""
            if train:
                text = text + f" {data['output'][i]}<eos>"
            texts.append(text)
        return texts
    return formatter

def get_baseline_rag_formatter_test_hamed_long(num_contexts, train=False):
    def formatter(data):
        assert "context" in data, 'context should be in the dataset'
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            context = data['context'][i][:num_contexts]
            combined_context = "\n\n".join([_get_documtent_text(doc) for doc in context])
            text = f"""Your task is to read the given information and write them again word by word in your output and nothing else.
            query: 
            {user_prompt}
            context: 
            {combined_context}
            response:"""
            if train:
                text = text + f" {data['output'][i]}<eos>"
            texts.append(text)
        return texts
    return formatter

def get_baseline_rag_cot_formatter(num_contexts):
    def formatter(data):
        assert "context" in data, 'context should be in the dataset'
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            context = data['context'][i][:num_contexts]
            combined_context = "\n\n".join([_get_documtent_text(doc) for doc in context])
            text = f"""Your task is to generate a comprehensive and factual response to the following query. You should first think step by step about the information that is needed to be present in the answer to the query and then generate a response that is both comprehensive and factually accurate. You should start your thinking by "thought:" and your final response to the query by "response:". You can use the information provided in the context to generate a more comprehensive and factual response.
            query: 
            {user_prompt}
            context: 
            {combined_context}
            thought:"""
            texts.append(text)
        return texts
    return formatter


def get_query_planning_formatter(train=False):
    def formatter(data):
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            text = f"""Your task is to convert the following search query into maximum 5 diverse aspects and perspectives that that cover all aspects of the original query. The aspects and perspectives should be non-overlapping and should not be redundant. The aspects and perspectives should cover all aspects that a comprehensive response to the original search query should cover.
            
            # your input:
                - query: the original search query
            # your output: Your output should be a valid json list of maximum 5 items enclosed in ```json ``` block that contains the following fields:
                - aspect: the aspect that covers a specific aspect of the original search query
                - query: the query that should be used to cover the specific aspect
                - reason: the reason why this aspect and query is important to cover in a comprehensive response to the original search query
        
            query: {user_prompt}
            output: ```json"""
            if train:
                text = text + f" {data['output'][i]}<eos>"
            texts.append(text)
        return texts
    return formatter

def get_query_planning_rag_formatter(num_contexts):
    def formatter(data):
        assert "plan" in data, 'plan should be in the dataset'
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            context = []
            plan = ""
            for p in data['plan'][i]:
                per_query_context = num_contexts // len(data['plan'][i])
                if 'aspect' not in p or 'reason' not in p:
                    print("Error in query")
                    continue
                context += p['context'][:per_query_context]
                plan += f"- aspect: {p['aspect']}\nreason: {p['reason']}\n"
            combined_context = "\n\n".join([_get_documtent_text(doc) for doc in context])
            if len(plan) == 0:
                plan = "No plan provided"
                print("Error in plan")
            if len(combined_context) == 0:
                combined_context = "No context provided"
                print("Error in context")
            text = f"""Your task is to generate a comprehensive and factual response to the given query. You can use the information provided in the context to generate a more comprehensive and factual response. Your response should cover the following aspects and perspectives that cover all aspects of the original query. You can use the following plan to generate a comprehensive response to the query.
        
            query: 
            {user_prompt}

            plan: To answer the query, you should cover the following aspects and perspectives:
            {plan}

            context: 
            {combined_context}

            response:"""
            texts.append(text)
        return texts
    return formatter

def get_query_planning_local_search_formatter(train=False):
    def formatter(data):
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            plan = ""
            for p in data['plan'][i]:
                if 'aspect' not in p or 'reason' not in p:
                    print("Error in query")
                    continue
                plan += f"- aspect: {p['aspect']}\nreason: {p['reason']}\n"
            if len(plan) == 0:
                plan = "No plan provided"
                print("Error in plan")
            neg_response = data['output_neg'][i]
            pos_response = data['output_pos'][i]
            text = f"""Your task is to improve the comprehensiveness and accuracy of the response generated for the query. To achieve this, provide a more detailed and factually accurate response, using the provided plan as a guide to ensure the response is both thorough and precise.

            query: {user_prompt}

            plan: To answer the query, you should cover the following aspects and perspectives:
            {plan}

            generated response: 
            {neg_response.strip()}
            
            improved response: 
            """
            if train:
                text = text + f" {pos_response.strip()}<eos>"
            texts.append(text)
        return texts
    return formatter

def get_query_planning_global_search_formatter(train=False, max_length=4096, tokenizer=None):
    def formatter(data):
        texts = []
        for i in range(len(data['query'])):
            user_prompt = data['query'][i]
            if train:
                neg_response = data['output_neg'][i]
                pos_response = data['output_pos'][i]
                pos_location = random.choice([0, 1])
                if pos_location == 0:
                    response_1 = pos_response
                    response_2 = neg_response
                    answer = 1
                else:
                    response_1 = neg_response
                    response_2 = pos_response
                    answer = 2
            else:
                response_1 = data['output_1'][i]
                response_2 = data['output_2'][i]
            text = f"""Your task is to choose the response that is more comprehensive and accurate between the two provided responses to the query. 

            query: {user_prompt}

            response 1: 
            {response_1.strip()}
            
            response 2:
            {response_2.strip()}

            selected output: """
            if train:
                text = text + f"{answer}<eos>"
                if tokenizer is not None:
                    tokens = tokenizer(text)['input_ids']
                    if len(tokens) > max_length:
                        print(f"Length of the text is {len(tokens)}")
                        continue
            texts.append(text)
        return texts
    return formatter